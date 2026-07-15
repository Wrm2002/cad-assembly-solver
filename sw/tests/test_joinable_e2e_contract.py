from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "sw"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    make_joint_graph,
)
from joinable_e2e import (  # noqa: E402
    build_pose_candidate_frontier,
    joint_axis_seed,
    prepare_roi_inference_graphs,
    roi_pose_candidates,
    run_pipeline,
)


def _roi_graph(prefix: str, face_count: int = 4) -> dict:
    faces = []
    for index in range(face_count):
        faces.append({
            "node_id": f"{prefix}_face_{index}",
            "entity_type": "face",
            "surface_type": "plane",
            "occt_topology_index": index + 1,
            "joinable_entity_type": "PlaneSurfaceType",
            "geometry_signature": f"{prefix}_face_signature_{index}",
            "area": 10.0 + index,
            "centroid": [float(index), 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
            "joinable_node_index": index,
        })
    edges = []
    links = []
    for index in range(1, face_count):
        edge_id = f"{prefix}_edge_{index}"
        incident = [faces[0]["node_id"], faces[index]["node_id"]]
        edges.append({
            "node_id": edge_id,
            "entity_type": "edge",
            "curve_type": "line",
            "occt_topology_index": index,
            "joinable_entity_type": "Line3DCurveType",
            "geometry_signature": f"{prefix}_edge_signature_{index}",
            "adjacent_face_ids": incident,
            "convexity": "concave",
            "topology_feature_status": "success",
            "is_seam_edge": False,
            "joinable_node_index": face_count + index - 1,
        })
        links.extend(
            {"src": face_id, "dst": edge_id}
            for face_id in incident
        )
    return {
        "metadata": {"checkpoint_pair_normalization_extent": 10.0},
        "nodes": faces + edges,
        "edges": links,
    }


class JoinableE2EContractTests(unittest.TestCase):
    def test_joint_graph_offsets_second_body_nodes(self):
        graph = make_joint_graph(2, 3)
        self.assertEqual(graph.edge_index.shape[1], 6)
        self.assertTrue(bool((graph.edge_index[0] < 2).all()))
        self.assertTrue(bool((graph.edge_index[1] >= 2).all()))
        self.assertTrue(bool((graph.edge_index[1] < 5).all()))

    def test_pose_seed_requires_axes_on_both_bodies(self):
        candidate = {
            "rank": 1,
            "logit": 3.0,
            "node_a": {
                "entity_id": "face_a",
                "axis_origin": [0, 0, 0],
                "axis_direction": [0, 0, 1],
            },
            "node_b": {
                "entity_id": "face_b",
                "centroid": [1, 2, 3],
                "normal": [1, 0, 0],
            },
        }
        seed = joint_axis_seed(candidate)
        self.assertIsNotNone(seed)
        assert seed is not None
        self.assertEqual(seed.fixed_direction, (0.0, 0.0, 1.0))
        self.assertEqual(seed.moving_direction, (1.0, 0.0, 0.0))
        candidate["node_b"].pop("normal")
        self.assertIsNone(joint_axis_seed(candidate))

    def test_roi_auto_preserves_small_graph_objects_and_skips_ranking(self):
        graph_a = _roi_graph("a", face_count=2)
        graph_b = _roi_graph("b", face_count=2)
        with patch(
            "joinable_e2e.rank_interface_rois",
            side_effect=AssertionError("small graph must not be ranked"),
        ):
            inference_a, inference_b, audit = prepare_roi_inference_graphs(
                graph_a, graph_b
            )

        self.assertIs(inference_a, graph_a)
        self.assertIs(inference_b, graph_b)
        self.assertEqual(audit["status"], "not_applied")
        self.assertEqual(audit["cartesian_reduction_ratio"], 1.0)
        self.assertFalse(audit["can_auto_accept"])

    def test_roi_auto_bounds_large_pair_and_emits_review_only_proposals(self):
        graph_a = _roi_graph("a")
        graph_b = _roi_graph("b")
        inference_a, inference_b, audit = prepare_roi_inference_graphs(
            graph_a,
            graph_b,
            combined_node_limit=10,
            cartesian_limit=1_000,
            maximum_faces=2,
            maximum_nodes=5,
            pair_proposal_limit=3,
        )

        self.assertEqual(audit["status"], "applied")
        self.assertLessEqual(len(inference_a["nodes"]), 5)
        self.assertLessEqual(len(inference_b["nodes"]), 5)
        self.assertLess(
            audit["inference_graph"]["cartesian_candidate_count"],
            audit["source_graph"]["cartesian_candidate_count"],
        )
        self.assertTrue(audit["pair_proposals"])
        self.assertTrue(all(
            row["review_required"] for row in audit["pair_proposals"]
        ))
        self.assertFalse(audit["uses_part_names_or_case_ids"])
        self.assertFalse(audit["can_auto_accept"])
        self.assertTrue(all(
            "source_joinable_node_index" in node
            for node in inference_a["nodes"]
        ))

    def test_roi_auto_keeps_small_side_full_in_mixed_size_pair(self):
        graph_a = _roi_graph("large")
        graph_b = _roi_graph("small", face_count=2)
        inference_a, inference_b, audit = prepare_roi_inference_graphs(
            graph_a,
            graph_b,
            combined_node_limit=8,
            cartesian_limit=1_000,
            maximum_faces=2,
            maximum_nodes=5,
        )

        self.assertIsNot(inference_a, graph_a)
        self.assertIs(inference_b, graph_b)
        self.assertEqual(audit["part_a"]["inference_scope"], "roi_subgraph")
        self.assertEqual(audit["part_b"]["inference_scope"], "full_graph")

    def test_pipeline_persists_roi_audit_without_changing_acceptance(self):
        graph_a = _roi_graph("a")
        graph_b = _roi_graph("b")

        def fake_inference(left, right, **_kwargs):
            return {
                "checkpoint": "fake",
                "device": "cpu",
                "input_features": "fake",
                "pair_scale": 1.0,
                "total_candidates": len(left["nodes"]) * len(right["nodes"]),
                "top_k": 0,
                "candidates": [],
            }

        with tempfile.TemporaryDirectory() as temp, patch(
            "joinable_e2e.extract_brep_graphs",
            return_value=(graph_a, graph_b),
        ), patch(
            "joinable_e2e.run_gnn_inference",
            side_effect=fake_inference,
        ), patch(
            "joinable_e2e.attach_axial_orientation_hypotheses",
            return_value=([], []),
        ) as orientation:
            output = run_pipeline(
                Path("part_a.step"),
                Path("part_b.step"),
                output_dir=Path(temp),
                run_search=False,
                roi_combined_node_limit=10,
                roi_cartesian_limit=1_000,
                roi_maximum_faces=2,
                roi_maximum_nodes=5,
            )

        self.assertLessEqual(len(orientation.call_args.args[0]["nodes"]), 5)
        self.assertLessEqual(len(orientation.call_args.args[1]["nodes"]), 5)
        self.assertEqual(output["interface_roi"]["status"], "applied")
        self.assertFalse(output["interface_roi"]["can_auto_accept"])
        self.assertFalse(output["acceptance_boundary"]["roi_can_auto_accept"])
        self.assertEqual(
            output["gnn_inference"]["graph_scope"]["part_a"],
            "roi_subgraph",
        )

    def test_roi_pose_frontier_promotes_interface_family_diversity(self):
        graph_a = _roi_graph("a", face_count=2)
        graph_b = _roi_graph("b", face_count=2)
        for graph in (graph_a, graph_b):
            cylinder = graph["nodes"][1]
            cylinder.update({
                "surface_type": "cylinder",
                "joinable_entity_type": "CylinderSurfaceType",
                "axis_origin": [0.0, 0.0, 0.0],
                "axis_direction": [0.0, 0.0, 1.0],
                "radius": 2.0,
            })
        audit = {
            "status": "applied",
            "pair_proposals": [
                {
                    "fixed_face_id": "a_face_1",
                    "moving_face_id": "b_face_1",
                    "score": 0.9,
                    "dimension_compatibility": 1.0,
                    "shared_interface_hints": ["cylindrical_insert_or_bore"],
                },
                {
                    "fixed_face_id": "a_face_0",
                    "moving_face_id": "b_face_0",
                    "score": 0.8,
                    "dimension_compatibility": 1.0,
                    "shared_interface_hints": ["planar_seating"],
                },
            ],
        }
        geometry = roi_pose_candidates(graph_a, graph_b, audit)
        learned = [{
            "rank": 1,
            "node_a": {
                "entity_id": "learned_a",
                "geometry_type": "cone",
                "axis_origin": [0.0, 0.0, 0.0],
                "axis_direction": [1.0, 0.0, 0.0],
            },
            "node_b": {
                "entity_id": "learned_b",
                "geometry_type": "cylinder",
                "axis_origin": [0.0, 0.0, 0.0],
                "axis_direction": [1.0, 0.0, 0.0],
            },
            "logit": 3.0,
            "probability": 0.5,
        }]

        frontier, frontier_audit = build_pose_candidate_frontier(
            learned, geometry
        )

        self.assertEqual(
            [row["proposal_source"] for row in frontier[:3]],
            ["joinable_gnn", "geometry_roi", "geometry_roi"],
        )
        self.assertEqual(
            [row["pose_candidate_family"] for row in frontier[:3]],
            ["axial", "axial", "planar"],
        )
        self.assertEqual(frontier_audit["family_coverage"], ["axial", "planar"])
        self.assertFalse(frontier_audit["can_auto_accept"])


if __name__ == "__main__":
    unittest.main()
