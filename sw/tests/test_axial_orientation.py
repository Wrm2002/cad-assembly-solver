from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import trimesh


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from pose_search import (  # noqa: E402
    JointAxisSeed,
    JoinablePoseSearch,
    extract_key_slot_evidence,
    generate_axial_rotation_hypotheses,
)


def _cylinders(angles: list[float]) -> dict:
    return {
        "cylinders": [
            {
                "origin": [
                    20.0 * math.cos(math.radians(angle)),
                    20.0 * math.sin(math.radians(angle)),
                    0.0,
                ],
                "axis": [0.0, 0.0, 1.0],
                "radius": 2.0,
            }
            for angle in angles
        ]
    }


def _planes(angles: list[float]) -> dict:
    return {
        "planes": [
            {
                "position": [
                    30.0 * math.cos(math.radians(angle)),
                    30.0 * math.sin(math.radians(angle)),
                    0.0,
                ],
                "normal": [
                    math.cos(math.radians(angle)),
                    math.sin(math.radians(angle)),
                    0.0,
                ],
                "area": 100.0,
            }
            for angle in angles
        ]
    }


def _closed_slot_graph() -> dict:
    """A synthetic B-Rep topology fixture, not a named project case."""

    return {
        "metadata": {"edge_topology_features": {"available": True}},
        "nodes": [
            {
                "node_id": "face_wall_left", "entity_type": "face",
                "surface_type": "plane", "centroid": [10.0, -2.0, 0.0],
                "normal": [0.0, 1.0, 0.0], "area": 40.0,
            },
            {
                "node_id": "face_wall_right", "entity_type": "face",
                "surface_type": "plane", "centroid": [10.0, 2.0, 0.0],
                "normal": [0.0, -1.0, 0.0], "area": 40.0,
            },
            {
                "node_id": "face_bottom", "entity_type": "face",
                "surface_type": "plane", "centroid": [12.0, 0.0, 0.0],
                "normal": [-1.0, 0.0, 0.0], "area": 20.0,
            },
            {
                "node_id": "edge_left_bottom", "entity_type": "edge",
                "topology_feature_status": "success", "convexity": "concave",
                "adjacent_face_ids": ["face_wall_left", "face_bottom"],
            },
            {
                "node_id": "edge_right_bottom", "entity_type": "edge",
                "topology_feature_status": "success", "convexity": "concave",
                "adjacent_face_ids": ["face_wall_right", "face_bottom"],
            },
        ],
    }


class AxialOrientationTests(unittest.TestCase):
    def test_uniform_pattern_generates_geometry_symmetry_without_functional_claim(self):
        result = generate_axial_rotation_hypotheses(
            _cylinders([0.0, 90.0, 180.0, 270.0]),
            _cylinders([0.0, 90.0, 180.0, 270.0]),
            fixed_axis_origin=[0, 0, 0],
            fixed_axis_direction=[0, 0, 1],
            moving_axis_origin=[0, 0, 0],
            moving_axis_direction=[0, 0, 1],
        )
        rotations = {
            round(float(row["rotation_degrees"]))
            for row in result["rotation_hypotheses"]
        }
        self.assertTrue({0, 90, 180, -90}.issubset(rotations))
        self.assertFalse(result["functional_orientation_available"])
        for row in result["rotation_hypotheses"]:
            if row["evidence_kind"] == "uniform_circular_pattern_symmetry":
                self.assertTrue(row["geometry_symmetry_only"])

    def test_nonuniform_pattern_retains_180_as_feature_correspondence(self):
        fixed = _cylinders([0.0, 55.0, 180.0])
        moving = _cylinders([180.0, -125.0, 0.0])
        result = generate_axial_rotation_hypotheses(
            fixed,
            moving,
            fixed_axis_origin=[0, 0, 0],
            fixed_axis_direction=[0, 0, 1],
            moving_axis_origin=[0, 0, 0],
            moving_axis_direction=[0, 0, 1],
        )
        rows = result["rotation_hypotheses"]
        self.assertTrue(result["geometric_directional_orientation_available"])
        self.assertFalse(result["functional_orientation_available"])
        matching = [
            row for row in rows
            if row["evidence_kind"] == "circular_pattern_correspondence"
            and abs(abs(float(row["rotation_degrees"])) - 180.0) < 1e-6
        ]
        self.assertTrue(matching)
        self.assertAlmostEqual(matching[0]["matching_fraction"], 1.0)

    def test_missing_features_is_explicitly_not_semantic_evidence(self):
        result = generate_axial_rotation_hypotheses(
            {"cylinders": []},
            {"cylinders": []},
            fixed_axis_origin=[0, 0, 0],
            fixed_axis_direction=[0, 0, 1],
            moving_axis_origin=[0, 0, 0],
            moving_axis_direction=[0, 0, 1],
        )
        self.assertEqual(
            [row["rotation_degrees"] for row in result["rotation_hypotheses"]],
            [0.0],
        )
        self.assertFalse(result["functional_orientation_available"])
        self.assertEqual(result["key_slot_status"], {
            "fixed": "unknown", "moving": "unknown",
        })

    def test_single_periodic_interface_preserves_derived_180_candidate(self):
        result = generate_axial_rotation_hypotheses(
            {"cylinders": []},
            _cylinders([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]),
            fixed_axis_origin=[0, 0, 0],
            fixed_axis_direction=[0, 0, 1],
            moving_axis_origin=[0, 0, 0],
            moving_axis_direction=[0, 0, 1],
        )
        rows = result["rotation_hypotheses"]
        self.assertTrue(any(
            abs(abs(float(row["rotation_degrees"])) - 180.0) < 1e-6
            and row["evidence_kind"] == "single_interface_periodicity"
            and row["geometry_symmetry_only"]
            for row in rows
        ))
        self.assertFalse(result["functional_orientation_available"])

    def test_planar_witness_is_explicitly_weak_not_key_slot_evidence(self):
        result = generate_axial_rotation_hypotheses(
            _planes([0.0, 180.0]),
            _planes([90.0, -90.0]),
            fixed_axis_origin=[0, 0, 0],
            fixed_axis_direction=[0, 0, 1],
            moving_axis_origin=[0, 0, 0],
            moving_axis_direction=[0, 0, 1],
        )
        self.assertTrue(result["fixed_planar_witnesses"])
        self.assertTrue(result["moving_planar_witnesses"])
        self.assertFalse(result["functional_orientation_available"])
        self.assertTrue(any(
            row["evidence_kind"] == "weak_planar_directional_witness"
            for row in result["rotation_hypotheses"]
        ))

    def test_joinable_search_preserves_explicit_rotation_seeds(self):
        fixed = trimesh.creation.box(extents=[10.0, 10.0, 2.0])
        moving = trimesh.creation.box(extents=[4.0, 4.0, 2.0])
        searcher = JoinablePoseSearch(
            fixed, moving, sample_count=256, budget=4, seed=9
        )
        seed = JointAxisSeed(
            moving_origin=(0.0, 0.0, 0.0),
            moving_direction=(0.0, 0.0, 1.0),
            fixed_origin=(0.0, 0.0, 0.0),
            fixed_direction=(0.0, 0.0, 1.0),
            rotation_seed_degrees=(0.0, 180.0),
        )
        rows = searcher.search([seed], top_k=1)
        self.assertTrue(any(
            row.candidate_origin == "axis_alignment_seed"
            and abs(row.rotation_seed_degrees - 180.0) < 1e-6
            for row in rows
        ))

    def test_closed_concave_three_face_topology_detects_slot(self):
        result = extract_key_slot_evidence(
            _closed_slot_graph(), [0, 0, 0], [0, 0, 1]
        )
        self.assertEqual(result["status"], "detected")
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(set(candidate["wall_face_ids"]), {
            "face_wall_left", "face_wall_right",
        })
        self.assertEqual(candidate["bottom_face_id"], "face_bottom")
        self.assertEqual(candidate["evidence_count"], 5)

    def test_planar_fragments_without_audited_topology_remain_unknown(self):
        result = extract_key_slot_evidence(
            {"nodes": _closed_slot_graph()["nodes"][:3]}, [0, 0, 0], [0, 0, 1]
        )
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["topology_available"])

    def test_audited_topology_without_concave_bottom_is_not_a_slot(self):
        graph = _closed_slot_graph()
        for node in graph["nodes"]:
            if node.get("entity_type") == "edge":
                node["convexity"] = "convex"
        result = extract_key_slot_evidence(graph, [0, 0, 0], [0, 0, 1])
        self.assertEqual(result["status"], "not_detected")
        self.assertTrue(result["topology_available"])

    def test_closed_slot_does_not_match_an_incompatible_joint_axis(self):
        result = extract_key_slot_evidence(
            _closed_slot_graph(), [0, 0, 0], [1, 0, 0]
        )
        self.assertEqual(result["status"], "not_detected")


if __name__ == "__main__":
    unittest.main()
