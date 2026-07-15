from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.interface_roi import (  # noqa: E402
    build_roi_subgraph,
    match_roi_pairs,
    rank_interface_rois,
)


def _face(node_id, area, *, surface="plane", radius=None):
    row = {
        "node_id": node_id,
        "entity_type": "face",
        "surface_type": surface,
        "area": area,
        "centroid": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 1.0],
        "joinable_node_index": int(node_id.split("_")[-1]),
        "joinable_entity_type_index": 0,
        "is_face": 1,
        "length": 0.0,
        "face_reversed": 0,
        "edge_reversed": 0,
    }
    if radius is not None:
        row["radius"] = radius
        row["axis_origin"] = [0.0, 0.0, 0.0]
        row["axis_direction"] = [0.0, 0.0, 1.0]
    return row


def _edge(node_id, faces, *, convexity="concave", curve="line", index=10):
    return {
        "node_id": node_id,
        "entity_type": "edge",
        "curve_type": curve,
        "adjacent_face_ids": list(faces),
        "convexity": convexity,
        "topology_feature_status": "success",
        "is_seam_edge": False,
        "joinable_node_index": index,
        "joinable_entity_type_index": 8,
        "is_face": 0,
        "length": 1.0,
        "face_reversed": 0,
        "edge_reversed": 0,
    }


def _graph(nodes):
    links = []
    for node in nodes:
        if node.get("entity_type") != "edge":
            continue
        for face_id in node.get("adjacent_face_ids") or []:
            links.append({"src": face_id, "dst": node["node_id"], "relation": "face_edge_adjacency"})
    return {
        "metadata": {"checkpoint_pair_normalization_extent": 10.0},
        "nodes": nodes,
        "edges": links,
    }


class InterfaceRoiTests(unittest.TestCase):
    def test_concave_bounded_local_face_outranks_repeated_large_planes(self):
        faces = [_face(f"face_{index}", 1000.0) for index in range(1, 4)]
        local = _face("face_4", 120.0)
        edges = [
            _edge(f"edge_{index}", ["face_4", "face_1"], index=10 + index)
            for index in range(1, 4)
        ]
        result = rank_interface_rois(_graph(faces + [local] + edges), maximum=4)
        self.assertIn(result["status"], {"success", "partial"})
        self.assertEqual(result["rois"][0]["seed_face_id"], "face_4")
        self.assertIn("bounded_channel_or_pocket", result["rois"][0]["interface_hints"])
        self.assertTrue(result["rois"][0]["review_required"])

    def test_stratified_frontier_retains_cylinder_family(self):
        planes = [_face(f"face_{index}", 100.0 + index) for index in range(1, 8)]
        cylinder = _face("face_8", 400.0, surface="cylinder", radius=5.0)
        edge = _edge("edge_1", ["face_8", "face_1"], curve="circle", index=20)
        result = rank_interface_rois(_graph(planes + [cylinder, edge]), maximum=4, per_family_minimum=1)
        self.assertTrue(any(row["surface_type"] == "cylinder" for row in result["rois"]))

    def test_roi_subgraph_has_no_dangling_links_and_contiguous_indices(self):
        face_a = _face("face_1", 10.0)
        face_b = _face("face_2", 20.0)
        face_c = _face("face_3", 30.0)
        edge_a = _edge("edge_1", ["face_1", "face_2"], index=10)
        edge_b = _edge("edge_2", ["face_2", "face_3"], index=11)
        graph = _graph([face_a, face_b, face_c, edge_a, edge_b])
        subgraph = build_roi_subgraph(
            graph,
            [{"seed_face_id": "face_1"}],
            neighborhood_hops=1,
            maximum_nodes=4,
        )
        ids = {node["node_id"] for node in subgraph["nodes"]}
        self.assertTrue(subgraph["edges"])
        self.assertTrue(all(link["src"] in ids and link["dst"] in ids for link in subgraph["edges"]))
        self.assertEqual(
            [node["joinable_node_index"] for node in subgraph["nodes"]],
            list(range(len(subgraph["nodes"]))),
        )

    def test_pair_matching_requires_compatible_surface_and_radius(self):
        fixed = {"rois": [{
            "roi_id": "a", "seed_face_id": "fa", "surface_type": "cylinder",
            "radius_mm": 10.0, "score": 0.9,
            "interface_hints": ["cylindrical_insert_or_bore"],
        }]}
        moving = {"rois": [
            {
                "roi_id": "b", "seed_face_id": "fb", "surface_type": "cylinder",
                "radius_mm": 10.2, "score": 0.8,
                "interface_hints": ["cylindrical_insert_or_bore"],
            },
            {
                "roi_id": "c", "seed_face_id": "fc", "surface_type": "cylinder",
                "radius_mm": 30.0, "score": 1.0,
                "interface_hints": ["cylindrical_insert_or_bore"],
            },
        ]}
        rows = match_roi_pairs(fixed, moving)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["moving_face_id"], "fb")
        self.assertTrue(rows[0]["review_required"])


if __name__ == "__main__":
    unittest.main()
