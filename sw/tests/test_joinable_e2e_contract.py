from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "sw"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cad_assembly_agent.tools.joinable_interface_predictor.pretrained_joinable_predictor import (  # noqa: E402
    make_joint_graph,
)
from joinable_e2e import joint_axis_seed  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
