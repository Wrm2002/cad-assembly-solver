import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from known_group_assembly import (
    _rigid_attachment_cluster,
    _set_placement_with_rigid_dependents,
)
from pose_search import matrix_to_placement, placement_to_matrix


class RigidDependentPropagationTests(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "selected": [
                {
                    "parts": ["shaft", "key"],
                    "relation_types": ["planar_mate"],
                },
                {
                    "parts": ["shaft", "flange"],
                    "relation_types": ["coaxial"],
                },
            ]
        }

    def test_cluster_crosses_planar_attachment_but_not_axial_edge(self):
        cluster = _rigid_attachment_cluster("shaft", self.graph)
        self.assertEqual(set(cluster), {"shaft", "key"})
        self.assertNotIn("flange", cluster)

    def test_world_delta_preserves_child_relative_transform(self):
        placements = {
            "shaft": matrix_to_placement(np.eye(4)),
            "key": matrix_to_placement(np.array([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ])),
            "flange": matrix_to_placement(np.array([
                [1.0, 0.0, 0.0, 50.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])),
        }
        old_relative = (
            np.linalg.inv(placement_to_matrix(placements["shaft"]))
            @ placement_to_matrix(placements["key"])
        )
        flange_before = placement_to_matrix(placements["flange"]).copy()
        new_root = matrix_to_placement(np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]))

        moved = _set_placement_with_rigid_dependents(
            placements,
            "shaft",
            new_root,
            self.graph,
        )

        new_relative = (
            np.linalg.inv(placement_to_matrix(placements["shaft"]))
            @ placement_to_matrix(placements["key"])
        )
        self.assertEqual(set(moved), {"shaft", "key"})
        self.assertTrue(np.allclose(new_relative, old_relative, atol=1e-9))
        self.assertTrue(np.allclose(
            placement_to_matrix(placements["flange"]),
            flange_before,
            atol=1e-9,
        ))

    def test_blocked_planar_neighbor_is_not_moved(self):
        graph = {
            "selected": [{
                "parts": ["root", "stationary"],
                "relation_types": ["pocket_mate"],
            }]
        }
        placements = {
            "root": matrix_to_placement(np.eye(4)),
            "stationary": matrix_to_placement(np.eye(4)),
        }
        moved = _set_placement_with_rigid_dependents(
            placements,
            "root",
            matrix_to_placement(np.array([
                [1.0, 0.0, 0.0, 5.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])),
            graph,
            blocked={"stationary"},
        )
        self.assertEqual(moved, ["root"])
        self.assertTrue(np.allclose(
            placement_to_matrix(placements["stationary"]), np.eye(4)
        ))


if __name__ == "__main__":
    unittest.main()
