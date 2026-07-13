from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from pose_search import (  # noqa: E402
    JointAxisSeed,
    JoinablePoseSearch,
    matrix_to_placement,
    placement_to_matrix,
)
from pose_search.transforms import (  # noqa: E402
    joint_parameter_matrix,
    transform_points,
)


class JoinablePoseTransformTests(unittest.TestCase):
    def test_axis_alignment_maps_both_origin_and_direction(self):
        matrix = joint_parameter_matrix(
            moving_origin=[1.0, 2.0, 3.0],
            moving_direction=[1.0, 0.0, 0.0],
            fixed_origin=[10.0, -4.0, 7.0],
            fixed_direction=[0.0, 0.0, 1.0],
            offset=5.0,
            rotation_degrees=37.0,
            axis_flip=False,
        )
        mapped_origin = transform_points(
            np.array([[1.0, 2.0, 3.0]]), matrix
        )[0]
        mapped_tip = transform_points(
            np.array([[2.0, 2.0, 3.0]]), matrix
        )[0]
        self.assertTrue(np.allclose(mapped_origin, [10.0, -4.0, 12.0]))
        self.assertTrue(
            np.allclose(mapped_tip - mapped_origin, [0.0, 0.0, 1.0])
        )
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0, places=7)

    def test_axis_flip_remains_a_proper_rigid_transform(self):
        matrix = joint_parameter_matrix(
            [0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 1],
            offset=0.0,
            rotation_degrees=0.0,
            axis_flip=True,
        )
        self.assertAlmostEqual(np.linalg.det(matrix[:3, :3]), 1.0, places=7)
        mapped = matrix[:3, :3] @ np.array([0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(mapped, [0.0, 0.0, -1.0]))

    def test_matrix_placement_round_trip(self):
        matrix = joint_parameter_matrix(
            [2, -1, 3], [1, 1, 0], [7, 4, -2], [0, 0, 1],
            offset=9.0,
            rotation_degrees=83.0,
            axis_flip=False,
        )
        placement = matrix_to_placement(matrix)
        reconstructed = placement_to_matrix(placement)
        self.assertTrue(np.allclose(matrix, reconstructed, atol=1e-8))


class JoinablePoseSearchTests(unittest.TestCase):
    def test_search_is_bounded_and_returns_rigid_pose(self):
        fixed = trimesh.creation.box(extents=[10.0, 10.0, 2.0])
        moving = trimesh.creation.box(extents=[4.0, 4.0, 2.0])
        searcher = JoinablePoseSearch(
            fixed,
            moving,
            sample_count=256,
            budget=8,
            seed=7,
        )
        seed = JointAxisSeed(
            moving_origin=(0.0, 0.0, 0.0),
            moving_direction=(0.0, 0.0, 1.0),
            fixed_origin=(0.0, 0.0, 0.0),
            fixed_direction=(0.0, 0.0, 1.0),
        )
        results = searcher.search([seed], top_k=1)
        self.assertGreaterEqual(len(results), 2)
        for result in results:
            self.assertLessEqual(abs(result.offset), result.offset_limit + 1e-8)
            self.assertAlmostEqual(result.transform_determinant, 1.0, places=6)
            self.assertTrue(np.isfinite(result.evaluation.cost))
        self.assertLessEqual(
            results[0].evaluation.cost, results[1].evaluation.cost
        )


if __name__ == "__main__":
    unittest.main()
