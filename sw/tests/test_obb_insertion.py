from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.obb_insertion import enumerate_axis_role_frames


def _obb(dimensions):
    return {
        "center": [0.0, 0.0, 0.0],
        "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "dimensions": list(dimensions),
    }


class OBBInsertionTests(unittest.TestCase):
    def test_all_frames_are_proper_and_review_only(self):
        rows = enumerate_axis_role_frames(
            _obb((387.87, 29.36, 426.05)),
            _obb((2.77, 133.35, 31.25)),
            maximum=24,
        )
        self.assertEqual(len(rows), 24)
        for row in rows:
            matrix = np.asarray(row["rotation_matrix"], dtype=float)
            self.assertTrue(np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6))
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=6)
            self.assertTrue(row["review_required"])
            self.assertFalse(row["can_auto_accept"])

    def test_dimm_upright_axis_role_is_retained(self):
        rows = enumerate_axis_role_frames(
            _obb((387.87, 29.36, 426.05)),
            _obb((2.77, 133.35, 31.25)),
            maximum=24,
        )
        self.assertTrue(any(
            row["axis_mapping"][2] == 1
            and row["axis_mapping"][1] in {0, 2}
            for row in rows
        ))

    def test_cpu_flat_axis_role_is_retained(self):
        rows = enumerate_axis_role_frames(
            _obb((387.87, 29.36, 426.05)),
            _obb((72.02, 6.17, 75.42)),
            maximum=24,
        )
        self.assertTrue(any(row["axis_mapping"][1] == 1 for row in rows))

    def test_psu_depth_axis_role_is_retained(self):
        rows = enumerate_axis_role_frames(
            _obb((446.15, 87.30, 796.44)),
            _obb((225.09, 76.66, 42.44)),
            maximum=24,
        )
        self.assertTrue(any(
            row["axis_mapping"][1] == 1
            and row["axis_mapping"][0] == 2
            for row in rows
        ))

    def test_rotated_obb_frames_remain_unique_and_proper(self):
        angle = np.deg2rad(31.0)
        rotated = _obb((12.0, 7.0, 3.0))
        rotated["axes"] = [
            [float(np.cos(angle)), float(np.sin(angle)), 0.0],
            [-float(np.sin(angle)), float(np.cos(angle)), 0.0],
            [0.0, 0.0, 1.0],
        ]

        rows = enumerate_axis_role_frames(
            _obb((20.0, 10.0, 5.0)), rotated, maximum=24
        )

        matrices = [np.asarray(row["rotation_matrix"]) for row in rows]
        self.assertEqual(len(matrices), 24)
        self.assertEqual(
            len({tuple(np.round(matrix, 8).ravel()) for matrix in matrices}),
            24,
        )
        for matrix in matrices:
            self.assertTrue(np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6))
            self.assertAlmostEqual(np.linalg.det(matrix), 1.0, places=6)

    def test_axis_mapping_contract_holds_for_rotated_obb(self):
        angle = np.deg2rad(23.0)
        fixed = _obb((20.0, 10.0, 5.0))
        fixed["axes"] = [
            [float(np.cos(angle)), 0.0, float(np.sin(angle))],
            [0.0, 1.0, 0.0],
            [-float(np.sin(angle)), 0.0, float(np.cos(angle))],
        ]
        moving = _obb((12.0, 7.0, 3.0))
        moving["axes"] = [
            [float(np.cos(angle)), float(np.sin(angle)), 0.0],
            [-float(np.sin(angle)), float(np.cos(angle)), 0.0],
            [0.0, 0.0, 1.0],
        ]

        rows = enumerate_axis_role_frames(fixed, moving, maximum=24)

        for row in rows:
            rotation = np.asarray(row["rotation_matrix"], dtype=float)
            for moving_axis, fixed_axis in enumerate(row["axis_mapping"]):
                actual = rotation @ np.asarray(moving["axes"][moving_axis])
                expected = (
                    row["axis_signs"][moving_axis]
                    * np.asarray(fixed["axes"][fixed_axis])
                )
                self.assertTrue(np.allclose(actual, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
