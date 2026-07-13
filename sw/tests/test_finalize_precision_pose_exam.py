from __future__ import annotations

import sys
import unittest
from pathlib import Path


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from finalize_precision_pose_exam import select_precision_hypothesis


class FinalizePrecisionPoseExamTests(unittest.TestCase):
    def test_valid_beats_review_and_failed(self):
        hypotheses = [{}, {}, {}]
        precision = [
            {"precision_status": "review", "axis_distance_mm": 0.0},
            {"precision_status": "valid", "axis_distance_mm": 0.1},
            {"precision_status": "failed", "axis_distance_mm": 0.0},
        ]
        self.assertEqual(select_precision_hypothesis(hypotheses, precision), 1)

    def test_review_prefers_more_prismatic_closure_then_fewer_free_dofs(self):
        hypotheses = [
            {
                "exact_validation": {"status": "valid"},
                "factor_residuals": [{"manifold_type": "axis_coincidence"}],
                "unresolved_manifold_dofs": ["tz", "rz", "tx"],
            },
            {
                "exact_validation": {"status": "valid"},
                "factor_residuals": [
                    {"manifold_type": "compound_prismatic_insertion_rigid"},
                    {"manifold_type": "compound_prismatic_insertion_rigid"},
                ],
                "unresolved_manifold_dofs": ["tz", "rz"],
            },
        ]
        precision = [
            {"precision_status": "review", "axis_distance_mm": 0.01},
            {"precision_status": "review", "axis_distance_mm": 0.05},
        ]
        self.assertEqual(select_precision_hypothesis(hypotheses, precision), 1)


if __name__ == "__main__":
    unittest.main()
