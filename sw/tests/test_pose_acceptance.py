from __future__ import annotations

import sys
import unittest
from pathlib import Path


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.pose_acceptance import contact_supported_exact_pose  # noqa: E402


class PoseAcceptanceTests(unittest.TestCase):
    def _row(self, exact: str, gaps: list[float]):
        return {
            "exact_validation": {"status": exact},
            "geometry_residual_audit": {
                "pair_scores": [
                    {"selected_constraint_edge": True, "contact_gap_normalized": gap}
                    for gap in gaps
                ]
            },
        }

    def test_exact_noncollision_with_supported_contact_is_valid(self):
        self.assertEqual(
            contact_supported_exact_pose(self._row("valid", [0.008, 0.02]))["status"],
            "valid",
        )

    def test_separated_but_collision_free_pose_is_review(self):
        result = contact_supported_exact_pose(self._row("valid", [0.01, 0.35]))
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["reason"], "occt_valid_but_selected_edges_are_separated")

    def test_collision_failure_stays_failed(self):
        self.assertEqual(
            contact_supported_exact_pose(self._row("failed", [0.0]))["status"],
            "failed",
        )


if __name__ == "__main__":
    unittest.main()
