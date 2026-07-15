from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from visual_joinable_pose_bridge import visual_analytic_pool  # noqa: E402


def _row(candidate_id: str, geometry: float = 0.8, semantic: float = 1.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "R": np.eye(3).tolist(),
        "t_mm": [0.0, 0.0, 0.0],
        "geometry_score": geometry,
        "semantic_region_score": semantic,
    }


class VisualJoinablePoseBridgeTests(unittest.TestCase):
    def test_visual_bonus_is_bounded_and_never_auto_accepts(self):
        pool, _ = visual_analytic_pool("a", "b", [_row("c")])
        candidate = pool["candidates"][0]
        self.assertAlmostEqual(candidate["prior"], 0.85)
        self.assertFalse(candidate["can_auto_accept"])
        self.assertTrue(candidate["proposal_only"])

    def test_prior_collision_rejection_is_not_resurrected(self):
        pool, audit = visual_analytic_pool(
            "a",
            "b",
            [_row("bad"), _row("unchecked", geometry=0.7)],
            prior_validations={
                "bad": {
                    "decision": "rejected_collision",
                    "collision_result": "collision_detected",
                }
            },
        )
        self.assertEqual([row["candidate_id"] for row in pool["candidates"]], ["unchecked"])
        self.assertEqual(audit["excluded"][0]["candidate_id"], "bad")


if __name__ == "__main__":
    unittest.main()
