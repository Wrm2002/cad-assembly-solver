from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from pose_search import (  # noqa: E402
    PairPoseSeed,
    compose_group_pose,
    compose_group_pose_hypotheses,
)


def seed(a: str, b: str, translation: list[float]) -> PairPoseSeed:
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return PairPoseSeed(
        part_a=a,
        part_b=b,
        transform_b_to_a=tuple(tuple(row) for row in matrix.tolist()),
        source=f"{a}-{b}",
    )


class GroupPoseCompositionTests(unittest.TestCase):
    def test_chain_composes_in_global_reference(self):
        result = compose_group_pose(
            ["a", "b", "c"],
            [seed("a", "b", [10, 0, 0]), seed("b", "c", [0, 5, 0])],
            reference_part="a",
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["placements"]["b"]["translate"], [10.0, 0.0, 0.0])
        self.assertEqual(result["placements"]["c"]["translate"], [10.0, 5.0, 0.0])

    def test_inconsistent_cycle_routes_to_review(self):
        result = compose_group_pose(
            ["a", "b", "c"],
            [
                seed("a", "b", [10, 0, 0]),
                seed("b", "c", [0, 5, 0]),
                seed("a", "c", [100, 100, 0]),
            ],
            reference_part="a",
        )
        self.assertEqual(result["status"], "inconsistent")
        self.assertTrue(result["review_required"])
        self.assertTrue(result["inconsistent_cycles"])

    def test_alternative_pair_poses_create_bounded_hypotheses(self):
        hypotheses = compose_group_pose_hypotheses(
            ["a", "b", "c"],
            [
                seed("a", "b", [10, 0, 0]),
                seed("a", "b", [-10, 0, 0]),
                seed("a", "c", [0, 5, 0]),
                seed("a", "c", [0, -5, 0]),
            ],
            maximum_candidates_per_pair=2,
            maximum_combinations=3,
        )
        self.assertEqual(len(hypotheses), 3)
        self.assertTrue(all(row["status"] == "complete" for row in hypotheses))


if __name__ == "__main__":
    unittest.main()
