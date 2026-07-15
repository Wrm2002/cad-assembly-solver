from __future__ import annotations

import sys
import json
import tempfile
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
    load_joinable_pair_pose_candidates,
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

    def test_no_search_report_exposes_manifold_initial_as_proposal(self):
        payload = {
            "part_a_fixed": "a.step",
            "part_b_moving": "b.step",
            "pose_search": {"enabled": False, "results": []},
            "joint_hypotheses": {
                "rows": [{
                    "initial_pose_b_in_a": np.eye(4).tolist(),
                    "confidence": 0.8,
                }]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "joinable_e2e_result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_joinable_pair_pose_candidates(path, limit=2)
        self.assertEqual(len(rows), 1)
        self.assertIn("proposal_only=true", rows[0].source)
        self.assertAlmostEqual(rows[0].score or 0.0, 0.2)


if __name__ == "__main__":
    unittest.main()
