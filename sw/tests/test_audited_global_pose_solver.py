from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from global_pose_solver import solve_bounded_global_pose  # noqa: E402


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def _pool(source: str, target: str, *rows: tuple[str, np.ndarray, float]) -> dict:
    return {
        "source": source,
        "target": target,
        "candidates": [
            {"candidate_id": key, "T_rel": transform, "prior": prior}
            for key, transform, prior in rows
        ],
    }


class AuditedGlobalPoseSolverTests(unittest.TestCase):
    def test_triangle_uses_independent_cycle_to_choose_consistent_pose(self):
        result = solve_bounded_global_pose(
            ["a", "b", "c"],
            [
                _pool("a", "b", ("ab", _translation(10), 0.8)),
                _pool("b", "c", ("bc", _translation(0, 8), 0.8)),
                _pool(
                    "a", "c",
                    ("ac_wrong", _translation(25, 0), 0.9),
                    ("ac_right", _translation(10, 8), 0.6),
                ),
            ],
            max_candidates_per_pair=2,
            max_topologies=3,
            max_hypotheses=16,
        )
        self.assertEqual(result["status"], "review_required")
        self.assertFalse(result["accepted"])
        self.assertTrue(result["hypotheses"])
        best = result["hypotheses"][0]
        self.assertTrue(best["all_independent_cycles_consistent"])
        self.assertLess(best["max_translation_residual_mm"], 1e-5)
        self.assertLess(best["max_rotation_residual_degrees"], 1e-5)
        self.assertAlmostEqual(best["part_poses"]["c"][0][3], 10.0, places=5)
        self.assertAlmostEqual(best["part_poses"]["c"][1][3], 8.0, places=5)

    def test_tree_only_solution_is_explicitly_review_only(self):
        result = solve_bounded_global_pose(
            ["a", "b", "c"],
            [
                _pool("a", "b", ("ab", _translation(10), 1.0)),
                _pool("b", "c", ("bc", _translation(0, 8), 1.0)),
            ],
        )
        self.assertEqual(result["status"], "review_required")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["hypotheses"][0]["all_independent_cycles_consistent"])
        self.assertEqual(result["hypotheses"][0]["independent_cycle_count"], 0)

    def test_disconnected_input_has_no_pose_hypothesis(self):
        result = solve_bounded_global_pose(
            ["a", "b", "c"],
            [_pool("a", "b", ("ab", _translation(10), 1.0))],
        )
        self.assertEqual(result["status"], "insufficient_connectivity")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["hypotheses"])

    def test_exact_validator_failure_cannot_become_acceptance(self):
        result = solve_bounded_global_pose(
            ["a", "b"],
            [_pool("a", "b", ("ab", _translation(10), 1.0))],
            exact_validator=lambda _: {"status": "failed", "reason": "solid_penetration"},
            validate_top_n=1,
        )
        self.assertEqual(result["hypotheses"][0]["exact_validation"]["status"], "failed")
        self.assertFalse(result["accepted"])
        self.assertTrue(result["review_required"])

    def test_budget_is_shared_across_multiple_connection_topologies(self):
        result = solve_bounded_global_pose(
            ["a", "b", "c"],
            [
                _pool("a", "b", ("ab", _translation(2), 1.0)),
                _pool("b", "c", ("bc", _translation(2), 1.0)),
                _pool("a", "c", ("ac", _translation(4), 1.0)),
            ],
            max_topologies=3,
            max_hypotheses=3,
        )
        topology_ids = {
            tuple(tuple(pair) for pair in row["tree_pairs"])
            for row in result["hypotheses"]
        }
        self.assertEqual(len(topology_ids), 3)

    def test_candidate_combination_frontier_keeps_far_pose_alternative(self):
        result = solve_bounded_global_pose(
            ["a", "b", "c"],
            [
                _pool(
                    "a", "b",
                    ("ab_near", _translation(0), 10.0),
                    ("ab_far", _translation(100), 0.1),
                ),
                _pool(
                    "b", "c",
                    ("bc_near", _translation(0, 0), 10.0),
                    ("bc_other", _translation(0, 1), 9.0),
                ),
            ],
            max_candidates_per_pair=2,
            max_topologies=1,
            max_hypotheses=2,
        )
        candidate_ids = {
            row["candidate_id"]
            for hypothesis in result["hypotheses"]
            for row in hypothesis["tree_candidates"]
        }
        self.assertIn("ab_far", candidate_ids)


if __name__ == "__main__":
    unittest.main()
