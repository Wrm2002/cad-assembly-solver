from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.manifold_solver import (  # noqa: E402
    _bounded_assignments,
    _closure_ranked_assignments,
    _normalise_pools,
    _parse_factor,
    factor_error,
    solve_manifold_pose_graph,
)


def _pose(translation=(0, 0, 0), axis=(0, 0, 1), degrees=0):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(
        np.asarray(axis, dtype=float) * math.radians(degrees)
    ).as_matrix()
    result[:3, 3] = translation
    return result


def _candidate(source, target, kind, free, frame_a=None, frame_b=None, confidence=1.0, key="h"):
    frame_a = np.eye(4) if frame_a is None else frame_a
    frame_b = np.eye(4) if frame_b is None else frame_b
    return {
        "candidate_id": key,
        "source": source,
        "target": target,
        "manifold_type": kind,
        "frame_a": frame_a.tolist(),
        "frame_b": frame_b.tolist(),
        "free_dof_mask": free,
        "initial_pose_b_in_a": (frame_a @ np.linalg.inv(frame_b)).tolist(),
        "confidence": confidence,
    }


class ManifoldPoseSolverTests(unittest.TestCase):
    def test_axis_factor_does_not_penalise_slide_or_spin(self):
        factor = _parse_factor(_candidate(
            "a", "b", "axis_coincidence", [0, 0, 1, 0, 0, 1]
        ), 0)
        poses = {"a": np.eye(4), "b": _pose((0, 0, 25), degrees=73)}
        np.testing.assert_allclose(factor_error(factor, poses), np.zeros(6), atol=1e-7)

    def test_plane_factor_does_not_penalise_in_plane_se2(self):
        factor = _parse_factor(_candidate(
            "a", "b", "plane_coincidence", [1, 1, 0, 0, 0, 1]
        ), 0)
        poses = {"a": np.eye(4), "b": _pose((12, -9, 0), degrees=33)}
        np.testing.assert_allclose(factor_error(factor, poses), np.zeros(6), atol=1e-7)

    def test_joint_solver_uses_cycle_to_reject_high_prior_inconsistent_choice(self):
        ab = _candidate("a", "b", "frame", [0] * 6, frame_b=_pose((-10, 0, 0)), key="ab")
        bc = _candidate("b", "c", "frame", [0] * 6, frame_b=_pose((0, -8, 0)), key="bc")
        ac_wrong = _candidate("a", "c", "frame", [0] * 6, frame_b=_pose((-25, 0, 0)), confidence=2, key="wrong")
        ac_right = _candidate("a", "c", "frame", [0] * 6, frame_b=_pose((-10, -8, 0)), confidence=0.5, key="right")
        result = solve_manifold_pose_graph(
            ["a", "b", "c"],
            [ab, bc, ac_wrong, ac_right],
            max_candidates_per_pair=2,
            max_topologies=3,
            max_hypotheses=12,
        )
        self.assertTrue(result["hypotheses"])
        best = result["hypotheses"][0]
        self.assertLess(best["optimizer"]["cost"], 1e-8)
        self.assertIn("right", best["cycle_candidate_ids"] + best["tree_candidate_ids"])

    def test_inconsistent_non_tree_edge_is_audited_but_not_activated(self):
        result = solve_manifold_pose_graph(
            ["a", "b", "c"],
            [
                _candidate("a", "b", "frame", [0] * 6, frame_b=_pose((-10, 0, 0)), key="ab"),
                _candidate("b", "c", "frame", [0] * 6, frame_b=_pose((0, -8, 0)), key="bc"),
                _candidate("a", "c", "frame", [0] * 6, frame_b=_pose((-100, 0, 0)), key="wrong"),
            ],
            max_topologies=1,
            max_hypotheses=1,
        )
        best = result["hypotheses"][0]
        self.assertEqual(best["cycle_candidate_ids"], [])
        self.assertFalse(best["cycle_residuals_before_joint_optimization"][0]["consistent"])

    def test_disconnected_frontier_is_explicit(self):
        result = solve_manifold_pose_graph(
            ["a", "b", "c"],
            [_candidate("a", "b", "axis", [0, 0, 1, 0, 0, 1])],
        )
        self.assertEqual(result["status"], "insufficient_connectivity")
        self.assertFalse(result["accepted"])

    def test_large_cartesian_frontier_is_bounded_and_mixed(self):
        lists = []
        for edge_index in range(3):
            values = []
            for candidate_index in range(40):
                row = _candidate(
                    f"p{edge_index}", f"p{edge_index + 1}", "axis",
                    [0, 0, 1, 0, 0, 1],
                    confidence=1.0 / (candidate_index + 1),
                    key=f"e{edge_index}c{candidate_index}",
                )
                row["initial_pose_b_in_a"] = _pose((candidate_index, 0, 0)).tolist()
                values.append(_parse_factor(row, candidate_index))
            lists.append(values)
        rows = _bounded_assignments(lists, 120)
        self.assertEqual(len(rows), 120)
        self.assertTrue(any(sum(f.candidate_id.endswith("0") for f in row) < 3 for row in rows))

    def test_small_symmetry_product_is_ranked_by_strong_cycle_closure(self):
        rows = [
            _candidate("a", "b", "frame", [0] * 6,
                       frame_b=_pose((-10, 0, 0)), key="ab"),
            _candidate("b", "c", "frame", [0] * 6,
                       frame_b=_pose((0, -50, 0)), confidence=2.0, key="bc_wrong"),
            _candidate("b", "c", "frame", [0] * 6,
                       frame_b=_pose((0, -8, 0)), confidence=0.5, key="bc_right"),
            _candidate("a", "c", "frame", [0] * 6,
                       frame_b=_pose((-10, -8, 0)), key="ac_cycle"),
        ]
        pools = _normalise_pools(["a", "b", "c"], rows, 4)
        topology = (("a", "b"), ("b", "c"))
        selected = _closure_ranked_assignments(
            [pools[key] for key in topology],
            1,
            parts=["a", "b", "c"],
            anchor="a",
            pools=pools,
            topology=topology,
            translation_scale_mm=2.0,
            rotation_scale_degrees=5.0,
            enumeration_limit=100,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(
            [factor.candidate_id for factor in selected[0]],
            ["ab", "bc_right"],
        )


if __name__ == "__main__":
    unittest.main()
