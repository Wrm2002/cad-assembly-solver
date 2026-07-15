from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from global_pose_solver.joinable_adapter import load_joinable_pose_pool  # noqa: E402


class JoinableGlobalPoseAdapterTests(unittest.TestCase):
    def test_completed_exact_boolean_with_collision_is_excluded(self):
        transform = np.eye(4).tolist()
        payload = {
            "pose_search": {"results": [
                {
                    "transform": transform,
                    "exact_collision": {
                        "status": "success",
                        "collisions": [{"parts": ["a", "b"]}],
                    },
                },
                {
                    "transform": transform,
                    "exact_collision": {"status": "not_checked"},
                },
            ]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            pool, audit = load_joinable_pose_pool("a", "b", path)
        self.assertEqual(len(pool["candidates"]), 1)
        self.assertEqual(audit["excluded"][0]["reason"], "pair_exact_collision")

    def test_collision_free_exact_result_is_retained(self):
        payload = {
            "pose_search": {"results": [{
                "transform": np.eye(4).tolist(),
                "exact_collision": {"status": "success", "collisions": []},
            }]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            pool, _ = load_joinable_pose_pool("a", "b", path)
        self.assertEqual(len(pool["candidates"]), 1)
        self.assertEqual(pool["candidates"][0]["pair_exact_status"], "success")

    def test_frontier_keeps_a_geometrically_distinct_pose(self):
        transforms = []
        for offset in (0.0, 1.0, 50.0):
            transform = np.eye(4)
            transform[0, 3] = offset
            transforms.append(transform.tolist())
        payload = {
            "pose_search": {"results": [
                {"transform": transforms[0], "evaluation": {"contact": 1.0}},
                {"transform": transforms[1], "evaluation": {"contact": 0.9}},
                {"transform": transforms[2], "evaluation": {"contact": 0.1}},
            ]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            pool, _ = load_joinable_pose_pool("a", "b", path, maximum_candidates=2)
        offsets = sorted(round(row["T_rel"][0][3]) for row in pool["candidates"])
        self.assertEqual(offsets, [0, 50])

    def test_manifold_initial_is_available_when_sdf_search_is_disabled(self):
        transform = np.eye(4)
        transform[2, 3] = 42.0
        payload = {
            "pose_search": {"enabled": False, "results": []},
            "joint_hypotheses": {"rows": [{
                "rank": 3,
                "confidence": 0.2,
                "manifold_type": "axis_coincidence",
                "entity_a": "face_1",
                "entity_b": "face_2",
                "initial_pose_b_in_a": transform.tolist(),
            }]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            pool, audit = load_joinable_pose_pool("carrier", "ear", path)
        self.assertEqual(len(pool["candidates"]), 1)
        self.assertEqual(
            pool["candidates"][0]["candidate_origin"],
            "joinable_constraint_manifold_initial",
        )
        self.assertEqual(pool["candidates"][0]["pair_exact_status"], "not_checked")
        self.assertEqual(audit["input_manifold_count"], 1)
        self.assertEqual(audit["parsed_manifold_initial_count"], 1)


if __name__ == "__main__":
    unittest.main()
