from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.report_adapter import load_manifold_pool  # noqa: E402


class ManifoldReportAdapterTests(unittest.TestCase):
    def test_relabels_bookkeeping_ids_without_using_source_names_as_features(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({
                "joint_hypotheses": {"rows": [{
                    "source": "part_0",
                    "target": "part_1",
                    "frame_a": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "frame_b": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "initial_pose_b_in_a": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "free_dof_mask": [0, 0, 1, 0, 0, 1],
                    "confidence": 0.7,
                }]}
            }), encoding="utf-8")
            pool, audit = load_manifold_pool("anonymous_0", "anonymous_1", path)
        self.assertEqual(pool["candidates"][0]["source"], "anonymous_0")
        self.assertEqual(pool["candidates"][0]["target"], "anonymous_1")
        self.assertEqual(audit["retained_count"], 1)

    def test_entity_pair_diversity_precedes_symmetry_variants(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        rows = []
        for index in range(4):
            rows.append({
                "entity_a": "same_a", "entity_b": "same_b",
                "frame_a": identity, "frame_b": identity,
                "initial_pose_b_in_a": identity,
                "free_dof_mask": [0, 0, 1, 0, 0, 1],
                "confidence": 0.9, "phase_degrees": index * 90,
            })
        rows.append({
            "entity_a": "other_a", "entity_b": "other_b",
            "frame_a": identity, "frame_b": identity,
            "initial_pose_b_in_a": identity,
            "free_dof_mask": [0, 0, 1, 0, 0, 1],
            "confidence": 0.5,
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({"joint_hypotheses": {"rows": rows}}), encoding="utf-8")
            pool, _ = load_manifold_pool("a", "b", path, maximum_candidates=2)
        entity_pairs = {(row["entity_a"], row["entity_b"]) for row in pool["candidates"]}
        self.assertEqual(len(entity_pairs), 2)

    def test_strongest_unoriented_pair_keeps_both_polarities(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        rows = []
        for polarity in (1, -1):
            rows.append({
                "entity_a": "top_a", "entity_b": "top_b", "polarity": polarity,
                "frame_a": identity, "frame_b": identity,
                "initial_pose_b_in_a": identity,
                "free_dof_mask": [1, 1, 0, 0, 0, 1], "confidence": 0.9,
            })
        rows.append({
            "entity_a": "lower_a", "entity_b": "lower_b", "polarity": 1,
            "frame_a": identity, "frame_b": identity,
            "initial_pose_b_in_a": identity,
            "free_dof_mask": [1, 1, 0, 0, 0, 1], "confidence": 0.7,
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({"joint_hypotheses": {"rows": rows}}), encoding="utf-8")
            pool, _ = load_manifold_pool("a", "b", path, maximum_candidates=2)
        self.assertEqual([row["polarity"] for row in pool["candidates"]], [1, -1])

    def test_learned_sidecar_is_additive_to_protected_baseline(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        baseline = [{
            "entity_a": f"base_{index}", "entity_b": f"base_{index}",
            "frame_a": identity, "frame_b": identity, "initial_pose_b_in_a": identity,
            "free_dof_mask": [0, 0, 1, 0, 0, 1], "confidence": 0.5,
        } for index in range(2)]
        learned = [{
            "entity_a": "learned", "entity_b": "learned",
            "frame_a": identity, "frame_b": identity, "initial_pose_b_in_a": identity,
            "free_dof_mask": [0, 0, 1, 0, 0, 1], "confidence": 0.99,
            "provenance": {"learned_pose_initial": True, "learned_pose_score": 8.0},
        }]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(json.dumps({"joint_hypotheses": {"rows": baseline + learned}}), encoding="utf-8")
            pool, audit = load_manifold_pool(
                "a", "b", path, maximum_candidates=2, maximum_learned_candidates=1
            )
        self.assertEqual(len(pool["candidates"]), 3)
        self.assertFalse(pool["candidates"][0].get("provenance", {}).get("learned_pose_initial", False))
        self.assertFalse(pool["candidates"][1].get("provenance", {}).get("learned_pose_initial", False))
        self.assertTrue(pool["candidates"][2]["provenance"]["learned_pose_initial"])
        self.assertEqual(audit["retained_baseline_count"], 2)
        self.assertEqual(audit["retained_learned_count"], 1)

    def test_compound_geometry_has_an_additive_recall_budget(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        baseline = [{
            "entity_a": "plane_a", "entity_b": "plane_b",
            "frame_a": identity, "frame_b": identity,
            "initial_pose_b_in_a": identity,
            "free_dof_mask": [1, 1, 0, 0, 0, 1], "confidence": 0.7,
        }]
        compound = [{
            "entity_a": "pattern_a", "entity_b": "pattern_b",
            "manifold_type": "compound_multi_axis_rigid",
            "frame_a": identity, "frame_b": identity,
            "initial_pose_b_in_a": identity,
            "free_dof_mask": [0, 0, 0, 0, 0, 0], "confidence": 0.99,
            "provenance": {
                "multi_interface_ransac": True,
                "independent_evidence_count": 6,
            },
        }]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            path.write_text(
                json.dumps({"joint_hypotheses": {"rows": baseline + compound}}),
                encoding="utf-8",
            )
            pool, audit = load_manifold_pool(
                "a", "b", path,
                maximum_candidates=1,
                maximum_compound_candidates=1,
            )
        self.assertEqual(len(pool["candidates"]), 2)
        self.assertEqual(audit["retained_baseline_count"], 1)
        self.assertEqual(audit["retained_compound_count"], 1)
        self.assertTrue(audit["compound_geometry_additive"])


if __name__ == "__main__":
    unittest.main()
