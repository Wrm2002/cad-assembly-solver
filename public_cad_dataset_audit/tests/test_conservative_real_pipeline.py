import sys
import json
import tempfile
import unittest
from pathlib import Path

AUDIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIT_ROOT))

from evaluate_conservative_real_benchmark import (
    _accepted_gate,
    _classify_candidate,
    _load_geometry_candidates,
    _route_overlapping_accepts,
)
from run_real_pose_validation import (
    _production_pose_record,
    _select_blind_queue,
)


class RealConservativePipelineTests(unittest.TestCase):
    def test_blind_queue_is_size_diverse_and_ignores_truth_fields(self):
        rows = []
        for size in (2, 3, 4):
            for index in range(2):
                rows.append(
                    {
                        "pool_id": "pool",
                        "group_id": f"G{size}_{index}",
                        "group_size": size,
                        "parts": [f"P{n}" for n in range(size)],
                        "candidate_priority_score": 1.0 - index / 10,
                        "geometry_score": 0.9,
                        "active_edge_density": 1.0,
                        "consistency": {
                            "group_consistency_score": 0.8
                        },
                        "geometry_evidence": {
                            "independent_evidence_count": 2
                        },
                        "evaluation_is_true_group": index == 1,
                    }
                )
        selected = _select_blind_queue(rows, maximum_per_pool=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {row["group_size"] for row in selected}, {2, 3, 4}
        )
        self.assertFalse(
            any(row["evaluation_is_true_group"] for row in selected)
        )

    def test_acceptance_requires_every_gate(self):
        candidate = {
            "group_size": 3,
            "geometry_score": 0.9,
            "review_required": False,
            "geometry_evidence": {
                "independent_evidence_count": 2
            },
            "consistency": {
                "group_consistency_score": 0.8,
                "weak_single_interface_match": False,
                "has_global_conflict": False,
            },
        }
        pose = {
            "final_pose_status": "valid",
            "collision_result": "success",
            "occt_common_volume": 0.0,
        }
        passed, reasons = _accepted_gate(candidate, pose)
        self.assertTrue(passed)
        self.assertEqual(reasons, [])
        candidate["review_required"] = True
        passed, reasons = _accepted_gate(candidate, pose)
        self.assertFalse(passed)
        self.assertIn("gate_failed:no_review_flag", reasons)

    def test_incomplete_pose_never_accepts(self):
        candidate = {
            "group_size": 2,
            "geometry_score": 1.0,
            "review_required": False,
            "geometry_evidence": {
                "independent_evidence_count": 3
            },
            "consistency": {
                "group_consistency_score": 1.0,
                "weak_single_interface_match": False,
                "has_global_conflict": False,
            },
        }
        passed, reasons = _accepted_gate(
            candidate,
            {
                "final_pose_status": "uncertain",
                "collision_result": "success",
                "occt_common_volume": 0.0,
            },
        )
        self.assertFalse(passed)
        self.assertIn("gate_failed:pose_status_valid", reasons)

    def test_outside_frontier_stays_deferred_review(self):
        candidate = {
            "group_id": "G1",
            "pool_id": "pool",
            "parts": ["a", "b"],
            "group_size": 2,
            "geometry_score": 0.9,
            "review_required": True,
            "geometry_evidence": {"independent_evidence_count": 1},
            "consistency": {
                "group_consistency_score": 0.8,
                "weak_single_interface_match": True,
                "has_global_conflict": False,
            },
            "decision_reasons": [],
        }
        result = _classify_candidate(
            candidate, None, in_review_frontier=False
        )
        self.assertEqual(result["final_decision"], "review")
        self.assertEqual(result["review_queue_state"], "deferred")
        self.assertIn(
            "deferred_outside_bounded_review_frontier",
            result["decision_reasons"],
        )

    def test_final_aggregation_loads_all_geometry_tiers(self):
        with tempfile.TemporaryDirectory() as temporary:
            phase_root = Path(temporary)
            directory = phase_root / "phase6_candidate_tiers"
            directory.mkdir()
            for name, group_id in (
                ("accepted_geometry_candidates.json", "accepted"),
                ("review_geometry_candidates.json", "review"),
                ("rejected_geometry_candidates.json", "rejected"),
            ):
                (directory / name).write_text(
                    json.dumps(
                        [
                            {
                                "group_id": group_id,
                                "pool_id": "pool",
                                "parts": [group_id, "shared"],
                                "group_size": 2,
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
            rows = _load_geometry_candidates(phase_root)
        self.assertEqual(
            {row["group_id"] for row in rows},
            {"accepted", "review", "rejected"},
        )

    def test_overlapping_accepts_have_no_greedy_winner(self):
        rows = [
            {
                "group_id": "G1",
                "pool_id": "pool",
                "parts": ["a", "b"],
                "decision_reasons": [],
            },
            {
                "group_id": "G2",
                "pool_id": "pool",
                "parts": ["a", "c"],
                "decision_reasons": [],
            },
        ]
        safe, review = _route_overlapping_accepts(rows)
        self.assertEqual(safe, [])
        self.assertEqual({row["group_id"] for row in review}, {"G1", "G2"})

    def test_production_pose_record_removes_evaluation_queue_markers(self):
        record = {
            "candidate_id": "G1",
            "queue_modes": [
                "blind_production",
                "evaluation_only_truth_audit",
            ],
            "evaluation_only": False,
            "production_eligible": True,
        }
        sanitized = _production_pose_record(record)
        self.assertNotIn("queue_modes", sanitized)
        self.assertNotIn("evaluation_only", sanitized)
        self.assertEqual(
            sanitized["validation_scope"], "bounded_production"
        )


if __name__ == "__main__":
    unittest.main()
