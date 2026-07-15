from __future__ import annotations

import unittest

from visual_pose_candidate_reranker import (
    classify_exact_collision_audit,
    rank_candidates,
)


def _candidate(candidate_id: str, *, closure: float, collision_count: int = 0):
    return {
        "candidate_id": candidate_id,
        "source_pose_rank": int(candidate_id[-1]),
        "manifest": f"{candidate_id}.json",
        "protected": candidate_id == "P1",
        "machine_evidence": {
            "collision_status": "success",
            "collision_count": collision_count,
            "closure_ratio": closure,
        },
    }


class VisualPoseCandidateRerankerTests(unittest.TestCase):
    def test_visual_preference_reranks_collision_free_candidates(self):
        rows = rank_candidates(
            [_candidate("P1", closure=1.0), _candidate("P2", closure=1.0)],
            {
                "preferred_candidate_ids": ["P2"],
                "candidate_assessments": [
                    {"candidate_id": "P1", "semantic_score": 0.4, "semantic_forbidden": False},
                    {"candidate_id": "P2", "semantic_score": 0.9, "semantic_forbidden": False},
                ],
            },
        )
        self.assertEqual(rows[0]["candidate_id"], "P2")
        self.assertFalse(rows[0]["can_auto_accept"])

    def test_known_collision_stays_below_semantic_preference(self):
        rows = rank_candidates(
            [_candidate("P1", closure=0.5), _candidate("P2", closure=1.0, collision_count=1)],
            {
                "preferred_candidate_ids": ["P2"],
                "candidate_assessments": [
                    {"candidate_id": "P1", "semantic_score": 0.5, "semantic_forbidden": False},
                    {"candidate_id": "P2", "semantic_score": 0.99, "semantic_forbidden": False},
                ],
            },
        )
        self.assertEqual(rows[0]["candidate_id"], "P1")

    def test_semantic_forbidden_is_demoted_but_preserved_for_audit(self):
        rows = rank_candidates(
            [_candidate("P1", closure=0.5), _candidate("P2", closure=1.0)],
            {
                "preferred_candidate_ids": ["P2"],
                "candidate_assessments": [
                    {"candidate_id": "P1", "semantic_score": 0.5, "semantic_forbidden": False},
                    {"candidate_id": "P2", "semantic_score": 0.99, "semantic_forbidden": True},
                ],
            },
        )
        self.assertEqual([row["candidate_id"] for row in rows], ["P1", "P2"])
        self.assertTrue(rows[1]["review_required"])

    def test_zero_semantic_score_is_not_replaced_by_neutral_default(self):
        rows = rank_candidates(
            [_candidate("P1", closure=1.0), _candidate("P2", closure=1.0)],
            {
                "preferred_candidate_ids": [],
                "candidate_assessments": [
                    {"candidate_id": "P1", "semantic_score": 0.0, "semantic_forbidden": False},
                    {"candidate_id": "P2", "semantic_score": 0.2, "semantic_forbidden": False},
                ],
            },
        )
        self.assertEqual(rows[0]["candidate_id"], "P2")
        self.assertEqual(rows[1]["semantic_score"], 0.0)

    def test_small_overlap_on_partial_topology_remains_review_uncertain(self):
        result = classify_exact_collision_audit({
            "status": "partial",
            "collision_result": "collision_detected",
            "coverage_audit": {"complete": False},
            "collisions": [{"minimum_part_volume_ratio": 0.0012}],
        })
        self.assertFalse(result["collision_free"])
        self.assertTrue(result["collision_uncertain"])

    def test_same_overlap_on_complete_topology_is_collision_failure(self):
        result = classify_exact_collision_audit({
            "status": "success",
            "collision_result": "collision_detected",
            "coverage_audit": {"complete": True},
            "collisions": [{"minimum_part_volume_ratio": 0.0012}],
        })
        self.assertFalse(result["collision_free"])
        self.assertFalse(result["collision_uncertain"])
        self.assertEqual(result["outcome"], "collision_failed")


if __name__ == "__main__":
    unittest.main()
