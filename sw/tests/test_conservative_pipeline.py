import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conservative_pipeline import (
    _passes_final_acceptance,
    bound_review_queue,
    geometry_tiers,
    route_overlapping_accepts_to_review,
)


CONFIG = {
    "geometry_threshold": 0.8,
    "group_consistency_threshold": 0.7,
    "minimum_independent_evidence": 2,
    "max_auto_accept_group_size": 5,
    "conflict_score_margin": 0.04,
}


class ConservativePipelineTests(unittest.TestCase):
    def test_low_geometry_rejects_and_single_planar_reviews(self):
        proposals = [
            {
                "schema_version": "1.0.0",
                "group_id": "low",
                "parts": ["a", "b"],
                "candidate_edges": [],
                "geometry_score": 0.79,
                "connected": True,
                "status": "candidate",
                "reasons": [],
            },
            {
                "schema_version": "1.0.0",
                "group_id": "plane",
                "parts": ["c", "d"],
                "candidate_edges": ["E1"],
                "geometry_score": 0.95,
                "connected": True,
                "status": "candidate",
                "reasons": [],
            },
        ]
        edges = [
            {
                "candidate_id": "E1",
                "parts": ["c", "d"],
                "candidate_type": "planar_mate",
                "audit_reason": {
                    "normal_quality": 1.0,
                    "distance_quality": 1.0,
                },
            }
        ]
        accepted, review, rejected = geometry_tiers(
            Path("pool"), proposals, edges, CONFIG
        )
        self.assertEqual(accepted, [])
        self.assertEqual([item["group_id"] for item in review], ["plane"])
        self.assertEqual([item["group_id"] for item in rejected], ["low"])

    def test_review_frontier_is_bounded_and_size_diverse(self):
        rows = []
        for size in (2, 3, 4):
            for index in range(5):
                rows.append(
                    {
                        "group_id": f"G{size}_{index}",
                        "parts": [f"p{n}" for n in range(size)],
                        "geometry_score": 1.0 - index / 100,
                        "consistency": {"group_consistency_score": 0.8},
                        "decision_reasons": [],
                    }
                )
        chosen, dominated = bound_review_queue(
            rows, maximum=6, per_size=2
        )
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len(dominated), 9)
        self.assertEqual(
            {len(item["parts"]) for item in chosen}, {2, 3, 4}
        )
        self.assertTrue(
            all(item["review_queue_state"] == "selected" for item in chosen)
        )
        self.assertEqual(
            sorted(
                item["review_ranking"]["review_rank"]
                for item in chosen
            ),
            list(range(1, 7)),
        )
        self.assertTrue(
            all(
                item["review_ranking"]["affects_auto_accept"] is False
                for item in chosen
            )
        )
        self.assertTrue(
            all(item["final_decision"] == "review" for item in dominated)
        )
        self.assertTrue(
            all(item["review_queue_state"] == "deferred" for item in dominated)
        )

    def test_overlapping_accepts_all_route_to_review(self):
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
        accepted, review = route_overlapping_accepts_to_review(rows)
        self.assertEqual(accepted, [])
        self.assertEqual({item["group_id"] for item in review}, {"G1", "G2"})
        self.assertTrue(
            all(item["final_decision"] == "review" for item in review)
        )

    def test_final_acceptance_requires_explicit_collision_success(self):
        item = {
            "geometry_tier": "accepted_for_pose_validation",
            "geometry_score": 0.95,
            "parts": ["a", "b"],
            "consistency": {
                "independent_evidence_count": 3,
                "group_consistency_score": 0.9,
                "weak_single_interface_match": False,
                "review_required": False,
                "has_global_conflict": False,
            },
        }
        pose = {
            "final_pose_status": "valid",
            "worker_status": "success",
            "collision_result": "not_run",
            "occt_common_volume": 0.0,
        }
        self.assertFalse(_passes_final_acceptance(item, pose, CONFIG))
        pose["collision_result"] = "success"
        self.assertTrue(_passes_final_acceptance(item, pose, CONFIG))

    def test_two_part_group_routes_to_review_when_minimum_is_three(self):
        config = {**CONFIG, "minimum_auto_accept_group_size": 3}
        proposal = {
            "schema_version": "1.0.0",
            "group_id": "binary",
            "parts": ["a", "b"],
            "candidate_edges": ["E1"],
            "geometry_score": 0.95,
            "connected": True,
            "status": "candidate",
            "reasons": [],
        }
        edge = {
            "candidate_id": "E1",
            "parts": ["a", "b"],
            "candidate_type": "clearance",
            "audit_reason": {
                "gap_quality": 0.95,
                "axis_dot_abs": 1.0,
                "area_reliability": 0.9,
            },
        }
        accepted, review, rejected = geometry_tiers(
            Path("pool"), [proposal], [edge], config
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])
        self.assertEqual([row["group_id"] for row in review], ["binary"])
        self.assertIn(
            "group_size_below_auto_accept_limit",
            review[0]["decision_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
