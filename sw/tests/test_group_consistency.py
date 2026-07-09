import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from global_optimizer.group_consistency import assess_group_consistency


class GroupConsistencyTests(unittest.TestCase):
    def test_single_planar_contact_requires_review(self):
        proposal = {
            "group_id": "G1",
            "parts": ["a", "b"],
            "candidate_edges": ["E1"],
            "geometry_score": 0.95,
            "connected": True,
        }
        edge = {
            "candidate_id": "E1",
            "parts": ["a", "b"],
            "candidate_type": "planar_mate",
            "audit_reason": {
                "normal_quality": 1.0,
                "distance_quality": 1.0,
                "area_reliability": 1.0,
            },
        }
        result = assess_group_consistency(proposal, [edge], [proposal])
        self.assertTrue(result["weak_single_interface_match"])
        self.assertTrue(result["review_required"])

    def test_cylindrical_multi_evidence_can_pass_without_conflict(self):
        proposal = {
            "group_id": "G1",
            "parts": ["a", "b"],
            "candidate_edges": ["E1"],
            "geometry_score": 0.95,
            "connected": True,
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
        result = assess_group_consistency(proposal, [edge], [proposal])
        self.assertGreaterEqual(result["independent_evidence_count"], 2)
        self.assertFalse(result["weak_single_interface_match"])
        self.assertFalse(result["has_global_conflict"])
        self.assertEqual(result["group_completeness_score"], 1.0)
        self.assertEqual(result["central_part_coverage"], 1.0)
        self.assertGreater(result["interface_diversity_score"], 0.0)

    def test_near_tied_overlap_forces_review(self):
        proposal = {
            "group_id": "G1",
            "parts": ["a", "b"],
            "candidate_edges": [],
            "geometry_score": 0.9,
        }
        alternative = {
            "group_id": "G2",
            "parts": ["a", "c"],
            "candidate_edges": [],
            "geometry_score": 0.89,
        }
        result = assess_group_consistency(
            proposal, [], [proposal, alternative]
        )
        self.assertTrue(result["has_global_conflict"])
        self.assertTrue(result["review_required"])

    def test_low_confidence_joinable_not_counted_as_evidence(self):
        """JoinABLe without provider_evidence.softmax_probability >= 0.85
        contributes no independent evidence."""
        proposal = {
            "group_id": "G1",
            "parts": ["a", "b"],
            "candidate_edges": ["E1"],
            "geometry_score": 0.99,
        }
        edge = {
            "candidate_id": "E1",
            "parts": ["a", "b"],
            "candidate_type": "joinable_interface_rank",
            "audit_reason": {"joinable_uniform_lift": 100.0},
        }
        result = assess_group_consistency(proposal, [edge], [proposal])
        self.assertEqual(result["independent_evidence_count"], 0)
        self.assertEqual(result["learned_evidence_count"], 0)
        self.assertEqual(result["corroborating_provider_count"], 1)
        self.assertFalse(
            result["provider_agreement_counts_as_independent_evidence"]
        )
        self.assertTrue(result["review_required"])

    def test_high_confidence_joinable_is_review_corroboration_only(self):
        """A high softmax can be recorded but is not a physical constraint."""
        proposal = {
            "group_id": "G1",
            "parts": ["a", "b"],
            "candidate_edges": ["E1"],
            "geometry_score": 0.99,
        }
        edge = {
            "candidate_id": "E1",
            "parts": ["a", "b"],
            "candidate_type": "joinable_interface_rank",
            "audit_reason": {"joinable_uniform_lift": 100.0},
            "provider_evidence": {
                "pretrained_joinable": {
                    "softmax_probability": 0.92,
                    "rank": 1,
                }
            },
        }
        result = assess_group_consistency(
            proposal,
            [edge],
            [proposal],
            learned_evidence_enabled=True,
            joinable_min_softmax=0.85,
        )
        self.assertEqual(result["independent_evidence_count"], 0)
        self.assertEqual(result["learned_evidence_count"], 1)
        self.assertIn(
            "learned_interface_consistency", result["learned_evidence"]
        )
        self.assertEqual(result["analytic_evidence_count"], 0)
        self.assertTrue(result["review_required"])

    def test_corroborated_provider_agreement_is_not_independent(self):
        """Correlated geometry providers may agree without adding physics."""
        proposal = {
            "group_id": "G1",
            "parts": ["a", "b"],
            "candidate_edges": ["E1", "E2"],
            "geometry_score": 0.99,
        }
        analytic_edge = {
            "candidate_id": "E1",
            "parts": ["a", "b"],
            "candidate_type": "coaxial",
            "audit_reason": {
                "gap_quality": 0.95,
                "axis_dot_abs": 1.0,
                "area_reliability": 0.9,
            },
        }
        learned_edge = {
            "candidate_id": "E2",
            "parts": ["a", "b"],
            "candidate_type": "joinable_interface_rank",
            "audit_reason": {"joinable_uniform_lift": 100.0},
            "provider_evidence": {
                "pretrained_joinable": {
                    "softmax_probability": 0.92,
                    "rank": 1,
                }
            },
        }
        result = assess_group_consistency(
            proposal,
            [analytic_edge, learned_edge],
            [proposal],
            learned_evidence_enabled=True,
            joinable_min_softmax=0.85,
        )
        self.assertGreaterEqual(result["independent_evidence_count"], 3)
        self.assertEqual(result["learned_evidence_count"], 1)
        self.assertEqual(result["corroborated_pair_count"], 1)
        self.assertTrue(result["provider_agreement_present"])
        self.assertFalse(
            result["provider_agreement_counts_as_independent_evidence"]
        )


if __name__ == "__main__":
    unittest.main()
