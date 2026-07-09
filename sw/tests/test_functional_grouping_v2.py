import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import GroupProposal, PairEdge
from family_templates import TEMPLATE_BY_FAMILY
from pair_edge import build_pair_edges
from proposal_postprocess import cluster_proposals
from role_estimator import estimate_roles
from run_frontier_pose_experiment import _decide


class FunctionalGroupingV2Tests(unittest.TestCase):
    def test_pair_edge_keeps_joinable_separate_from_physical_evidence(self):
        rows = [
            {
                "candidate_id": "A",
                "parts": ["a", "b"],
                "candidate_type": "clearance",
                "geometry_score": 0.9,
                "audit_reason": {
                    "gap_quality": 0.9,
                    "axis_dot_abs": 1.0,
                    "area_reliability": 0.9,
                },
            },
            {
                "candidate_id": "J",
                "parts": ["a", "b"],
                "candidate_type": "joinable_interface_rank",
                "geometry_score": 0.0,
                "provider_evidence": {
                    "pretrained_joinable": {
                        "rank": 1,
                        "softmax_probability": 0.95,
                    }
                },
                "audit_reason": {},
            },
        ]
        edge = build_pair_edges(rows)[0]
        PairEdge.model_validate(edge)
        self.assertTrue(edge["provider_agreement_present"])
        self.assertFalse(
            edge["provider_agreement_counts_as_independent_evidence"]
        )
        self.assertNotIn(
            "learned_interface_consistency", edge["physical_evidence"]
        )

    def test_role_estimator_ignores_embedded_truth_semantics(self):
        feature = {
            "part_id": "p.step",
            "bbox": {"size": [5.0, 5.0, 30.0]},
            "volume": 500.0,
            "planar_faces": [{}, {}],
            "cylindrical_faces": [
                {"parameters": {"surface_polarity": "convex"}}
            ],
            "holes": [],
            "functional_semantics": {
                "part_role": "cover",
                "assembly_family": "cover_base",
            },
        }
        result = estimate_roles([feature], [])["p.step"]
        self.assertFalse(result["evaluation_semantics_used"])
        self.assertGreater(
            result["role_scores"]["shaft"],
            result["role_scores"]["cover"],
        )

    def test_templates_distinguish_complete_assembly_from_binary_subgroup(self):
        shaft = TEMPLATE_BY_FAMILY["shaft_hub_key"]
        bearing = TEMPLATE_BY_FAMILY["bearing_housing"]
        self.assertIn("key", shaft.complete_roles)
        self.assertIn("shaft", bearing.complete_roles)
        self.assertIn("end_cover", bearing.complete_roles)
        self.assertNotIn("axial_retainer", shaft.complete_roles)

    def test_clustering_never_connects_different_families(self):
        common = dict(
            candidate_edges=[], geometry_score=0.9, connected=True,
            status="candidate", reasons=[], center_part_ids=["a"],
        )
        rows = [
            GroupProposal(
                group_id="g1", parts=["a", "b", "c"],
                assembly_family="cover_base", **common
            ),
            GroupProposal(
                group_id="g2", parts=["a", "b", "d"],
                assembly_family="bearing_housing", **common
            ),
        ]
        _, clusters = cluster_proposals(rows)
        self.assertEqual(len(clusters), 2)

    def test_uncalibrated_role_template_gate_cannot_auto_accept(self):
        row = {
            "group_id": "g",
            "parts": ["a", "b", "c"],
            "proposal_cluster_id": "pc",
            "completeness_status": "family_complete",
            "missing_required_relations": [],
            "geometry_score": 0.95,
            "independent_evidence_count": 4,
            "ranking_features": {"learned_only_critical_edge": 0.0},
            "review_rank_score": 7.0,
            "pose_validation": {
                "final_pose_status": "valid",
                "collision_result": "success",
            },
        }
        accepted, review, rejected = _decide([row])
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])
        self.assertEqual(review[0]["final_decision"], "review")
        self.assertIn(
            "role_template_calibration_gate_closed",
            review[0]["decision_reasons"],
        )


if __name__ == "__main__":
    unittest.main()

