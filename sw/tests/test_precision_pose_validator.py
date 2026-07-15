from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.manifold_solver import solve_manifold_pose_graph  # noqa: E402
from learned_joint.precision_pose_validator import validate_precision_pose  # noqa: E402


def _exact(status: str = "valid", collisions=None):
    return {
        "status": status,
        "occt": {
            "status": "success",
            "method": "occt_boolean_common_volume",
            "collisions": list(collisions or []),
            "errors": [],
        },
    }


def _row(kind: str, translation=0.0, rotation=0.0, **extra):
    result = {
        "manifold_type": kind,
        "projected_translation_residual_mm": translation,
        "projected_rotation_residual_degrees": rotation,
    }
    result.update(extra)
    return result


class PrecisionPoseValidatorTests(unittest.TestCase):
    def test_exact_failure_is_failed_and_reports_common_volume(self):
        result = validate_precision_pose({
            "exact_validation": _exact("failed", [{"intersection_volume_mm3": 3.5}]),
            "factor_residuals": [],
        })
        self.assertEqual(result["precision_status"], "failed")
        self.assertEqual(result["reason"], "exact_validation_failed")
        self.assertEqual(result["occt_common_volume_mm3"], 3.5)

    def test_occt_valid_without_contact_support_is_review(self):
        result = validate_precision_pose({
            "exact_validation": _exact(),
            "factor_residuals": [
                _row("axis_coincidence"),
                _row("plane_coincidence"),
            ],
        })
        self.assertEqual(result["precision_status"], "review")
        self.assertEqual(result["reason"], "occt_valid_but_contact_support_missing")

    def test_single_plane_stays_review_even_with_contact_gate(self):
        result = validate_precision_pose(
            {
                "exact_validation": _exact(),
                "factor_residuals": [_row("plane_coincidence", translation=0.01)],
            },
            contact_gate={"status": "valid"},
        )
        self.assertEqual(result["precision_status"], "review")
        self.assertEqual(result["independent_evidence_types"], ["planar_contact"])
        self.assertIn("single_interface_constraint_requires_review", result["reasons"])

    def test_multi_axis_ransac_supplies_two_independent_evidence_types(self):
        result = validate_precision_pose({
            "exact_validation": _exact(),
            "factor_residuals": [_row(
                "compound_multi_axis_rigid",
                translation=0.03,
                rotation=0.2,
                provenance={
                    "multi_interface_ransac": True,
                    "residual_mm": 0.04,
                    "clearance_mm": 0.02,
                    "offset_mm": 17.0,
                    # This untrusted integer must not replace typed evidence.
                    "independent_evidence_count": 99,
                },
            )],
            "unresolved_manifold_dofs": [],
        })
        self.assertEqual(result["precision_status"], "valid")
        self.assertEqual(result["axis_distance_mm"], 0.03)
        self.assertEqual(result["axis_angle_degrees"], 0.2)
        self.assertEqual(result["plane_gap_mm"], 0.02)
        self.assertEqual(result["hole_pattern_rms_mm"], 0.04)
        self.assertEqual(
            result["independent_evidence_types"],
            ["axial_support_contact", "repeated_axis_pattern"],
        )

    def test_legacy_ransac_offset_is_conservative_gap_fallback(self):
        result = validate_precision_pose({
            "exact_validation": _exact(),
            "factor_residuals": [_row(
                "compound_multi_axis_rigid",
                provenance={
                    "multi_interface_ransac": True,
                    "residual_mm": 0.01,
                    "offset_mm": 0.04,
                },
            )],
        })
        self.assertEqual(result["precision_status"], "valid")
        self.assertEqual(result["plane_gap_mm"], 0.04)

    def test_single_axis_plus_collision_free_contact_stays_review(self):
        result = validate_precision_pose(
            {
                "exact_validation": _exact(),
                "factor_residuals": [_row("axis_coincidence")],
            },
            contact_gate={"supported": True},
        )
        self.assertEqual(result["precision_status"], "review")
        self.assertIn("single_interface_constraint_requires_review", result["reasons"])

    def test_prismatic_without_measured_insertion_depth_is_review(self):
        result = validate_precision_pose(
            {
                "exact_validation": _exact(),
                "factor_residuals": [_row(
                    "compound_prismatic_insertion_rigid",
                    precision_evidence={
                        "evidence_types": ["prismatic_profile_fit", "surface_contact"]
                    },
                )],
            },
            contact_gate={"supported": True},
        )
        self.assertEqual(result["precision_status"], "review")
        self.assertIn("prismatic_insertion_depth_not_verified", result["reasons"])

    def test_verified_prismatic_depth_can_pass_general_gate(self):
        result = validate_precision_pose(
            {
                "exact_validation": _exact(),
                "factor_residuals": [_row(
                    "compound_prismatic_insertion_rigid",
                    precision_evidence={
                        "evidence_types": ["prismatic_profile_fit"],
                        "insertion_depth_mm": 4.0,
                        "insertion_depth_verified": True,
                    },
                )],
            },
            contact_gate={"supported": True},
        )
        self.assertEqual(result["precision_status"], "valid")
        self.assertEqual(result["insertion_depth_mm"], 4.0)

    def test_injected_tolerance_fails_out_of_spec_pattern(self):
        result = validate_precision_pose(
            {
                "exact_validation": _exact(),
                "factor_residuals": [_row(
                    "compound_multi_axis_rigid",
                    provenance={
                        "multi_interface_ransac": True,
                        "residual_mm": 0.08,
                        "clearance_mm": 0.01,
                    },
                )],
            },
            tolerances={"maximum_hole_pattern_rms_mm": 0.05},
        )
        self.assertEqual(result["precision_status"], "failed")
        self.assertEqual(result["reason"], "hole_pattern_rms_exceeds_tolerance")

    def test_names_roles_and_case_ids_cannot_change_decision(self):
        base = {
            "exact_validation": _exact(),
            "factor_residuals": [_row(
                "compound_multi_axis_rigid",
                provenance={
                    "multi_interface_ransac": True,
                    "residual_mm": 0.02,
                    "clearance_mm": 0.01,
                },
            )],
        }
        renamed = copy.deepcopy(base)
        renamed.update({"case_id": "case_1", "part_name": "special", "part_role": "hub"})
        renamed["factor_residuals"][0]["candidate_id"] = "case_1_magic"
        left = validate_precision_pose(base)
        right = validate_precision_pose(renamed)
        for key in (
            "precision_status", "reason", "independent_evidence_count",
            "independent_evidence_types", "hole_pattern_rms_mm", "plane_gap_mm",
        ):
            self.assertEqual(left[key], right[key])

    def test_solver_factor_audit_carries_only_filtered_precision_evidence(self):
        identity = np.eye(4).tolist()
        candidate = {
            "candidate_id": "opaque",
            "source": "a",
            "target": "b",
            "manifold_type": "compound_multi_axis_rigid",
            "frame_a": identity,
            "frame_b": identity,
            "initial_pose_b_in_a": identity,
            "free_dof_mask": [0, 0, 0, 0, 0, 0],
            "confidence": 0.9,
            "provenance": {
                "multi_interface_ransac": True,
                "residual_mm": 0.03,
                "clearance_mm": 0.01,
                "case_id": "must_not_escape",
                "part_role": "must_not_escape",
            },
        }
        solved = solve_manifold_pose_graph(
            ["a", "b"], [candidate], max_hypotheses=1, max_topologies=1
        )
        audit = solved["hypotheses"][0]["factor_residuals"][0]
        self.assertEqual(
            audit["precision_evidence"]["evidence_types"],
            ["axial_support_contact", "repeated_axis_pattern"],
        )
        self.assertNotIn("provenance", audit)
        self.assertNotIn("case_id", repr(audit))
        self.assertNotIn("part_role", repr(audit))


if __name__ == "__main__":
    unittest.main()
