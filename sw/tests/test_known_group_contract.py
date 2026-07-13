import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import KnownGroupAssemblyResult
from known_group_assembly import (
    _apply_joinable_support,
    _conservative_pose_output,
    _portable_components,
    _relative_transform,
)


class KnownGroupContractTests(unittest.TestCase):
    def test_portable_components_support_independent_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_dir = root / "inputs" / "case"
            output_dir = root / "independent" / "exam" / "case3"
            case_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            source = case_dir / "part.step"
            source.touch()

            components = _portable_components(
                [{"source": str(source), "label": "part"}],
                case_dir,
                output_dir,
            )

            resolved = (output_dir / components[0]["source"]).resolve()
            self.assertEqual(resolved, source.resolve())

    def test_relative_transform_uses_part_b_frame(self):
        placements = {
            "a": {"translate": [5.0, 0.0, 0.0]},
            "b": {"translate": [2.0, 0.0, 0.0]},
        }
        matrix = _relative_transform("a", "b", placements)
        self.assertAlmostEqual(matrix[0][3], 3.0)
        self.assertAlmostEqual(matrix[1][3], 0.0)

    def test_minimal_single_part_document(self):
        document = KnownGroupAssemblyResult.model_validate({
            "assembly_id": "single",
            "parts": ["only.step"],
            "reference_part": "only.step",
            "assembly_connected": True,
            "pose_status": "valid",
            "direct_connections": [],
            "assembly_relations": [],
            "components": [{"source": "../only.step"}],
            "collision_validation": {"status": "success"},
            "candidate_summary": {},
        })
        self.assertEqual(document.schema_version, "2.0.0")

    def test_joinable_axial_candidate_boundedly_supports_clearance(self):
        match = {
            "type": "clearance",
            "parts": ("shaft.step", "hub.step"),
            "score": 0.7,
            "confidence": "medium",
        }
        learned = {
            ("hub.step", "shaft.step"): {
                "pair_id": "p1",
                "top_interface_candidates": [{
                    "rank": 1,
                    "family_hint": "coaxial_or_cylindrical",
                    "softmax_probability": 0.5,
                }],
            }
        }
        result = _apply_joinable_support([match], learned)[0]
        self.assertGreater(result["score"], 0.7)
        self.assertLessEqual(result["score"], 0.78)
        self.assertEqual(result["joinable_support"]["rank"], 1)

    def test_pose_valid_without_precision_gate_is_review_not_accepted(self):
        result = _conservative_pose_output(
            {
                "assembly_id": "anonymous",
                "parts": ["a", "b"],
                "assembly_connected": True,
                "pose_status": "valid",
                "collision_validation": {
                    "status": "success",
                    "selected_pose_rank": 1,
                },
                "direct_connections": [],
                "unresolved_parts": [],
            },
            [{"rank": 1, "constraint_closure": {"fully_closed": True}}],
        )
        self.assertEqual(result["accepted_groups"], [])
        self.assertEqual(len(result["review_groups"]), 1)
        self.assertEqual(
            result["review_groups"][0]["decision_reasons"],
            ["precision_pose_validation_missing"],
        )

    def test_pose_and_precision_valid_can_be_accepted(self):
        result = _conservative_pose_output(
            {
                "assembly_id": "anonymous",
                "parts": ["a", "b"],
                "assembly_connected": True,
                "pose_status": "valid",
                "precision_pose_validation": {
                    "precision_status": "valid",
                    "independent_evidence_count": 2,
                },
                "collision_validation": {
                    "status": "success",
                    "selected_pose_rank": 1,
                },
                "direct_connections": [],
                "unresolved_parts": [],
            },
            [{"rank": 1, "constraint_closure": {"fully_closed": True}}],
        )
        self.assertEqual(len(result["accepted_groups"]), 1)
        self.assertEqual(result["review_groups"], [])


if __name__ == "__main__":
    unittest.main()
