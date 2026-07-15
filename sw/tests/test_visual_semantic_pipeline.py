import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from multimodal_reviewer import QwenVLReviewer, _chat_completions_url
from visual_semantic_pipeline import (
    ASSEMBLY_SYNTHESIS_SYSTEM_PROMPT,
    CARRIER_REGION_SYSTEM_PROMPT,
    PART_ROLE_SYSTEM_PROMPT,
    PartRoleAnalysis,
    VisualSemanticPipeline,
    _fallback_part,
    _model_validator,
    _normalize_hypothesis_output,
    _normalize_part_output,
    _normalize_regions_output,
    fuse_candidate_quotas,
)


def _part_output():
    return {
        "part_id": "P01",
        "part_role": "inserted_module",
        "possible_names": ["power module"],
        "assembly_family_candidates": ["service_module_bay"],
        "functional_description": "A serviceable module with an exterior face.",
        "functional_faces": [
            {
                "face_id": "F01",
                "role": "service_face",
                "evidence": ["visible connector"],
                "confidence": 0.8,
            }
        ],
        "likely_assembly_actions": ["insert"],
        "principal_axis_semantics": "long axis is insertion axis",
        "symmetry": {
            "has_symmetry": False,
            "symmetry_type": "none",
            "orientation_ambiguity": False,
        },
        "risks": [],
        "confidence": 0.8,
        "review_required": True,
    }


def _region_output():
    return {
        "carrier": {
            "role": "equipment_chassis",
            "evidence": ["multiple service bays"],
            "confidence": 0.8,
        },
        "region_assessments": [
            {
                "region_id": "R02",
                "region_type": "service_bay",
                "compatible_part_roles": ["inserted_module"],
                "opening_direction": "outward",
                "internal_direction": "inward",
                "possible_insertion_axis": "depth",
                "visible_interfaces": ["rails", "stop"],
                "semantic_score": 0.9,
                "reasons": ["service opening and rails"],
                "possible_equivalent_slot": False,
                "forbidden_for_current_part": False,
            }
        ],
        "preferred_region_ids": ["R02"],
        "forbidden_region_ids": [{"region_id": "R03", "reason": "top panel"}],
        "equivalent_region_groups": [],
        "confidence": 0.85,
        "review_required": True,
    }


def _hypothesis_output():
    return {
        "carrier": {
            "role": "equipment_chassis",
            "evidence": ["service bay"],
            "confidence": 0.8,
        },
        "part": {
            "role": "inserted_module",
            "possible_names": ["power module"],
            "functional_description": "serviceable module",
            "evidence": ["service face"],
            "confidence": 0.8,
        },
        "assembly_hypothesis": {
            "assembly_family": "service_module_bay",
            "relation": "guided insertion",
            "assembly_action": "insert",
            "target_region_type": "service_bay",
            "preferred_region_ids": [
                {
                    "region_id": "R02",
                    "semantic_score": 0.9,
                    "reasons": ["rails and exterior opening"],
                    "possible_equivalent_slot": False,
                }
            ],
            "forbidden_region_ids": [{"region_id": "R03", "reason": "top panel"}],
        },
        "orientation_constraints": {
            "external_face_ids": ["F01"],
            "internal_face_ids": [],
            "mounting_face_ids": [],
            "insertion_axis_relative_to_part": "long axis",
            "service_face_must_remain_visible": True,
            "mirror_transform_allowed": False,
            "reasons": ["maintenance access"],
        },
        "required_geometry_evidence": [
            {
                "interface_type": "passable_opening_and_guides",
                "part_feature_ids": [],
                "carrier_region_ids": ["R02"],
                "importance": "required",
                "reason": "module must enter the bay",
            }
        ],
        "ambiguity": {
            "has_multiple_valid_regions": False,
            "equivalent_region_ids": [],
            "cannot_be_resolved_from_images": False,
            "reason": "one region is stronger",
        },
        "risk": {
            "possible_visual_misclassification": False,
            "possible_hidden_interface": True,
            "possible_scale_ambiguity": False,
            "possible_symmetry": False,
            "notes": ["geometry validation remains required"],
        },
        "semantic_confidence": 0.84,
        "review_required": True,
        "suggested_action": "prioritize_regions",
    }


class VisualSemanticPipelineTests(unittest.TestCase):
    def test_endpoint_accepts_root_or_complete_path(self):
        root = "https://example.test/compatible-mode/v1"
        complete = root + "/chat/completions"
        self.assertEqual(_chat_completions_url(root), complete)
        self.assertEqual(_chat_completions_url(complete), complete)

    def test_three_prompt_chain_and_isolated_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "input.png"
            Image.new("RGB", (12, 12), "white").save(image)
            prompts = []

            def transport(payload, api_key, timeout):
                self.assertEqual(api_key, "test-key")
                system = payload["messages"][0]["content"]
                prompts.append(system)
                if system == PART_ROLE_SYSTEM_PROMPT:
                    output = _part_output()
                elif system == CARRIER_REGION_SYSTEM_PROMPT:
                    output = _region_output()
                elif system == ASSEMBLY_SYNTHESIS_SYSTEM_PROMPT:
                    output = _hypothesis_output()
                else:
                    raise AssertionError("unexpected prompt")
                return {
                    "model": "test-model",
                    "choices": [{"message": {"content": json.dumps(output)}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 10},
                }

            reviewer = QwenVLReviewer(
                {"vision_model": "test-model", "vision_max_attempts": 1},
                Path(temporary) / "cache",
                transport=transport,
            )
            pipeline = VisualSemanticPipeline(reviewer)
            with mock.patch.dict(os.environ, {"QWEN_API_KEY": "test-key"}, clear=False):
                first = pipeline.analyze_part(
                    "P01", [image], {"functional_face_ids": ["F01"]}, mode="live"
                )
                second = pipeline.analyze_regions(
                    "P01", [image], first["output"], [{"region_id": "R02"}], {}, mode="live"
                )
                third = pipeline.synthesize(
                    "P01", [image], first["output"], second["output"], {}, mode="live"
                )
                cached = pipeline.analyze_part(
                    "P01", [image], {"functional_face_ids": ["F01"]}, mode="live"
                )

            self.assertEqual([first["status"], second["status"], third["status"]], ["ok"] * 3)
            self.assertEqual(third["output"]["suggested_action"], "prioritize_regions")
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(len(prompts), 3)
            cache_text = "\n".join(path.read_text() for path in (Path(temporary) / "cache").glob("*.json"))
            self.assertNotIn("test-key", cache_text)
            self.assertNotIn("Authorization", cache_text)
            self.assertNotIn("data:image", cache_text)

    def test_invalid_json_abstains_without_exposing_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "input.png"
            Image.new("RGB", (8, 8), "white").save(image)

            def transport(*_):
                return {"choices": [{"message": {"content": "not json"}}]}

            reviewer = QwenVLReviewer(
                {"vision_model": "test-model", "vision_max_attempts": 1},
                Path(temporary) / "cache",
                transport=transport,
            )
            with mock.patch.dict(os.environ, {"QWEN_API_KEY": "test-key"}, clear=False):
                record = reviewer.structured_review(
                    "stage",
                    [image],
                    "context",
                    system_prompt="prompt",
                    prompt_version="v1",
                    validate_output=_model_validator(PartRoleAnalysis),
                    fallback_output=_fallback_part("P01"),
                )
            self.assertEqual(record["status"], "abstain")
            self.assertEqual(record["output"]["part_role"], "unknown")
            self.assertNotIn("test-key", json.dumps(record))

    def test_missing_key_abstains(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "input.png"
            Image.new("RGB", (8, 8), "white").save(image)
            reviewer = QwenVLReviewer({}, Path(temporary) / "cache")
            with mock.patch.dict(
                os.environ,
                {
                    "QWEN_API_KEY": "",
                    "Qwen_API_KEY": "",
                    "DASHSCOPE_API_KEY": "",
                    "OPENAI_API_KEY": "",
                },
                clear=False,
            ):
                record = reviewer.structured_review(
                    "stage",
                    [image],
                    "context",
                    system_prompt="prompt",
                    prompt_version="v1",
                    validate_output=_model_validator(PartRoleAnalysis),
                    fallback_output=_fallback_part("P01"),
                )
            self.assertEqual(record["abstention_reason"], "api_key_missing")

    def test_semantic_quota_cannot_remove_protected_geometry(self):
        candidates = [
            {
                "candidate_id": "C1",
                "region_id": "R01",
                "geometry_score": 0.95,
                "candidate_sources": ["analytic"],
                "protected": True,
            },
            {
                "candidate_id": "C2",
                "region_id": "R02",
                "geometry_score": 0.40,
                "candidate_sources": ["analytic"],
            },
            {
                "candidate_id": "C3",
                "region_id": "R03",
                "geometry_score": 0.80,
                "candidate_sources": ["joinable"],
            },
        ]
        fused = fuse_candidate_quotas(
            candidates,
            _region_output(),
            total_k=3,
            geometry_quota=1,
            semantic_quota=1,
            protected_quota=1,
        )
        ids = [row["candidate_id"] for row in fused]
        self.assertIn("C1", ids)
        self.assertIn("C2", ids)
        self.assertFalse(any(row["can_auto_accept_from_semantics"] for row in fused))

    def test_model_shape_compatibility_is_normalized_without_inventing_role(self):
        part = _normalize_part_output(
            {
                "part_id": "P01",
                "part_role": "power_supply_module",
                "functional_faces": {"F05": "service_face"},
                "confidence": 0.7,
            }
        )
        self.assertEqual(part["functional_faces"][0]["face_id"], "F05")
        self.assertEqual(part["functional_faces"][0]["confidence"], 0.7)

        regions = _normalize_regions_output(
            {
                "carrier": "equipment_chassis",
                "preferred_region_ids": [{"region_id": "R02"}],
                "forbidden_region_ids": ["R03"],
            }
        )
        self.assertEqual(regions["carrier"]["role"], "equipment_chassis")
        self.assertEqual(regions["preferred_region_ids"], ["R02"])

        hypothesis = _normalize_hypothesis_output(
            {
                "carrier": "equipment_chassis",
                "part": "power_supply_module",
                "assembly_hypothesis": {
                    "assembly_action": "insert",
                    "preferred_region_ids": ["R02"],
                },
            }
        )
        self.assertEqual(hypothesis["part"]["role"], "power_supply_module")
        self.assertEqual(
            hypothesis["assembly_hypothesis"]["preferred_region_ids"][0]["region_id"],
            "R02",
        )


if __name__ == "__main__":
    unittest.main()
