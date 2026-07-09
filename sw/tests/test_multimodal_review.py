import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

from PIL import Image

from multimodal_reviewer import QwenVLReviewer
from semantic_pool import (
    _production_semantic_hint,
    _resolve_step_path,
)


class MultimodalReviewTests(unittest.TestCase):
    def test_step_path_resolves_source_file_and_existing_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            pool = Path(temporary)
            parts = pool / "parts"
            parts.mkdir()
            source = parts / "part_001.step"
            source.write_bytes(b"STEP placeholder")
            resolved = _resolve_step_path(
                pool,
                "part_001.step",
                {"source_file": str(source)},
            )
            self.assertEqual(resolved, source.resolve())

    def test_synthetic_functional_truth_is_excluded_by_default(self):
        info = {
            "functional_semantics": {
                "source": "function_grounded_metadata",
                "part_role": "shaft",
                "part_name": "ground-truth shaft",
                "interface_types": ["cylindrical_shaft"],
                "source_template_disclosed_for_evaluation_only": False,
            }
        }
        result = _production_semantic_hint(
            info, allow_evaluation_only_semantics=False
        )
        self.assertFalse(result["available"])
        self.assertEqual(
            result["excluded_reason"],
            "evaluation_truth_not_available_in_production",
        )

    def test_production_cad_metadata_can_be_included(self):
        info = {
            "functional_semantics": {
                "source": "production_cad_metadata",
                "part_role": "shaft",
                "part_name": "drive shaft",
                "interface_types": ["cylindrical_shaft"],
            }
        }
        result = _production_semantic_hint(
            info, allow_evaluation_only_semantics=False
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["fields"]["part_role"], "shaft")

    def test_off_mode_never_calls_qwen_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "candidate.png"
            Image.new("RGB", (8, 8), "white").save(image)
            calls = []

            def transport(*args):
                calls.append(args)
                raise AssertionError("transport must not be called")

            reviewer = QwenVLReviewer(
                {"vision_model": "qwen-vl-plus"},
                Path(temporary) / "cache",
                transport=transport,
            )
            record = reviewer.review(
                "G1", [image], "anonymous parts", mode="off"
            )
            self.assertEqual(record["decision"]["verdict"], "abstain")
            self.assertEqual(
                record["decision"]["reason_codes"],
                ["semantic_disabled"],
            )
            self.assertEqual(calls, [])

    def test_off_mode_ignores_existing_live_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "candidate.png"
            Image.new("RGB", (8, 8), "white").save(image)

            def transport(*args):
                return {
                    "model": "qwen-vl-plus",
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"schema_version":"1.0.0",'
                                    '"proposal_id":"G1","verdict":"accept",'
                                    '"plausibility_score":0.9,'
                                    '"confidence":0.9,'
                                    '"reason_codes":["test"],'
                                    '"explanation":"test",'
                                    '"risk_flags":[]}'
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {},
                }

            reviewer = QwenVLReviewer(
                {"vision_model": "qwen-vl-plus"},
                Path(temporary) / "cache",
                transport=transport,
            )
            with mock.patch.dict(
                os.environ, {"QWEN_API_KEY": "test-key"}, clear=False
            ):
                live = reviewer.review(
                    "G1", [image], "anonymous parts", mode="live"
                )
            self.assertEqual(live["decision"]["verdict"], "accept")
            off = reviewer.review(
                "G1", [image], "anonymous parts", mode="off"
            )
            self.assertEqual(off["decision"]["verdict"], "abstain")
            self.assertFalse(off["cache_hit"])


if __name__ == "__main__":
    unittest.main()
