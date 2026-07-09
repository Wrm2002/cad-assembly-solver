import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_review import DeepSeekReviewer


CONFIG = {
    "base_url": "https://example.invalid",
    "model": "test-model",
    "prompt_version": "test-v1",
    "timeout_seconds": 1,
    "maximum_attempts": 2,
    "maximum_output_tokens": 200,
}


class SemanticReviewTests(unittest.TestCase):
    def test_valid_response_is_cached_and_replayed(self):
        calls = []

        def transport(payload, api_key, timeout):
            calls.append((payload, api_key, timeout))
            decision = {
                "schema_version": "1.0.0",
                "proposal_id": "G1",
                "verdict": "abstain",
                "plausibility_score": 0.5,
                "confidence": 0.4,
                "reason_codes": ["ambiguous_geometry"],
                "explanation": "Insufficient functional evidence.",
                "risk_flags": [],
            }
            return {
                "model": "test-model",
                "choices": [{
                    "message": {"content": json.dumps(decision)},
                    "finish_reason": "stop",
                }],
                "usage": {"total_tokens": 10},
            }

        with tempfile.TemporaryDirectory() as folder:
            reviewer = DeepSeekReviewer(CONFIG, folder, transport=transport)
            old = __import__("os").environ.get("DEEPSEEK_API_KEY")
            __import__("os").environ["DEEPSEEK_API_KEY"] = "sk-test"
            try:
                first = reviewer.review({"proposal_id": "G1", "parts": []})
                second = reviewer.review({"proposal_id": "G1", "parts": []})
            finally:
                if old is None:
                    __import__("os").environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    __import__("os").environ["DEEPSEEK_API_KEY"] = old
            self.assertEqual(len(calls), 1)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])

    def test_invalid_provider_output_abstains(self):
        def transport(payload, api_key, timeout):
            return {
                "choices": [{
                    "message": {"content": "{\"verdict\":\"accept\"}"},
                    "finish_reason": "stop",
                }]
            }

        with tempfile.TemporaryDirectory() as folder:
            reviewer = DeepSeekReviewer(CONFIG, folder, transport=transport)
            old = __import__("os").environ.get("DEEPSEEK_API_KEY")
            __import__("os").environ["DEEPSEEK_API_KEY"] = "sk-test"
            try:
                result = reviewer.review({"proposal_id": "G1", "parts": []})
            finally:
                if old is None:
                    __import__("os").environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    __import__("os").environ["DEEPSEEK_API_KEY"] = old
            self.assertEqual(result["decision"]["verdict"], "abstain")
            self.assertEqual(
                result["decision"]["reason_codes"], ["provider_failure"]
            )

    def test_off_mode_never_calls_transport(self):
        def transport(*args):
            raise AssertionError("transport must not run")

        with tempfile.TemporaryDirectory() as folder:
            reviewer = DeepSeekReviewer(CONFIG, folder, transport=transport)
            result = reviewer.review(
                {"proposal_id": "G1", "parts": []}, mode="off"
            )
            self.assertEqual(result["decision"]["verdict"], "abstain")


if __name__ == "__main__":
    unittest.main()
