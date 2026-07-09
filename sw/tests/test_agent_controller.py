import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_controller import semantic_application_allowed


class AgentControllerTests(unittest.TestCase):
    def test_missing_or_failed_calibration_closes_gate(self):
        self.assertFalse(semantic_application_allowed(None))
        self.assertFalse(
            semantic_application_allowed(
                {"semantic_reranking_enabled": False}
            )
        )

    def test_only_explicit_calibration_pass_opens_gate(self):
        self.assertTrue(
            semantic_application_allowed(
                {
                    "semantic_reranking_enabled": True,
                    "semantic_application_mode": "rerank",
                }
            )
        )

    def test_explanation_only_config_keeps_gate_closed(self):
        self.assertFalse(
            semantic_application_allowed(
                {
                    "semantic_reranking_enabled": True,
                    "semantic_application_mode": "rerank",
                },
                {"application_mode": "explanation_only"},
            )
        )


if __name__ == "__main__":
    unittest.main()
