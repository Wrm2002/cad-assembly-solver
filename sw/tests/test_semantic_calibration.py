import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_semantic_review import calibration_gate


class SemanticCalibrationGateTests(unittest.TestCase):
    def test_requires_all_calibration_and_holdout_conditions(self):
        safe = {
            "auto_accept_precision_not_decreased": True,
            "false_positive_count_not_increased": True,
        }
        self.assertTrue(
            calibration_gate(
                semantic_auc=0.8,
                semantic_brier=0.1,
                geometry_brier=0.2,
                verdicts={"accept", "reject", "abstain"},
                holdout=safe,
            )
        )
        self.assertFalse(
            calibration_gate(
                semantic_auc=0.8,
                semantic_brier=0.1,
                geometry_brier=0.2,
                verdicts={"accept", "abstain"},
                holdout=safe,
            )
        )
        self.assertFalse(
            calibration_gate(
                semantic_auc=0.8,
                semantic_brier=0.1,
                geometry_brier=0.2,
                verdicts={"accept", "reject", "abstain"},
                holdout={},
            )
        )


if __name__ == "__main__":
    unittest.main()
