import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_joinable_holdout import exact_paired_binomial, wilson


class JoinableHoldoutAuditTests(unittest.TestCase):
    def test_exact_paired_binomial_matches_27_joint_discordance(self):
        self.assertAlmostEqual(exact_paired_binomial(5, 2), 0.453125)

    def test_wilson_interval_contains_observed_fraction(self):
        low, high = wilson(2, 27)
        self.assertLessEqual(low, 2 / 27)
        self.assertGreaterEqual(high, 2 / 27)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
