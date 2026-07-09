import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_dataset_pose_validation import failure_kind, wilson_interval


class DatasetPoseAuditTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(93, 100)
        self.assertLess(low, 0.93)
        self.assertGreater(high, 0.93)

    def test_failure_classification_prioritizes_worker_failure(self):
        self.assertEqual(
            failure_kind({"worker_failed": True}),
            "worker_crash_or_timeout",
        )

    def test_collision_failure_is_explicit(self):
        row = {
            "validation_result": {
                "collision_count": 1,
                "unsolved_parts": [],
                "metrics": {
                    "exact_collision_check_status": "success",
                    "assembly_step_build_status": "success",
                },
            }
        }
        self.assertEqual(
            failure_kind(row), "confirmed_solid_penetration"
        )


if __name__ == "__main__":
    unittest.main()
