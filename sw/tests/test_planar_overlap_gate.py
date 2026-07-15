import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constraints import PLANAR_MATE
from known_group_assembly import _constraint_satisfied


class PlanarOverlapGateTests(unittest.TestCase):
    def setUp(self):
        self.match = {"type": PLANAR_MATE, "parts": ["a", "b"]}
        self.base_record = {
            "valid": True,
            "normal_angle_deg": 0.0,
            "plane_distance": 0.0,
        }

    def test_distant_coplanar_bounded_faces_do_not_close(self):
        record = {**self.base_record, "bounded_overlap_ratio": 0.0}
        self.assertFalse(_constraint_satisfied(self.match, record))

    def test_positive_bounded_overlap_closes_planar_constraint(self):
        record = {**self.base_record, "bounded_overlap_ratio": 0.5}
        self.assertTrue(_constraint_satisfied(self.match, record))

    def test_tiny_sliver_below_minimum_does_not_close(self):
        record = {**self.base_record, "bounded_overlap_ratio": 0.005}
        self.assertFalse(_constraint_satisfied(self.match, record))

    def test_missing_footprint_preserves_legacy_constraint_behavior(self):
        self.assertTrue(_constraint_satisfied(self.match, self.base_record))


if __name__ == "__main__":
    unittest.main()
