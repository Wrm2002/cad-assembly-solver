import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constraints import CLEARANCE, COAXIAL
from known_group_assembly import _constraint_satisfied
from placement_validation import constraint_residual


def _cylinder(radius, origin, axis=(0.0, 0.0, 1.0)):
    return {
        "radius": float(radius),
        "origin": [float(value) for value in origin],
        "axis": [float(value) for value in axis],
    }


def _match(kind=COAXIAL):
    return {
        "type": kind,
        "parts": ["a", "b"],
        "feat_a_idx": 0,
        "feat_b_idx": 0,
    }


class DominantAxisGuardTests(unittest.TestCase):
    def test_small_selected_holes_cannot_hide_offset_dominant_axes(self):
        features = {
            "a": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [0, 0, 0]),
            ]},
            "b": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [12, 0, 0]),
            ]},
        }

        record = constraint_residual(_match(), features, {"a": {}, "b": {}})

        self.assertEqual(record["axis_angle_deg"], 0.0)
        self.assertEqual(record["radial_distance"], 0.0)
        self.assertTrue(record["dominant_axis_guard_required"])
        self.assertFalse(record["dominant_axis_guard_passed"])
        self.assertAlmostEqual(
            record["dominant_axis_radial_distance_mm"], 12.0
        )
        self.assertIn(
            "radial_distance_exceeds_threshold",
            record["dominant_axis_guard_reason"],
        )
        self.assertFalse(_constraint_satisfied(_match(), record))

    def test_required_guard_passes_for_aligned_dominant_axes(self):
        features = {
            "a": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [0, 0, 0]),
            ]},
            "b": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [0.5, 0, 0]),
            ]},
        }

        record = constraint_residual(_match(), features, {"a": {}, "b": {}})

        self.assertTrue(record["dominant_axis_guard_required"])
        self.assertTrue(record["dominant_axis_guard_passed"])
        self.assertAlmostEqual(record["dominant_axis_angle_deg"], 0.0)
        self.assertAlmostEqual(
            record["dominant_axis_radial_distance_mm"], 0.5
        )
        self.assertTrue(_constraint_satisfied(_match(), record))

    def test_required_guard_rejects_dominant_axis_angle_over_two_degrees(self):
        angle = math.radians(3.0)
        tilted = [math.sin(angle), 0.0, math.cos(angle)]
        features = {
            "a": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [0, 0, 0]),
            ]},
            "b": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [0, 0, 0], tilted),
            ]},
        }

        record = constraint_residual(_match(), features, {"a": {}, "b": {}})

        self.assertAlmostEqual(record["dominant_axis_angle_deg"], 3.0)
        self.assertEqual(record["dominant_axis_radial_distance_mm"], 0.0)
        self.assertFalse(record["dominant_axis_guard_passed"])
        self.assertFalse(_constraint_satisfied(_match(), record))

    def test_guard_is_not_triggered_when_either_selected_axis_is_dominant(self):
        features = {
            "a": {"cylinders": [_cylinder(10.0, [0, 0, 0])]},
            "b": {"cylinders": [
                _cylinder(2.0, [0, 0, 0]),
                _cylinder(10.0, [25, 0, 0]),
            ]},
        }

        record = constraint_residual(_match(), features, {"a": {}, "b": {}})

        self.assertFalse(record["dominant_axis_guard_required"])
        self.assertIsNone(record["dominant_axis_guard_passed"])
        self.assertEqual(record["dominant_axis_selected_radius_ratio_a"], 1.0)
        self.assertTrue(_constraint_satisfied(_match(), record))

    def test_clearance_closure_also_requires_the_guard(self):
        bbox = {"min": [-5, -5, -5], "max": [5, 5, 5]}
        features = {
            "a": {
                "bbox": bbox,
                "cylinders": [
                    _cylinder(2.0, [0, 0, 0]),
                    _cylinder(10.0, [0, 0, 0]),
                ],
            },
            "b": {
                "bbox": bbox,
                "cylinders": [
                    _cylinder(2.3, [0, 0, 0]),
                    _cylinder(10.0, [8, 0, 0]),
                ],
            },
        }
        match = _match(CLEARANCE)

        record = constraint_residual(match, features, {"a": {}, "b": {}})

        self.assertEqual(record["axial_overlap_ratio"], 1.0)
        self.assertFalse(record["dominant_axis_guard_passed"])
        self.assertFalse(_constraint_satisfied(match, record))

    def test_legacy_axial_record_without_guard_fields_remains_compatible(self):
        record = {
            "valid": True,
            "axis_angle_deg": 0.0,
            "radial_distance": 0.0,
        }
        self.assertTrue(_constraint_satisfied(_match(), record))


if __name__ == "__main__":
    unittest.main()
