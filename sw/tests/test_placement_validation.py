import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from placement_validation import bbox_collisions, transform_point


class PlacementValidationTests(unittest.TestCase):
    def test_axis_angle_then_translation(self):
        point = transform_point(
            [1, 0, 0],
            {
                "rotate_sequence": [{"axis_angle": [0, 0, 1, 90]}],
                "translate": [1, 2, 3],
            },
        )
        self.assertAlmostEqual(point[0], 1.0, places=6)
        self.assertAlmostEqual(point[1], 3.0, places=6)
        self.assertAlmostEqual(point[2], 3.0, places=6)

    def test_bbox_collision(self):
        features = {
            "a": {"bbox": {"min": [0, 0, 0], "max": [10, 10, 10]}},
            "b": {"bbox": {"min": [0, 0, 0], "max": [10, 10, 10]}},
        }
        collisions = bbox_collisions(
            features,
            {"a": {"translate": [0, 0, 0]}, "b": {"translate": [9, 0, 0]}},
        )
        self.assertEqual(len(collisions), 1)
        self.assertFalse(collisions[0]["severe"])

    def test_separated_boxes_do_not_collide(self):
        features = {
            "a": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            "b": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}},
        }
        self.assertFalse(
            bbox_collisions(features, {"a": {}, "b": {"translate": [5, 0, 0]}})
        )


if __name__ == "__main__":
    unittest.main()
