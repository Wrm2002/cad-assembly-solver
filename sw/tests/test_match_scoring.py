import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from match_scoring import score_match


class MatchScoringTests(unittest.TestCase):
    def setUp(self):
        self.features = {
            "a.step": {
                "cylinders": [
                    {"radius": 10.0, "axis": [0, 0, 1], "area": 2000.0},
                    {"radius": 20.0, "axis": [0, 0, 1], "area": 4000.0},
                ],
                "planes": [
                    {"normal": [0, 0, 1], "area": 5000.0},
                ],
            },
            "b.step": {
                "cylinders": [
                    {"radius": 10.1, "axis": [0, 0, -1], "area": 1900.0},
                    {"radius": 30.0, "axis": [0, 0, 1], "area": 4000.0},
                ],
                "planes": [
                    {"normal": [0, 0, -1], "area": 4800.0},
                ],
            },
        }

    def test_good_coaxial_is_high_confidence(self):
        result = score_match(
            {
                "type": "coaxial",
                "parts": ("a.step", "b.step"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "radius_match": 0.1,
            },
            self.features,
        )
        self.assertGreaterEqual(result["score"], 0.75)
        self.assertEqual(result["confidence"], "high")
        self.assertIn("radius_diff", result["reason"])

    def test_large_clearance_is_penalized(self):
        result = score_match(
            {
                "type": "clearance",
                "parts": ("a.step", "b.step"),
                "feat_a_idx": 0,
                "feat_b_idx": 1,
                "gap": 20.0,
            },
            self.features,
        )
        self.assertLess(result["score"], 0.5)
        self.assertEqual(result["confidence"], "low")

    def test_large_opposing_planes_score_high(self):
        result = score_match(
            {
                "type": "planar_mate",
                "parts": ("a.step", "b.step"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "distance": 0.1,
            },
            self.features,
        )
        self.assertGreaterEqual(result["score"], 0.75)
        self.assertGreater(result["reason"]["area_ratio"], 0.9)

    def test_planar_align_gets_discount(self):
        self.features["b.step"]["planes"][0]["normal"] = [0, 0, 1]
        result = score_match(
            {
                "type": "planar_align",
                "parts": ("a.step", "b.step"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "distance": 0.1,
            },
            self.features,
        )
        self.assertEqual(result["reason"]["planar_align_discount"], 0.75)
        self.assertLess(result["score"], 0.75)

    def test_pocket_reason_is_structured(self):
        result = score_match(
            {
                "type": "pocket_mate",
                "parts": ("a.step", "b.step"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "_size_a": [10.0, 20.0],
                "_size_b": [10.2, 19.8],
                "_dir_a": [0, 0, 1],
                "_dir_b": [0, 0, -1],
                "_wall_a": [1, 0, 0],
                "_wall_b": [-1, 0, 0],
            },
            self.features,
        )
        self.assertGreaterEqual(result["score"], 0.9)
        self.assertIn("pocket_size_similarity", result["reason"])


if __name__ == "__main__":
    unittest.main()
