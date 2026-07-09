import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_real_cases import projected_face_comparisons


class RealValidationTests(unittest.TestCase):
    def test_projected_comparisons_match_nested_face_loops(self):
        features = [
            {
                "part_id": "a",
                "planar_faces": [{}, {}],
                "cylindrical_faces": [{}, {}, {}],
            },
            {
                "part_id": "b",
                "planar_faces": [{}, {}, {}, {}],
                "cylindrical_faces": [{}, {}],
            },
        ]
        self.assertEqual(projected_face_comparisons(features), 14)
        self.assertEqual(
            projected_face_comparisons(features, ["a"]), 0
        )


if __name__ == "__main__":
    unittest.main()
