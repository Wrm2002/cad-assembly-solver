import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_proxy import (
    build_proxy,
    compact_ranges,
    range_cardinality,
)


class FeatureProxyTests(unittest.TestCase):
    def test_compact_ranges_is_reversible_by_count(self):
        ranges = compact_ranges([7, 1, 2, 3, 9, 8])
        self.assertEqual(ranges, [[1, 3], [7, 9]])
        self.assertEqual(range_cardinality(ranges), 6)

    def test_equivalent_features_cluster_without_losing_members(self):
        full = {
            "part_id": "p.step",
            "planar_faces": [
                {
                    "feature_id": f"p.step:plane:{index}",
                    "parameters": {
                        "position": [float(index), 0.0, 2.0],
                        "normal": [0.0, 0.0, 1.0],
                        "area": 10.0 + index,
                    },
                }
                for index in range(3)
            ],
            "cylindrical_faces": [
                {
                    "feature_id": f"p.step:cylinder:{index}",
                    "parameters": {
                        "origin": [0.0, 0.0, float(index)],
                        "axis": [0.0, 0.0, 1.0],
                        "radius": 5.0,
                        "area": 20.0,
                    },
                }
                for index in range(2)
            ],
            "holes": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "full.json"
            source.write_text("{}", encoding="utf-8")
            proxy = build_proxy(full, source_path=source)
        self.assertEqual(len(proxy["plane_interfaces"]), 1)
        self.assertEqual(len(proxy["cylindrical_interfaces"]), 1)
        self.assertEqual(len(proxy["plane_families"]), 1)
        self.assertEqual(len(proxy["cylinder_families"]), 1)
        self.assertTrue(
            proxy["compression"]["all_members_accounted_for"]
        )
        self.assertEqual(
            proxy["plane_interfaces"][0]["member_count"], 3
        )


if __name__ == "__main__":
    unittest.main()
