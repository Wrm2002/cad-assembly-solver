import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sw_dataset_generator.templates import build_case_spec
from sw_dataset_generator.templates.library import FAMILIES_BY_SIZE


class TemplateTests(unittest.TestCase):
    def test_all_group_sizes_are_deterministic_and_valid(self):
        for size in range(1, 7):
            first = build_case_spec(size, 123)
            second = build_case_spec(size, 123)
            self.assertEqual(first, second)
            self.assertEqual(len(first["parts"]), size)
            self.assertEqual(len(first["placements"]), size)
            self.assertTrue(first["hard_negatives"])
            self.assertEqual(len(first["part_semantics"]), size)
            for semantics in first["part_semantics"].values():
                self.assertTrue(semantics["part_category"])
                self.assertTrue(semantics["expected_interfaces"])
                self.assertFalse(semantics["group_identity_disclosed"])

    def test_sizes_four_to_six_have_three_families(self):
        for size in (4, 5, 6):
            observed = {
                build_case_spec(size, seed)["template"]
                for seed in range(12)
            }
            self.assertEqual(observed, set(FAMILIES_BY_SIZE[size]))


if __name__ == "__main__":
    unittest.main()
