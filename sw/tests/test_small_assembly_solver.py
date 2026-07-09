import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from small_assembly_solver import (
    _bbox_collision_penalty,
    solve_small_assembly,
)


class SmallAssemblySolverTests(unittest.TestCase):
    def test_satisfied_interface_discount_preserves_unrelated_collision_penalty(self):
        collisions = [
            {
                "parts": ["shaft", "housing"],
                "minimum_part_volume_ratio": 1.0,
            },
            {
                "parts": ["shaft", "wrong_part"],
                "minimum_part_volume_ratio": 1.0,
            },
        ]
        penalty = _bbox_collision_penalty(
            collisions,
            {("housing", "shaft")},
        )
        self.assertAlmostEqual(penalty, 2.1)

    def test_single_part(self):
        result = solve_small_assembly(
            {"only.step": {"cylinders": [], "planes": [], "bbox": None}},
            [],
        )
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["unsolved_parts"])

    def test_two_coaxial_parts(self):
        features = {
            "a.step": {
                "cylinders": [{"radius": 10, "origin": [0, 0, 0], "axis": [0, 0, 1]}],
                "planes": [],
                "bbox": {"min": [-10, -10, 0], "max": [10, 10, 20]},
            },
            "b.step": {
                "cylinders": [{"radius": 10, "origin": [0, 0, 5], "axis": [0, 0, 1]}],
                "planes": [],
                "bbox": {"min": [-10, -10, 5], "max": [10, 10, 25]},
            },
        }
        matches = [{
            "type": "coaxial", "parts": ("a.step", "b.step"),
            "feat_a_idx": 0, "feat_b_idx": 0, "score": 0.9,
        }]
        result = solve_small_assembly(features, matches, beam_width=5)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["expanded_states"], 0)
        self.assertIn("b.step", result["placements"])

    def test_disconnected_part_returns_partial(self):
        features = {
            name: {"cylinders": [], "planes": [], "bbox": None}
            for name in ("a", "b")
        }
        result = solve_small_assembly(features, [])
        self.assertEqual(result["status"], "partial_success")
        self.assertEqual(len(result["unsolved_parts"]), 1)

    def test_three_part_chain_propagates_global_reference_placement(self):
        def part(z):
            return {
                "cylinders": [],
                "planes": [
                    {"normal": [0, 0, 1], "position": [0, 0, z], "area": 1000},
                    {"normal": [0, 0, -1], "position": [0, 0, z + 10], "area": 1000},
                ],
                "bbox": {"min": [-5, -5, z], "max": [5, 5, z + 10]},
            }

        features = {"a": part(0), "b": part(20), "c": part(40)}
        matches = [
            {
                "type": "planar_mate", "parts": ("a", "b"),
                "feat_a_idx": 0, "feat_b_idx": 1, "score": 0.9,
            },
            {
                "type": "planar_mate", "parts": ("b", "c"),
                "feat_a_idx": 0, "feat_b_idx": 1, "score": 0.9,
            },
        ]
        result = solve_small_assembly(features, matches, beam_width=5)
        self.assertEqual(result["status"], "success")
        self.assertNotEqual(
            result["placements"]["b"]["translate"],
            result["placements"]["c"]["translate"],
        )

    def test_placement_priority_defers_key_until_axial_skeleton(self):
        def part(size):
            return {
                "cylinders": [],
                "planes": [
                    {
                        "normal": [1, 0, 0],
                        "position": [size, 0, 0],
                        "area": 100,
                    }
                ],
                "bbox": {
                    "min": [-size, -size, -size],
                    "max": [size, size, size],
                },
            }

        features = {
            "hub": part(10),
            "shaft": part(8),
            "key": part(2),
        }
        features["hub"]["cylinders"] = [
            {"radius": 8.2, "origin": [0, 0, 0], "axis": [0, 0, 1]}
        ]
        features["shaft"]["cylinders"] = [
            {"radius": 8.0, "origin": [0, 0, 0], "axis": [0, 0, 1]}
        ]
        matches = [
            {
                "type": "clearance",
                "parts": ("hub", "shaft"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "score": 0.95,
            },
            {
                "type": "planar_mate",
                "parts": ("hub", "shaft"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "score": 0.8,
            },
            {
                "type": "planar_mate",
                "parts": ("hub", "key"),
                "feat_a_idx": 0,
                "feat_b_idx": 0,
                "score": 0.9,
            },
        ]
        result = solve_small_assembly(
            features,
            matches,
            beam_width=10,
            target_branching=2,
            placement_priority={"shaft": 2.5, "key": 0.0},
            preferred_axial_pairs={("hub", "shaft")},
        )
        self.assertEqual(result["selected_mates"][0]["target"], "shaft")
        self.assertEqual(
            result["selected_mates"][0]["evidence"][0]["type"],
            "clearance",
        )
        self.assertEqual(result["selected_mates"][1]["target"], "key")


if __name__ == "__main__":
    unittest.main()
