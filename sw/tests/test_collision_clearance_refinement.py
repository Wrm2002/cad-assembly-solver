from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.collision_clearance_refinement import (
    HARD_MAXIMUM_ITERATIONS,
    propose_collision_clearance_refinement,
)


def _placements(*parts):
    return {part: {"translate": [0.0, 0.0, 0.0]} for part in parts}


def _obb(dimensions=(100.0, 100.0, 100.0)):
    return {
        "center": [0.0, 0.0, 0.0],
        "axes": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "dimensions": list(dimensions),
    }


def _collision(parts, intersections):
    return {
        "parts": list(parts),
        "intersection_volume_mm3": sum(
            row.get("intersection_volume_mm3", 0.0) for row in intersections
        ),
        "solid_intersections": intersections,
    }


def _solid(volume, vector=None, indices=(0, 0)):
    row = {
        "solid_indices": list(indices),
        "intersection_volume_mm3": float(volume),
    }
    if vector is not None:
        row["clearance_translation_for_second_part_mm"] = list(vector)
    return row


def _history(parts, moving_part, dimensions=(100.0, 100.0, 100.0)):
    return [{
        "parts": list(parts),
        "moving_part": moving_part,
        "part_obbs": {moving_part: _obb(dimensions)},
    }]


class CollisionClearanceRefinementTests(unittest.TestCase):
    def test_selects_maximum_volume_solid_intersection(self):
        placements = _placements("fixed", "moving")
        original = copy.deepcopy(placements)
        exact = {
            "status": "success",
            "collisions": [_collision(
                ("fixed", "moving"),
                [
                    _solid(2.0, [1.0, 0.0, 0.0], (0, 0)),
                    _solid(9.0, [0.0, 2.0, 0.0], (1, 3)),
                    _solid(4.0, [0.0, 0.0, 3.0], (2, 2)),
                ],
            )],
        }

        result = propose_collision_clearance_refinement(
            placements,
            exact,
            {"moving"},
            _history(("fixed", "moving"), "moving"),
        )

        self.assertEqual(result["status"], "proposed")
        self.assertEqual(len(result["proposals"]), 1)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["selected_solid_intersection_index"], 1)
        self.assertEqual(proposal["selected_solid_indices"], [1, 3])
        self.assertEqual(proposal["selected_intersection_volume_mm3"], 9.0)
        self.assertEqual(proposal["translation_mm"], [0.0, 2.0, 0.0])
        self.assertEqual(
            result["proposed_placements"]["moving"]["translate"],
            [0.0, 2.0, 0.0],
        )
        self.assertEqual(placements, original)

    def test_over_relative_obb_limit_abstains_without_clipping(self):
        exact = {
            "status": "success",
            "collisions": [_collision(
                ("anchor", "leaf"),
                [_solid(5.0, [2.0, 0.0, 0.0])],
            )],
        }

        result = propose_collision_clearance_refinement(
            _placements("anchor", "leaf"),
            exact,
            {"leaf"},
            _history(("anchor", "leaf"), "leaf", (10.0, 10.0, 10.0)),
        )

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])
        self.assertIn(
            "clearance_translation_exceeds_relative_obb_limit",
            result["rejection_reasons"],
        )
        rejection = result["rejections"][0]
        self.assertAlmostEqual(
            rejection["maximum_translation_norm_mm"], 3.0 ** 0.5
        )
        self.assertEqual(
            result["proposed_placements"]["leaf"]["translate"],
            [0.0, 0.0, 0.0],
        )

    def test_multiple_collisions_emit_only_one_vector_for_part(self):
        exact = {
            "status": "success",
            "collisions": [
                _collision(
                    ("left", "shared"),
                    [_solid(4.0, [1.0, 0.0, 0.0])],
                ),
                _collision(
                    ("right", "shared"),
                    [_solid(12.0, [0.0, 2.0, 0.0])],
                ),
            ],
        }
        history = [
            *_history(("left", "shared"), "shared"),
            *_history(("right", "shared"), "shared"),
        ]

        result = propose_collision_clearance_refinement(
            _placements("left", "right", "shared"),
            exact,
            {"shared"},
            history,
        )

        self.assertEqual(len(result["proposals"]), 1)
        self.assertEqual(result["proposals"][0]["collision_index"], 1)
        self.assertEqual(
            result["proposals"][0]["translation_mm"], [0.0, 2.0, 0.0]
        )
        self.assertIn(
            "one_translation_vector_per_part_per_round_guard",
            result["rejection_reasons"],
        )
        self.assertEqual(
            result["audit"]["candidate_translation_count_before_part_guard"],
            2,
        )
        self.assertEqual(result["audit"]["proposed_translation_count"], 1)

    def test_missing_vector_on_maximum_volume_does_not_fall_back(self):
        exact = {
            "status": "success",
            "collisions": [_collision(
                ("base", "insert"),
                [
                    _solid(20.0, None),
                    _solid(3.0, [0.5, 0.0, 0.0]),
                ],
            )],
        }

        result = propose_collision_clearance_refinement(
            _placements("base", "insert"),
            exact,
            {"insert"},
            _history(("base", "insert"), "insert"),
        )

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])
        self.assertIn(
            "maximum_volume_intersection_has_no_valid_clearance_vector",
            result["rejection_reasons"],
        )
        self.assertEqual(result["rejections"][0]["selected_solid_intersection_index"], 0)

    def test_iteration_and_auto_accept_guards_are_hard(self):
        exact = {
            "status": "success",
            "collisions": [_collision(
                ("one", "two"), [_solid(4.0, [1.0, 0.0, 0.0])]
            )],
        }

        at_limit = propose_collision_clearance_refinement(
            _placements("one", "two"),
            exact,
            {"two"},
            _history(("one", "two"), "two"),
            iteration_index=HARD_MAXIMUM_ITERATIONS,
        )
        expanded = propose_collision_clearance_refinement(
            _placements("one", "two"),
            exact,
            {"two"},
            _history(("one", "two"), "two"),
            maximum_iterations=HARD_MAXIMUM_ITERATIONS + 1,
        )

        for result in (at_limit, expanded):
            self.assertEqual(result["status"], "abstain")
            self.assertTrue(result["proposal_only"])
            self.assertTrue(result["review_required"])
            self.assertFalse(result["can_auto_accept"])
            self.assertFalse(result["collision_free_claimed"])
            self.assertEqual(
                result["collision_status_after_proposal"], "not_evaluated"
            )
            self.assertTrue(result["exact_revalidation_required"])
            self.assertFalse(result["audit"]["exact_collision_rerun_performed"])
            self.assertFalse(
                result["audit"]["filename_or_case_id_heuristics_used"]
            )
        self.assertIn(
            "fixed_iteration_limit_reached", at_limit["rejection_reasons"]
        )
        self.assertIn(
            "maximum_iterations_exceeds_hard_guard",
            expanded["rejection_reasons"],
        )

    def test_first_part_vector_is_inverted_when_it_is_only_allowed_mover(self):
        exact = {
            "status": "success",
            "collisions": [_collision(
                ("movable", "fixed"), [_solid(6.0, [0.0, 0.0, 1.5])]
            )],
        }
        history = [{
            "parts": ["movable", "fixed"],
            "moving_part": "movable",
            "moving_obb": _obb(),
        }]

        result = propose_collision_clearance_refinement(
            _placements("movable", "fixed"),
            exact,
            {"movable"},
            history,
        )

        proposal = result["proposals"][0]
        self.assertEqual(proposal["translation_mm"], [0.0, 0.0, -1.5])
        self.assertTrue(proposal["vector_inverted_from_second_part"])


if __name__ == "__main__":
    unittest.main()
