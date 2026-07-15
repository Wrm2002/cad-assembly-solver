from __future__ import annotations

import unittest

from run_case1234_locked_visual_workflow import (
    _canonical_pose,
    _numbers_close,
    validate_visual_decision,
)


class LockedCase1234WorkflowTests(unittest.TestCase):
    def test_visual_abstain_cannot_pass_as_success(self):
        errors = validate_visual_decision({
            "status": "unresolved",
            "visual_semantics_used_for_ranking": False,
            "geometry_scores_exposed_to_visual_model": False,
            "semantic_auto_accept_enabled": False,
            "selected_manifest": None,
        })
        self.assertIn("visual semantics were not used for ranking", errors)
        self.assertIn("no selected review manifest", errors)

    def test_geometry_score_leak_is_rejected(self):
        errors = validate_visual_decision({
            "status": "review",
            "visual_semantics_used_for_ranking": True,
            "geometry_scores_exposed_to_visual_model": True,
            "semantic_auto_accept_enabled": False,
            "selected_manifest": "selected.json",
        })
        self.assertIn("geometry scores leaked into visual model input", errors)

    def test_pose_comparison_is_label_order_independent_and_tolerant(self):
        left = _canonical_pose({"components": [
            {"label": "B", "source": "b.step", "placement": {"translate": [1, 2, 3]}},
            {"label": "A", "source": "a.step", "placement": {"translate": [0, 0, 0]}},
        ]})
        right = _canonical_pose({"components": [
            {"label": "A", "source": "a.step", "placement": {"translate": [0.0, 0.0, 0.0]}},
            {"label": "B", "source": "b.step", "placement": {"translate": [1.0000001, 2, 3]}},
        ]})
        self.assertTrue(_numbers_close(left, right))


if __name__ == "__main__":
    unittest.main()
