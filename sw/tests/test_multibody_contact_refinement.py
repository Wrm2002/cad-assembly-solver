from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from global_pose_solver import MultiBodyContactRefiner  # noqa: E402


def _pose(x: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[0, 3] = x
    return value


class MultiBodyContactRefinementTests(unittest.TestCase):
    def test_nonedge_overlap_is_reduced_without_auto_accept(self):
        box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        refiner = MultiBodyContactRefiner(
            {"a": box, "b": box}, [], sample_count=256, seed=7
        )
        result = refiner.refine(
            {"a": _pose(0.0), "b": _pose(0.5)},
            translation_bound_mm=4.0,
            maxiter=40,
        )
        before = result["initial_pair_scores"][0]["overlap"]
        after = result["final_pair_scores"][0]["overlap"]
        self.assertLess(after, before)
        self.assertFalse(result["accepted"])
        self.assertTrue(result["review_required"])


if __name__ == "__main__":
    unittest.main()
