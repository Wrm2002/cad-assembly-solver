from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constraints import PLANAR_MATE  # noqa: E402
from known_group_assembly import _obb_insertion_candidates_for_connection  # noqa: E402


def _obb(dimensions):
    return {
        "center": [0.0, 0.0, 0.0],
        "axes": np.eye(3).tolist(),
        "dimensions": list(dimensions),
    }


class ObbInsertionDepthRecallTests(unittest.TestCase):
    def test_frontier_keeps_nonzero_depths_before_duplicate_zero_depths(self):
        features = {
            "carrier.stp": {"obb": _obb([100.0, 100.0, 100.0])},
            "module.stp": {"obb": _obb([10.0, 20.0, 30.0])},
        }
        source = {
            "placements": {
                "carrier.stp": {"translate": [0.0, 0.0, 0.0]},
                "module.stp": {"translate": [0.0, 0.0, 0.0]},
            },
            "total_score": 1.0,
        }
        connection = {
            "connection_id": "depth-edge",
            "parts": ["carrier.stp", "module.stp"],
            "relation_types": [PLANAR_MATE],
            "matches": [{"type": PLANAR_MATE}],
        }
        frame = {
            "axis_mapping": [0, 1, 2],
            "axis_signs": [1, 1, 1],
            "dimension_order_score": 1.0,
            "rotation_axis_angle": None,
        }
        axis_data = (
            np.asarray([1.0, 0.0, 0.0]),
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0]),
            np.zeros(3),
            None,
        )
        with patch(
            "known_group_assembly.enumerate_axis_role_frames",
            return_value=[frame],
        ), patch(
            "known_group_assembly._joinable_axis_data_for_match",
            return_value=axis_data,
        ):
            rows = _obb_insertion_candidates_for_connection(
                [source],
                connection,
                features,
                max_candidates=6,
                refinement_phase="test",
            )

        depths = [
            row["obb_insertion"]["sampled_depth_fraction"]
            for row in rows
        ]
        self.assertIn(0.0, depths)
        self.assertTrue(any(depth > 0.0 for depth in depths))
        self.assertGreaterEqual(len(set(depths)), 2)
        self.assertTrue(all(row["proposal_only"] for row in rows))
        self.assertTrue(all(row["review_required"] for row in rows))


if __name__ == "__main__":
    unittest.main()
