from __future__ import annotations

import sys
import unittest
from pathlib import Path


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from annotate_compound_pair_exact import annotate_frontier, build_pair_probe


class CompoundPairExactAnnotationTests(unittest.TestCase):
    def test_only_compound_rows_are_probed_and_raw_indices_survive_sorting(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, 1]]
        frontier = {"joint_hypotheses": {"rows": [
            {"manifold_type": "plane_coincidence", "initial_pose_b_in_a": identity},
            {"manifold_type": "compound_multi_axis_rigid", "initial_pose_b_in_a": identity},
            {"manifold_type": "compound_prismatic_insertion_rigid", "initial_pose_b_in_a": identity},
        ]}}
        probe, indices = build_pair_probe(frontier, "a", "b")
        self.assertEqual(indices, [1, 2])
        probe["hypotheses"][0]["exact_validation"] = {
            "status": "failed",
            "occt": {"collisions": [{"intersection_volume_mm3": 2.5}]},
        }
        probe["hypotheses"][1]["exact_validation"] = {
            "status": "valid", "occt": {"collisions": []}
        }
        # Simulate the exact validator sorting valid candidates first.
        probe["hypotheses"].reverse()
        result = annotate_frontier(frontier, probe)
        rows = result["joint_hypotheses"]["rows"]
        self.assertNotIn("provenance", rows[0])
        self.assertFalse(rows[1]["provenance"]["pair_exact_collision_free"])
        self.assertEqual(
            rows[1]["provenance"]["pair_exact_intersection_volume_mm3"], 2.5
        )
        self.assertTrue(rows[2]["provenance"]["pair_exact_collision_free"])


if __name__ == "__main__":
    unittest.main()
