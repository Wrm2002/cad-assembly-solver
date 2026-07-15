import json
import tempfile
import unittest
from pathlib import Path

from case5_folded_flange_insertion import physical_holes
from run_case5_visual_semantic_workflow import (
    Checkpoints,
    parse_cpu_affinity,
    select_diverse_collision_shortlist,
)


class Case5WorkflowSafetyTests(unittest.TestCase):
    def test_counterbore_faces_count_as_one_physical_hole(self):
        rows = [
            {
                "face_index": 1,
                "radius": 1.0,
                "centre": [0.0, 4.0, 8.0],
                "axis": [1.0, 0.0, 0.0],
            },
            {
                "face_index": 2,
                "radius": 2.0,
                "centre": [2.5, 4.0, 8.0],
                "axis": [-1.0, 0.0, 0.0],
            },
            {
                "face_index": 3,
                "radius": 1.0,
                "centre": [0.0, 10.0, 8.0],
                "axis": [1.0, 0.0, 0.0],
            },
        ]
        clustered = physical_holes(rows)
        self.assertEqual(len(clustered), 2)
        self.assertEqual(len(clustered[0].members), 2)

    def test_cpu_affinity_range_parser(self):
        self.assertEqual(parse_cpu_affinity("0-3,7"), (0, 1, 2, 3, 7))

    def test_collision_shortlist_prefers_distinct_pose_families(self):
        rotation_a = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        rotation_b = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        rotation_c = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        rows = [
            {"candidate_id": "A1", "R": rotation_a, "candidate_sources": ["analytic", "hole_pattern"]},
            {"candidate_id": "A2", "R": rotation_a, "candidate_sources": ["analytic", "hole_pattern"]},
            {"candidate_id": "A3", "R": rotation_a, "candidate_sources": ["analytic", "hole_pattern"]},
            {"candidate_id": "B1", "R": rotation_b, "candidate_sources": ["analytic", "folded_flange"]},
            {"candidate_id": "C1", "R": rotation_c, "candidate_sources": ["analytic", "hole_pattern"]},
        ]
        selected = select_diverse_collision_shortlist(rows, 4)
        self.assertEqual(
            [row["candidate_id"] for row in selected], ["A1", "B1", "C1", "A2"]
        )

    def test_checkpoint_rejects_corrupted_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
            checkpoints = Checkpoints(root, resume=True)
            checkpoints.complete("render", [artifact])
            self.assertTrue(checkpoints.valid("render"))

            artifact.write_text(json.dumps({"ok": False}), encoding="utf-8")
            self.assertFalse(checkpoints.valid("render"))


if __name__ == "__main__":
    unittest.main()
