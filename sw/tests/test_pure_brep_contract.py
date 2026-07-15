from __future__ import annotations

import sys
import unittest
from pathlib import Path


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.fusion_contract import (  # noqa: E402
    audit_records,
    free_dof_mask,
    iter_forbidden_model_keys,
    project_brep_graph,
)


class PureBRepContractTests(unittest.TestCase):
    def test_projection_drops_names_paths_and_properties(self):
        projected = project_brep_graph({
            "nodes": [{
                "id": 1,
                "surface_type": "CylinderSurfaceType",
                "radius": 2.0,
                "file_name": "exam_answer.step",
                "part_role": "shaft",
            }],
            "links": [{"source": 1, "target": 2, "name": "leak"}],
            "properties": {"name": "secret"},
        })
        self.assertEqual(projected["nodes"][0]["radius"], 2.0)
        self.assertNotIn("file_name", projected["nodes"][0])
        self.assertNotIn("part_role", projected["nodes"][0])
        self.assertNotIn("name", projected["links"][0])
        self.assertNotIn("properties", projected)

    def test_forbidden_model_key_audit_is_recursive(self):
        paths = list(iter_forbidden_model_keys({
            "geometry": {"part_name": "answer"},
            "safe": [1, 2, 3],
        }))
        self.assertEqual(paths, ["model_input.geometry.part_name"])

    def test_joint_types_supervise_dof_not_named_solver_rules(self):
        self.assertEqual(free_dof_mask("CylindricalJointType"), [0, 0, 1, 0, 0, 1])
        self.assertEqual(free_dof_mask("PlanarJointType"), [1, 1, 0, 0, 0, 1])

    def test_audit_rejects_split_group_overlap(self):
        base = {
            "record_id": "a",
            "model_input": {"representation": "paired_brep_topology"},
            "supervision": [],
            "storage": {"split_group_hash": "same"},
        }
        rows = [{**base, "split": "train"}, {**base, "record_id": "b", "split": "test"}]
        result = audit_records(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["split_group_overlap_count"], 1)


if __name__ == "__main__":
    unittest.main()
