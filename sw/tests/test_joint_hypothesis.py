from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from learned_joint.hypothesis import (  # noqa: E402
    attach_pose_initials,
    build_joint_hypotheses,
    frame_from_entity,
)


def _entity(entity_id: str, kind: str, origin=(0, 0, 0), axis=(0, 0, 1)):
    key = "normal" if "Plane" in kind else "axis_direction"
    return {
        "entity_id": entity_id,
        "geometry_type": kind,
        "axis_origin": list(origin),
        "centroid": list(origin),
        key: list(axis),
    }


class JointHypothesisTests(unittest.TestCase):
    def test_axis_pair_retains_axial_translation_and_rotation(self):
        rows = build_joint_hypotheses("a", "b", [{
            "rank": 1,
            "probability": 0.8,
            "node_a": _entity("fa", "CylinderSurfaceType"),
            "node_b": _entity("fb", "Circle3DCurveType", origin=(5, 0, 2)),
        }])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].manifold_type, "axis_coincidence")
        self.assertEqual(rows[0].free_dof_mask, (0, 0, 1, 0, 0, 1))
        self.assertFalse(rows[0].provenance["fixed_relative_pose"])

    def test_polarity_is_retained_before_additional_phase_variants(self):
        rows = build_joint_hypotheses("a", "b", [{
            "rank": 1,
            "probability": 0.8,
            "node_a": _entity("fa", "CylinderSurfaceType"),
            "node_b": _entity("fb", "Circle3DCurveType"),
            "rotation_hypotheses": [0.0, 90.0],
        }])
        self.assertEqual({rows[0].polarity, rows[1].polarity}, {-1, 1})
        self.assertEqual(rows[0].phase_degrees, 0.0)
        self.assertEqual(rows[1].phase_degrees, 0.0)

    def test_plane_pair_retains_in_plane_se2_freedom(self):
        rows = build_joint_hypotheses("a", "b", [{
            "rank": 1,
            "probability": 0.5,
            "node_a": _entity("fa", "PlaneSurfaceType"),
            "node_b": _entity("fb", "PlaneSurfaceType", origin=(1, 2, 3)),
        }], enumerate_polarity=False)
        self.assertEqual(rows[0].free_dof_mask, (1, 1, 0, 0, 0, 1))
        self.assertEqual(rows[0].symmetry_class["group"], "SE2")

    def test_asymmetric_local_witness_constrains_axial_phase(self):
        left = _entity("fa", "CylinderSurfaceType")
        right = _entity("fb", "CylinderSurfaceType")
        for row in (left, right):
            row["local_direction"] = [1, 0, 0]
            row["local_asymmetry_score"] = 0.8
        rows = build_joint_hypotheses("a", "b", [{
            "rank": 1, "probability": 0.8,
            "node_a": left, "node_b": right,
        }])
        self.assertEqual(rows[0].free_dof_mask, (0, 0, 1, 0, 0, 0))
        self.assertTrue(rows[0].symmetry_class["local_asymmetric_witness"])
        self.assertIn(-180.0, {row.phase_degrees for row in rows})

    def test_pair_pose_is_initial_point_not_fixed_constraint(self):
        rows = build_joint_hypotheses("a", "b", [{
            "rank": 1, "probability": 0.8,
            "node_a": _entity("fa", "CylinderSurfaceType"),
            "node_b": _entity("fb", "CylinderSurfaceType"),
        }])
        transform = np.eye(4)
        transform[2, 3] = 25.0
        enriched = attach_pose_initials(rows, [{
            "entity_a": "fa", "entity_b": "fb",
            "transform": transform.tolist(),
            "axis_flip": False,
            "rotation_seed_degrees": 0.0,
            "evaluation": {"cost": -1.0, "contact": 0.2, "overlap": 0.0},
            "exact_collision": {"status": "success", "collisions": []},
        }])
        self.assertEqual(enriched[0].initial_pose_b_in_a[2][3], 25.0)
        self.assertEqual(enriched[0].free_dof_mask, (0, 0, 1, 0, 0, 1))
        self.assertFalse(enriched[0].provenance["initial_pose_is_constraint"])

    def test_frame_is_proper_rigid_for_arbitrary_axis(self):
        frame = frame_from_entity(_entity("e", "CylinderSurfaceType", axis=(1, 2, 3)))
        self.assertIsNotNone(frame)
        self.assertAlmostEqual(float(np.linalg.det(frame[:3, :3])), 1.0, places=7)
        np.testing.assert_allclose(frame[:3, :3].T @ frame[:3, :3], np.eye(3), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
