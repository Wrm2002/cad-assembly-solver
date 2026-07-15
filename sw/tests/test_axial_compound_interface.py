from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SW_DIR = Path(__file__).resolve().parents[1]
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

from pose_search.axial_compound_interface import (  # noqa: E402
    construct_compound_transform,
    correspondence_phase_degrees,
    phase_residual_degrees,
    recall_axial_compound_candidates,
    validate_axial_compound_pose,
)


def _unit(value: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    return vector / np.linalg.norm(vector)


def _axis_basis(axis: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = _unit(axis)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, direction))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = _unit(np.cross(direction, seed))
    second = _unit(np.cross(direction, first))
    return first, second


def _radial_point(
    axis: list[float], angle_degrees: float, radius: float
) -> list[float]:
    first, second = _axis_basis(axis)
    radians = math.radians(angle_degrees)
    point = radius * (math.cos(radians) * first + math.sin(radians) * second)
    return point.tolist()


def _flange_summary(
    axis: list[float],
    end_normal: list[float],
    end_position: list[float],
    *,
    hole_angles: tuple[float, ...] = (0, 60, 120, 180, 240, 300),
) -> dict:
    cylinders = [{
        "origin": [0.0, 0.0, 0.0],
        "axis": list(axis),
        "radius": 20.0,
        "area": 500.0,
    }]
    cylinders.extend({
        "origin": _radial_point(axis, angle, 30.0),
        "axis": list(axis),
        "radius": 2.0,
        "area": 25.0,
    } for angle in hole_angles)
    return {
        "cylinders": cylinders,
        "planes": [{
            "position": list(end_position),
            "normal": list(end_normal),
            "area": 1000.0,
        }],
    }


def _slot_graph(*, end_normal: list[float]) -> dict:
    axis = [0.0, 0.0, 1.0]
    nodes: list[dict] = [{
        "node_id": "main_cylinder",
        "entity_type": "face",
        "surface_type": "cylinder",
        "axis_origin": [0.0, 0.0, 0.0],
        "axis_direction": axis,
        "radius": 20.0,
        "area": 500.0,
    }]
    for index, angle in enumerate((0, 60, 120, 180, 240, 300)):
        nodes.append({
            "node_id": f"hole_{index}",
            "entity_type": "face",
            "surface_type": "cylinder",
            "axis_origin": _radial_point(axis, angle, 30.0),
            "axis_direction": axis,
            "radius": 2.0,
            "area": 25.0,
        })
    nodes.extend([
        {
            "node_id": "end_face",
            "entity_type": "face",
            "surface_type": "plane",
            "centroid": [0.0, 0.0, 5.0],
            "normal": list(end_normal),
            "area": 1000.0,
        },
        {
            "node_id": "slot_wall_left",
            "entity_type": "face",
            "surface_type": "plane",
            "centroid": [10.0, -2.0, 0.0],
            "normal": [0.0, 1.0, 0.0],
            "area": 40.0,
        },
        {
            "node_id": "slot_wall_right",
            "entity_type": "face",
            "surface_type": "plane",
            "centroid": [10.0, 2.0, 0.0],
            "normal": [0.0, -1.0, 0.0],
            "area": 40.0,
        },
        {
            "node_id": "slot_bottom",
            "entity_type": "face",
            "surface_type": "plane",
            "centroid": [12.0, 0.0, 0.0],
            "normal": [-1.0, 0.0, 0.0],
            "area": 20.0,
        },
        {
            "node_id": "edge_slot_left",
            "entity_type": "edge",
            "topology_feature_status": "success",
            "convexity": "concave",
            "adjacent_face_ids": ["slot_wall_left", "slot_bottom"],
        },
        {
            "node_id": "edge_slot_right",
            "entity_type": "edge",
            "topology_feature_status": "success",
            "convexity": "concave",
            "adjacent_face_ids": ["slot_wall_right", "slot_bottom"],
        },
    ])
    return {
        "metadata": {"edge_topology_features": {"available": True}},
        "nodes": nodes,
    }


def _desired_proposals(candidate: dict) -> list[dict]:
    polarity = candidate["required_axis_polarity_for_opposed_faces"]
    return [
        row for row in candidate["proposals"]
        if row["axis_polarity"] == polarity
    ]


class AxialCompoundInterfaceTests(unittest.TestCase):
    def test_recalls_end_faces_in_unrelated_local_x_and_y_frames(self):
        fixed = _flange_summary(
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        )
        moving = _flange_summary(
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 7.0, 0.0],
        )
        # Raw normals are orthogonal because the parts have not been placed.
        self.assertAlmostEqual(np.dot(
            fixed["planes"][0]["normal"], moving["planes"][0]["normal"]
        ), 0.0)

        result = recall_axial_compound_candidates(fixed, moving)
        self.assertEqual(result["status"], "recalled")
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(
            candidate["normal_recall_method"],
            "each_end_face_vs_its_own_main_axis",
        )
        self.assertEqual(candidate["required_axis_polarity_for_opposed_faces"], -1)
        for row in candidate["proposals"]:
            self.assertAlmostEqual(row["rotation_determinant"], 1.0, places=7)
        proposal = _desired_proposals(candidate)[0]

        validation = validate_axial_compound_pose(
            candidate, proposal, collision_free=True
        )
        self.assertTrue(validation["compound_constraints_satisfied"])
        self.assertAlmostEqual(validation["radial_center_residual_mm"], 0.0, places=7)
        self.assertAlmostEqual(
            validation["end_face_distance_residual_mm"], 0.0, places=7
        )
        # Six geometrically equivalent phases are proposals, not auto closure.
        self.assertTrue(validation["proposal_only"])
        self.assertFalse(validation["is_closed"])

    def test_c6_orbit_quotients_sixty_degrees_but_not_thirty(self):
        result = recall_axial_compound_candidates(
            _flange_summary([0, 0, 1], [0, 0, 1], [0, 0, 5]),
            _flange_summary([0, 0, 1], [0, 0, -1], [0, 0, -5]),
        )
        candidate = result["candidates"][0]
        proposal = _desired_proposals(candidate)[0]
        orbit = proposal["phase_orbit_degrees"]
        self.assertEqual(proposal["interface_symmetry_order"], 6)
        self.assertEqual(proposal["whole_part_symmetry_order"], 6)
        self.assertEqual(len(orbit), 6)
        self.assertAlmostEqual(phase_residual_degrees(60.0, orbit), 0.0)
        self.assertAlmostEqual(phase_residual_degrees(30.0, orbit), 30.0)
        self.assertEqual(proposal["phase_status"], "periodic_interface_only")
        self.assertTrue(proposal["review_required"])

    def test_topological_slot_reduces_c6_to_c1_and_selects_phase(self):
        result = recall_axial_compound_candidates(
            _slot_graph(end_normal=[0.0, 0.0, 1.0]),
            _slot_graph(end_normal=[0.0, 0.0, -1.0]),
        )
        self.assertEqual(len(result["fixed_directional_witnesses"]), 1)
        self.assertEqual(
            result["fixed_directional_witnesses"][0]["kind"],
            "topological_key_slot",
        )
        candidate = result["candidates"][0]
        proposal = [
            row for row in _desired_proposals(candidate)
            if not row["proposal_only"]
        ][0]
        self.assertEqual(proposal["interface_symmetry_order"], 6)
        self.assertEqual(proposal["whole_part_symmetry_order"], 1)
        self.assertEqual(len(proposal["interface_phase_orbit_degrees"]), 6)
        self.assertEqual(len(proposal["phase_orbit_degrees"]), 1)
        self.assertEqual(
            proposal["phase_status"], "resolved_by_asymmetric_witness"
        )
        self.assertAlmostEqual(proposal["phase_residual_deg"], 0.0)

    def test_axis_polarity_changes_phase_sign_without_reflection(self):
        fixed_angle = 20.0
        moving_angle = 70.0
        self.assertAlmostEqual(
            correspondence_phase_degrees(fixed_angle, moving_angle, 1), -50.0
        )
        self.assertAlmostEqual(
            correspondence_phase_degrees(fixed_angle, moving_angle, -1), 90.0
        )

        result = recall_axial_compound_candidates(
            _flange_summary([0, 0, 1], [0, 0, 1], [0, 0, 5]),
            _flange_summary([0, 0, 1], [0, 0, -1], [0, 0, -5]),
        )
        candidate = result["candidates"][0]
        moving_first, moving_second = _axis_basis([0, 0, 1])
        moving_vector = (
            math.cos(math.radians(moving_angle)) * moving_first
            + math.sin(math.radians(moving_angle)) * moving_second
        )
        fixed_first, fixed_second = _axis_basis([0, 0, 1])
        for polarity in (1, -1):
            phase = correspondence_phase_degrees(
                fixed_angle, moving_angle, polarity
            )
            transform = construct_compound_transform(
                candidate,
                axis_polarity=polarity,
                phase_degrees=phase,
            )
            self.assertAlmostEqual(
                np.linalg.det(transform[:3, :3]), 1.0, places=7
            )
            mapped = transform[:3, :3] @ moving_vector
            mapped_angle = math.degrees(math.atan2(
                float(np.dot(mapped, fixed_second)),
                float(np.dot(mapped, fixed_first)),
            ))
            self.assertAlmostEqual(mapped_angle, fixed_angle, places=7)

    def test_collision_free_wrong_phase_cannot_close_compound_interface(self):
        result = recall_axial_compound_candidates(
            _slot_graph(end_normal=[0.0, 0.0, 1.0]),
            _slot_graph(end_normal=[0.0, 0.0, -1.0]),
        )
        candidate = result["candidates"][0]
        proposal = [
            row for row in _desired_proposals(candidate)
            if not row["proposal_only"]
        ][0]
        correct = validate_axial_compound_pose(
            candidate, proposal, collision_free=True
        )
        self.assertTrue(correct["is_closed"])

        wrong_transform = construct_compound_transform(
            candidate,
            axis_polarity=proposal["axis_polarity"],
            phase_degrees=proposal["phase_degrees"] + 30.0,
        )
        wrong = validate_axial_compound_pose(
            candidate,
            proposal,
            collision_free=True,
            transform=wrong_transform,
        )
        self.assertTrue(wrong["collision_free"])
        self.assertAlmostEqual(wrong["phase_residual_deg"], 30.0, places=7)
        self.assertFalse(wrong["checks"]["phase_in_active_orbit"])
        self.assertFalse(wrong["compound_constraints_satisfied"])
        self.assertFalse(wrong["is_closed"])


if __name__ == "__main__":
    unittest.main()
