from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.edge_slot_interface import recall_edge_slot_interface_proposals


def _obb(center, axes, dimensions):
    return {
        "center": list(center),
        "axes": np.asarray(axes, dtype=float).tolist(),
        "dimensions": list(dimensions),
    }


def _slot(center, long_axis, spacing_axis, *, length=100.0, width=3.0):
    normal = np.cross(long_axis, spacing_axis)
    return {
        "normal": np.asarray(normal, dtype=float).tolist(),
        "centroid": np.asarray(center, dtype=float).tolist(),
        "footprint_axes": [
            np.asarray(long_axis, dtype=float).tolist(),
            np.asarray(spacing_axis, dtype=float).tolist(),
        ],
        "footprint_dimensions": [float(length), float(width)],
        "area": float(length * width),
    }


def _wall(center, long_axis, normal, *, length=100.0, width=1.211):
    long_axis = np.asarray(long_axis, dtype=float)
    normal = np.asarray(normal, dtype=float)
    short_axis = np.cross(normal, long_axis)
    short_axis /= np.linalg.norm(short_axis)
    return {
        "normal": normal.tolist(),
        "centroid": np.asarray(center, dtype=float).tolist(),
        "footprint_axes": [long_axis.tolist(), short_axis.tolist()],
        "footprint_dimensions": [float(length), float(width)],
        "area": float(length * width),
    }


def _channel(center, long_axis, spacing_axis, *, length=100.0, floor_width=5.2):
    center = np.asarray(center, dtype=float)
    long_axis = np.asarray(long_axis, dtype=float)
    spacing_axis = np.asarray(spacing_axis, dtype=float)
    opening_axis = np.cross(long_axis, spacing_axis)
    opening_axis /= np.linalg.norm(opening_axis)
    wall_opening_offset = 5.2
    wall_spacing_offset = 0.83365
    negative_wall_normal = -0.564 * opening_axis - 0.826 * spacing_axis
    positive_wall_normal = -0.564 * opening_axis + 0.826 * spacing_axis
    return [
        _slot(
            center,
            long_axis,
            spacing_axis,
            length=length,
            width=floor_width,
        ),
        _wall(
            center
            + wall_opening_offset * opening_axis
            - wall_spacing_offset * spacing_axis,
            long_axis,
            negative_wall_normal,
            length=length,
        ),
        _wall(
            center
            + wall_opening_offset * opening_axis
            + wall_spacing_offset * spacing_axis,
            long_axis,
            positive_wall_normal,
            length=length,
        ),
    ]


def _moving(dimensions=(98.0, 28.0, 1.5), axes=np.eye(3)):
    return {"obb": _obb((0.0, 0.0, 0.0), axes, dimensions)}


class EdgeSlotInterfaceTests(unittest.TestCase):
    def test_repeated_internal_slot_family_generates_review_only_poses(self):
        planes = [
            plane
            for y in (-30.0, -10.0, 10.0, 30.0)
            for plane in _channel(
                (0.0, y, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            )
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (180.0, 120.0, 4.0)),
            "planes": planes,
        }

        result = recall_edge_slot_interface_proposals(stationary, _moving())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["audit"]["repeated_slot_family_count"], 1)
        self.assertEqual(result["audit"]["length_compatible_family_count"], 1)
        self.assertGreater(len(result["proposals"]), 0)
        for proposal in result["proposals"]:
            self.assertEqual(proposal["slot_family_size"], 4)
            self.assertEqual(proposal["independent_evidence_count"], 4)
            self.assertEqual(
                set(proposal["evidence_families"]),
                {
                    "moving_slot_length_compatibility",
                    "bounded_mirrored_channel_walls",
                    "repeated_equally_spaced_internal_slot_family",
                    "carrier_thin_axis_insertion_role",
                },
            )
            self.assertEqual(len(proposal["wall_plane_indices"]), 2)
            self.assertAlmostEqual(proposal["channel_gap"], 1.6673, places=4)
            self.assertEqual(
                proposal["insertion_axis"],
                proposal["floor_evidence"]["floor_opening_normal"],
            )
            self.assertTrue(proposal["proposal_only"])
            self.assertTrue(proposal["review_required"])
            self.assertFalse(proposal["can_auto_accept"])
            self.assertIn("placement", proposal)
            self.assertIn("transform_matrix", proposal)
            self.assertEqual(proposal["semantic_fields_used"], [])

    def test_isolated_outward_boundary_face_abstains(self):
        # Its short edge reaches the carrier boundary at y=-60 exactly.
        boundary = _slot(
            (0.0, -58.0, 2.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            length=100.0,
            width=4.0,
        )
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (180.0, 120.0, 4.0)),
            "planes": [boundary],
        }

        result = recall_edge_slot_interface_proposals(stationary, _moving())

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])
        self.assertEqual(result["audit"]["outward_boundary_rejected_count"], 1)
        self.assertEqual(result["audit"]["internal_slot_plane_count"], 0)

    def test_length_incompatible_repeated_family_abstains(self):
        planes = [
            plane
            for y in (-24.0, -8.0, 8.0, 24.0)
            for plane in _channel(
                (0.0, y, 2.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                length=55.0,
            )
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (180.0, 120.0, 4.0)),
            "planes": planes,
        }

        result = recall_edge_slot_interface_proposals(stationary, _moving())

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["audit"]["repeated_slot_family_count"], 0)
        self.assertEqual(result["audit"]["length_compatible_family_count"], 0)
        self.assertEqual(result["audit"]["roi_length_rejected_count"], 12)

    def test_rotated_slot_frame_always_emits_proper_rigid_rotation(self):
        angle = np.deg2rad(37.0)
        spacing_axis = np.asarray([np.cos(angle), np.sin(angle), 0.0])
        long_axis = np.asarray([-np.sin(angle), np.cos(angle), 0.0])
        normal = np.asarray([0.0, 0.0, 1.0])
        carrier_axes = np.vstack((spacing_axis, long_axis, normal))
        planes = [
            plane
            for pitch in (-30.0, -10.0, 10.0, 30.0)
            for plane in _channel(
                pitch * spacing_axis + 2.0 * normal,
                long_axis,
                spacing_axis,
            )
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), carrier_axes, (120.0, 180.0, 4.0)),
            "planes": planes,
        }

        result = recall_edge_slot_interface_proposals(stationary, _moving())

        self.assertEqual(result["status"], "success")
        for proposal in result["proposals"]:
            rotation = np.asarray(proposal["rotation_matrix"], dtype=float)
            transform = np.asarray(proposal["transform_matrix"], dtype=float)
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7))
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=7)
            self.assertTrue(np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]))
            transformed_long = rotation @ np.asarray([1.0, 0.0, 0.0])
            transformed_insert = rotation @ np.asarray([0.0, 1.0, 0.0])
            self.assertGreater(abs(float(np.dot(transformed_long, long_axis))), 0.999999)
            self.assertGreater(abs(float(np.dot(transformed_insert, normal))), 0.999999)

    def test_repeated_rail_distractors_without_mirrored_walls_cannot_win(self):
        correct = [
            plane
            for y in (-30.0, -10.0, 10.0, 30.0)
            for plane in _channel(
                (0.0, y, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            )
        ]
        # Repeated and length-compatible, but they are bare rails/faces rather
        # than bounded channels and therefore must never supply family members.
        distractors = [
            _slot(
                (25.0, y, 2.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                length=100.0,
                width=3.0,
            )
            for y in (-45.0, -25.0, -5.0, 15.0)
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (180.0, 120.0, 4.0)),
            "planes": distractors + correct,
        }

        result = recall_edge_slot_interface_proposals(stationary, _moving())

        self.assertEqual(result["status"], "success")
        used_floors = {row["floor_plane_index"] for row in result["proposals"]}
        self.assertTrue(used_floors)
        self.assertTrue(all(index >= len(distractors) for index in used_floors))
        self.assertGreaterEqual(
            result["audit"]["floor_without_mirror_walls_count"],
            len(distractors),
        )

    def test_dominant_functional_body_obb_is_preferred_over_full_obb(self):
        planes = [
            plane
            for y in (-30.0, -10.0, 10.0, 30.0)
            for plane in _channel(
                (0.0, y, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            )
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (180.0, 120.0, 4.0)),
            "planes": planes,
        }
        derived = {
            "status": "proposed",
            "functional_body_obb": _obb(
                (0.0, 0.0, 0.0), np.eye(3), (98.0, 28.0, 1.5)
            ),
        }
        # The fallback is intentionally incompatible; success proves that the
        # derived functional body controls both the ROI and the transform.
        moving = _moving(dimensions=(220.0, 80.0, 25.0))
        with patch(
            "pose_search.edge_slot_interface.derive_dominant_planar_envelope",
            return_value=derived,
        ):
            result = recall_edge_slot_interface_proposals(stationary, moving)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["audit"]["moving_obb_source"], "dominant_planar_envelope")

    def test_realistic_24_channel_metrics_are_bounded_by_proposal_limit(self):
        planes = [
            plane
            for index in range(24)
            for plane in _channel(
                (0.0, -86.71 + 7.54 * index, 0.21),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                length=128.76,
                floor_width=5.2,
            )
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (170.0, 220.0, 4.0)),
            "planes": planes,
        }

        result = recall_edge_slot_interface_proposals(
            stationary,
            _moving(dimensions=(130.0, 30.0, 1.5)),
            maximum_proposals=10,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["audit"]["bounded_channel_floor_count"], 24)
        self.assertEqual(result["proposals"][0]["slot_family_size"], 24)
        self.assertAlmostEqual(result["proposals"][0]["slot_pitch"], 7.54, places=6)
        self.assertEqual(len(result["proposals"]), 10)
        self.assertEqual(result["audit"]["raw_proposal_count"], 48)

    def test_linear_extent_roi_discards_bulk_faces_before_normalisation(self):
        distractors = [
            {
                "normal": [0.0, 0.0, 1.0],
                "centroid": [float(index % 30), float(index // 30), 2.0],
                "footprint_axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "footprint_dimensions": [8.0, 8.0],
            }
            for index in range(3000)
        ]
        channels = [
            plane
            for y in (-30.0, -10.0, 10.0, 30.0)
            for plane in _channel(
                (0.0, y, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            )
        ]
        stationary = {
            "obb": _obb((0.0, 0.0, 0.0), np.eye(3), (180.0, 120.0, 4.0)),
            "planes": distractors + channels,
        }

        result = recall_edge_slot_interface_proposals(stationary, _moving())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["audit"]["raw_stationary_plane_count"], 3012)
        self.assertEqual(result["audit"]["roi_stationary_plane_count"], 12)
        self.assertEqual(result["audit"]["usable_stationary_plane_count"], 12)
        self.assertEqual(result["audit"]["roi_aspect_rejected_count"], 3000)


if __name__ == "__main__":
    unittest.main()
