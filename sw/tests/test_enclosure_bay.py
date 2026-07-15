from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.enclosure_bay import propose_enclosure_bay_placements


def _plane(center, normal, u_axis, v_axis, dimensions):
    return {
        "centroid": list(center),
        "normal": list(normal),
        "footprint_axes": [list(u_axis), list(v_axis)],
        "footprint_dimensions": list(dimensions),
        "area": float(dimensions[0] * dimensions[1]),
    }


def _prism_planes(center, axes, dimensions, *, first_index=0):
    center = np.asarray(center, dtype=float)
    axes = np.asarray(axes, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    rows = []
    for normal_axis in range(3):
        footprint_axes = [index for index in range(3) if index != normal_axis]
        for sign in (-1.0, 1.0):
            footprint_dimensions = dimensions[footprint_axes]
            rows.append({
                "index": first_index + len(rows),
                "centroid": (
                    center
                    + sign
                    * 0.5
                    * dimensions[normal_axis]
                    * axes[normal_axis]
                ).tolist(),
                "normal": (sign * axes[normal_axis]).tolist(),
                "footprint_axes": axes[footprint_axes].tolist(),
                "footprint_dimensions": footprint_dimensions.tolist(),
                "area": float(np.prod(footprint_dimensions)),
            })
    return rows


def _stationary(*, rails=True, roof=True):
    planes = [
        # Two atomic slots: [-77,-1] and [0,74].  The close -1/0 pair is the
        # divider; normals face into the neighbouring cavities.
        _plane((-77, 21, 90), (1, 0, 0), (0, 1, 0), (0, 0, 1), (40, 180)),
        _plane((-1, 21, 90), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (40, 180)),
        _plane((0, 21, 90), (1, 0, 0), (0, 1, 0), (0, 0, 1), (40, 180)),
        _plane((74, 21, 90), (-1, 0, 0), (0, 1, 0), (0, 0, 1), (40, 180)),
    ]
    if rails:
        planes.extend([
            _plane((-39, 1, 90), (0, 1, 0), (1, 0, 0), (0, 0, 1), (35, 140)),
            _plane((37, 1, 90), (0, 1, 0), (1, 0, 0), (0, 0, 1), (35, 140)),
        ])
    if roof:
        planes.append(
            _plane((-1.5, 41, 90), (0, -1, 0), (1, 0, 0), (0, 0, 1), (151, 180))
        )
    return {
        "planes": planes,
        "opening_direction": [0, 0, -1],
    }


def _moving(*, protruding_bbox=True):
    result = {
        "functional_body_obb": {
            "center": [0, -0.8, -20.1],
            "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "dimensions": [184.2, 73.5, 40.0],
        },
        "external_end_direction": [-1, 0, 0],
        "topology": {"solid_count": 0, "shell_count": 12},
    }
    if protruding_bbox:
        result["full_bbox"] = {
            "min": [-122.0, -39.0, -42.0],
            "max": [103.0, 41.0, 3.0],
        }
    return result


def _moving_part_summary(*, protrusion=True):
    center = np.asarray([0.0, -0.8, -20.1])
    dimensions = np.asarray([184.2, 73.5, 40.0])
    planes = _prism_planes(center, np.eye(3), dimensions)
    if protrusion:
        planes.extend(_prism_planes(
            (-107.0, -0.8, -20.1),
            np.eye(3),
            (30.0, 20.0, 15.0),
            first_index=100,
        ))
    return {
        "planes": planes,
        "bbox": {
            "min": [-122.0, -37.55, -40.1],
            "max": [92.1, 35.95, -0.1],
        },
        "topology": {"solid_count": 0, "shell_count": 12},
    }


class EnclosureBayProposalTests(unittest.TestCase):
    def test_two_slots_return_preferred_and_reverse_proper_poses(self):
        result = propose_enclosure_bay_placements(_stationary(), _moving())

        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["slot_count"], 2)
        self.assertEqual(len(result["proposals"]), 4)
        audit = result["functional_body_envelope_audit"]
        self.assertEqual(audit["status"], "provided")
        self.assertEqual(audit["source"], "functional_body_obb")
        self.assertIsNone(audit["confidence"])
        by_slot = {
            index: [row for row in result["proposals"] if row["slot_index"] == index]
            for index in (0, 1)
        }
        self.assertEqual([len(by_slot[index]) for index in (0, 1)], [2, 2])
        for row in result["proposals"]:
            rotation = np.asarray(row["rotation_matrix"], dtype=float)
            transform = np.asarray(row["transform_4x4"], dtype=float)
            self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=7)
            self.assertTrue(np.allclose(transform[:3, :3], rotation))
            self.assertTrue(np.allclose(transform[:3, 3], row["translation"]))
            self.assertTrue(row["proposal_only"])
            self.assertTrue(row["review_required"])
            self.assertFalse(row["can_auto_accept"])
            self.assertGreaterEqual(row["independent_evidence_count"], 4)

        preferred = [
            row for row in result["proposals"]
            if row["opening_polarity"] == "preferred"
        ]
        reverse = [
            row for row in result["proposals"]
            if row["opening_polarity"] == "opposite"
        ]
        self.assertEqual(len(preferred), 2)
        self.assertEqual(len(reverse), 2)
        self.assertTrue(all(
            row["proposal_score"] > reverse[index]["proposal_score"]
            for index, row in enumerate(preferred)
        ))
        expected_rotation = np.asarray([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        for row in preferred:
            self.assertTrue(np.allclose(row["rotation_matrix"], expected_rotation))

    def test_missing_body_is_derived_from_optional_part_summary(self):
        result = propose_enclosure_bay_placements(
            _stationary(),
            {"external_end_direction": [-1, 0, 0]},
            moving_part_summary=_moving_part_summary(),
        )

        self.assertEqual(result["status"], "proposed")
        self.assertEqual(len(result["proposals"]), 4)
        audit = result["functional_body_envelope_audit"]
        self.assertEqual(audit["status"], "proposed")
        self.assertEqual(audit["source"], "derived_dominant_planar_envelope")
        self.assertGreater(audit["confidence"], 0.7)
        self.assertEqual(
            audit["derivation_evidence"]["evidence_pattern"],
            "multiple_opposing_pairs",
        )
        self.assertTrue(audit["excluded_protrusion_risk"])
        self.assertTrue(audit["combined_protrusion_risk"])
        self.assertTrue(result["functional_body_protrusion_risk"])
        for row in result["proposals"]:
            self.assertEqual(
                row["functional_body_source"],
                "derived_dominant_planar_envelope",
            )
            self.assertEqual(row["functional_body_derivation_status"], "proposed")
            self.assertTrue(row["functional_body_excluded_protrusion_risk"])
            self.assertTrue(row["proposal_only"])
            self.assertTrue(row["review_required"])
            self.assertFalse(row["can_auto_accept"])

    def test_whole_part_obb_is_not_used_when_planar_derivation_abstains(self):
        summary = {
            "planes": [_moving_part_summary()["planes"][0]],
            "obb": {
                "center": [0, 0, 0],
                "axes": np.eye(3).tolist(),
                "dimensions": [184.2, 73.5, 40.0],
            },
        }
        result = propose_enclosure_bay_placements(
            _stationary(),
            moving_part_summary=summary,
        )

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])
        self.assertIn("derivation abstained", result["reason"])
        self.assertEqual(
            result["functional_body_envelope_audit"]["status"], "abstain"
        )
        self.assertTrue(result["proposal_only"])
        self.assertTrue(result["review_required"])
        self.assertFalse(result["can_auto_accept"])

    def test_single_plane_abstains(self):
        stationary = {
            "planes": [_stationary()["planes"][0]],
            "opening_direction": [0, 0, -1],
        }
        result = propose_enclosure_bay_placements(stationary, _moving())
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])
        self.assertTrue(result["proposal_only"])
        self.assertTrue(result["review_required"])
        self.assertFalse(result["can_auto_accept"])
        self.assertEqual(
            result["functional_body_envelope_audit"]["status"], "provided"
        )

    def test_opposing_walls_without_repeated_rails_abstain(self):
        result = propose_enclosure_bay_placements(
            _stationary(rails=False), _moving()
        )
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])

    def test_protruding_full_bbox_does_not_veto_functional_body_fit(self):
        result = propose_enclosure_bay_placements(_stationary(), _moving())
        self.assertEqual(result["status"], "proposed")
        self.assertTrue(any(
            row["full_bbox_protrusion_risk"] for row in result["proposals"]
        ))
        self.assertTrue(result["functional_body_protrusion_risk"])
        self.assertTrue(
            result["functional_body_envelope_audit"][
                "placement_frame_full_envelope_protrusion_risk"
            ]
        )
        for row in result["proposals"]:
            self.assertFalse(row["full_bbox_used_as_rejection_gate"])
            self.assertTrue(row["open_shell"])
            self.assertEqual(row["collision_status"], "unchecked")
            self.assertFalse(row["can_auto_accept"])

    def test_common_roof_recovers_depth_from_segmented_divider(self):
        stationary = _stationary()
        # A stamped central guide may only cover the middle of the insertion
        # path.  Repeated rails plus the common roof still provide independent
        # full-depth evidence.
        stationary["planes"][1]["footprint_dimensions"][1] = 100.0
        stationary["planes"][2]["footprint_dimensions"][1] = 100.0

        result = propose_enclosure_bay_placements(stationary, _moving())

        self.assertEqual(result["status"], "proposed")
        preferred = [
            row for row in result["proposals"]
            if row["opening_polarity"] == "preferred"
        ]
        self.assertEqual(len(preferred), 2)
        self.assertTrue(all(
            abs(row["slot_dimensions_lateral_height_depth"][2] - 180.0) < 1e-6
            for row in preferred
        ))

    def test_area_without_planar_footprint_does_not_invent_a_square(self):
        stationary = _stationary()
        for plane in stationary["planes"]:
            plane.pop("footprint_axes")
            plane.pop("footprint_dimensions")
            plane["area"] = 20000.0

        result = propose_enclosure_bay_placements(stationary, _moving())

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])

    def test_translation_maps_functional_body_center_to_each_slot(self):
        result = propose_enclosure_bay_placements(_stationary(), _moving())
        body_center = np.asarray([0.0, -0.8, -20.1])
        preferred = sorted(
            (
                row for row in result["proposals"]
                if row["opening_polarity"] == "preferred"
            ),
            key=lambda row: row["slot_index"],
        )
        expected = [np.asarray([-39.0, 21.0, 90.0]), np.asarray([37.0, 21.0, 90.0])]
        for row, target in zip(preferred, expected):
            rotation = np.asarray(row["rotation_matrix"])
            translation = np.asarray(row["translation"])
            self.assertTrue(np.allclose(rotation @ body_center + translation, target))


if __name__ == "__main__":
    unittest.main()
