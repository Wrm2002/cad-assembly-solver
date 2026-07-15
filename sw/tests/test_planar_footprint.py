from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.planar_footprint import recall_planar_footprint_proposals


def _moving_summary(dimensions=(60.0, 4.0, 90.0), cylinders=None):
    return {
        "obb": {
            "center": [0.0, 0.0, 0.0],
            "axes": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "dimensions": list(dimensions),
        },
        "cylinders": list(cylinders or []),
    }


def _plane(y, dimensions=(60.0, 90.0), x=100.0, z=200.0):
    return {
        "normal": [0.0, 1.0, 0.0],
        "centroid": [x, y, z],
        "footprint_axes": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "footprint_dimensions": list(dimensions),
        "area": float(dimensions[0] * dimensions[1]),
    }


def _cylinder(point, *, radius=2.0, polarity="concave"):
    return {
        "radius": radius,
        "axis": [0.0, 1.0, 0.0],
        "centroid": list(point),
        "surface_polarity": polarity,
    }


class PlanarFootprintTests(unittest.TestCase):
    def test_carrier_perpendicular_wall_recalls_outward_edge_insertion(self):
        stationary = {
            "obb": {
                "center": [100.0, 0.0, 0.0],
                "axes": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "dimensions": [300.0, 200.0, 2.0],
            },
            "planes": [
                _plane(10.0, dimensions=(100.0, 26.0)),
                _plane(12.0, dimensions=(100.0, 26.0)),
            ],
        }
        moving = _moving_summary(dimensions=(100.0, 4.0, 30.0))

        result = recall_planar_footprint_proposals(stationary, moving)

        self.assertEqual(result["status"], "success")
        outward = [
            row for row in result["proposals"]
            if row["in_plane_alignment_strategy"] == "outward_edge"
        ]
        self.assertTrue(outward)
        self.assertGreater(
            result["audit"]["outward_edge_alignment_proposal_count"], 0
        )
        self.assertTrue(all(
            row["interface_orientation_class"]
            == "carrier_perpendicular_wall"
            for row in outward
        ))
        self.assertEqual(
            {round(row["in_plane_edge_alignment_offset_mm"], 6)
             for row in outward},
            {2.0},
        )
        # The wall is on the +Z side of the carrier centre.  Aligning the
        # lower plate edge to the lower wall edge therefore shifts the moving
        # centre outward from z=200 to z=202.
        self.assertEqual(
            {round(float(row["transform_matrix"][2][3]), 6)
             for row in outward},
            {202.0},
        )
        self.assertTrue(all(row["review_required"] for row in outward))
        self.assertTrue(all(not row["can_auto_accept"] for row in outward))

    def test_carrier_parallel_surface_does_not_invent_edge_insertion(self):
        stationary = {
            "obb": {
                "center": [100.0, 0.0, 200.0],
                "axes": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "dimensions": [300.0, 2.0, 400.0],
            },
            "planes": [_plane(10.0), _plane(12.0)],
        }

        result = recall_planar_footprint_proposals(
            stationary, _moving_summary()
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            {row["in_plane_alignment_strategy"]
             for row in result["proposals"]},
            {"center"},
        )
        self.assertEqual(
            {row["interface_orientation_class"]
             for row in result["proposals"]},
            {"carrier_parallel_surface"},
        )

    def test_parallel_co_centered_layers_generate_guarded_equivalent_poses(self):
        stationary = {"planes": [_plane(10.0), _plane(12.0)]}

        result = recall_planar_footprint_proposals(stationary, _moving_summary())

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["proposals"])
        self.assertEqual(result["audit"]["multi_plane_supported_count"], 2)
        self.assertEqual(
            {row["anchor_plane_index"] for row in result["proposals"]}, {0, 1}
        )
        self.assertEqual(
            {row["phase_degrees"] for row in result["proposals"]}, {0, 180}
        )
        for row in result["proposals"]:
            matrix = np.asarray(row["rotation_matrix"], dtype=float)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=6)
            self.assertGreaterEqual(row["independent_evidence_count"], 2)
            self.assertTrue(row["has_multi_evidence_support"])
            self.assertTrue(row["proposal_only"])
            self.assertTrue(row["review_required"])
            self.assertFalse(row["can_auto_accept"])
            self.assertEqual(
                row["transform_frame"],
                "stationary_local_from_moving_local",
            )
            self.assertEqual(row["semantic_fields_used"], [])

    def test_single_planar_contact_abstains_by_default(self):
        result = recall_planar_footprint_proposals(
            {"planes": [_plane(10.0)]}, _moving_summary()
        )

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["proposals"], [])
        self.assertIn("lacked", result["reason"])

    def test_size_gate_rejects_unrelated_large_plane_layers(self):
        stationary = {
            "planes": [
                _plane(10.0, dimensions=(240.0, 360.0)),
                _plane(12.0, dimensions=(240.0, 360.0)),
            ]
        }

        result = recall_planar_footprint_proposals(stationary, _moving_summary())

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["audit"]["size_compatible_plane_count"], 0)
        self.assertEqual(result["proposals"], [])

    def test_area_without_2d_extent_is_not_invented_as_a_square(self):
        plane = _plane(10.0)
        plane.pop("footprint_dimensions")
        result = recall_planar_footprint_proposals(
            {"planes": [plane]}, _moving_summary(), require_multi_plane=False
        )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["audit"]["usable_stationary_plane_count"], 0)

    def test_distance_only_cylinder_array_never_becomes_evidence(self):
        moving_cylinders = [
            _cylinder((-20.0, 0.0, -30.0), polarity="unknown"),
            _cylinder((20.0, 0.0, -30.0), polarity="unknown"),
            _cylinder((-20.0, 0.0, 30.0), polarity="unknown"),
            _cylinder((20.0, 0.0, 30.0), polarity="unknown"),
        ]
        # Same pairwise rectangle distances, but translated away from the
        # footprint anchor and without known surface polarity/layer semantics.
        stationary_cylinders = [
            _cylinder((1080.0, 10.0, 1170.0), polarity="unknown"),
            _cylinder((1120.0, 10.0, 1170.0), polarity="unknown"),
            _cylinder((1080.0, 10.0, 1230.0), polarity="unknown"),
            _cylinder((1120.0, 10.0, 1230.0), polarity="unknown"),
        ]
        result = recall_planar_footprint_proposals(
            {
                "planes": [_plane(10.0)],
                "cylinders": stationary_cylinders,
            },
            _moving_summary(cylinders=moving_cylinders),
            require_multi_plane=False,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["audit"]["cylinder_layout_evidence_count"], 0)
        self.assertFalse(result["audit"]["distance_only_cylinder_signature_used"])
        for row in result["proposals"]:
            self.assertEqual(row["independent_evidence_count"], 1)
            self.assertFalse(row["has_multi_evidence_support"])
            self.assertFalse(row["distance_only_cylinder_signature_used"])

    def test_radius_polarity_layer_and_position_layout_adds_third_evidence(self):
        moving_cylinders = [
            _cylinder((-20.0, 0.0, -30.0)),
            _cylinder((20.0, 0.0, -30.0)),
            _cylinder((-20.0, 0.0, 30.0)),
            _cylinder((20.0, 0.0, 30.0)),
        ]
        # A 4 mm-thick moving OBB seats one support face on y=10, so one
        # support polarity maps its centre-line cylinders to y=12.
        stationary_cylinders = [
            _cylinder((80.0, 12.0, 170.0)),
            _cylinder((120.0, 12.0, 170.0)),
            _cylinder((80.0, 12.0, 230.0)),
            _cylinder((120.0, 12.0, 230.0)),
        ]
        result = recall_planar_footprint_proposals(
            {
                "planes": [_plane(10.0), _plane(12.0)],
                "cylinders": stationary_cylinders,
            },
            _moving_summary(cylinders=moving_cylinders),
        )

        supported = [
            row for row in result["proposals"]
            if row["independent_evidence_count"] == 3
        ]
        self.assertTrue(supported)
        self.assertGreater(result["audit"]["cylinder_layout_evidence_count"], 0)
        evidence = [
            item for item in supported[0]["evidence"]
            if item["type"] == "cylinder_layout_correspondence"
        ][0]
        self.assertGreaterEqual(evidence["correspondence_count"], 3)
        self.assertTrue(evidence["requires_known_polarity"])
        self.assertTrue(evidence["requires_installation_layer"])
        self.assertTrue(evidence["requires_transformed_position_correspondence"])
        self.assertFalse(evidence["distance_only_signature_used"])

    def test_support_surface_not_obb_center_is_aligned_to_plane(self):
        result = recall_planar_footprint_proposals(
            {"planes": [_plane(10.0), _plane(12.0)]},
            _moving_summary(dimensions=(60.0, 4.0, 90.0)),
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            {round(row["support_offset_mm"], 6) for row in result["proposals"]},
            {2.0},
        )
        self.assertEqual(
            {
                round(float(row["transform_matrix"][1][3]), 6)
                for row in result["proposals"]
                if row["anchor_plane_index"] == 0
            },
            {8.0, 12.0},
        )
        self.assertTrue(all(
            row["support_surface_aligned"] for row in result["proposals"]
        ))

    def test_fixed_bound_preserves_equivalent_sites_and_support_sides(self):
        result = recall_planar_footprint_proposals(
            {
                "planes": [
                    _plane(10.0, x=100.0),
                    _plane(12.0, x=100.0),
                    _plane(10.0, x=400.0),
                    _plane(12.0, x=400.0),
                ]
            },
            _moving_summary(),
            maximum=4,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["proposals"]), 4)
        self.assertEqual(
            len({row["equivalence_class_id"] for row in result["proposals"]}),
            2,
        )
        self.assertEqual(
            {row["support_polarity"] for row in result["proposals"]},
            {-1, 1},
        )
        anchors_by_equivalence = {}
        for row in result["proposals"]:
            anchors_by_equivalence.setdefault(
                row["equivalence_class_id"], set()
            ).add(row["anchor_plane_index"])
        self.assertTrue(all(
            len(anchor_indices) == 2
            for anchor_indices in anchors_by_equivalence.values()
        ))
        self.assertEqual(result["audit"]["equivalence_anchor_stratum_count"], 4)

    def test_non_thin_moving_part_abstains(self):
        result = recall_planar_footprint_proposals(
            {"planes": [_plane(10.0), _plane(12.0)]},
            _moving_summary(dimensions=(60.0, 40.0, 90.0)),
        )

        self.assertEqual(result["status"], "abstain")
        self.assertIn("not sufficiently thin", result["reason"])

    @unittest.skipUnless(importlib.util.find_spec("OCC") is not None, "OCCT unavailable")
    def test_occt_shapes_are_supported_without_names_or_case_metadata(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        stationary = BRepPrimAPI_MakeBox(60.0, 2.0, 90.0).Shape()
        moving = BRepPrimAPI_MakeBox(60.0, 4.0, 90.0).Shape()

        result = recall_planar_footprint_proposals(
            stationary, moving, size_tolerance=0.10
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["audit"]["stationary_source"], "occt_shape")
        self.assertEqual(result["audit"]["moving_source"], "occt_shape")
        self.assertTrue(result["proposals"])
        self.assertFalse(result["audit"]["anonymous_semantic_fields_used"])


if __name__ == "__main__":
    unittest.main()
