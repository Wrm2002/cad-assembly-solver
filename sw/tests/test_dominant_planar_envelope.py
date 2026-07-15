from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_search.dominant_planar_envelope import (
    derive_dominant_planar_envelope,
)


def _proper_axes() -> np.ndarray:
    first = np.asarray([1.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.asarray([-1.0, 1.0, 2.0])
    second -= np.dot(second, first) * first
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    return np.asarray([first, second, third])


def _prism_planes(center, axes, dimensions, *, first_index=0):
    center = np.asarray(center, dtype=float)
    axes = np.asarray(axes, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    rows = []
    for normal_axis in range(3):
        footprint_axes = [index for index in range(3) if index != normal_axis]
        for sign in (-1.0, 1.0):
            face_center = (
                center
                + sign * 0.5 * dimensions[normal_axis] * axes[normal_axis]
            )
            footprint_dimensions = dimensions[footprint_axes]
            rows.append({
                "index": first_index + len(rows),
                "centroid": face_center.tolist(),
                "normal": (sign * axes[normal_axis]).tolist(),
                "footprint_axes": axes[footprint_axes].tolist(),
                "footprint_dimensions": footprint_dimensions.tolist(),
                "area": float(np.prod(footprint_dimensions)),
            })
    return rows


def _assert_same_obb(test, result, center, dimensions, *, atol=1e-6):
    test.assertEqual(result["status"], "proposed")
    obb = result["functional_body_obb"]
    test.assertTrue(np.allclose(obb["center"], center, atol=atol))
    test.assertTrue(np.allclose(
        sorted(obb["dimensions"]), sorted(dimensions), atol=atol
    ))
    basis = np.asarray(obb["axes"], dtype=float)
    test.assertTrue(np.allclose(basis @ basis.T, np.eye(3), atol=1e-7))
    test.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=7)
    test.assertTrue(result["proposal_only"])
    test.assertTrue(result["review_required"])
    test.assertFalse(result["can_auto_accept"])


class DominantPlanarEnvelopeTests(unittest.TestCase):
    def test_long_handle_bbox_does_not_expand_dominant_body(self):
        body_center = np.asarray([10.0, -4.0, 7.0])
        body_dimensions = np.asarray([180.0, 74.0, 40.0])
        planes = _prism_planes(body_center, np.eye(3), body_dimensions)
        # A small handle is a complete prism and extends far beyond the body,
        # but it explains much less planar area than the enclosure shell.
        planes.extend(_prism_planes(
            (10.0, -4.0, 69.5),
            np.eye(3),
            (24.0, 18.0, 85.0),
            first_index=100,
        ))
        result = derive_dominant_planar_envelope({
            "planes": planes,
            "bbox": {
                "min": [-80.0, -41.0, -13.0],
                "max": [100.0, 33.0, 112.0],
            },
        })

        _assert_same_obb(self, result, body_center, body_dimensions)
        self.assertTrue(result["excluded_protrusion_risk"])
        self.assertTrue(result["full_envelope_protrusion_risk"])
        self.assertFalse(result["full_envelope_used_as_body_fit"])
        self.assertGreater(result["excluded_planar_area"], 0.0)
        self.assertGreater(
            result["derivation_evidence"]["explained_area_fraction"], 0.75
        )

    def test_random_planes_and_single_normal_family_abstain(self):
        single_family = []
        for index, coordinate in enumerate((-12.0, -4.0, 5.0, 19.0)):
            single_family.append({
                "centroid": [coordinate, 0.0, 0.0],
                "normal": [1.0 if index % 2 else -1.0, 0.0, 0.0],
                "footprint_axes": [[0, 1, 0], [0, 0, 1]],
                "footprint_dimensions": [20.0, 30.0],
                "area": 600.0,
            })
        result = derive_dominant_planar_envelope({"planes": single_family})
        self.assertEqual(result["status"], "abstain")
        self.assertIsNone(result["functional_body_obb"])

        random_planes = [
            {
                "centroid": [0, 0, 0],
                "normal": [1, 0, 0],
                "footprint_axes": [[0, 1, 0], [0, 0, 1]],
                "footprint_dimensions": [11, 17],
                "area": 187,
            },
            {
                "centroid": [8, 3, -2],
                "normal": [0, 1, 0],
                "footprint_axes": [[1, 0, 0], [0, 0, 1]],
                "footprint_dimensions": [7, 13],
                "area": 91,
            },
            {
                "centroid": [-5, 4, 9],
                "normal": [0, 0, 1],
                "footprint_axes": [[1, 0, 0], [0, 1, 0]],
                "footprint_dimensions": [19, 5],
                "area": 95,
            },
        ]
        result = derive_dominant_planar_envelope({"planes": random_planes})
        self.assertEqual(result["status"], "abstain")
        self.assertIsNone(result["functional_body_obb"])

    def test_rotated_body_returns_a_proper_orthonormal_basis(self):
        axes = _proper_axes()
        self.assertAlmostEqual(float(np.linalg.det(axes)), 1.0, places=7)
        center = np.asarray([23.0, -17.0, 41.0])
        dimensions = np.asarray([120.0, 55.0, 32.0])
        result = derive_dominant_planar_envelope({
            "planes": _prism_planes(center, axes, dimensions),
        })

        _assert_same_obb(self, result, center, dimensions)
        inferred = np.asarray(result["functional_body_obb"]["axes"])
        for source_axis in axes:
            self.assertGreater(
                max(abs(float(np.dot(source_axis, axis))) for axis in inferred),
                1.0 - 1e-7,
            )
        self.assertEqual(
            result["derivation_evidence"]["evidence_pattern"],
            "multiple_opposing_pairs",
        )

    def test_open_body_can_use_end_face_plus_two_sides(self):
        dimensions = np.asarray([100.0, 60.0, 30.0])
        complete = _prism_planes((3, -2, 9), np.eye(3), dimensions)
        # Keep the -X end and the +/-Y sides.  Their audited footprints give
        # three independent faces and recover all three body dimensions without
        # pretending that the open +X end or top/bottom faces were observed.
        sparse = [complete[0], complete[2], complete[3]]
        result = derive_dominant_planar_envelope({"planes": sparse})

        _assert_same_obb(self, result, (3, -2, 9), dimensions)
        self.assertEqual(
            result["derivation_evidence"]["evidence_pattern"],
            "end_face_plus_two_sides",
        )
        self.assertEqual(
            len(result["derivation_evidence"]["opposing_pair_axes"]), 1
        )

    def test_small_protruding_step_is_excluded_not_absorbed(self):
        body_center = np.asarray([0.0, 0.0, 0.0])
        body_dimensions = np.asarray([100.0, 60.0, 30.0])
        planes = _prism_planes(body_center, np.eye(3), body_dimensions)
        # A shallow step attached to the +Z face.  No bbox is supplied: risk
        # must also be detectable from excluded planar footprints themselves.
        planes.extend(_prism_planes(
            (24.0, 7.0, 21.0),
            np.eye(3),
            (26.0, 18.0, 12.0),
            first_index=200,
        ))
        result = derive_dominant_planar_envelope({"planes": planes})

        _assert_same_obb(self, result, body_center, body_dimensions)
        self.assertTrue(result["excluded_protrusion_risk"])
        self.assertFalse(result["full_envelope_protrusion_risk"])
        self.assertGreater(result["excluded_protrusion_area"], 0.0)
        self.assertTrue(set(range(200, 206)).intersection(
            result["excluded_protrusion_plane_indices"]
        ))

    def test_area_without_footprint_does_not_invent_a_rectangle(self):
        rows = _prism_planes((0, 0, 0), np.eye(3), (80, 50, 20))
        for row in rows:
            row.pop("footprint_axes")
            row.pop("footprint_dimensions")
            row["area"] = 10000.0
        result = derive_dominant_planar_envelope({"planes": rows})
        self.assertEqual(result["status"], "abstain")


if __name__ == "__main__":
    unittest.main()
