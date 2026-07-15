import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from placement_validation import (
    _bbox_clearance_translation_for_second,
    _bbox_overlap_volume,
    bbox_collisions,
    constraint_residual,
    exact_shape_collisions_solid_broadphase,
    transform_point,
)
from constraints import PLANAR_MATE

try:
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt

    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False


class PlacementValidationTests(unittest.TestCase):
    @staticmethod
    def _bounded_plane(center, normal, dimensions=(10.0, 10.0)):
        return {
            "position": list(center),
            "centroid": list(center),
            "normal": list(normal),
            "footprint_axes": [[1, 0, 0], [0, 1, 0]],
            "footprint_dimensions": list(dimensions),
        }

    def test_axis_angle_then_translation(self):
        point = transform_point(
            [1, 0, 0],
            {
                "rotate_sequence": [{"axis_angle": [0, 0, 1, 90]}],
                "translate": [1, 2, 3],
            },
        )
        self.assertAlmostEqual(point[0], 1.0, places=6)
        self.assertAlmostEqual(point[1], 3.0, places=6)
        self.assertAlmostEqual(point[2], 3.0, places=6)

    def test_bbox_collision(self):
        features = {
            "a": {"bbox": {"min": [0, 0, 0], "max": [10, 10, 10]}},
            "b": {"bbox": {"min": [0, 0, 0], "max": [10, 10, 10]}},
        }
        collisions = bbox_collisions(
            features,
            {"a": {"translate": [0, 0, 0]}, "b": {"translate": [9, 0, 0]}},
        )
        self.assertEqual(len(collisions), 1)
        self.assertFalse(collisions[0]["severe"])

    def test_separated_boxes_do_not_collide(self):
        features = {
            "a": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            "b": {"bbox": {"min": [0, 0, 0], "max": [1, 1, 1]}},
        }
        self.assertFalse(
            bbox_collisions(features, {"a": {}, "b": {"translate": [5, 0, 0]}})
        )

    def test_bbox_strict_containment_is_explicit_not_collision_free(self):
        features = {
            "host": {"bbox": {"min": [0, 0, 0], "max": [10, 10, 10]}},
            "insert": {"bbox": {"min": [2, 2, 2], "max": [4, 4, 4]}},
        }
        rows = bbox_collisions(features, {"host": {}, "insert": {}})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_strict_containment"])
        self.assertEqual(rows[0]["smaller_part"], "insert")
        self.assertTrue(rows[0]["severe"])

    def test_exact_broadphase_bbox_overlap_volume(self):
        volume, overlap = _bbox_overlap_volume(
            (0.0, 0.0, 0.0, 4.0, 5.0, 6.0),
            (2.0, 1.0, 3.0, 8.0, 4.0, 10.0),
        )
        self.assertEqual(overlap, (2.0, 3.0, 3.0))
        self.assertEqual(volume, 18.0)

    def test_exact_broadphase_bbox_touch_is_not_penetration(self):
        volume, overlap = _bbox_overlap_volume(
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 2.0, 1.0, 1.0),
        )
        self.assertEqual(overlap, (0.0, 1.0, 1.0))
        self.assertEqual(volume, 0.0)

    def test_exact_broadphase_bbox_respects_tolerance(self):
        volume, overlap = _bbox_overlap_volume(
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (0.9999995, 0.0, 0.0, 2.0, 1.0, 1.0),
        )
        self.assertGreater(overlap[0], 0.0)
        self.assertEqual(volume, 0.0)

    def test_bbox_clearance_translation_uses_smallest_separating_move(self):
        translation = _bbox_clearance_translation_for_second(
            (147.561, 21.415, -238.977, 161.061, 23.415, -191.377),
            (123.439, 21.659, -244.432, 186.351, 84.571, -210.423),
        )
        self.assertEqual(translation[0], 0.0)
        self.assertAlmostEqual(translation[1], 1.806, places=6)
        self.assertEqual(translation[2], 0.0)

    def test_bbox_clearance_translation_handles_contained_box(self):
        translation = _bbox_clearance_translation_for_second(
            (0.0, 0.0, 0.0, 10.0, 10.0, 10.0),
            (4.0, 4.0, 4.0, 6.0, 6.0, 6.0),
        )
        self.assertAlmostEqual(
            sum(abs(value) for value in translation), 6.05, places=6
        )

    def test_planar_residual_reports_finite_footprint_overlap(self):
        features = {
            "a": {"planes": [self._bounded_plane([0, 0, 0], [0, 0, 1])]},
            "b": {"planes": [self._bounded_plane([3, 0, 0], [0, 0, -1])]},
        }
        record = constraint_residual(
            {
                "type": PLANAR_MATE,
                "parts": ["a", "b"],
                "feat_a_idx": 0,
                "feat_b_idx": 0,
            },
            features,
            {"a": {}, "b": {}},
        )
        self.assertTrue(record["bounded_footprint_available"])
        self.assertAlmostEqual(record["bounded_overlap_ratio"], 0.7)
        self.assertAlmostEqual(record["bounded_overlap_area_mm2"], 70.0)
        self.assertAlmostEqual(record["tangential_distance"], 3.0)

    def test_planar_residual_exposes_zero_overlap_for_distant_coplanar_faces(self):
        features = {
            "a": {"planes": [self._bounded_plane([0, 0, 0], [0, 0, 1])]},
            "b": {"planes": [self._bounded_plane([25, 0, 0], [0, 0, -1])]},
        }
        record = constraint_residual(
            {
                "type": PLANAR_MATE,
                "parts": ["a", "b"],
                "feat_a_idx": 0,
                "feat_b_idx": 0,
            },
            features,
            {"a": {}, "b": {}},
        )
        self.assertEqual(record["plane_distance"], 0.0)
        self.assertEqual(record["normal_angle_deg"], 0.0)
        self.assertEqual(record["bounded_overlap_ratio"], 0.0)
        self.assertAlmostEqual(record["tangential_distance_mm"], 25.0)

    def test_planar_residual_without_footprint_keeps_legacy_fields(self):
        features = {
            "a": {"planes": [{
                "position": [0, 0, 0], "normal": [0, 0, 1]
            }]},
            "b": {"planes": [{
                "position": [20, 0, 0], "normal": [0, 0, -1]
            }]},
        }
        record = constraint_residual(
            {
                "type": PLANAR_MATE,
                "parts": ["a", "b"],
                "feat_a_idx": 0,
                "feat_b_idx": 0,
            },
            features,
            {"a": {}, "b": {}},
        )
        self.assertFalse(record["bounded_footprint_available"])
        self.assertNotIn("bounded_overlap_ratio", record)
        self.assertEqual(record["residual"], 0.0)


@unittest.skipUnless(OCC_AVAILABLE, "pythonocc-core is required")
class SolidBroadphaseCoverageTests(unittest.TestCase):
    @staticmethod
    def _compound(*shapes):
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for shape in shapes:
            builder.Add(compound, shape)
        return compound

    @staticmethod
    def _orphan_face(z=3.0):
        return BRepBuilderAPI_MakeFace(
            gp_Pln(gp_Pnt(0.0, 0.0, z), gp_Dir(0.0, 0.0, 1.0)),
            -1.0,
            1.0,
            -1.0,
            1.0,
        ).Face()

    @staticmethod
    def _run(shapes):
        def load_shape(path):
            return shapes[Path(path).name]

        components = [
            {"source": source, "placement": {}}
            for source in shapes
        ]
        with patch("build_assembly.load_step", side_effect=load_shape):
            return exact_shape_collisions_solid_broadphase(".", components)

    def test_solid_only_component_has_complete_collision_coverage(self):
        result = self._run({
            "solid.step": BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(),
        })
        audit = result["component_audit"]["solid.step"]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["collision_result"], "no_collision_detected")
        self.assertTrue(result["collision_free"])
        self.assertTrue(result["coverage_audit"]["complete"])
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["topology_face_count"], 6)
        self.assertEqual(audit["uncovered_face_count"], 0)

    def test_mixed_solid_and_orphan_face_is_uncertain_not_collision_free(self):
        mixed = self._compound(
            BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(),
            self._orphan_face(),
        )
        result = self._run({"mixed.step": mixed})
        audit = result["component_audit"]["mixed.step"]
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["collision_result"], "uncertain")
        self.assertIsNone(result["collision_free"])
        self.assertFalse(result["coverage_audit"]["complete"])
        self.assertEqual(audit["solid_count"], 1)
        self.assertEqual(audit["topology_face_count"], 7)
        self.assertEqual(audit["solid_covered_face_count"], 6)
        self.assertEqual(audit["uncovered_face_count"], 1)
        self.assertIn(
            "mixed.step", result["mixed_solid_open_shell_components"]
        )

    def test_partial_coverage_preserves_detected_solid_collision(self):
        mixed = self._compound(
            BRepPrimAPI_MakeBox(2.0, 2.0, 2.0).Shape(),
            self._orphan_face(),
        )
        overlapping = BRepPrimAPI_MakeBox(
            gp_Pnt(1.0, 0.0, 0.0), 2.0, 2.0, 2.0
        ).Shape()
        result = self._run({
            "mixed.step": mixed,
            "overlapping.step": overlapping,
        })
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["collision_result"], "collision_detected")
        self.assertFalse(result["collision_free"])
        self.assertEqual(len(result["collisions"]), 1)
        self.assertGreater(
            result["collisions"][0]["intersection_volume_mm3"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
