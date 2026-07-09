import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constraints import match_features
from features import extract_features


class CylinderPolarityTests(unittest.TestCase):
    def _write_step(self, shape, path):
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)

    def test_occt_marks_external_pin_convex_and_bore_concave(self):
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCC.Core.BRepPrimAPI import (
            BRepPrimAPI_MakeBox,
            BRepPrimAPI_MakeCylinder,
        )
        from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pin_path = root / "pin.step"
            plate_path = root / "plate_with_bore.step"
            pin = BRepPrimAPI_MakeCylinder(2.0, 10.0).Shape()
            plate = BRepPrimAPI_MakeBox(
                gp_Pnt(-10.0, -10.0, 0.0),
                20.0,
                20.0,
                6.0,
            ).Shape()
            cutter = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(0.0, 0.0, -1.0), gp_Dir(0.0, 0.0, 1.0)),
                2.3,
                8.0,
            ).Shape()
            bored_plate = BRepAlgoAPI_Cut(plate, cutter).Shape()
            self._write_step(pin, pin_path)
            self._write_step(bored_plate, plate_path)

            pin_features = extract_features(str(pin_path))
            plate_features = extract_features(str(plate_path))
            self.assertIn(
                "convex",
                {
                    item.get("surface_polarity")
                    for item in pin_features["cylinders"]
                },
            )
            self.assertIn(
                "concave",
                {
                    item.get("surface_polarity")
                    for item in plate_features["cylinders"]
                },
            )
            self.assertTrue(
                all(
                    item.get("area", 0.0) > 0.0
                    for item in pin_features["cylinders"]
                )
            )
            plate_z_normals = {
                round(float(item["normal"][2]), 6)
                for item in plate_features["planes"]
                if abs(float(item["normal"][2])) > 0.9
            }
            self.assertEqual(plate_z_normals, {-1.0, 1.0})

    def test_clearance_requires_convex_inside_concave_when_known(self):
        compatible = {
            "pin.step": {
                "cylinders": [
                    {
                        "radius": 2.0,
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "surface_polarity": "convex",
                    }
                ],
                "planes": [],
                "bbox": {"min": [-2, -2, 0], "max": [2, 2, 10]},
            },
            "bore.step": {
                "cylinders": [
                    {
                        "radius": 2.3,
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "surface_polarity": "concave",
                    }
                ],
                "planes": [],
                "bbox": {"min": [-10, -10, 0], "max": [10, 10, 6]},
            },
        }
        matches = match_features(
            compatible,
            {"radius_tolerance_mm": 0.1, "clearance_minimum_mm": 0.1},
        )
        clearances = [item for item in matches if item["type"] == "clearance"]
        self.assertEqual(len(clearances), 1)
        self.assertEqual(
            tuple(clearances[0]["parts"]),
            ("pin.step", "bore.step"),
        )

        incompatible = {
            key: {
                **value,
                "cylinders": [
                    {
                        **value["cylinders"][0],
                        "surface_polarity": "convex",
                    }
                ],
            }
            for key, value in compatible.items()
        }
        matches = match_features(
            incompatible,
            {"radius_tolerance_mm": 0.1, "clearance_minimum_mm": 0.1},
        )
        self.assertFalse(
            any(item["type"] == "clearance" for item in matches)
        )

    def test_pose_mode_preserves_repeated_bore_hypotheses(self):
        features = {
            "pin.step": {
                "cylinders": [
                    {
                        "radius": 2.0,
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [0.0, 0.0, 0.0],
                        "surface_polarity": "convex",
                    }
                ],
                "planes": [],
                "bbox": {"min": [-2, -2, 0], "max": [2, 2, 10]},
            },
            "base.step": {
                "cylinders": [
                    {
                        "radius": 2.3,
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [-10.0, 0.0, 0.0],
                        "surface_polarity": "concave",
                    },
                    {
                        "radius": 2.3,
                        "axis": [0.0, 0.0, 1.0],
                        "origin": [10.0, 0.0, 0.0],
                        "surface_polarity": "concave",
                    },
                ],
                "planes": [],
                "bbox": {"min": [-20, -10, 0], "max": [20, 10, 6]},
            },
        }
        compact = match_features(
            features,
            {
                "radius_tolerance_mm": 0.1,
                "clearance_minimum_mm": 0.1,
            },
        )
        pose = match_features(
            features,
            {
                "radius_tolerance_mm": 0.1,
                "clearance_minimum_mm": 0.1,
                "preserve_cylindrical_face_hypotheses": True,
            },
        )
        self.assertEqual(
            sum(item["type"] == "clearance" for item in compact),
            1,
        )
        self.assertEqual(
            sum(item["type"] == "clearance" for item in pose),
            2,
        )


if __name__ == "__main__":
    unittest.main()
