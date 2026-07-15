from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import extract_features


@unittest.skipUnless(importlib.util.find_spec("OCC") is not None, "OCCT unavailable")
class PlanarFootprintFeatureTests(unittest.TestCase):
    def test_trimmed_uv_dimensions_and_axes_are_preserved(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import (
            STEPControl_AsIs,
            STEPControl_Writer,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rectangular_box.step"
            writer = STEPControl_Writer()
            writer.Transfer(
                BRepPrimAPI_MakeBox(60.0, 4.0, 90.0).Shape(),
                STEPControl_AsIs,
            )
            self.assertEqual(writer.Write(str(path)), IFSelect_RetDone)
            features = extract_features(str(path), use_occt=True)

        self.assertEqual(len(features["planes"]), 6)
        self.assertEqual(features["topology"]["solid_count"], 1)
        self.assertGreaterEqual(features["topology"]["shell_count"], 1)
        dimensions = {
            tuple(round(float(value), 6) for value in sorted(row["footprint_dimensions"]))
            for row in features["planes"]
        }
        self.assertEqual(dimensions, {(4.0, 60.0), (4.0, 90.0), (60.0, 90.0)})
        for row in features["planes"]:
            self.assertEqual(len(row["footprint_axes"]), 2)
            self.assertEqual(len(row["footprint_axes"][0]), 3)
            self.assertEqual(len(row["footprint_axes"][1]), 3)


if __name__ == "__main__":
    unittest.main()
