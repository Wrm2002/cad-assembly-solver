import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from step_simplifier import preservation_checks


def stats(*, valid=True, solids=2, faces=10, volume=100.0, bounds=None):
    return {
        "valid": valid,
        "volume_mm3": volume,
        "bounds_mm": bounds or [0.0, 0.0, 0.0, 10.0, 5.0, 2.0],
        "topology": {"solids": solids, "faces": faces},
    }


class StepSimplifierTests(unittest.TestCase):
    def test_accepts_face_reduction_with_preserved_geometry(self):
        result = preservation_checks(
            stats(),
            stats(faces=6, volume=100.000001),
            volume_relative_tolerance=1e-6,
            bounds_absolute_tolerance_mm=1e-5,
        )
        self.assertTrue(result["accepted"])

    def test_rejects_lost_solid(self):
        result = preservation_checks(
            stats(solids=2),
            stats(solids=1, faces=6),
            volume_relative_tolerance=1e-6,
            bounds_absolute_tolerance_mm=1e-5,
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["solid_count_preserved"])

    def test_rejects_volume_change(self):
        result = preservation_checks(
            stats(volume=100.0),
            stats(faces=6, volume=99.0),
            volume_relative_tolerance=1e-6,
            bounds_absolute_tolerance_mm=1e-5,
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["volume_preserved"])


if __name__ == "__main__":
    unittest.main()
