import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_assembly_manifest_stl import (
    _resolved_component_source,
    _safe_name,
)


class ExportAssemblyManifestStlTests(unittest.TestCase):
    def test_relative_component_source_resolves_from_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "exam" / "assembly_manifest.json"
            manifest.parent.mkdir()
            expected = root / "inputs" / "part.step"
            expected.parent.mkdir()
            expected.touch()
            source = _resolved_component_source(
                manifest, {"source": "../inputs/part.step"}
            )
            self.assertEqual(source, expected.resolve())

    def test_safe_name_removes_path_punctuation(self):
        self.assertEqual(_safe_name("fan/cage (left)"), "fan_cage_left")


if __name__ == "__main__":
    unittest.main()
