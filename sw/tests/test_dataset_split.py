import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_dataset_split import create_split


class DatasetSplitTests(unittest.TestCase):
    def test_stratified_split_is_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(10):
                case = root / f"group_4_{index:06d}"
                case.mkdir()
                (case / "gt.json").write_text(
                    json.dumps(
                        {
                            "group_size": 4,
                            "template": "family_a",
                        }
                    ),
                    encoding="utf-8",
                )
            first = create_split(root, 7)
            second = create_split(root, 7)
            self.assertEqual(first, second)
            counts = {
                split: sum(
                    item["split"] == split
                    for item in first["assignments"].values()
                )
                for split in ("train", "calibration", "test")
            }
            self.assertEqual(counts, {"train": 6, "calibration": 2, "test": 2})


if __name__ == "__main__":
    unittest.main()
