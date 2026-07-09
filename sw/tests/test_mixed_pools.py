import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_mixed_pools import build_pools


def make_case(root, size, index):
    case = root / f"group_{size}_{index:06d}"
    step = case / "step"
    step.mkdir(parents=True)
    parts = []
    placements = {}
    for number in range(1, size + 1):
        name = f"part_{number:02d}.step"
        (step / name).write_text("STEP", encoding="utf-8")
        parts.append(name)
        placements[name] = {"translate": [0, 0, 0], "rotate": []}
    gt = {
        "group_size": size,
        "parts": parts,
        "true_mates": [],
        "placements": placements,
    }
    (case / "gt.json").write_text(json.dumps(gt), encoding="utf-8")


class MixedPoolTests(unittest.TestCase):
    def test_pool_names_do_not_expose_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "output"
            source.mkdir()
            for size in (1, 2, 3):
                for index in (1, 2, 3):
                    make_case(source, size, index)
            report = build_pools(source, output, num_pools=2, seed=42)
            self.assertEqual(report["num_pools"], 2)
            gt = json.loads(
                (output / "pool_001" / "pool_gt.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(len(gt["parts"]), 5)
            self.assertEqual(len(gt["parts"]), len(set(gt["parts"])))
            self.assertTrue(all(name.startswith("part_") for name in gt["parts"]))
            self.assertTrue(all("group_" not in name for name in gt["parts"]))


if __name__ == "__main__":
    unittest.main()
