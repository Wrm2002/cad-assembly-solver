import json
import tempfile
import unittest
from pathlib import Path

from sw_dataset_generator.audit_dataset import audit


class AuditDatasetTests(unittest.TestCase):
    def test_complete_single_part_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "group_1_000001"
            (case / "native").mkdir(parents=True)
            (case / "step").mkdir()
            gt = {
                "group_size": 1,
                "parts": ["part_01.step"],
                "placements": {"part_01.step": {"translate": [0, 0, 0]}},
            }
            (case / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
            for path in (
                case / "generated_spec.json",
                case / "native" / "assembly.sldasm",
                case / "native" / "part_01.sldprt",
                case / "step" / "assembly_gt.step",
                case / "step" / "part_01.step",
            ):
                path.write_text("x", encoding="utf-8")
            report = audit(temporary)
            self.assertEqual(report["status"], "success")
            self.assertEqual(report["valid_cases_by_group"]["1"], 1)


if __name__ == "__main__":
    unittest.main()
