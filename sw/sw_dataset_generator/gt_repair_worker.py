"""Isolated worker for one ground-truth transform repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sw_dataset_generator.repair_ground_truth import repaired_document
else:
    from .repair_ground_truth import repaired_document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    case = Path(args.case_dir).resolve()
    gt, changes = repaired_document(case)
    if args.apply:
        text = json.dumps(gt, indent=2, ensure_ascii=False) + "\n"
        (case / "gt.json").write_text(text, encoding="utf-8")
        (case / "step" / "gt.json").write_text(text, encoding="utf-8")
    print(json.dumps({"case_id": case.name, "changes": changes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
