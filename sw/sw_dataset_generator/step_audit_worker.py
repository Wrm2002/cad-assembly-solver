"""Isolated worker: probe one case so an OCCT crash cannot kill full audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sw_dataset_generator.audit_dataset import _geometry_consistency, _step_readable
else:
    from .audit_dataset import _geometry_consistency, _step_readable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    case = Path(args.case_dir).resolve()
    gt = json.loads((case / "gt.json").read_text(encoding="utf-8"))
    step_paths = [
        case / "step" / "assembly_gt.step",
        *[case / "step" / name for name in gt["parts"]],
    ]
    errors = []
    for path in step_paths:
        readable, error = _step_readable(path)
        if not readable:
            errors.append({"file": path.name, "error": error})
    result = {
        "case_id": case.name,
        "step_files_checked": len(step_paths),
        "step_errors": errors,
        "geometry": _geometry_consistency(case, gt) if args.geometry and not errors else None,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
