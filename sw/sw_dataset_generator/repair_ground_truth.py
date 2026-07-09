"""Repair component transforms from SolidWorks insertion-point semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from part_index import index_part


def repaired_document(case):
    case = Path(case)
    gt = json.loads((case / "gt.json").read_text(encoding="utf-8"))
    changes = {}
    for name in gt["parts"]:
        feature = index_part(case / "step" / name, Path(name).stem)
        center = [
            (feature.bbox.minimum[axis] + feature.bbox.maximum[axis]) / 2.0
            for axis in range(3)
        ]
        original = gt["placements"][name]
        insertion = original.get(
            "component_insertion_point",
            original.get("translate", [0.0, 0.0, 0.0]),
        )
        corrected = [
            float(insertion[axis]) - center[axis] for axis in range(3)
        ]
        changes[name] = {
            "old_translate": original.get("translate"),
            "bbox_center_local": center,
            "component_insertion_point": insertion,
            "new_translate": corrected,
        }
        gt["placements"][name] = {
            "translate": corrected,
            "rotate": original.get("rotate", []),
            "component_insertion_point": insertion,
            "derivation": "component_insertion_point - local_part_bbox_center",
        }
    gt["placement_semantics"] = {
        "coordinate_frame": "assembly",
        "length_unit": "mm",
        "transform_convention": "global_point = local_point + translate",
        "solidworks_insertion_note": (
            "AddComponent5 insertion coordinates locate the component bbox "
            "center for these generated primitives."
        ),
    }
    return gt, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default="gt_repair_report.json")
    parser.add_argument("--cases", nargs="*", help="Optional case IDs to repair")
    args = parser.parse_args()
    root = Path(args.dataset_root).resolve()
    report = {"applied": args.apply, "cases": [], "errors": []}
    worker = Path(__file__).with_name("gt_repair_worker.py")
    output = root / args.report
    selected = set(args.cases or [])
    cases = sorted(
        path for path in root.iterdir()
        if path.is_dir() and (not selected or path.name in selected)
    )
    for case in cases:
        try:
            command = [sys.executable, str(worker), str(case)]
            if args.apply:
                command.append("--apply")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(
                    f"worker returncode={result.returncode}: {result.stderr[-1000:]}"
                )
            item = json.loads(result.stdout.strip().splitlines()[-1])
            report["cases"].append(item)
            print(f"{case.name}: {'applied' if args.apply else 'diagnosed'}", flush=True)
        except Exception as exc:
            report["errors"].append({"case_id": case.name, "error": str(exc)})
        finally:
            report["num_cases"] = len(report["cases"])
            report["num_errors"] = len(report["errors"])
            output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    print(output)
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
