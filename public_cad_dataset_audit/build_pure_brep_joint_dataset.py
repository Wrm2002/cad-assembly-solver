"""Build the Stage-1 leakage-safe Fusion 360 B-Rep supervision index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))

from learned_joint.fusion_contract import audit_records, load_json, make_record  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    joint_root = args.dataset_root / "joint" if (args.dataset_root / "joint").is_dir() else args.dataset_root
    paths = sorted(joint_root.glob("joint_set_*.json"))
    if args.limit > 0:
        paths = paths[: args.limit]
    records = []
    failures = []
    for path in paths:
        try:
            record = make_record(path, load_json(path))
            if record is not None:
                records.append(record)
        except Exception as exc:  # dataset audit must retain malformed rows
            failures.append({"source": str(path), "error": f"{type(exc).__name__}:{exc}"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (args.output_dir / f"fusion360_pure_brep_{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "dev", "test")
    }
    try:
        for record in records:
            handles[record["split"]].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    audit = audit_records(records)
    audit.update({
        "source_joint_file_count": len(paths),
        "conversion_failure_count": len(failures),
        "conversion_failures": failures[:100],
        "output_dir": str(args.output_dir.resolve()),
    })
    write_json(args.output_dir / "pure_brep_contract_audit.json", audit)
    write_json(PROJECT_ROOT / "reports" / "pure_brep_contract_audit.json", audit)
    print(json.dumps({
        "records": len(records),
        "failures": len(failures),
        "passed": audit["passed"],
        "splits": audit["split_counts"],
    }, ensure_ascii=False))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
