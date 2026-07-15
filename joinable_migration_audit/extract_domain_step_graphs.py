"""Resume-safe isolated OCCT extraction for a domain-adaptation subset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = PROJECT_ROOT / "joinable_migration_audit" / "step_to_brep_graph_probe.py"
DEFAULT_SUBSET = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_300"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def valid_existing(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    metadata = data.get("metadata", {})
    return (
        metadata.get("extraction_status") == "success"
        and metadata.get("released_checkpoint_minimal_features_available")
        is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")

    split_path = args.subset_root / "subset_split.json"
    subset = read_json(split_path)
    raw_root = args.subset_root / "j1.0.0" / "joint"
    output_root = args.subset_root / "occt_checkpoint_graphs"
    output_root.mkdir(parents=True, exist_ok=True)
    shard_suffix = (
        ""
        if args.shard_count == 1
        else f"_shard_{args.shard_index}_of_{args.shard_count}"
    )
    progress_path = (
        args.subset_root
        / f"graph_extraction_progress{shard_suffix}.json"
    )
    report_path = (
        args.subset_root
        / f"graph_extraction_report{shard_suffix}.json"
    )

    body_ids = set()
    joint_ids = [
        item
        for split_items in subset["splits"].values()
        for item in split_items
    ]
    failures = []
    for joint_id in joint_ids:
        path = raw_root / f"{joint_id}.json"
        if not path.is_file():
            failures.append(
                {
                    "body_id": None,
                    "status": "joint_json_missing",
                    "reason": str(path),
                }
            )
            continue
        joint = read_json(path)
        body_ids.add(str(joint["body_one"]))
        body_ids.add(str(joint["body_two"]))

    rows = []
    all_body_list = sorted(body_ids)
    body_list = all_body_list[args.shard_index :: args.shard_count]
    for index, body_id in enumerate(body_list, 1):
        source = raw_root / f"{body_id}.step"
        output = output_root / f"{body_id}.brep_graph.json"
        if valid_existing(output):
            status = "success_cached"
            reason = None
        elif not source.is_file():
            status = "source_step_missing"
            reason = str(source)
        else:
            command = [
                sys.executable,
                str(EXTRACTOR),
                "--worker",
                "--worker-input",
                str(source),
                "--worker-output",
                str(output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(1, args.timeout_seconds),
                    check=False,
                )
                if completed.returncode == 0 and valid_existing(output):
                    status = "success"
                    reason = None
                else:
                    status = "worker_failed"
                    reason = (
                        f"exit:{completed.returncode}:"
                        f"{completed.stderr[-1000:]}"
                    )
            except subprocess.TimeoutExpired:
                status = "worker_timeout"
                reason = f"timeout_after:{args.timeout_seconds}s"

        row = {
            "body_id": body_id,
            "source_step": str(source),
            "output_graph": str(output),
            "status": status,
            "reason": reason,
        }
        rows.append(row)
        if not status.startswith("success"):
            failures.append(row)
        write_json(
            progress_path,
            {
                "stage": "occt_graph_extraction",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "completed": index,
                "total": len(body_list),
                "success_count": sum(
                    item["status"].startswith("success") for item in rows
                ),
                "failure_count": len(failures),
                "current_body": body_id,
            },
        )
        if index % 20 == 0 or index == len(body_list):
            print(
                f"graphs {index}/{len(body_list)}, "
                f"failures={len(failures)}",
                flush=True,
            )

    report = {
        "schema_version": "1.0.0",
        "subset_root": str(args.subset_root),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "total_body_count": len(all_body_list),
        "body_count": len(body_list),
        "success_count": sum(
            row["status"].startswith("success") for row in rows
        ),
        "failure_count": len(failures),
        "rows": rows,
        "failure_reasons": [
            row.get("reason") for row in failures if row.get("reason")
        ],
        "unavailable_fields": [],
    }
    write_json(report_path, report)
    write_json(
        progress_path,
        {
            "stage": "complete",
            "completed": len(body_list),
            "total": len(body_list),
            "success_count": report["success_count"],
            "failure_count": report["failure_count"],
        },
    )
    return 0 if report["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
