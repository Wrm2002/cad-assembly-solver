"""Audit generated known groups with one isolated CAD process per case."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from geometry_pipeline import _pipeline_fingerprint


def wilson_interval(successes: int, total: int, z: float = 1.959964):
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def failure_kind(row: dict[str, Any]) -> str:
    if row.get("worker_failed"):
        return "worker_crash_or_timeout"
    validation = row.get("validation_result", {})
    metrics = validation.get("metrics", {})
    if validation.get("unsolved_parts"):
        return "unsolved_parts"
    if metrics.get("exact_collision_check_status") != "success":
        return "exact_collision_check_unavailable"
    if int(validation.get("collision_count", 0)) > 0:
        return "confirmed_solid_penetration"
    maximum = validation.get("max_constraint_residual")
    if maximum is not None and float(maximum) > 5.0:
        return "constraint_residual"
    if metrics.get("assembly_step_build_status") != "success":
        return "assembly_step_build_failure"
    return "other_validation_failure"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output_root")
    parser.add_argument("--group-size", nargs="+", type=int, default=[4, 5])
    parser.add_argument("--limit-per-size", type=int)
    parser.add_argument("--timeout-per-case", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    dataset = Path(args.dataset_root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = _pipeline_fingerprint(config)

    cases_by_size = {size: [] for size in args.group_size}
    for case in sorted(path for path in dataset.iterdir() if path.is_dir()):
        gt_path = case / "gt.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        size = int(gt.get("group_size", 0))
        if size in cases_by_size:
            cases_by_size[size].append(case)
    if args.limit_per_size:
        cases_by_size = {
            size: cases[: args.limit_per_size]
            for size, cases in cases_by_size.items()
        }

    rows = []
    worker = Path(__file__).parent / "dataset_pose_case_worker.py"
    total_cases = sum(len(cases) for cases in cases_by_size.values())
    status_path = output / "run_status.json"

    def write_status(state: str, current_case: str | None = None):
        status_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "state": state,
                    "pid": os.getpid(),
                    "completed_cases": len(rows),
                    "total_cases": total_cases,
                    "current_case": current_case,
                    "accepted_cases": sum(
                        bool(row["accepted"]) for row in rows
                    ),
                    "worker_failures": sum(
                        bool(row.get("worker_failed")) for row in rows
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    write_status("running")
    for size in args.group_size:
        for case in cases_by_size[size]:
            write_status("running", case.name)
            case_output = output / "cases" / case.name
            result_path = case_output / "case_result.json"
            if args.resume and result_path.is_file():
                try:
                    existing = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    if existing.get("pipeline_fingerprint") == fingerprint:
                        rows.append(existing)
                        print(
                            f"{case.name}: cached "
                            f"accepted={existing['accepted']}",
                            flush=True,
                        )
                        write_status("running")
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            case_output.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    [
                        sys.executable,
                        str(worker),
                        str(case),
                        str(case_output),
                        "--config",
                        str(config_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.timeout_per_case,
                )
                if process.returncode == 0 and result_path.is_file():
                    row = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    row = {
                        "schema_version": "1.0.0",
                        "case_id": case.name,
                        "group_size": size,
                        "accepted": False,
                        "worker_failed": True,
                        "worker_returncode": process.returncode,
                        "stdout_tail": process.stdout[-2000:],
                        "stderr_tail": process.stderr[-4000:],
                        "pipeline_fingerprint": fingerprint,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    result_path.write_text(
                        json.dumps(row, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
            except subprocess.TimeoutExpired as exc:
                row = {
                    "schema_version": "1.0.0",
                    "case_id": case.name,
                    "group_size": size,
                    "accepted": False,
                    "worker_failed": True,
                    "worker_timeout": True,
                    "stdout_tail": (exc.stdout or "")[-2000:],
                    "stderr_tail": (exc.stderr or "")[-4000:],
                    "pipeline_fingerprint": fingerprint,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                result_path.write_text(
                    json.dumps(row, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            rows.append(row)
            print(
                f"{case.name}: accepted={row['accepted']} "
                f"elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )
            write_status("running")

    summaries = {}
    for size in args.group_size:
        group = [row for row in rows if row["group_size"] == size]
        accepted = sum(bool(row["accepted"]) for row in group)
        failures = Counter(
            failure_kind(row) for row in group if not row["accepted"]
        )
        elapsed = [float(row["elapsed_seconds"]) for row in group]
        summaries[str(size)] = {
            "cases": len(group),
            "accepted": accepted,
            "acceptance_rate": accepted / len(group) if group else None,
            "wilson_95_interval": wilson_interval(accepted, len(group)),
            "worker_failures": sum(
                bool(row.get("worker_failed")) for row in group
            ),
            "failure_types": dict(sorted(failures.items())),
            "mean_elapsed_seconds": (
                sum(elapsed) / len(elapsed) if elapsed else None
            ),
            "maximum_elapsed_seconds": max(elapsed, default=None),
        }
    report = {
        "schema_version": "1.0.0",
        "dataset_root": str(dataset),
        "pipeline_fingerprint": fingerprint,
        "group_sizes": args.group_size,
        "resume_enabled": args.resume,
        "summaries": summaries,
        "cases": rows,
    }
    report_path = output / "pose_validation_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_status("complete")
    print(json.dumps(summaries, ensure_ascii=False))
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
