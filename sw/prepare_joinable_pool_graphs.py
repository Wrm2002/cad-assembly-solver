"""Prepare checkpoint-compatible B-Rep graphs for mixed-pool JoinABLe inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = (
    PROJECT_ROOT
    / "joinable_migration_audit"
    / "step_to_brep_graph_probe.py"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _valid_graph(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read(path)
    except (OSError, ValueError):
        return False
    metadata = payload.get("metadata") or {}
    return (
        metadata.get("extraction_status") == "success"
        and metadata.get("released_checkpoint_minimal_features_available")
        is True
    )


def prepare(
    mixed_pool_root: str | Path,
    *,
    timeout_seconds: int = 90,
    pool_limit: int = 0,
    part_limit_per_pool: int = 0,
) -> dict[str, Any]:
    root = Path(mixed_pool_root).resolve()
    manifest = _read(root / "mixed_pool_manifest.json")
    rows = []
    pools = manifest.get("pools", [])
    if pool_limit > 0:
        pools = pools[:pool_limit]
    for pool in pools:
        pool_id = str(pool["pool_id"])
        pool_dir = root / pool_id
        pool_input = _read(pool_dir / "pool_input.json")
        parts = pool_input.get("parts", [])
        if part_limit_per_pool > 0:
            parts = parts[:part_limit_per_pool]
        graph_dir = pool_dir / "joinable_graphs"
        graph_dir.mkdir(exist_ok=True)
        for part in parts:
            part_id = str(part["part_id"])
            source = pool_dir / str(part["file"])
            # Keep the complete part_id, including ".step", because the batch
            # inference contract addresses graphs by the pool-local part ID.
            output = graph_dir / f"{part_id}.brep_graph.json"
            if _valid_graph(output):
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
                        timeout=max(1, timeout_seconds),
                        check=False,
                    )
                    if completed.returncode == 0 and _valid_graph(output):
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
                    reason = f"timeout_after:{timeout_seconds}s"
            rows.append(
                {
                    "pool_id": pool_id,
                    "part_id": part_id,
                    "source_step": str(source),
                    "output_graph": str(output),
                    "status": status,
                    "reason": reason,
                }
            )
        print(
            f"prepared {pool_id}: "
            f"{sum(row['pool_id'] == pool_id and row['status'].startswith('success') for row in rows)}"
            f"/{sum(row['pool_id'] == pool_id for row in rows)}",
            flush=True,
        )
    report = {
        "schema_version": "1.0.0",
        "mixed_pool_root": str(root),
        "extractor": str(EXTRACTOR),
        "attempted_count": len(rows),
        "success_count": sum(
            row["status"].startswith("success") for row in rows
        ),
        "failure_count": sum(
            not row["status"].startswith("success") for row in rows
        ),
        "rows": rows,
    }
    _write(root / "joinable_graph_preparation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--pool-limit", type=int, default=0)
    parser.add_argument("--part-limit-per-pool", type=int, default=0)
    args = parser.parse_args()
    report = prepare(
        args.mixed_pool_root,
        timeout_seconds=args.timeout_seconds,
        pool_limit=args.pool_limit,
        part_limit_per_pool=args.part_limit_per_pool,
    )
    print(
        json.dumps(
            {
                "success_count": report["success_count"],
                "failure_count": report["failure_count"],
            }
        )
    )
    return 0 if report["failure_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
