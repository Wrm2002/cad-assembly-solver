"""Resume-safe OCCT graph extraction for anonymized real mixed pools."""

from __future__ import annotations

import argparse
import hashlib
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_graph(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        graph = read_json(path)
    except Exception:
        return False
    metadata = graph.get("metadata") or {}
    return (
        metadata.get("extraction_status") == "success"
        and metadata.get("released_checkpoint_minimal_features_available")
        is True
        and bool(graph.get("nodes"))
        and bool(graph.get("edges"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    manifest = read_json(root / "mixed_pool_manifest.json")
    rows = []
    failures = []
    total = sum(
        int(pool.get("part_count") or 0)
        for pool in manifest.get("pools", [])
    )
    completed_count = 0
    progress_path = root / "joinable_graph_extraction_progress.json"
    for pool in manifest.get("pools", []):
        pool_id = str(pool["pool_id"])
        pool_dir = root / pool_id
        pool_input = read_json(pool_dir / "pool_input.json")
        graph_dir = pool_dir / "joinable_graphs"
        graph_dir.mkdir(parents=True, exist_ok=True)
        for part in pool_input.get("parts", []):
            part_id = str(part["part_id"])
            source = (pool_dir / part["geometry_path"]).resolve()
            output = graph_dir / f"{part_id}.brep_graph.json"
            if valid_graph(output):
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
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=max(1, args.timeout_seconds),
                    )
                    if process.returncode == 0 and valid_graph(output):
                        status = "success"
                        reason = None
                    else:
                        status = "worker_failed"
                        reason = (
                            f"exit:{process.returncode}:"
                            f"{process.stderr[-1000:]}"
                        )
                except subprocess.TimeoutExpired:
                    status = "worker_timeout"
                    reason = (
                        f"timeout_after:{args.timeout_seconds}s"
                    )
            graph = read_json(output) if valid_graph(output) else {}
            metadata = graph.get("metadata") or {}
            row = {
                "pool_id": pool_id,
                "split": pool.get("split"),
                "part_id": part_id,
                "source_step": str(source),
                "source_geometry_sha256": (
                    sha256_file(source) if source.is_file() else None
                ),
                "output_graph": str(output.resolve()),
                "status": status,
                "node_count": len(graph.get("nodes") or []),
                "face_count": metadata.get("num_faces"),
                "edge_count": metadata.get("num_edges"),
                "checkpoint_extent": metadata.get(
                    "checkpoint_pair_normalization_extent"
                ),
                "reason": reason,
            }
            rows.append(row)
            completed_count += 1
            if not status.startswith("success"):
                failures.append(row)
            write_json(
                progress_path,
                {
                    "stage": "occt_graph_extraction",
                    "completed": completed_count,
                    "total": total,
                    "success_count": sum(
                        item["status"].startswith("success")
                        for item in rows
                    ),
                    "failure_count": len(failures),
                    "current_pool": pool_id,
                    "current_part": part_id,
                },
            )
            if completed_count % 5 == 0 or completed_count == total:
                print(
                    f"graphs {completed_count}/{total}, "
                    f"failures={len(failures)}",
                    flush=True,
                )
    successful_by_hash = {}
    for row in rows:
        if row["status"].startswith("success"):
            successful_by_hash.setdefault(
                row["source_geometry_sha256"], row
            )
    recovered_count = 0
    for row in rows:
        if row["status"].startswith("success"):
            continue
        donor = successful_by_hash.get(row["source_geometry_sha256"])
        if donor is None:
            continue
        donor_graph = read_json(Path(donor["output_graph"]))
        donor_graph["part_id"] = row["part_id"]
        donor_graph["source_step_path"] = row["source_step"]
        donor_graph["source_geometry_sha256"] = row[
            "source_geometry_sha256"
        ]
        donor_graph.setdefault("metadata", {})[
            "exact_geometry_hash_reuse"
        ] = {
            "donor_pool_id": donor["pool_id"],
            "donor_part_id": donor["part_id"],
            "sha256": row["source_geometry_sha256"],
        }
        write_json(Path(row["output_graph"]), donor_graph)
        if valid_graph(Path(row["output_graph"])):
            row["status"] = "success_reused_exact_geometry_hash"
            row["reason"] = None
            row["node_count"] = len(donor_graph.get("nodes") or [])
            metadata = donor_graph.get("metadata") or {}
            row["face_count"] = metadata.get("num_faces")
            row["edge_count"] = metadata.get("num_edges")
            row["checkpoint_extent"] = metadata.get(
                "checkpoint_pair_normalization_extent"
            )
            recovered_count += 1
    failures = [
        row for row in rows if not row["status"].startswith("success")
    ]
    report = {
        "schema_version": "1.0.0",
        "dataset_id": manifest.get("dataset_id"),
        "python_executable": sys.executable,
        "requested_count": total,
        "attempted_count": len(rows),
        "success_count": sum(
            row["status"].startswith("success") for row in rows
        ),
        "failure_count": len(failures),
        "combined_node_count": sum(
            int(row.get("node_count") or 0) for row in rows
        ),
        "maximum_node_count": max(
            (int(row.get("node_count") or 0) for row in rows),
            default=0,
        ),
        "exact_geometry_hash_recovery_count": recovered_count,
        "rows": rows,
        "failure_reasons": [
            f"{row['pool_id']}:{row['part_id']}:{row['reason']}"
            for row in failures
        ],
        "unavailable_fields": [
            "designer_selected_interface_truth_in_occt_topology"
        ],
    }
    write_json(root / "joinable_graph_extraction_report.json", report)
    write_json(
        progress_path,
        {
            "stage": "complete",
            "completed": len(rows),
            "total": total,
            "success_count": report["success_count"],
            "failure_count": report["failure_count"],
        },
    )
    print(
        f"Mixed-pool STEP graphs: {report['success_count']}/"
        f"{report['attempted_count']} successful"
    )
    return 0 if not failures and len(rows) == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
