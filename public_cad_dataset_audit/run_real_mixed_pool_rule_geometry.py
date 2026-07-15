"""Run the existing bounded analytic geometry baseline on real mixed pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR = (
    PROJECT_ROOT
    / "cad_assembly_agent"
    / "tools"
    / "joinable_interface_predictor"
    / "rule_interface_predictor.py"
)
sys.path.insert(0, str(PREDICTOR.parent))
from rule_interface_predictor import rank_candidates  # noqa: E402


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


def valid_descriptors(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = read_json(path)
    except Exception:
        return False
    return bool(data.get("entities")) and not data.get("failure_reasons")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    manifest = read_json(root / "mixed_pool_manifest.json")
    descriptor_rows = []
    successful_by_hash: dict[str, Path] = {}
    descriptor_failures = []
    total_parts = sum(
        int(pool.get("part_count") or 0)
        for pool in manifest.get("pools", [])
    )
    completed = 0
    for pool in manifest.get("pools", []):
        pool_id = str(pool["pool_id"])
        pool_dir = root / pool_id
        pool_input = read_json(pool_dir / "pool_input.json")
        descriptor_dir = pool_dir / "geometry_descriptors"
        descriptor_dir.mkdir(parents=True, exist_ok=True)
        for part in pool_input.get("parts", []):
            part_id = str(part["part_id"])
            source = (pool_dir / part["geometry_path"]).resolve()
            output = descriptor_dir / f"{part_id}.descriptors.json"
            source_hash = str(part["geometry_sha256"])
            if valid_descriptors(output):
                status = "success_cached"
                reason = None
            elif source_hash in successful_by_hash:
                donor = successful_by_hash[source_hash]
                payload = read_json(donor)
                payload["source_step_path"] = str(source)
                payload["source_geometry_sha256"] = source_hash
                payload["exact_geometry_hash_reuse"] = str(
                    donor.resolve()
                )
                write_json(output, payload)
                status = "success_reused_exact_geometry_hash"
                reason = None
            else:
                command = [
                    sys.executable,
                    str(PREDICTOR),
                    "--extract",
                    "--source",
                    str(source),
                    "--output",
                    str(output),
                    "--per-bucket",
                    str(args.per_bucket),
                ]
                try:
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=max(1, args.timeout_seconds),
                        check=False,
                    )
                    if process.returncode == 0 and valid_descriptors(output):
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
            if status.startswith("success"):
                successful_by_hash.setdefault(source_hash, output)
            else:
                descriptor_failures.append(
                    f"{pool_id}:{part_id}:{reason}"
                )
            data = read_json(output) if valid_descriptors(output) else {}
            descriptor_rows.append(
                {
                    "pool_id": pool_id,
                    "part_id": part_id,
                    "status": status,
                    "source_geometry_sha256": source_hash,
                    "descriptor_path": str(output.resolve()),
                    "retained_entity_count": data.get(
                        "retained_entity_count"
                    ),
                    "topology_counts": data.get("topology_counts"),
                    "reason": reason,
                }
            )
            completed += 1
            if completed % 5 == 0 or completed == total_parts:
                print(
                    f"descriptors {completed}/{total_parts}, "
                    f"failures={len(descriptor_failures)}",
                    flush=True,
                )

    pair_rows = []
    pair_failures = []
    for pool in manifest.get("pools", []):
        pool_id = str(pool["pool_id"])
        pool_dir = root / pool_id
        pool_input = read_json(pool_dir / "pool_input.json")
        part_ids = sorted(
            str(part["part_id"])
            for part in pool_input.get("parts", [])
        )
        prediction_dir = pool_dir / "rule_pair_predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for part_a, part_b in combinations(part_ids, 2):
            descriptor_a = (
                pool_dir
                / "geometry_descriptors"
                / f"{part_a}.descriptors.json"
            )
            descriptor_b = (
                pool_dir
                / "geometry_descriptors"
                / f"{part_b}.descriptors.json"
            )
            output = prediction_dir / f"{part_a}__{part_b}.json"
            if not valid_descriptors(descriptor_a) or not valid_descriptors(
                descriptor_b
            ):
                payload = {
                    "schema_version": "1.0.0",
                    "candidates": [],
                    "failure_reasons": [
                        "one_or_both_descriptor_files_unusable"
                    ],
                    "unavailable_fields": [
                        "ranked_geometry_candidates"
                    ],
                }
                write_json(output, payload)
                status = "descriptor_unusable"
            else:
                code = rank_candidates(
                    descriptor_a,
                    descriptor_b,
                    output,
                    args.top_k,
                    args.neighbors,
                )
                payload = read_json(output)
                status = "success" if code == 0 else "prediction_failed"
            candidates = payload.get("candidates") or []
            top = candidates[0] if candidates else {}
            row = {
                "pair_id": f"{pool_id}:{part_a}:{part_b}",
                "pool_id": pool_id,
                "split": pool.get("split"),
                "part_a": part_a,
                "part_b": part_b,
                "status": status,
                "geometry_score": top.get("score"),
                "joint_family_candidate": top.get(
                    "joint_family_candidate"
                ),
                "score_evidence": top.get("score_evidence"),
                "top_candidates": candidates[:5],
                "evaluated_compatible_neighbor_pairs": payload.get(
                    "evaluated_compatible_neighbor_pairs"
                ),
                "failure_reasons": payload.get("failure_reasons", []),
                "unavailable_fields": payload.get(
                    "unavailable_fields", []
                ),
            }
            pair_rows.append(row)
            if status != "success":
                pair_failures.append(
                    f"{row['pair_id']}:{'|'.join(row['failure_reasons'])}"
                )
        print(f"rule geometry completed {pool_id}", flush=True)
    report = {
        "schema_version": "1.0.0",
        "purpose": (
            "Independent deterministic geometry compatibility evidence for "
            "the real mixed-pool JoinABLe graph."
        ),
        "dataset_id": manifest.get("dataset_id"),
        "descriptor_policy": {
            "per_geometry_scale_bucket_limit": args.per_bucket,
            "isolated_worker_per_part": True,
            "exact_hash_reuse": True,
        },
        "pair_policy": {
            "top_k": args.top_k,
            "compatible_size_neighbors": args.neighbors,
            "score_is_calibrated_probability": False,
        },
        "part_count": len(descriptor_rows),
        "descriptor_success_count": sum(
            row["status"].startswith("success")
            for row in descriptor_rows
        ),
        "descriptor_failure_count": len(descriptor_failures),
        "pair_count": len(pair_rows),
        "pair_success_count": sum(
            row["status"] == "success" for row in pair_rows
        ),
        "pair_failure_count": len(pair_failures),
        "descriptors": descriptor_rows,
        "pairs": pair_rows,
        "failure_reasons": descriptor_failures + pair_failures,
        "unavailable_fields": [
            "learned_joinable_probability",
            "designer_selected_interface_truth",
            "functional_semantic_validity",
        ],
    }
    write_json(args.output.resolve(), report)
    print(
        f"Rule geometry pairs: {report['pair_success_count']}/"
        f"{report['pair_count']}"
    )
    return 0 if not report["failure_reasons"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
