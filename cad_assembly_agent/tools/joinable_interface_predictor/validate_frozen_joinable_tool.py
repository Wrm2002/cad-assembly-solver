"""Validate hashes and the frozen JoinABLe tool contract.

This validator is intentionally lightweight.  It does not load the 1.9 GB
official test pickle.  The full inference evidence is frozen by hash; the
checkpoint and every runtime source file are independently re-hashed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = Path(__file__).resolve().parent


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


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=TOOL_ROOT / "frozen_tool_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOOL_ROOT / "frozen_tool_validation.json",
    )
    args = parser.parse_args()
    failures: list[str] = []
    manifest = read_json(args.manifest.resolve())

    records = [
        ("checkpoint", manifest.get("checkpoint") or {})
    ]
    records.extend(
        ("source", record)
        for record in manifest.get("source_files", [])
    )
    checked = []
    for kind, record in records:
        value = record.get("path")
        path = resolve_record_path(str(value)) if value else Path()
        exists = bool(value and path.is_file())
        actual_hash = sha256_file(path) if exists else None
        expected_hash = record.get("sha256")
        hash_matches = bool(
            actual_hash and expected_hash and actual_hash == expected_hash
        )
        checked.append(
            {
                "kind": kind,
                "path": str(path.resolve()) if value else None,
                "exists": exists,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_matches": hash_matches,
            }
        )
        if not exists:
            failures.append(f"frozen_file_missing:{value}")
        elif not hash_matches:
            failures.append(f"frozen_file_hash_mismatch:{value}")

    reference = manifest.get("reference_evaluation") or {}
    report_value = reference.get("report_path")
    report_path = (
        resolve_record_path(str(report_value))
        if report_value else Path()
    )
    report_exists = bool(report_value and report_path.is_file())
    report_hash = sha256_file(report_path) if report_exists else None
    if not report_exists:
        failures.append("frozen_reference_report_missing")
    elif report_hash != reference.get("report_sha256"):
        failures.append("frozen_reference_report_hash_mismatch")

    smoke = manifest.get("pair_interface_smoke_evidence") or {}
    smoke_value = smoke.get("path")
    smoke_path = (
        resolve_record_path(str(smoke_value)) if smoke_value else Path()
    )
    smoke_exists = bool(smoke_value and smoke_path.is_file())
    smoke_hash = sha256_file(smoke_path) if smoke_exists else None
    smoke_valid = (
        smoke_exists
        and smoke_hash == smoke.get("sha256")
        and smoke.get("top_1_joinable_node_pair") == [2, 3]
        and int(smoke.get("candidate_count") or 0) == 10
        and int(
            smoke.get("failure_count")
            if smoke.get("failure_count") is not None
            else -1
        )
        == 0
        and smoke.get("can_change_accepted_groups") is False
    )
    if not smoke_valid:
        failures.append("frozen_pair_interface_smoke_evidence_invalid")

    metrics_valid = (
        int(reference.get("sample_count") or 0) == 1857
        and float(reference.get("top_1_accuracy") or 0.0) >= 0.79
        and float(reference.get("top_5_recall") or 0.0) >= 0.87
        and float(reference.get("top_10_recall") or 0.0) >= 0.90
        and int(
            reference.get("missing_positive_sample_count")
            if reference.get("missing_positive_sample_count") is not None
            else -1
        )
        == 0
    )
    if not metrics_valid:
        failures.append("frozen_reference_metrics_gate_failed")

    output = {
        "schema_version": "1.0.0",
        "tool_id": manifest.get("tool_id"),
        "tool_version": manifest.get("tool_version"),
        "status": "passed" if not failures else "failed",
        "checked_files": checked,
        "reference_report": {
            "path": str(report_path.resolve()) if report_value else None,
            "exists": report_exists,
            "expected_sha256": reference.get("report_sha256"),
            "actual_sha256": report_hash,
            "hash_matches": (
                report_exists
                and report_hash == reference.get("report_sha256")
            ),
            "metrics_gate_passed": metrics_valid,
        },
        "pair_interface_smoke_evidence": {
            "path": str(smoke_path.resolve()) if smoke_value else None,
            "exists": smoke_exists,
            "expected_sha256": smoke.get("sha256"),
            "actual_sha256": smoke_hash,
            "valid": smoke_valid,
        },
        "gate_policy_verified": (
            (manifest.get("gate_policy") or {}).get(
                "can_change_accepted_groups"
            )
            is False
        ),
        "failure_reasons": failures,
        "unavailable_fields": [
            "fresh_full_test_inference_not_run_by_lightweight_validator"
        ],
    }
    if not output["gate_policy_verified"]:
        output["failure_reasons"].append(
            "unsafe_gate_policy_can_change_accepted_groups"
        )
        output["status"] = "failed"
    write_json(args.output.resolve(), output)
    print(f"Frozen JoinABLe validation: {output['status']}")
    return 0 if output["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
