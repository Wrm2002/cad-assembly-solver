"""Freeze the reproduced official JoinABLe pair-interface tool.

The freeze is evidence based: it hashes the released checkpoint, the runtime
adapter, the public predictor entrypoint, and the vendored model sources.  It
also records the already completed full official-test evaluation.  No model is
trained or modified by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOL_ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    PROJECT_ROOT
    / "joinable_migration_audit"
    / "vendor"
    / "JoinABLe"
    / "pretrained"
    / "paper"
    / "last_run_0.ckpt"
)
REFERENCE_REPORT = (
    PROJECT_ROOT
    / "joinable_gpu_reproduction"
    / "official_inference_full_report.json"
)
COMPATIBILITY_ADAPTER = (
    PROJECT_ROOT / "joinable_gpu_reproduction" / "joinable_compat.py"
)
PREDICTOR = TOOL_ROOT / "pretrained_joinable_predictor.py"
CONTRACT = TOOL_ROOT / "JOINABLE_TOOL_CONTRACT.md"
SMOKE_PREDICTION = TOOL_ROOT / "frozen_tool_smoke_prediction.json"


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


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def source_files() -> list[Path]:
    model_root = (
        PROJECT_ROOT
        / "joinable_migration_audit"
        / "vendor"
        / "JoinABLe"
        / "models"
    )
    files = [PREDICTOR, COMPATIBILITY_ADAPTER, CONTRACT]
    files.extend(sorted(model_root.glob("*.py")))
    return files


def build_manifest(reference_report: Path) -> dict[str, Any]:
    failures: list[str] = []
    unavailable: list[str] = []
    required_paths = [
        CHECKPOINT,
        reference_report,
        PREDICTOR,
        COMPATIBILITY_ADAPTER,
        CONTRACT,
    ]
    for path in required_paths:
        if not path.is_file():
            failures.append(f"required_file_missing:{path}")

    report: dict[str, Any] = {}
    if reference_report.is_file():
        try:
            report = read_json(reference_report)
        except Exception as exc:  # pragma: no cover - defensive audit path
            failures.append(
                f"reference_report_unreadable:{type(exc).__name__}:{exc}"
            )

    evaluation = report.get("evaluation") or {}
    checkpoint_evidence = report.get("checkpoint") or {}
    expected_metrics = {
        "sample_count": 1857,
        "top_1_accuracy": 0.7932148626817448,
        "top_5_recall": 0.875605815831987,
        "top_10_recall": 0.9079159935379645,
        "missing_positive_sample_count": 0,
    }
    for field, expected in expected_metrics.items():
        actual = evaluation.get(field)
        if actual is None:
            failures.append(f"reference_metric_missing:{field}")
            continue
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                failures.append(
                    f"reference_metric_changed:{field}:{actual}!={expected}"
                )
        elif actual != expected:
            failures.append(
                f"reference_metric_changed:{field}:{actual}!={expected}"
            )
    if checkpoint_evidence.get("strict_weight_load") is not True:
        failures.append("strict_checkpoint_weight_load_not_proven")

    hashed_sources = []
    for path in source_files():
        if not path.is_file():
            failures.append(f"source_file_missing:{path}")
            continue
        hashed_sources.append(
            {
                "path": relative_or_absolute(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    checkpoint_record: dict[str, Any] = {
        "path": relative_or_absolute(CHECKPOINT),
        "exists": CHECKPOINT.is_file(),
        "sha256": sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None,
        "bytes": CHECKPOINT.stat().st_size if CHECKPOINT.is_file() else None,
        "epoch": checkpoint_evidence.get("epoch"),
        "global_step": checkpoint_evidence.get("global_step"),
        "strict_weight_load": checkpoint_evidence.get(
            "strict_weight_load", False
        ),
        "modified": False,
    }
    if not checkpoint_record["exists"]:
        unavailable.append("released_checkpoint")

    smoke_evidence: dict[str, Any] = {
        "path": relative_or_absolute(SMOKE_PREDICTION),
        "available": SMOKE_PREDICTION.is_file(),
    }
    if SMOKE_PREDICTION.is_file():
        try:
            smoke = read_json(SMOKE_PREDICTION)
            top = (smoke.get("candidates") or [{}])[0]
            smoke_evidence.update(
                {
                    "sha256": sha256_file(SMOKE_PREDICTION),
                    "predictor": smoke.get("predictor"),
                    "candidate_count": len(smoke.get("candidates") or []),
                    "failure_count": len(
                        smoke.get("failure_reasons") or []
                    ),
                    "top_1_joinable_node_pair": [
                        (top.get("part_a_entity") or {}).get(
                            "joinable_node_index"
                        ),
                        (top.get("part_b_entity") or {}).get(
                            "joinable_node_index"
                        ),
                    ],
                    "checkpoint_sha256": (
                        smoke.get("checkpoint") or {}
                    ).get("sha256"),
                    "can_change_accepted_groups": (
                        smoke.get("gate_policy") or {}
                    ).get("can_change_accepted_groups"),
                }
            )
            if smoke_evidence["candidate_count"] != 10:
                failures.append("smoke_prediction_candidate_count_changed")
            if smoke_evidence["failure_count"] != 0:
                failures.append("smoke_prediction_contains_failure")
            if smoke_evidence["top_1_joinable_node_pair"] != [2, 3]:
                failures.append("smoke_prediction_top_1_changed")
            if (
                smoke_evidence["checkpoint_sha256"]
                != checkpoint_record["sha256"]
            ):
                failures.append("smoke_prediction_checkpoint_hash_mismatch")
            if smoke_evidence["can_change_accepted_groups"] is not False:
                failures.append("smoke_prediction_unsafe_gate_policy")
        except Exception as exc:  # pragma: no cover - defensive audit path
            failures.append(
                f"smoke_prediction_unreadable:{type(exc).__name__}:{exc}"
            )
    else:
        unavailable.append("fresh_pair_interface_smoke_prediction")

    return {
        "schema_version": "1.0.0",
        "tool_id": "joinable_pair_interface_predictor",
        "tool_version": "1.0.0-frozen",
        "freeze_status": "frozen" if not failures else "failed",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Rank candidate B-Rep entity pairs for one known pair of parts."
        ),
        "explicit_non_goals": [
            "mixed_pool_grouping",
            "functional_semantic_validation",
            "collision_free_pose_proof",
            "automatic_group_acceptance",
        ],
        "entrypoint": relative_or_absolute(PREDICTOR),
        "contract": relative_or_absolute(CONTRACT),
        "checkpoint": checkpoint_record,
        "source_files": hashed_sources,
        "reference_evaluation": {
            "report_path": relative_or_absolute(reference_report),
            "report_sha256": (
                sha256_file(reference_report)
                if reference_report.is_file()
                else None
            ),
            "dataset": "official JoinABLe filtered test split",
            **{
                field: evaluation.get(field)
                for field in expected_metrics
            },
            "limitations": report.get("limitations", []),
        },
        "pair_interface_smoke_evidence": smoke_evidence,
        "input_contract": {
            "part_a_graph": "audited OCCT B-Rep graph JSON",
            "part_b_graph": "audited OCCT B-Rep graph JSON",
            "maximum_combined_node_count": 950,
            "required_extraction_status": "success",
        },
        "output_contract": {
            "ranked_candidates": True,
            "required_candidate_fields": [
                "candidate_id",
                "rank",
                "part_a_entity",
                "part_b_entity",
                "score",
                "softmax_probability",
                "review_required",
            ],
            "failure_reasons_always_present": True,
            "unavailable_fields_always_present": True,
        },
        "gate_policy": {
            "shadow_mode": True,
            "can_change_accepted_groups": False,
            "requires_pose_and_collision_validation": True,
            "requires_multi_evidence_gate": True,
        },
        "promotion_policy": {
            "official_checkpoint_is_production_baseline": True,
            "adapted_checkpoints_promoted": False,
            "reason": (
                "Expanded STEP-domain adaptation did not improve the "
                "independent test result consistently."
            ),
        },
        "failure_reasons": failures,
        "unavailable_fields": unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-report", type=Path, default=REFERENCE_REPORT
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOOL_ROOT / "frozen_tool_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.reference_report.resolve())
    write_json(args.output.resolve(), manifest)
    print(
        f"JoinABLe tool freeze status: {manifest['freeze_status']} "
        f"({args.output.resolve()})"
    )
    return 0 if manifest["freeze_status"] == "frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
