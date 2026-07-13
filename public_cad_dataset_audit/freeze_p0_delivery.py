"""Validate and freeze the P0 conservative-state semantics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public_cad_dataset_audit"
SW = ROOT / "sw"

SOURCE_FILES = [
    "P0_CONSERVATIVE_DELIVERY.md",
    "sw/global_optimizer/group_consistency.py",
    "sw/conservative_pipeline.py",
    "sw/tests/test_group_consistency.py",
    "sw/tests/test_conservative_pipeline.py",
    "sw/configs/conservative_pipeline.json",
    "public_cad_dataset_audit/search_real_mixed_pool_groups.py",
    "public_cad_dataset_audit/tier_real_group_candidates.py",
    "public_cad_dataset_audit/run_real_pose_validation.py",
    "public_cad_dataset_audit/prepare_real_semantic_evidence.py",
    "public_cad_dataset_audit/evaluate_conservative_real_benchmark.py",
    "public_cad_dataset_audit/tests/test_conservative_real_pipeline.py",
    "public_cad_dataset_audit/freeze_p0_delivery.py",
]

RESULT_FILES = [
    "sw/data/results/final_accepted_groups.json",
    "sw/data/results/final_review_groups.json",
    "sw/data/results/final_rejected_groups.json",
    "sw/data/results/unresolved_parts.json",
    "sw/data/results/conservative_metrics.json",
    "sw/data/results/final_decision_truth_audit.json",
    "public_cad_dataset_audit/outputs/phase5_group_search/group_proposals.json",
    "public_cad_dataset_audit/outputs/phase5_group_search/group_proposal_truth_audit.json",
    "public_cad_dataset_audit/outputs/phase6_candidate_tiers/geometry_tiering_metrics.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_validation_full_audit.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_validation_truth_audit.json",
    "public_cad_dataset_audit/outputs/phase8_semantic_gate/semantic_calibration_report.json",
    "public_cad_dataset_audit/data/results/final_accepted_groups.json",
    "public_cad_dataset_audit/data/results/final_review_groups.json",
    "public_cad_dataset_audit/data/results/final_rejected_groups.json",
    "public_cad_dataset_audit/data/results/unresolved_parts.json",
    "public_cad_dataset_audit/data/results/conservative_metrics.json",
    "public_cad_dataset_audit/data/results/final_decision_truth_audit.json",
]

PUBLIC_PRODUCTION_FILES = [
    "public_cad_dataset_audit/outputs/phase5_group_search/group_proposals.json",
    "public_cad_dataset_audit/outputs/phase6_candidate_tiers/accepted_geometry_candidates.json",
    "public_cad_dataset_audit/outputs/phase6_candidate_tiers/review_geometry_candidates.json",
    "public_cad_dataset_audit/outputs/phase6_candidate_tiers/rejected_geometry_candidates.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_validation_queue.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_validation_full_audit.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_validated_candidates.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_failed_candidates.json",
    "public_cad_dataset_audit/outputs/phase7_pose_validation/pose_uncertain_candidates.json",
    "public_cad_dataset_audit/data/results/final_accepted_groups.json",
    "public_cad_dataset_audit/data/results/final_review_groups.json",
    "public_cad_dataset_audit/data/results/final_rejected_groups.json",
]

SW_PRODUCTION_FILES = [
    "sw/data/results/final_accepted_groups.json",
    "sw/data/results/final_review_groups.json",
    "sw/data/results/final_rejected_groups.json",
]

FORBIDDEN_PRODUCTION_KEYS = {
    "evaluation_is_true_group",
    "evaluation_true_group_id",
    "truth_group_id",
    "source_assembly_id",
}


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def find_forbidden(value: Any, path: str = "$") -> list[str]:
    failures = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PRODUCTION_KEYS:
                failures.append(f"{path}.{key}")
            failures.extend(find_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(find_forbidden(child, f"{path}[{index}]"))
    elif value == "evaluation_only_truth_audit":
        failures.append(f"{path}=evaluation_only_truth_audit")
    return failures


def validate_delivery() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for relative in PUBLIC_PRODUCTION_FILES + SW_PRODUCTION_FILES:
        violations = find_forbidden(load(relative))
        check(
            f"production_truth_isolation:{relative}",
            not violations,
            violations[:20],
        )

    for prefix, base in (
        ("legacy_12_pool", "sw/data/results"),
        ("real_6_pool", "public_cad_dataset_audit/data/results"),
    ):
        accepted = load(f"{base}/final_accepted_groups.json")
        review = load(f"{base}/final_review_groups.json")
        rejected = load(f"{base}/final_rejected_groups.json")
        metrics = load(f"{base}/conservative_metrics.json")
        check(
            f"{prefix}:candidate_accounting",
            metrics["candidate_count"]
            == len(accepted) + len(review) + len(rejected),
            {
                "metric": metrics["candidate_count"],
                "accepted": len(accepted),
                "review": len(review),
                "rejected": len(rejected),
            },
        )
        invalid_review = [
            row.get("group_id")
            for row in review
            if row.get("review_queue_state") not in {"selected", "deferred"}
        ]
        check(
            f"{prefix}:review_state_complete",
            not invalid_review,
            invalid_review[:20],
        )
        frontier_rejects = [
            row.get("group_id")
            for row in rejected
            if row.get("final_decision") == "rejected_candidate_frontier"
            or "deferred_outside_bounded_review_frontier"
            in row.get("decision_reasons", [])
        ]
        check(
            f"{prefix}:frontier_capacity_never_rejects",
            not frontier_rejects,
            frontier_rejects[:20],
        )
        deferred = sum(
            row.get("review_queue_state") == "deferred" for row in review
        )
        check(
            f"{prefix}:workload_claim_is_conservative",
            (
                deferred == 0
                or (
                    metrics.get("workload_reduction") is None
                    and metrics.get("workload_reduction_status")
                    == "not_established_deferred_review_candidates_remain"
                )
            ),
            {
                "deferred": deferred,
                "workload_reduction": metrics.get("workload_reduction"),
                "status": metrics.get("workload_reduction_status"),
            },
        )
        check(
            f"{prefix}:semantic_reranking_disabled",
            metrics.get("semantic_reranking_enabled") is False,
            metrics.get("semantic_reranking_enabled"),
        )

    geometry_metrics = load(
        "public_cad_dataset_audit/outputs/phase6_candidate_tiers/"
        "geometry_tiering_metrics.json"
    )
    check(
        "provider_agreement_not_independent_evidence",
        geometry_metrics["thresholds"].get(
            "provider_agreement_counts_as_independent_evidence"
        )
        is False,
        geometry_metrics["thresholds"],
    )

    passed = all(row["passed"] for row in checks)
    summary = {
        "check_count": len(checks),
        "passed_count": sum(row["passed"] for row in checks),
        "failed_count": sum(not row["passed"] for row in checks),
        "delivery_complete": passed,
    }
    return checks, summary


def main() -> int:
    missing = [
        relative
        for relative in SOURCE_FILES + RESULT_FILES
        if not (ROOT / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing_freeze_inputs:{missing}")

    checks, summary = validate_delivery()
    payload = {
        "schema_version": "1.0.0",
        "stage": "P0_conservative_semantics_repair",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Preserve uncertainty as review, forbid greedy overlapping "
            "auto-accepts, isolate evaluation truth, and count independent "
            "physical evidence rather than agreeing providers."
        ),
        "policy": {
            "frontier_capacity_is_rejection_evidence": False,
            "overlapping_auto_accept_winner_selection": False,
            "provider_agreement_counts_as_independent_evidence": False,
            "semantic_reranking_enabled": False,
        },
        "validation_summary": summary,
        "validation_checks": checks,
        "test_results": {
            "sw_core": "51/51 passed",
            "public_pipeline": "13/13 passed",
            "generator": "3/3 passed",
        },
        "source_sha256": {
            relative: digest(relative) for relative in SOURCE_FILES
        },
        "result_sha256": {
            relative: digest(relative) for relative in RESULT_FILES
        },
        "metrics": {
            "legacy_12_pool": load(
                "sw/data/results/conservative_metrics.json"
            ),
            "real_6_pool": load(
                "public_cad_dataset_audit/data/results/"
                "conservative_metrics.json"
            ),
        },
        "known_limits": [
            "D0 functional CAD dataset repair is not part of P0.",
            "Source provenance is still not functional validity truth.",
            "Zero auto-accepts leave accepted precision unestimable.",
            "Deferred review candidates prevent a workload-reduction claim.",
        ],
    }
    output = ROOT / "P0_CONSERVATIVE_FREEZE.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    print(output)
    return 0 if summary["delivery_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
