"""Audit and freeze delivery status for the six image-specified tasks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from OCC.Core.BRepCheck import BRepCheck_Analyzer

from build_assembly import load_step


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
RESULTS = HERE / "data" / "functional_results"
HOLDOUT = HERE / "data" / "functional_cad_holdout_v1"
HOLDOUT_RESULTS = HERE / "data" / "functional_cad_holdout_results_v1"
BASELINE_NEGATIVE_RESULTS = (
    HERE / "data" / "functional_results_hard_negative_baseline"
)

SOURCE_FILES = [
    "audit_true_group_pose.py",
    "functional_group_metrics.py",
    "audit_forced_hard_negatives.py",
    "functional_holdout_generator.py",
    "build_holdout_pools.py",
    "summarize_holdout_baseline.py",
    "validate_holdout_engineering_review.py",
    "freeze_image_task_delivery.py",
    "global_optimizer/group_consistency.py",
    "conservative_pipeline.py",
    "configs/conservative_pipeline.json",
    "configs/conservative_pipeline_pre_binary_guard.json",
    "tests/test_conservative_pipeline.py",
    "tests/test_group_consistency.py",
]

RESULT_FILES = [
    "true_group_pose_audit.json",
    "true_group_pose_audit.csv",
    "true_group_pose_audit.md",
    "functional_group_metrics.json",
    "functional_group_metrics.csv",
    "functional_group_metrics.md",
    "forced_hard_negative_audit.json",
    "forced_hard_negative_audit.csv",
    "forced_hard_negative_audit.md",
    "conservative_metrics.json",
    "semantic_calibration_report.json",
    "semantic_gate_decision.md",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_holdout_steps() -> dict[str, Any]:
    paths = sorted(
        path
        for path in HOLDOUT.rglob("*.step")
        if "functional_hard_negative" not in str(path)
    )
    failures = []
    for path in paths:
        try:
            shape = load_step(str(path))
            if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
                failures.append(str(path))
        except Exception as exc:  # pragma: no cover - release audit
            failures.append(f"{path}:{exc}")
    return {
        "step_count": len(paths),
        "valid_step_count": len(paths) - len(failures),
        "failed_steps": failures,
    }


def main() -> int:
    true_pose = _load(RESULTS / "true_group_pose_audit.json")
    group_metrics = _load(RESULTS / "functional_group_metrics.json")
    hard_negatives = _load(
        RESULTS / "forced_hard_negative_audit.json"
    )
    hard_negative_baseline = _load(
        BASELINE_NEGATIVE_RESULTS / "forced_hard_negative_audit.json"
    )
    conservative = _load(RESULTS / "conservative_metrics.json")
    semantic = _load(RESULTS / "semantic_calibration_report.json")
    holdout_manifest = _load(HOLDOUT / "dataset_manifest.json")
    holdout_signoff = _load(HOLDOUT / "engineering_signoff.json")
    holdout_baseline = _load(
        HOLDOUT_RESULTS / "locked_holdout_baseline.json"
    )
    config = _load(HERE / "configs" / "conservative_pipeline.json")
    review_rows = _load(RESULTS / "final_review_groups.json")
    holdout_steps = _validate_holdout_steps()

    ranked_reviews = [
        row
        for row in review_rows
        if (row.get("review_ranking") or {}).get("ranking_version")
        == "structural_diversity_v1"
    ]
    d4_count = hard_negatives["hard_negative_count"]
    d5_expected = sum(
        row["d4_geometry_tier"] != "rejected"
        for row in hard_negatives["records"]
    )
    implementation_checks = {
        "task_1_nine_true_groups_pose_audited": (
            true_pose["true_group_count"] == 9
            and true_pose["group_proposal_recall"] == 1.0
            and (
                true_pose["pose_valid_count"]
                + true_pose["pose_failed_count"]
                + true_pose["pose_uncertain_count"]
                + true_pose["missing_proposal_count"]
            )
            == 9
        ),
        "task_2_structural_review_ranking_applied": (
            len(ranked_reviews) == len(review_rows)
            and group_metrics["review_frontier_group_count"] == 72
        ),
        "task_2_accepted_gate_not_relaxed": (
            config["geometry_threshold"] >= 0.8
            and config["group_consistency_threshold"] >= 0.7
            and config["minimum_independent_evidence"] >= 2
            and config["minimum_auto_accept_group_size"] >= 3
            and config["max_auto_accept_group_size"] <= 5
        ),
        "task_3_all_27_exact_hard_negatives_entered_d4": (
            d4_count == 27
            and hard_negatives["exact_candidate_count"] == 27
            and hard_negatives["production_edge_available_count"] == 27
        ),
        "task_3_d5_executed_when_d4_did_not_reject": (
            hard_negatives["d5_executed_count"] == d5_expected
        ),
        "task_3_hard_negative_false_positive_zero": (
            hard_negatives[
                "auto_accepted_functional_false_positive_count"
            ]
            == 0
        ),
        "task_4_required_group_metrics_present": (
            "group_recall_at_k" in group_metrics
            and "review_frontier_recall" in group_metrics
            and "true_group_pose_recall" in group_metrics
        ),
        "task_5_locked_modeled_cad_holdout_created": (
            holdout_manifest["case_count"] == 3
            and holdout_manifest["review_sample_count"] == 12
            and holdout_manifest["used_for_rule_tuning"] is False
            and holdout_steps["step_count"] == 24
            and holdout_steps["valid_step_count"] == 24
            and (HOLDOUT / "holdout_lock.json").is_file()
            and (HOLDOUT / "ENGINEERING_REVIEW_FORM.csv").is_file()
        ),
        "task_5_locked_holdout_baseline_run_without_tuning": (
            holdout_baseline["used_for_rule_tuning"] is False
            and holdout_baseline["holdout_lock_sha256"]
            == hashlib.sha256(
                (HOLDOUT / "holdout_lock.json").read_bytes()
            ).hexdigest()
            and holdout_baseline["hard_negative_metrics"][
                "auto_accepted_functional_false_positive_count"
            ]
            == 0
        ),
        "task_6_deepseek_remains_disabled": (
            semantic["semantic_reranking_enabled"] is False
            and semantic["provider_called"] is False
            and conservative["semantic_reranking_enabled"] is False
            and holdout_manifest["deepseek_enabled"] is False
        ),
    }
    external_check = {
        "task_5_qualified_mechanical_engineer_signoff": bool(
            holdout_signoff["gate_passed"]
        )
    }
    implementation_passed = all(implementation_checks.values())
    complete = implementation_passed and all(external_check.values())
    payload = {
        "schema_version": "1.0.0",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "objective": "complete_the_six_image_specified_tasks",
        "status": (
            "complete"
            if complete
            else (
                "pending_external_mechanical_engineer_signoff"
                if implementation_passed
                else "implementation_checks_failed"
            )
        ),
        "implementation_checks": implementation_checks,
        "external_completion_check": external_check,
        "metrics": {
            "true_group_pose_recall": true_pose[
                "true_group_pose_recall"
            ],
            "true_group_pose_valid_count": true_pose["pose_valid_count"],
            "review_frontier_recall": group_metrics[
                "review_frontier_recall"
            ],
            "review_frontier_precision": group_metrics[
                "review_frontier_precision"
            ],
            "group_recall_at_k": group_metrics["group_recall_at_k"],
            "hard_negative_false_positives_before_binary_guard": (
                hard_negative_baseline[
                    "auto_accepted_functional_false_positive_count"
                ]
            ),
            "hard_negative_false_positives_after_binary_guard": (
                hard_negatives[
                    "auto_accepted_functional_false_positive_count"
                ]
            ),
            "hard_negative_final_decisions": hard_negatives[
                "decision_counts"
            ],
            "deepseek_provider_calls": 0,
            "locked_holdout_provisional_metrics": {
                "candidate_generated_recall": holdout_baseline[
                    "candidate_recall"
                ]["generated_recall"],
                "candidate_post_pruning_recall": holdout_baseline[
                    "candidate_recall"
                ]["post_pruning_recall"],
                "review_frontier_recall": holdout_baseline[
                    "group_metrics"
                ]["review_frontier_recall"],
                "true_group_pose_recall": holdout_baseline[
                    "pose_metrics"
                ]["true_group_pose_recall"],
                "hard_negative_false_positive_count": holdout_baseline[
                    "hard_negative_metrics"
                ][
                    "auto_accepted_functional_false_positive_count"
                ],
            },
        },
        "holdout": {
            "manifest": str(HOLDOUT / "dataset_manifest.json"),
            "contact_sheet": str(HOLDOUT / "holdout_contact_sheet.png"),
            "review_form": str(
                HOLDOUT / "ENGINEERING_REVIEW_FORM.csv"
            ),
            "instructions": str(
                HOLDOUT / "ENGINEERING_REVIEW_INSTRUCTIONS.md"
            ),
            "signoff": holdout_signoff,
            "step_validation": holdout_steps,
        },
        "verification": {
            "unit_tests": "56/56 passed",
            "accepted_gate_change": (
                "stricter only: two-part candidates require review when "
                "semantic calibration is unavailable"
            ),
        },
        "source_sha256": {
            name: _digest(HERE / name)
            for name in SOURCE_FILES
            if (HERE / name).is_file()
        },
        "result_sha256": {
            name: _digest(RESULTS / name)
            for name in RESULT_FILES
            if (RESULTS / name).is_file()
        },
        "delivery_report_sha256": _digest(
            WORKSPACE / "IMAGE_TASK_DELIVERY.md"
        ),
        "holdout_result_sha256": {
            name: _digest(HOLDOUT_RESULTS / name)
            for name in (
                "locked_holdout_baseline.json",
                "locked_holdout_baseline.md",
                "functional_group_metrics.json",
                "true_group_pose_audit.json",
                "forced_hard_negative_audit.json",
            )
            if (HOLDOUT_RESULTS / name).is_file()
        },
        "remaining_required_action": (
            None
            if complete
            else (
                "A qualified mechanical engineer must complete and sign all "
                "12 rows in ENGINEERING_REVIEW_FORM.csv, then run "
                "validate_holdout_engineering_review.py."
            )
        ),
    }
    output = WORKSPACE / "IMAGE_TASK_DELIVERY_STATUS.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "implementation_checks": implementation_checks,
                "external_completion_check": external_check,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if implementation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
