"""Freeze machine-checkable invariants for the conservative route alignment."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_count(path: Path) -> tuple[int | None, bool]:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:200]:
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    match = re.search(r"Ran\s+(\d+)\s+tests?", text)
    return (
        int(match.group(1)) if match else None,
        text.rstrip().endswith("OK"),
    )


def main() -> int:
    conservative = _load(HERE / "configs" / "conservative_pipeline.json")
    pool_config = _load(HERE / "configs" / "pool_pipeline.json")
    calibration = _load(
        HERE
        / "data"
        / "multimodal_calibration_v1"
        / "multimodal_calibration_report.json"
    )
    topology_metrics = _load(
        HERE
        / "data"
        / "topology_route_audit_results"
        / "conservative_metrics.json"
    )
    topology_groups = _load(
        HERE
        / "data"
        / "topology_route_audit_results"
        / "functional_group_metrics.json"
    )
    topology_negatives = _load(
        HERE
        / "data"
        / "topology_route_audit_results"
        / "forced_hard_negative_audit.json"
    )
    topology_holdout = _load(
        HERE
        / "data"
        / "topology_pose_holdout_audit_v1"
        / "true_group_pose_topology_audit.json"
    )
    aligned = _load(
        HERE
        / "data"
        / "aligned_route_audit_results"
        / "conservative_metrics.json"
    )
    aligned_groups = _load(
        HERE
        / "data"
        / "aligned_route_audit_results"
        / "functional_group_metrics.json"
    )
    relaxed = _load(
        HERE
        / "data"
        / "deepseek_route_audit_results"
        / "conservative_metrics.json"
    )
    relaxed_groups = _load(
        HERE
        / "data"
        / "deepseek_route_audit_results"
        / "functional_group_metrics.json"
    )

    main_test_count, main_tests_passed = _test_count(
        HERE / "data" / "strict_route_full_test.log"
    )
    dataset_test_count, dataset_tests_passed = _test_count(
        HERE / "data" / "strict_route_dataset_test.log"
    )
    invariant_values = {
        "geometry_threshold_is_0_80": (
            float(conservative["geometry_threshold"]) == 0.80
        ),
        "group_consistency_threshold_is_0_70": (
            float(conservative["group_consistency_threshold"]) == 0.70
        ),
        "learned_evidence_disabled_for_decisions": (
            conservative["learned_evidence_enabled"] is False
            and conservative["learned_evidence_application_mode"]
            == "review_corroboration_only"
        ),
        "semantic_is_explanation_only": (
            conservative["semantic_application_mode"] == "explanation_only"
            and pool_config["multimodal_review"]["application_mode"]
            == "explanation_only"
        ),
        "evaluation_semantics_excluded_from_multimodal_input": (
            pool_config["multimodal_review"][
                "allow_evaluation_only_semantics"
            ]
            is False
        ),
        "multimodal_calibration_gate_closed": (
            calibration["calibration_gate_passed"] is False
            and calibration["semantic_reranking_enabled"] is False
        ),
        "topology_hard_negative_auto_accepts_zero": (
            topology_negatives[
                "auto_accepted_functional_false_positive_count"
            ]
            == 0
        ),
        "topology_production_false_positives_zero": (
            topology_metrics["false_positive_count"] == 0
        ),
        "functional_true_group_pose_recall_is_complete": (
            float(topology_groups["true_group_pose_recall"]) == 1.0
        ),
        "main_tests_passed": main_tests_passed,
        "dataset_tests_passed": dataset_tests_passed,
    }
    tracked = [
        HERE / "configs" / "conservative_pipeline.json",
        HERE / "configs" / "pool_pipeline.json",
        HERE / "features.py",
        HERE / "part_index.py",
        HERE / "constraints.py",
        HERE / "match_pruning.py",
        HERE / "small_assembly_solver.py",
        HERE / "geometry_pipeline.py",
        HERE / "global_optimizer" / "group_consistency.py",
        HERE / "semantic_pool.py",
        HERE / "multimodal_reviewer.py",
        HERE / "multimodal_calibration.py",
        HERE / "pose_topology_regression_audit.py",
        ROOT / "TECHNICAL_REVIEW_AND_MODIFICATIONS.md",
    ]
    payload = {
        "schema_version": "1.0.0",
        "artifact_role": "route_alignment_lock",
        "route_lock_passed": all(invariant_values.values()),
        "invariants": invariant_values,
        "tests": {
            "main": {
                "passed": main_tests_passed,
                "count": main_test_count,
                "log": str(
                    HERE / "data" / "strict_route_full_test.log"
                ),
            },
            "dataset_generator": {
                "passed": dataset_tests_passed,
                "count": dataset_test_count,
                "log": str(
                    HERE / "data" / "strict_route_dataset_test.log"
                ),
            },
        },
        "benchmarks": {
            "deepseek_relaxed_audit": {
                "accepted": relaxed["accepted_group_count"],
                "review": relaxed["review_group_count"],
                "rejected": relaxed["rejected_group_count"],
                "false_positive_count": relaxed["false_positive_count"],
                "review_frontier_recall": relaxed_groups[
                    "review_frontier_recall"
                ],
                "true_group_pose_recall": relaxed_groups[
                    "true_group_pose_recall"
                ],
            },
            "aligned_pre_topology": {
                "accepted": aligned["accepted_group_count"],
                "review": aligned["review_group_count"],
                "rejected": aligned["rejected_group_count"],
                "false_positive_count": aligned["false_positive_count"],
                "review_frontier_recall": aligned_groups[
                    "review_frontier_recall"
                ],
                "true_group_pose_recall": aligned_groups[
                    "true_group_pose_recall"
                ],
            },
            "topology_route": {
                "accepted": topology_metrics["accepted_group_count"],
                "review": topology_metrics["review_group_count"],
                "rejected": topology_metrics["rejected_group_count"],
                "false_positive_count": topology_metrics[
                    "false_positive_count"
                ],
                "review_frontier_recall": topology_groups[
                    "review_frontier_recall"
                ],
                "group_recall_at_2000": topology_groups[
                    "group_recall_at_k"
                ]["2000"]["group_recall"],
                "true_group_pose_recall": topology_groups[
                    "true_group_pose_recall"
                ],
                "hard_negative_decisions": topology_negatives[
                    "decision_counts"
                ],
                "hard_negative_auto_accepts": topology_negatives[
                    "auto_accepted_functional_false_positive_count"
                ],
            },
            "topology_holdout_pose": {
                "true_group_count": topology_holdout["true_group_count"],
                "pose_valid_count": topology_holdout["pose_valid_count"],
                "true_group_pose_recall": topology_holdout[
                    "true_group_pose_recall"
                ],
            },
        },
        "semantic_gate": {
            "provider_mode": calibration["provider_mode"],
            "verdict_counts": calibration["all_verdict_counts"],
            "calibration_gate_passed": calibration[
                "calibration_gate_passed"
            ],
            "semantic_reranking_enabled": calibration[
                "semantic_reranking_enabled"
            ],
            "gate_failure_reasons": calibration[
                "gate_failure_reasons"
            ],
        },
        "known_limits": [
            "No group is auto-accepted, so auto_accept_precision is not estimable.",
            "Topology benchmark review-frontier recall is 0 despite pose recall 1.0.",
            "Topology-varied holdout pose recall remains 0.5.",
            "JoinABLe has not been integrated into the functional pool path.",
            "No auditable live multimodal calibration has been run.",
            "External mechanical-engineer signoff is still pending.",
        ],
        "sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in tracked
        },
    }
    output = ROOT / "STRICT_ROUTE_ALIGNMENT_STATUS.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(f"route_lock_passed={payload['route_lock_passed']}")
    return 0 if payload["route_lock_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
