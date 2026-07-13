"""Evaluate the public real mixed-pool benchmark with conservative gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    r"D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "public_cad_dataset_audit" / "data" / "results"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parts_key(parts: list[str]) -> tuple[str, ...]:
    return tuple(sorted(parts))


def _naive_score(item: dict[str, Any]) -> float:
    return (
        0.4 * float(item.get("candidate_priority_score", 0.0) or 0.0)
        + 0.3 * float(item.get("geometry_score", 0.0) or 0.0)
        + 0.3
        * float(
            (item.get("consistency") or {}).get(
                "group_consistency_score", 0.0
            )
            or 0.0
        )
    )


def _truth(dataset_root: Path) -> tuple[set[tuple[str, tuple[str, ...]]], set[str]]:
    groups: set[tuple[str, tuple[str, ...]]] = set()
    parts: set[str] = set()
    for pool in sorted(dataset_root.glob("pool_*")):
        gt = _load(pool / "pool_gt.json")
        for part in gt["parts"]:
            parts.add(f"{pool.name}:{part['part_id']}")
        for group in gt["true_groups"]:
            groups.add((pool.name, _parts_key(group["part_ids"])))
    return groups, parts


def _accepted_gate(
    item: dict[str, Any],
    pose: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    consistency = item.get("consistency") or {}
    geometry = item.get("geometry_evidence") or {}
    reasons = []
    checks = {
        "pose_status_valid": bool(
            pose and pose.get("final_pose_status") == "valid"
        ),
        "collision_free": bool(
            pose
            and pose.get("collision_result") == "success"
            and float(pose.get("occt_common_volume", 0.0) or 0.0) == 0.0
        ),
        "geometry_threshold": float(
            item.get("geometry_score", 0.0) or 0.0
        )
        >= 0.80,
        "independent_evidence": int(
            geometry.get("independent_evidence_count", 0) or 0
        )
        >= 2,
        "group_consistency_threshold": float(
            consistency.get("group_consistency_score", 0.0) or 0.0
        )
        >= 0.70,
        "not_weak_single_interface": not bool(
            consistency.get("weak_single_interface_match", True)
        ),
        "no_review_flag": not bool(item.get("review_required", True)),
        "no_global_conflict": not bool(
            consistency.get("has_global_conflict", True)
        ),
        "group_size_limit": int(item["group_size"]) <= 5,
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"gate_failed:{name}")
    return all(checks.values()), reasons


def _load_geometry_candidates(phase_root: Path) -> list[dict[str, Any]]:
    """Load every D4 tier exactly once.

    Final aggregation must not silently ignore pre-pose accepted or rejected
    candidates merely because a previous benchmark happened to put everything
    in review.
    """
    directory = phase_root / "phase6_candidate_tiers"
    candidates = []
    seen = set()
    for tier, name in (
        ("accepted_for_pose_validation", "accepted_geometry_candidates.json"),
        ("review", "review_geometry_candidates.json"),
        ("rejected", "rejected_geometry_candidates.json"),
    ):
        for item in _load(directory / name):
            candidate_id = str(item["group_id"])
            if candidate_id in seen:
                raise ValueError(f"duplicate_geometry_candidate:{candidate_id}")
            seen.add(candidate_id)
            record = {
                key: value
                for key, value in item.items()
                if not key.startswith("evaluation_")
                and key not in {"truth_group_id", "source_assembly_id"}
            }
            record.setdefault("pre_pose_tier", tier)
            candidates.append(record)
    return candidates


def _route_overlapping_accepts(
    accepted: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only mutually non-overlapping accepts without greedy tie-breaking.

    If two otherwise acceptable groups consume the same part, neither is
    automatically selected.  Every conflicting group is routed to review.
    """
    conflicting_ids = set()
    for index, first in enumerate(accepted):
        first_parts = set(first["parts"])
        for second in accepted[index + 1 :]:
            if first["pool_id"] != second["pool_id"]:
                continue
            if first_parts & set(second["parts"]):
                conflicting_ids.update(
                    {first["group_id"], second["group_id"]}
                )
    safe = []
    review = []
    for item in accepted:
        if item["group_id"] not in conflicting_ids:
            safe.append(item)
            continue
        item["final_decision"] = "review"
        item["review_queue_state"] = "selected"
        item["decision_reasons"] = sorted(
            set(item.get("decision_reasons", []))
            | {"overlapping_auto_accept_candidates_require_review"}
        )
        review.append(item)
    return safe, review


def _classify_candidate(
    item: dict[str, Any],
    pose: dict[str, Any] | None,
    *,
    in_review_frontier: bool,
) -> dict[str, Any]:
    """Apply conservative final-state semantics to one production candidate."""
    passed, failed_gates = _accepted_gate(item, pose)
    record = {
        **item,
        "pose_validation": pose,
        "semantic_application_mode": "explanation_only",
        "semantic_reranking_enabled": False,
        "semantic_review": {
            "semantic_validity": "unknown",
            "suggested_action": "abstain",
            "affects_final_decision": False,
        },
        "decision_reasons": list(item.get("decision_reasons", [])),
    }
    if item.get("pre_pose_tier") == "rejected":
        record["final_decision"] = "rejected"
        record["decision_reasons"].append(
            "pre_pose_geometry_gate_confirmed_rejection"
        )
    elif passed:
        record["final_decision"] = "accepted"
        record["decision_reasons"].append(
            "passed_all_conservative_acceptance_gates"
        )
    elif (
        pose
        and pose.get("final_pose_status") == "failed"
        and pose.get("worker_status") == "success"
    ):
        record["final_decision"] = "rejected"
        record["decision_reasons"].append(
            "bounded_pose_search_confirmed_failure"
        )
        record["failed_acceptance_gates"] = failed_gates
    else:
        record["final_decision"] = "review"
        record["review_queue_state"] = (
            "selected" if in_review_frontier else "deferred"
        )
        record["decision_reasons"].extend(failed_gates)
        record["decision_reasons"].append(
            "retained_in_bounded_human_review_frontier"
            if in_review_frontier
            else "deferred_outside_bounded_review_frontier"
        )
        if not in_review_frontier:
            record["review_boundary"] = (
                "Review-frontier capacity is not rejection evidence. "
                "This candidate remains unresolved and review-required."
            )
    record["decision_reasons"] = sorted(set(record["decision_reasons"]))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    args = parser.parse_args()

    phase_root = PROJECT_ROOT / "public_cad_dataset_audit" / "outputs"
    candidates = _load_geometry_candidates(phase_root)
    pose_audit = _load(
        phase_root
        / "phase7_pose_validation"
        / "pose_validation_full_audit.json"
    )
    semantic_gate = _load(
        phase_root
        / "phase8_semantic_gate"
        / "semantic_calibration_report.json"
    )
    pose_by_candidate = {
        row["candidate_id"]: row
        for row in pose_audit["records"]
        if row.get("production_eligible")
    }
    blind_frontier = set(pose_by_candidate)
    truth_groups, all_parts = _truth(args.dataset_root)

    accepted = []
    review = []
    rejected = []
    truth_by_candidate = {}
    for item in candidates:
        pose = pose_by_candidate.get(item["group_id"])
        truth_by_candidate[item["group_id"]] = (
            item["pool_id"], _parts_key(item["parts"])
        ) in truth_groups
        record = _classify_candidate(
            item,
            pose,
            in_review_frontier=item["group_id"] in blind_frontier,
        )
        if record["final_decision"] == "accepted":
            accepted.append(record)
        elif record["final_decision"] == "rejected":
            rejected.append(record)
        else:
            review.append(record)

    accepted, overlap_review = _route_overlapping_accepts(accepted)
    review.extend(overlap_review)
    final_records = accepted + review + rejected
    score_rows = [
        (
            {
                "pool_id": record["pool_id"],
                "candidate_id": record["group_id"],
                "parts": "|".join(record["parts"]),
                "group_size": record["group_size"],
                "candidate_priority_score": record.get(
                    "candidate_priority_score"
                ),
                "geometry_score": record.get("geometry_score"),
                "group_consistency_score": (
                    record.get("consistency") or {}
                ).get("group_consistency_score"),
                "independent_evidence_count": (
                    record.get("geometry_evidence") or {}
                ).get("independent_evidence_count"),
                "pose_status": (
                    (record.get("pose_validation") or {}).get(
                        "final_pose_status", "not_run"
                    )
                ),
                "semantic_mode": "explanation_only",
                "final_decision": record["final_decision"],
                "review_queue_state": record.get("review_queue_state"),
                "evaluation_is_true_group": truth_by_candidate[
                    record["group_id"]
                ],
                "decision_reasons": "|".join(record["decision_reasons"]),
            }
        )
        for record in final_records
    ]

    accepted_part_keys = {
        f"{row['pool_id']}:{part}"
        for row in accepted
        for part in row["parts"]
    }
    unresolved = [
        {
            "pool_id": key.split(":", 1)[0],
            "part_id": key.split(":", 1)[1],
            "reason": "not_covered_by_any_auto_accepted_group",
            "review_candidates": [
                row["group_id"]
                for row in review
                if row["pool_id"] == key.split(":", 1)[0]
                and key.split(":", 1)[1] in row["parts"]
                and row.get("review_queue_state") == "selected"
            ][:20],
            "deferred_review_candidate_count": sum(
                row["pool_id"] == key.split(":", 1)[0]
                and key.split(":", 1)[1] in row["parts"]
                and row.get("review_queue_state") == "deferred"
                for row in review
            ),
        }
        for key in sorted(all_parts - accepted_part_keys)
    ]

    accepted_tp = sum(
        truth_by_candidate[row["group_id"]] for row in accepted
    )
    false_positives = len(accepted) - accepted_tp
    review_truth = sum(
        truth_by_candidate[row["group_id"]] for row in review
    )
    selected_review = [
        row
        for row in review
        if row.get("review_queue_state") == "selected"
    ]
    deferred_review = [
        row
        for row in review
        if row.get("review_queue_state") == "deferred"
    ]
    selected_review_truth = sum(
        truth_by_candidate[row["group_id"]] for row in selected_review
    )
    proposal_truth = sum(
        (
            row["pool_id"],
            _parts_key(row["parts"]),
        )
        in truth_groups
        for row in candidates
    )
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_pool[item["pool_id"]].append(item)
    naive = [
        max(rows, key=lambda row: (_naive_score(row), row["group_id"]))
        for rows in by_pool.values()
    ]
    naive_tp = sum(
        (row["pool_id"], _parts_key(row["parts"])) in truth_groups
        for row in naive
    )
    naive_fp = len(naive) - naive_tp
    metrics = {
        "schema_version": "1.0.0",
        "benchmark": "fusion360_real_mixed_pools_6_pool_v1",
        "pool_count": len(by_pool),
        "input_part_count": len(all_parts),
        "candidate_count": len(candidates),
        "true_group_count": len(truth_groups),
        "candidate_generation_true_group_recall": (
            proposal_truth / len(truth_groups) if truth_groups else None
        ),
        "accepted_group_count": len(accepted),
        "accepted_true_positive_count": accepted_tp,
        "auto_accept_precision": (
            accepted_tp / len(accepted) if accepted else None
        ),
        "auto_accept_precision_status": (
            "not_estimable_no_auto_accepts"
            if not accepted
            else (
                "passed"
                if accepted_tp / len(accepted) >= 0.90
                else "failed"
            )
        ),
        "false_positive_count": false_positives,
        "review_group_count": len(review),
        "review_true_group_count": review_truth,
        "review_true_group_recall": (
            review_truth / len(truth_groups) if truth_groups else None
        ),
        "review_rate": len(review) / len(candidates) if candidates else None,
        "review_frontier_group_count": len(selected_review),
        "review_frontier_true_group_count": selected_review_truth,
        "review_frontier_true_group_recall": (
            selected_review_truth / len(truth_groups)
            if truth_groups
            else None
        ),
        "deferred_review_group_count": len(deferred_review),
        "operator_queue_rate": (
            len(selected_review) / len(candidates) if candidates else None
        ),
        "review_frontier_compression": (
            1.0 - len(selected_review) / len(candidates)
            if candidates
            else None
        ),
        "rejected_candidate_count": len(rejected),
        "unresolved_parts_count": len(unresolved),
        "workload_reduction": None if deferred_review else (
            1.0 - len(review) / len(candidates) if candidates else None
        ),
        "workload_reduction_status": (
            "not_established_deferred_review_candidates_remain"
            if deferred_review
            else "estimated_from_immediate_review_queue"
        ),
        "rejected_reason_coverage": (
            sum(bool(row["decision_reasons"]) for row in rejected)
            / len(rejected)
            if rejected
            else 1.0
        ),
        "semantic_reranking_enabled": semantic_gate[
            "semantic_reranking_enabled"
        ],
        "naive_score_only_top1_per_pool": {
            "accepted_count": len(naive),
            "true_positive_count": naive_tp,
            "false_positive_count": naive_fp,
            "precision": naive_tp / len(naive) if naive else None,
            "warning": (
                "Evaluation-only counterfactual; not a historical baseline."
            ),
        },
        "limitations": [
            "Source assembly provenance is not functional semantic truth.",
            "No auto-accepted groups means precision cannot be estimated.",
            "Deferred review candidates prevent a workload-reduction claim.",
            "Semantic calibration is disabled.",
        ],
    }

    _write(args.output_root / "final_accepted_groups.json", accepted)
    _write(args.output_root / "final_review_groups.json", review)
    _write(args.output_root / "final_rejected_groups.json", rejected)
    _write(args.output_root / "unresolved_parts.json", unresolved)
    _write(args.output_root / "conservative_metrics.json", metrics)
    _write(
        args.output_root / "final_decision_truth_audit.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "evaluation_only",
            "truth_type": "source_assembly_provenance_not_functional_truth",
            "records": [
                {
                    "pool_id": row["pool_id"],
                    "candidate_id": row["group_id"],
                    "parts": row["parts"],
                    "final_decision": row["final_decision"],
                    "review_queue_state": row.get("review_queue_state"),
                    "evaluation_is_true_group": truth_by_candidate[
                        row["group_id"]
                    ],
                }
                for row in final_records
            ],
        },
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "candidate_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    with (args.output_root / "false_positive_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [
            {
                "mode": "conservative_auto_accept",
                "pool_id": row["pool_id"],
                "candidate_id": row["group_id"],
                "parts": "|".join(row["parts"]),
                "is_true_group": truth_by_candidate[row["group_id"]],
                "false_positive": not truth_by_candidate[row["group_id"]],
                "reason": "passed_all_conservative_gates",
            }
            for row in accepted
        ] + [
            {
                "mode": "naive_score_only_counterfactual",
                "pool_id": row["pool_id"],
                "candidate_id": row["group_id"],
                "parts": "|".join(row["parts"]),
                "is_true_group": (
                    row["pool_id"], _parts_key(row["parts"])
                )
                in truth_groups,
                "false_positive": (
                    row["pool_id"], _parts_key(row["parts"])
                )
                not in truth_groups,
                "reason": "top_naive_score_per_pool",
            }
            for row in naive
        ]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pre_pose_accepted_count = sum(
        row.get("pre_pose_tier") == "accepted_for_pose_validation"
        for row in candidates
    )
    accept_summary = (
        "No group passed every gate. Precision is therefore not estimable "
        "and is not reported as 100%."
        if not accepted
        else (
            f"{len(accepted)} groups passed every gate; provenance precision "
            f"is {accepted_tp / len(accepted):.2%}. Functional precision "
            "remains unavailable."
        )
    )
    unresolved_summary = (
        f"{len(unresolved)} parts are not covered by an auto-accepted group."
    )
    report = f"""# Conservative real mixed-pool assembly report

## 1. Input

- Pools: {len(by_pool)}
- Parts: {len(all_parts)}
- Candidate groups: {len(candidates)}

## 2. Candidate recall audit

- True provenance groups retained in the full candidate set:
  {proposal_truth}/{len(truth_groups)}
- This is candidate recall, not functional correctness.

## 3. Geometry candidate tiers

- Pre-pose candidates eligible for pose validation: {pre_pose_accepted_count}
- Near-tied/global conflicts forced candidates to review.

## 4. Final decisions

- Accepted: {len(accepted)}
- Review-required total: {len(review)}
- Immediate review frontier: {len(selected_review)}
- Deferred review: {len(deferred_review)}
- Confirmed rejected: {len(rejected)}
- Unresolved parts: {len(unresolved)}

## 5. Pose validation

- Bounded jobs: {pose_audit['queue_job_count']}
- Completed: {pose_audit['completed_job_count']}
- Status: {pose_audit['queue_status_counts']}
- Pose success is physical feasibility only.

## 6. Semantic gate

- Enabled for reranking: false
- Mode: explanation-only
- Reason: failed calibration and absent functional semantic labels.

## 7. Automatic accepts and evidence

{accept_summary}

## 8. Review groups

{len(review)} candidates remain review-required.  The immediate operator
frontier contains {len(selected_review)} candidates and
{selected_review_truth}/{len(truth_groups)} exact provenance truth groups.
Deferred candidates are not relabelled as rejected.

## 9. Rejected candidates

Rejected rows require either the hard geometry gate or a completed,
definitive pose failure.  Review-frontier capacity is never rejection
evidence.

## 10. Unresolved parts

{unresolved_summary}

## 11. False-positive risk

- Conservative auto-accept false positives: {false_positives}
- Naive score-only top-one-per-pool false positives: {naive_fp}
- The conservative policy removes automatic false accepts but currently
  sacrifices coverage.

## 12. Next smallest step

Annotate a compact real holdout with functional roles and acceptable groups,
then calibrate the review frontier.  Do not increase beam width, model count,
or LLM prompting before those labels exist.
"""
    (args.output_root / "assembly_report.md").write_text(
        report, encoding="utf-8"
    )
    review_text = f"""# Step 9 independent systemic review

The conservative gate produced {len(accepted)} automatic accepts and
{false_positives} provenance false positives.  The bounded immediate review
frontier contains {len(selected_review)} of {len(review)} review-required
candidates and retains {selected_review_truth}/{len(truth_groups)} exact source
groups.  The remaining {len(deferred_review)} candidates stay deferred review,
not rejected.  Consequently, review-frontier compression is reported but
workload reduction is not established.  The bottleneck remains
grouping/ranking quality and missing functional truth, not GPU speed or pose
beam width.
"""
    (args.output_root / "benchmark_system_review.md").write_text(
        review_text, encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
