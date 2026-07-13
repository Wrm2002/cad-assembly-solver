"""Precision-first D4-D7 pipeline for frozen mixed-part pools.

The pipeline never forces a complete partition. Geometry, physical pose, and
semantic evidence remain separate, and uncertain candidates are routed to
human review instead of being silently accepted.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

from contracts import GroupProposal
from geometry_pipeline import isolated_solve_and_validate_group
from global_grouping import run as run_grouping
from global_optimizer.group_consistency import assess_group_consistency


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _parts_key(parts: list[str]) -> frozenset[str]:
    return frozenset(parts)


def _truth_sets(pool: Path) -> set[frozenset[str]]:
    gt_path = pool / "pool_gt.json"
    if not gt_path.is_file():
        return set()
    return {
        _parts_key(group["parts"])
        for group in _load(gt_path).get("true_groups", [])
    }


def _passes_final_acceptance(
    item: dict[str, Any],
    pose: dict[str, Any] | None,
    config: dict[str, Any],
) -> bool:
    consistency = item["consistency"]
    return bool(
        item["geometry_tier"] == "accepted_for_pose_validation"
        and pose
        and pose["final_pose_status"] == "valid"
        and pose["worker_status"] == "success"
        and pose["collision_result"] == "success"
        and float(pose.get("occt_common_volume", 0.0) or 0.0) == 0.0
        and float(item["geometry_score"])
        >= float(config["geometry_threshold"])
        and consistency["independent_evidence_count"]
        >= int(config["minimum_independent_evidence"])
        and float(consistency["group_consistency_score"])
        >= float(config["group_consistency_threshold"])
        and not consistency["weak_single_interface_match"]
        and not consistency["review_required"]
        and not consistency["has_global_conflict"]
        and len(item["parts"])
        >= int(config.get("minimum_auto_accept_group_size", 2))
        and len(item["parts"]) <= int(config["max_auto_accept_group_size"])
    )


def route_overlapping_accepts_to_review(
    accepted: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Route every overlapping accepted candidate to review.

    No score-greedy winner is selected when two otherwise acceptable groups
    consume the same part.
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
    safe, review = [], []
    for item in accepted:
        if item["group_id"] not in conflicting_ids:
            safe.append(item)
            continue
        item["final_decision"] = "review"
        item["decision_reasons"].append(
            "overlapping_auto_accept_candidates_require_review"
        )
        item["decision_reasons"] = sorted(set(item["decision_reasons"]))
        review.append(item)
    return safe, review


def geometry_tiers(
    pool: Path,
    proposals: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted, review, rejected = [], [], []
    for proposal in proposals:
        consistency = assess_group_consistency(
            proposal,
            edges,
            proposals,
            conflict_score_margin=float(config["conflict_score_margin"]),
            learned_evidence_enabled=bool(
                config.get("learned_evidence_enabled", False)
            ),
            joinable_min_softmax=float(
                config.get("joinable_min_softmax", 0.85)
            ),
        )
        item = {
            **proposal,
            "pool_id": pool.name,
            "group_size": len(proposal["parts"]),
            "consistency": consistency,
            "geometry_tier": None,
            "decision_reasons": [],
        }
        if float(proposal["geometry_score"]) < float(
            config["geometry_threshold"]
        ):
            item["geometry_tier"] = "rejected"
            item["decision_reasons"].append("geometry_score_below_threshold")
            rejected.append(item)
            continue
        review_reasons = []
        if len(proposal["parts"]) < int(
            config.get("minimum_auto_accept_group_size", 2)
        ):
            review_reasons.append("group_size_below_auto_accept_limit")
        if len(proposal["parts"]) > int(
            config["max_auto_accept_group_size"]
        ):
            review_reasons.append("group_size_exceeds_auto_accept_limit")
        if consistency["independent_evidence_count"] < int(
            config["minimum_independent_evidence"]
        ):
            review_reasons.append("insufficient_independent_evidence")
        if consistency["weak_single_interface_match"]:
            review_reasons.append("weak_single_interface_match")
        if consistency["group_consistency_score"] < float(
            config["group_consistency_threshold"]
        ):
            review_reasons.append("group_consistency_below_threshold")
        if consistency["has_global_conflict"]:
            review_reasons.append("global_conflict")
        if consistency["blocks_larger_better_group"]:
            review_reasons.append("blocks_larger_near_tied_group")
        if review_reasons:
            item["geometry_tier"] = "review"
            item["decision_reasons"].extend(review_reasons)
            review.append(item)
        else:
            item["geometry_tier"] = "accepted_for_pose_validation"
            item["decision_reasons"].append(
                "passed_conservative_geometry_gate"
            )
            accepted.append(item)
    key = lambda item: (
        float(item["geometry_score"]),
        float(item["consistency"]["group_consistency_score"]),
        len(item["parts"]),
        item["group_id"],
    )
    return (
        sorted(accepted, key=key, reverse=True),
        sorted(review, key=key, reverse=True),
        sorted(rejected, key=key, reverse=True),
    )


def _pose_record(
    pool: Path,
    item: dict[str, Any],
    validation: dict[str, Any] | None,
    *,
    checked: bool,
) -> dict[str, Any]:
    run_dir = pool / "validation" / item["group_id"]
    search = (
        _load(run_dir / "search_report.json")
        if (run_dir / "search_report.json").is_file()
        else {}
    )
    metrics = (validation or {}).get("metrics", {})
    pose_audit = search.get("pose_candidate_audit", [])
    accepted = bool(metrics.get("physical_pose_valid", metrics.get("accepted")))
    exact_status = metrics.get("collision_result") or metrics.get(
        "exact_collision_check_status", "not_run"
    )
    worker_status = metrics.get(
        "worker_status", "success" if validation else "not_run"
    )
    if accepted:
        status = "valid"
    elif not checked or worker_status != "success":
        status = "uncertain"
    else:
        rejected_reasons = [
            entry.get("rejection_reason")
            for entry in pose_audit
            if entry.get("rejection_reason")
        ]
        complete_count = int(
            search.get("complete_pose_candidate_count", 0) or 0
        )
        checked_count = len(pose_audit)
        definite = bool(rejected_reasons) and all(
            reason in {"constraint_residual", "solid_penetration"}
            for reason in rejected_reasons
        ) and complete_count > 0 and complete_count <= checked_count
        status = "failed" if definite else "uncertain"
    collisions = metrics.get("exact_collisions", [])
    common_volumes = [
        float(
            row.get(
                "intersection_volume_mm3",
                row.get("intersection_volume", 0.0),
            )
            or 0.0
        )
        for row in collisions
    ]
    reasons = metrics.get("rejection_reason_per_rank") or [
        {
            "rank": row.get("rank"),
            "reason": row.get("rejection_reason"),
        }
        for row in pose_audit
        if row.get("rejection_reason")
    ]
    return {
        "schema_version": "1.0.0",
        "pool_id": pool.name,
        "candidate_id": item["group_id"],
        "parts": item["parts"],
        "group_size": len(item["parts"]),
        "checked_pose_count": int(
            metrics.get(
                "checked_pose_count",
                metrics.get("exact_pose_candidates_checked", len(pose_audit)),
            )
            or 0
        ),
        "best_pose_rank": metrics.get(
            "best_pose_rank",
            metrics.get("selected_pose_candidate_rank"),
        ),
        "rejection_reason_per_rank": reasons,
        "selected_constraint_residual": (
            validation.get("max_constraint_residual")
            if validation
            else None
        ),
        "collision_result": exact_status,
        "occt_common_volume": max(common_volumes, default=0.0)
        if checked
        else None,
        "worker_status": worker_status,
        "final_pose_status": status,
        "validation_result": (
            str(run_dir / "validation_result.json") if checked else None
        ),
        "assembly_manifest": (
            str(run_dir / "assembly_manifest.json")
            if (run_dir / "assembly_manifest.json").is_file()
            else None
        ),
        "assembly_step": (
            str(run_dir / "assembly.step")
            if (run_dir / "assembly.step").is_file()
            else None
        ),
    }


def validate_bounded_candidates(
    pool: Path,
    accepted_geometry: list[dict[str, Any]],
    review_geometry: list[dict[str, Any]],
    pipeline_config_path: Path,
    pipeline_config: dict[str, Any],
    conservative_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    maximum = int(
        conservative_config["maximum_pose_validations_per_pool"]
    )
    queue = (accepted_geometry + review_geometry)[:maximum]
    queue_ids = {item["group_id"] for item in queue}
    records: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(queue, start=1):
        print(
            f"{pool.name}: pose {position}/{len(queue)} "
            f"{item['group_id']} size={len(item['parts'])}",
            flush=True,
        )
        proposal = GroupProposal.model_validate(
            {
                key: item[key]
                for key in (
                    "schema_version",
                    "group_id",
                    "parts",
                    "candidate_edges",
                    "geometry_score",
                    "connected",
                    "status",
                    "reasons",
                )
            }
        )
        validation = isolated_solve_and_validate_group(
            pool, proposal, pipeline_config_path, pipeline_config
        )
        records[item["group_id"]] = _pose_record(
            pool, item, validation, checked=True
        )
    for item in accepted_geometry + review_geometry:
        if item["group_id"] not in queue_ids:
            records[item["group_id"]] = _pose_record(
                pool, item, None, checked=False
            )
    return records


def _baseline_rows(pool: Path) -> list[dict[str, Any]]:
    path = pool / "validation" / "validated_group_assignment.json"
    if not path.is_file():
        return []
    truth = _truth_sets(pool)
    assignment = _load(path)
    return [
        {
            "mode": "before_conservative_gate",
            "pool_id": pool.name,
            "candidate_id": item["group_id"],
            "parts": "|".join(item["parts"]),
            "group_size": len(item["parts"]),
            "is_true_group": _parts_key(item["parts"]) in truth,
            "false_positive": _parts_key(item["parts"]) not in truth,
            "geometry_score": item.get("geometry_score"),
            "final_decision": "accepted_legacy",
            "reason": "legacy selected group passed physical validation",
        }
        for item in assignment.get("selected_groups", [])
        if len(item["parts"]) > 1
    ]


def _route_baseline_audit_rows(
    baseline_rows: list[dict[str, Any]],
    final_accepted: list[dict[str, Any]],
    final_review: list[dict[str, Any]],
    final_rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the conservative destination to every legacy accepted row."""

    post_gate_by_candidate = {
        (item["pool_id"], item["group_id"]): item
        for item in (final_accepted + final_review + final_rejected)
    }
    routed_rows = []
    for row in baseline_rows:
        routed = post_gate_by_candidate.get(
            (row["pool_id"], row["candidate_id"])
        )
        routed_rows.append(
            {
                **row,
                "post_gate_decision": (
                    routed.get("final_decision")
                    if routed is not None
                    else "not_generated_or_not_retained"
                ),
                "post_gate_review_queue_state": (
                    routed.get("review_queue_state", "")
                    if routed is not None
                    else ""
                ),
                "post_gate_reason": (
                    "|".join(routed.get("decision_reasons", []))
                    if routed is not None
                    else "candidate absent from final tier artifacts"
                ),
            }
        )
    return routed_rows


def _metrics(
    accepted: list[dict[str, Any]],
    review: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    truth_by_candidate: dict[str, bool],
) -> dict[str, Any]:
    accepted_tp = sum(
        truth_by_candidate.get(item["group_id"], False) for item in accepted
    )
    accepted_count = len(accepted)
    false_positives = accepted_count - accepted_tp
    total = accepted_count + len(review) + len(rejected)
    selected_review = [
        item
        for item in review
        if item.get("review_queue_state") == "selected"
    ]
    deferred_review = [
        item
        for item in review
        if item.get("review_queue_state") == "deferred"
    ]
    rejected_with_reason = sum(
        bool(item.get("decision_reasons")) for item in rejected
    )
    return {
        "schema_version": "1.0.0",
        "auto_accept_precision": (
            accepted_tp / accepted_count if accepted_count else None
        ),
        "accepted_group_count": accepted_count,
        "accepted_true_positive_count": accepted_tp,
        "false_positive_count": false_positives,
        "review_group_count": len(review),
        "review_rate": len(review) / total if total else None,
        "review_frontier_group_count": len(selected_review),
        "deferred_review_group_count": len(deferred_review),
        "operator_queue_rate": (
            len(selected_review) / total if total else None
        ),
        "review_frontier_compression": (
            1.0 - len(selected_review) / total if total else None
        ),
        "rejected_group_count": len(rejected),
        "unresolved_parts_count": len(unresolved),
        "workload_reduction": None if deferred_review else (
            1.0 - len(review) / total if total else None
        ),
        "workload_reduction_status": (
            "not_established_deferred_review_candidates_remain"
            if deferred_review
            else "estimated_from_immediate_review_queue"
        ),
        "rejected_reason_coverage": (
            rejected_with_reason / len(rejected) if rejected else 1.0
        ),
        "candidate_count": total,
        "semantic_reranking_enabled": False,
        "precision_target": 0.9,
        "precision_target_status": (
            "not_estimable_no_auto_accepts"
            if not accepted_count
            else (
                "passed"
                if accepted_tp / accepted_count >= 0.9
                else "failed"
            )
        ),
    }


def bound_review_queue(
    review: list[dict[str, Any]],
    *,
    maximum: int,
    per_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a bounded, structurally diverse operator review frontier.

    Automatic acceptance uses the unchanged conservative gate.  These metrics
    affect review ordering only and never turn a review into an acceptance.
    """

    def base_priority(item: dict[str, Any]) -> float:
        consistency = item["consistency"]
        completeness = float(
            consistency.get(
                "group_completeness_score",
                consistency.get("interface_coverage", 0.0),
            )
        )
        central_coverage = float(
            consistency.get(
                "central_part_coverage",
                float(consistency.get("has_central_part_structure", False)),
            )
        )
        interface_diversity = float(
            consistency.get(
                "interface_diversity_score",
                min(
                    float(
                        consistency.get(
                            "independent_evidence_count", 0
                        )
                    )
                    / 3.0,
                    1.0,
                ),
            )
        )
        consistency_score = float(
            consistency["group_consistency_score"]
        )
        score = (
            0.30 * completeness
            + 0.20 * central_coverage
            + 0.30 * interface_diversity
            + 0.20 * consistency_score
        )
        item["review_ranking"] = {
            "ranking_version": "structural_diversity_v1",
            "group_completeness_score": round(completeness, 8),
            "central_part_coverage": round(central_coverage, 8),
            "interface_diversity_score": round(
                interface_diversity, 8
            ),
            "group_consistency_score": round(consistency_score, 8),
            "base_priority_score": round(score, 8),
            "selection_novelty": None,
            "final_priority_score": None,
            "affects_auto_accept": False,
        }
        return score

    ranked = list(review)
    for item in ranked:
        base_priority(item)

    def novelty(
        item: dict[str, Any], selected: list[dict[str, Any]]
    ) -> float:
        if not selected:
            return 1.0
        parts = set(item["parts"])
        maximum_overlap = 0.0
        for chosen in selected:
            chosen_parts = set(chosen["parts"])
            union = parts | chosen_parts
            overlap = len(parts & chosen_parts) / len(union) if union else 1.0
            maximum_overlap = max(maximum_overlap, overlap)
        return 1.0 - maximum_overlap

    def choose_one(
        candidates: list[dict[str, Any]],
        selected: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        scored = []
        for item in candidates:
            item_novelty = novelty(item, selected)
            final_score = (
                0.80
                * float(item["review_ranking"]["base_priority_score"])
                + 0.20 * item_novelty
            )
            scored.append(
                (
                    final_score,
                    float(item["review_ranking"]["base_priority_score"]),
                    float(item["geometry_score"]),
                    item["group_id"],
                    item_novelty,
                    item,
                )
            )
        _, _, _, _, item_novelty, chosen = max(scored)
        chosen["review_ranking"]["selection_novelty"] = round(
            item_novelty, 8
        )
        chosen["review_ranking"]["final_priority_score"] = round(
            0.80
            * float(chosen["review_ranking"]["base_priority_score"])
            + 0.20 * item_novelty,
            8,
        )
        return chosen

    chosen_ids = set()
    chosen = []
    for size in range(2, 7):
        size_candidates = [
            row for row in ranked if len(row["parts"]) == size
        ]
        while (
            size_candidates
            and sum(len(row["parts"]) == size for row in chosen) < per_size
        ):
            item = choose_one(size_candidates, chosen)
            if item is None:
                break
            chosen.append(item)
            chosen_ids.add(item["group_id"])
            size_candidates = [
                row
                for row in size_candidates
                if row["group_id"] != item["group_id"]
            ]
    while len(chosen) < maximum:
        remaining = [
            item
            for item in ranked
            if item["group_id"] not in chosen_ids
        ]
        item = choose_one(remaining, chosen)
        if item is None:
            break
        chosen.append(item)
        chosen_ids.add(item["group_id"])
    ranked = sorted(
        ranked,
        key=lambda item: (
            float(item["review_ranking"]["base_priority_score"]),
            float(item["geometry_score"]),
            item["group_id"],
        ),
        reverse=True,
    )
    for item in ranked:
        if len(chosen) >= maximum:
            break
        if item["group_id"] not in chosen_ids:
            chosen.append(item)
            chosen_ids.add(item["group_id"])
    dominated = [
        item for item in ranked if item["group_id"] not in chosen_ids
    ]
    for review_rank, item in enumerate(chosen, start=1):
        item["review_queue_state"] = "selected"
        item["review_ranking"]["review_rank"] = review_rank
        item["decision_reasons"].append(
            "retained_in_bounded_human_review_frontier"
        )
        item["decision_reasons"] = sorted(set(item["decision_reasons"]))
    for deferred_rank, item in enumerate(dominated, start=len(chosen) + 1):
        item["final_decision"] = "review"
        item["review_queue_state"] = "deferred"
        item["review_ranking"]["review_rank"] = deferred_rank
        item["decision_reasons"].append(
            "deferred_outside_bounded_review_frontier"
        )
        item["decision_reasons"] = sorted(set(item["decision_reasons"]))
    return chosen, dominated


def run(
    root: str | Path,
    output_dir: str | Path,
    *,
    pipeline_config_path: str | Path,
    conservative_config_path: str | Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output_dir).resolve()
    pipeline_config_path = Path(pipeline_config_path).resolve()
    conservative_config_path = Path(conservative_config_path).resolve()
    pipeline_config = _load(pipeline_config_path)
    conservative = _load(conservative_config_path)
    pools = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "pool_gt.json").is_file()
    )
    baseline_rows = [
        row for pool in pools for row in _baseline_rows(pool)
    ]
    all_accepted_geometry, all_review_geometry, all_rejected_geometry = (
        [],
        [],
        [],
    )
    final_accepted, final_review, final_rejected, unresolved = [], [], [], []
    pose_validated, pose_failed, pose_uncertain = [], [], []
    candidate_score_rows = []
    truth_by_candidate: dict[str, bool] = {}

    for pool in pools:
        run_grouping(pool, pipeline_config_path)
        proposals = _load(pool / "grouping" / "group_proposals.json")
        edges = _load(pool / "index" / "pruned_candidates.json")
        accepted_geometry, review_geometry, rejected_geometry = (
            geometry_tiers(pool, proposals, edges, conservative)
        )
        all_accepted_geometry.extend(accepted_geometry)
        all_review_geometry.extend(review_geometry)
        all_rejected_geometry.extend(rejected_geometry)
        pose_records = validate_bounded_candidates(
            pool,
            accepted_geometry,
            review_geometry,
            pipeline_config_path,
            pipeline_config,
            conservative,
        )
        truth = _truth_sets(pool)
        accepted_pool, review_pool, rejected_pool = [], [], []
        for item in accepted_geometry + review_geometry + rejected_geometry:
            pose = pose_records.get(item["group_id"])
            truth_by_candidate[item["group_id"]] = (
                _parts_key(item["parts"]) in truth
            )
            item = {
                **item,
                "pose": pose,
                "semantic_application_mode": conservative[
                    "semantic_application_mode"
                ],
            }
            if item["geometry_tier"] == "rejected":
                item["final_decision"] = "rejected"
                rejected_pool.append(item)
            elif pose and pose["final_pose_status"] == "failed":
                item["final_decision"] = "rejected"
                item["decision_reasons"].append("physical_pose_failed")
                rejected_pool.append(item)
            elif _passes_final_acceptance(item, pose, conservative):
                item["final_decision"] = "accepted"
                accepted_pool.append(item)
            else:
                item["final_decision"] = "review"
                if pose and pose["final_pose_status"] == "uncertain":
                    item["decision_reasons"].append(
                        "pose_uncertain_or_not_checked"
                    )
                item["decision_reasons"] = sorted(
                    set(item["decision_reasons"])
                )
                review_pool.append(item)
            if pose:
                {
                    "valid": pose_validated,
                    "failed": pose_failed,
                    "uncertain": pose_uncertain,
                }[pose["final_pose_status"]].append(pose)

        accepted_pool, overlap_review = route_overlapping_accepts_to_review(
            accepted_pool
        )
        review_pool.extend(overlap_review)
        selected_review, deferred_review = bound_review_queue(
            review_pool,
            maximum=int(
                conservative["maximum_review_groups_per_pool"]
            ),
            per_size=int(conservative["review_groups_per_size"]),
        )
        review_pool = selected_review + deferred_review
        final_accepted.extend(accepted_pool)
        final_review.extend(review_pool)
        final_rejected.extend(rejected_pool)

        part_ids = sorted(
            path.name
            for path in (pool / "parts").iterdir()
            if path.suffix.lower() in {".step", ".stp"}
        )
        accepted_parts = {
            part for item in accepted_pool for part in item["parts"]
        }
        for part in part_ids:
            if part in accepted_parts:
                continue
            related = [
                item["group_id"]
                for item in review_pool
                if part in item["parts"]
                and item.get("review_queue_state") == "selected"
            ][:20]
            deferred_count = sum(
                part in item["parts"]
                and item.get("review_queue_state") == "deferred"
                for item in review_pool
            )
            unresolved.append(
                {
                    "pool_id": pool.name,
                    "part_id": part,
                    "reason": (
                        "only review candidates available"
                        if related
                        else "no accepted or review candidate"
                    ),
                    "review_candidate_ids": related,
                    "deferred_review_candidate_count": deferred_count,
                }
            )
        for item in accepted_pool + review_pool + rejected_pool:
            candidate_score_rows.append(
                {
                    "pool_id": pool.name,
                    "candidate_id": item["group_id"],
                    "parts": "|".join(item["parts"]),
                    "group_size": len(item["parts"]),
                    "geometry_score": item["geometry_score"],
                    "group_consistency_score": item["consistency"][
                        "group_consistency_score"
                    ],
                    "independent_evidence_count": item["consistency"][
                        "independent_evidence_count"
                    ],
                    "pose_status": (
                        item["pose"]["final_pose_status"]
                        if item.get("pose")
                        else "not_applicable"
                    ),
                    "semantic_reranking_enabled": False,
                    "final_decision": item["final_decision"],
                    "review_queue_state": item.get("review_queue_state"),
                    "evaluation_is_true_group": truth_by_candidate[
                        item["group_id"]
                    ],
                    "decision_reasons": "|".join(item["decision_reasons"]),
                }
            )
        print(
            f"{pool.name}: accepted={len(accepted_pool)} "
            f"review={len(review_pool)} rejected={len(rejected_pool)}",
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    _write(output / "accepted_geometry_candidates.json", all_accepted_geometry)
    _write(output / "review_geometry_candidates.json", all_review_geometry)
    _write(output / "rejected_geometry_candidates.json", all_rejected_geometry)
    _write(output / "pose_validated_candidates.json", pose_validated)
    _write(output / "pose_failed_candidates.json", pose_failed)
    _write(output / "pose_uncertain_candidates.json", pose_uncertain)
    _write(output / "final_accepted_groups.json", final_accepted)
    _write(output / "final_review_groups.json", final_review)
    _write(output / "final_rejected_groups.json", final_rejected)
    _write(output / "unresolved_parts.json", unresolved)
    metrics = _metrics(
        final_accepted,
        final_review,
        final_rejected,
        unresolved,
        truth_by_candidate,
    )
    baseline_accepted = len(baseline_rows)
    baseline_tp = sum(
        bool(row["is_true_group"]) for row in baseline_rows
    )
    baseline_assignment_paths = [
        pool / "validation" / "validated_group_assignment.json"
        for pool in pools
    ]
    baseline_available = all(
        path.is_file() for path in baseline_assignment_paths
    )
    baseline_unresolved = (
        sum(
            1
            for path in baseline_assignment_paths
            for item in _load(path).get("selected_groups", [])
            if len(item["parts"]) == 1
        )
        if baseline_available
        else None
    )
    baseline = {
        "status": (
            "available"
            if baseline_available
            else "unavailable_no_legacy_validated_assignment"
        ),
        "accepted_group_count": (
            baseline_accepted if baseline_available else None
        ),
        "accepted_true_positive_count": (
            baseline_tp if baseline_available else None
        ),
        "auto_accept_precision": (
            baseline_tp / baseline_accepted
            if baseline_available and baseline_accepted
            else None
        ),
        "false_positive_count": (
            baseline_accepted - baseline_tp
            if baseline_available
            else None
        ),
        "review_group_count": 0 if baseline_available else None,
        "unresolved_parts_count": baseline_unresolved,
    }
    comparison = {
        "before": baseline,
        "after": dict(metrics),
        "false_positive_reduction": (
            baseline["false_positive_count"]
            - metrics["false_positive_count"]
            if baseline["false_positive_count"] is not None
            else None
        ),
    }
    metrics = {**metrics, "baseline_comparison": comparison}
    _write(output / "conservative_metrics.json", metrics)
    _write(output / "baseline_comparison.json", comparison)
    _write(
        output / "final_decision_truth_audit.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "evaluation_only",
            "truth_type": "functional_validity",
            "source_id_is_production_truth": False,
            "records": [
                {
                    "pool_id": item["pool_id"],
                    "candidate_id": item["group_id"],
                    "parts": item["parts"],
                    "final_decision": item["final_decision"],
                    "review_queue_state": item.get("review_queue_state"),
                    "evaluation_is_true_group": truth_by_candidate[
                        item["group_id"]
                    ],
                }
                for item in (
                    final_accepted + final_review + final_rejected
                )
            ],
        },
    )

    routed_baseline_rows = _route_baseline_audit_rows(
        baseline_rows, final_accepted, final_review, final_rejected
    )
    fp_rows = routed_baseline_rows + [
        {
            "mode": "after_conservative_gate",
            "pool_id": item["pool_id"],
            "candidate_id": item["group_id"],
            "parts": "|".join(item["parts"]),
            "group_size": len(item["parts"]),
            "is_true_group": truth_by_candidate[item["group_id"]],
            "false_positive": not truth_by_candidate[item["group_id"]],
            "geometry_score": item["geometry_score"],
            "final_decision": "accepted",
            "reason": "|".join(item["decision_reasons"]),
            "post_gate_decision": "accepted",
            "post_gate_review_queue_state": "",
            "post_gate_reason": "|".join(item["decision_reasons"]),
        }
        for item in final_accepted
    ]
    with (output / "false_positive_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "mode",
            "pool_id",
            "candidate_id",
            "parts",
            "group_size",
            "is_true_group",
            "false_positive",
            "geometry_score",
            "final_decision",
            "reason",
            "post_gate_decision",
            "post_gate_review_queue_state",
            "post_gate_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fp_rows)
    with (output / "candidate_scores.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(candidate_score_rows[0])
        )
        writer.writeheader()
        writer.writerows(candidate_score_rows)

    tier_report = "\n".join(
        [
            "# Geometry Candidate Tiering",
            "",
            f"- Accepted for pose validation: {len(all_accepted_geometry)}",
            f"- Review: {len(all_review_geometry)}",
            f"- Rejected: {len(all_rejected_geometry)}",
            "",
            "A geometry-tier pass is not a final assembly decision.",
            "",
        ]
    )
    (output / "geometry_candidate_tiering.md").write_text(
        tier_report, encoding="utf-8"
    )
    pose_report = "\n".join(
        [
            "# Pose Validation Report",
            "",
            f"- Valid: {len(pose_validated)}",
            f"- Failed: {len(pose_failed)}",
            f"- Uncertain/not checked: {len(pose_uncertain)}",
            "",
            "Pose validity proves physical feasibility only, not common "
            "assembly provenance.",
            "",
        ]
    )
    (output / "pose_validation_report.md").write_text(
        pose_report, encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Conservative Assembly Report",
            "",
            "## 1. Input parts",
            f"{sum(len(list((pool / 'parts').glob('*.step'))) for pool in pools)} "
            f"parts across {len(pools)} frozen pools.",
            "",
            "## 2. Candidate recall audit",
            "See `candidate_recall_audit.md` and the two CSV exception files.",
            "",
            "## 3. Geometry candidates",
            f"{metrics['candidate_count']} total proposals.",
            "",
            "## 4. Final tiers",
            f"Accepted={metrics['accepted_group_count']}, "
            f"review={metrics['review_group_count']}, "
            f"rejected={metrics['rejected_group_count']}.",
            "",
            "## 5. Pose validation",
            f"Valid={len(pose_validated)}, failed={len(pose_failed)}, "
            f"uncertain={len(pose_uncertain)}.",
            "",
            "## 6. Semantic gate",
            "Disabled; DeepSeek is explanation-only.",
            "",
            "## 7. Auto-accepted groups",
            f"{metrics['accepted_group_count']} groups; every record contains "
            "its evidence and gate reasons.",
            "",
            "## 8. Review groups",
            f"{metrics['review_group_count']} review-required groups: "
            f"{metrics['review_frontier_group_count']} selected for the "
            f"immediate operator frontier and "
            f"{metrics['deferred_review_group_count']} deferred. Deferred "
            "review is not rejection.",
            "",
            "## 9. Rejected groups",
            f"{metrics['rejected_group_count']} groups; reason coverage "
            f"{metrics['rejected_reason_coverage']:.2%}.",
            "",
            "## 10. Unresolved parts",
            f"{metrics['unresolved_parts_count']} pool-local parts.",
            "",
            "## 11. False-positive risk",
            f"Legacy baseline: {baseline['status']}; before false positives: "
            f"{baseline['false_positive_count']}; after: "
            f"{metrics['false_positive_count']}. Auto-accept precision is "
            f"{metrics['auto_accept_precision']}.",
            "",
            "## 12. Next step",
            "Reduce the immediate review frontier with deterministic local "
            "interface ranking while preserving measured truth-candidate "
            "recall. Keep semantic reranking disabled.",
            "",
        ]
    )
    (output / "assembly_report.md").write_text(report, encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="mixed_pools_v1")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "data" / "results"),
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(
            Path(__file__).parent / "configs" / "pool_pipeline.json"
        ),
    )
    parser.add_argument(
        "--conservative-config",
        default=str(
            Path(__file__).parent
            / "configs"
            / "conservative_pipeline.json"
        ),
    )
    args = parser.parse_args()
    result = run(
        args.root,
        args.output,
        pipeline_config_path=args.pipeline_config,
        conservative_config_path=args.conservative_config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
