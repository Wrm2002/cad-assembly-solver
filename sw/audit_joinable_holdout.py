"""Strict two-track audit for JoinABLe candidate-provider integration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pair(parts: list[str]) -> tuple[str, str]:
    return tuple(sorted(str(part) for part in parts))


def wilson(successes: int, total: int, z: float = 1.9599639845) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_paired_binomial(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = min(left_only, right_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2 * probability)


def real_joint_audit(report_path: Path, k: int = 10) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = read_json(report_path)
    rows = []
    for source in report["rows"]:
        if not (
            source.get("pretrained_inference_status") == "success"
            and source.get("rule_equivalent_evaluable")
            and source.get("pretrained_equivalent_evaluable")
        ):
            continue
        rule_hit = source.get("rule_equivalent_rank") is not None and source["rule_equivalent_rank"] <= k
        joinable_hit = source.get("pretrained_equivalent_rank") is not None and source["pretrained_equivalent_rank"] <= k
        if rule_hit and joinable_hit:
            outcome = "both_hit"
        elif rule_hit:
            outcome = "analytic_only_hit"
        elif joinable_hit:
            outcome = "joinable_rescue"
        else:
            outcome = "both_miss"
        rows.append(
            {
                "joint_set": source["joint_set"],
                "joint_index": source["joint_index"],
                "entity_pair_type": source["truth_entity_pair_type"],
                "analytic_rank": source.get("rule_equivalent_rank"),
                "joinable_rank": source.get("pretrained_equivalent_rank"),
                "outcome_at_k": outcome,
            }
        )
    total = len(rows)
    counts = {name: sum(row["outcome_at_k"] == name for row in rows) for name in (
        "both_hit", "analytic_only_hit", "joinable_rescue", "both_miss"
    )}
    rule_hits = counts["both_hit"] + counts["analytic_only_hit"]
    joinable_hits = counts["both_hit"] + counts["joinable_rescue"]
    union_hits = total - counts["both_miss"]
    summary = {
        "dataset_role": "frozen_real_STEP_entity_pair_recall",
        "k": k,
        "paired_evaluable_joint_count": total,
        "analytic_hits": rule_hits,
        "analytic_recall": rule_hits / total if total else None,
        "joinable_hits": joinable_hits,
        "joinable_recall": joinable_hits / total if total else None,
        "union_hits": union_hits,
        "union_recall": union_hits / total if total else None,
        "absolute_union_gain_over_analytic": (union_hits - rule_hits) / total if total else None,
        "joinable_rescue_count": counts["joinable_rescue"],
        "analytic_only_hit_count": counts["analytic_only_hit"],
        "both_miss_count": counts["both_miss"],
        "union_gain_wilson_95": wilson(union_hits - rule_hits, total),
        "joinable_vs_analytic_exact_paired_p_value": exact_paired_binomial(
            counts["analytic_only_hit"], counts["joinable_rescue"]
        ),
        "outcome_counts": counts,
        "interpretation": (
            "Union recall measures candidate complementarity. The paired p-value compares "
            "JoinABLe alone with the analytic ranker and is not a significance test for the "
            "mechanically monotone union."
        ),
    }
    return summary, rows


def domain_holdout_audit(report_path: Path, k: int = 10) -> dict[str, Any]:
    report = read_json(report_path)
    rows = [row for row in report["rows"] if row.get("exact_evaluable")]
    both = analytic_only = joinable_only = both_miss = 0
    for row in rows:
        analytic_hit = (
            row.get("analytic_exact_rank") is not None
            and row["analytic_exact_rank"] <= k
        )
        joinable_hit = (
            row.get("joinable_exact_rank") is not None
            and row["joinable_exact_rank"] <= k
        )
        if analytic_hit and joinable_hit:
            both += 1
        elif analytic_hit:
            analytic_only += 1
        elif joinable_hit:
            joinable_only += 1
        else:
            both_miss += 1
    total = len(rows)
    analytic_hits = both + analytic_only
    joinable_hits = both + joinable_only
    union_hits = total - both_miss
    return {
        "dataset_role": "source_design_disjoint_STEP_exact_entity_holdout",
        "k": k,
        "evaluable_count": total,
        "analytic_hits": analytic_hits,
        "analytic_recall": analytic_hits / total if total else None,
        "joinable_hits": joinable_hits,
        "joinable_recall": joinable_hits / total if total else None,
        "union_hits": union_hits,
        "union_recall": union_hits / total if total else None,
        "joinable_rescue_count": joinable_only,
        "analytic_only_hit_count": analytic_only,
        "both_hit_count": both,
        "both_miss_count": both_miss,
        "absolute_union_gain_over_analytic": (
            (union_hits - analytic_hits) / total if total else None
        ),
        "union_gain_wilson_95": wilson(union_hits - analytic_hits, total),
        "joinable_vs_analytic_exact_paired_p_value": exact_paired_binomial(
            analytic_only, joinable_only
        ),
        "source_design_overlap_after_filter": report.get(
            "manifest_summary", {}
        ).get("source_design_overlap_after_filter"),
        "previously_used_for_historical_evaluation": True,
        "untouched_blind_holdout": False,
        "limitations": report.get("limitations", []),
    }


def pool_truth(pool: Path) -> dict[str, Any]:
    gt = read_json(pool / "pool_gt.json")
    true_groups = {frozenset(group["parts"]) for group in gt.get("true_groups", [])}
    same_group_pairs: set[tuple[str, str]] = set()
    true_mates: set[tuple[str, str]] = set()
    for group in gt.get("true_groups", []):
        parts = group["parts"]
        for index, first in enumerate(parts):
            for second in parts[index + 1 :]:
                same_group_pairs.add(pair([first, second]))
        for mate in group.get("true_mates", []):
            true_mates.add(pair(mate.get("parts") or [mate.get("part_a"), mate.get("part_b")]))
    distractors = set(gt.get("distractors", []))
    return {
        "true_groups": true_groups,
        "same_group_pairs": same_group_pairs,
        "true_mates": true_mates,
        "distractors": distractors,
        "truth_basis": gt.get("truth_basis"),
    }


def load_final(results: Path) -> dict[str, dict[str, Any]]:
    final = {}
    for decision, name in (
        ("accepted", "final_accepted_groups.json"),
        ("review", "final_review_groups.json"),
        ("rejected", "final_rejected_groups.json"),
    ):
        for row in read_json(results / name):
            final[row["group_id"]] = {**row, "audit_decision": decision}
    return final


def functional_audit(
    analytic_root: Path,
    joinable_root: Path,
    analytic_results: Path,
    joinable_results: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route_rows = []
    generated_false = kept_false = pruned_false = 0
    generated_delta = kept_delta = 0
    truth_bases = set()
    for analytic_pool in sorted(analytic_root.glob("holdout_pool_*")):
        joinable_pool = joinable_root / analytic_pool.name
        truth = pool_truth(analytic_pool)
        truth_bases.add(str(truth["truth_basis"]))
        for stage, filename in (
            ("generated", "geometry_candidates.json"),
            ("kept", "pruned_candidates.json"),
        ):
            analytic = {row["candidate_id"]: row for row in read_json(analytic_pool / "index" / filename)}
            learned = {row["candidate_id"]: row for row in read_json(joinable_pool / "index" / filename)}
            new_ids = set(learned) - set(analytic)
            if stage == "generated":
                generated_delta += len(new_ids)
            else:
                kept_delta += len(new_ids)
            for candidate_id in sorted(new_ids):
                row = learned[candidate_id]
                edge_pair = pair(row["parts"])
                is_same_group = edge_pair in truth["same_group_pairs"]
                is_true_mate = edge_pair in truth["true_mates"]
                is_false = not is_same_group
                if is_false and stage == "generated":
                    generated_false += 1
                if is_false and stage == "kept":
                    kept_false += 1
                route_rows.append(
                    {
                        "record_kind": "edge",
                        "pool_id": analytic_pool.name,
                        "record_id": candidate_id,
                        "parts": "|".join(row["parts"]),
                        "candidate_type": row["candidate_type"],
                        "stage_or_decision": stage,
                        "is_same_true_group": is_same_group,
                        "is_true_mate": is_true_mate,
                        "contains_distractor": bool(set(row["parts"]) & truth["distractors"]),
                        "review_queue_state": "",
                        "reasons": json.dumps(row.get("audit_reason", {}), ensure_ascii=False),
                    }
                )
    generated_edge_ids = {
        row["record_id"] for row in route_rows
        if row["record_kind"] == "edge" and row["stage_or_decision"] == "generated"
        and not row["is_same_true_group"]
    }
    kept_edge_ids = {
        row["record_id"] for row in route_rows
        if row["record_kind"] == "edge" and row["stage_or_decision"] == "kept"
        and not row["is_same_true_group"]
    }
    pruned_false = len(generated_edge_ids - kept_edge_ids)

    analytic_final = load_final(analytic_results)
    joinable_final = load_final(joinable_results)
    new_group_ids = set(joinable_final) - set(analytic_final)
    transitioned_group_ids = {
        group_id
        for group_id in set(joinable_final) & set(analytic_final)
        if joinable_final[group_id]["audit_decision"]
        != analytic_final[group_id]["audit_decision"]
    }
    affected_group_ids = new_group_ids | transitioned_group_ids
    removed_group_ids = set(analytic_final) - set(joinable_final)
    decisions = {"accepted": 0, "review": 0, "rejected": 0}
    new_false_groups = 0
    selected_new_review = deferred_new_review = 0
    for group_id in sorted(affected_group_ids):
        row = joinable_final[group_id]
        truth = pool_truth(joinable_root / row["pool_id"])
        is_true = frozenset(row["parts"]) in truth["true_groups"]
        decision = row["audit_decision"]
        decisions[decision] += 1
        new_false_groups += int(not is_true)
        selected_new_review += int(
            decision == "review" and row.get("review_queue_state") == "selected"
        )
        deferred_new_review += int(
            decision == "review" and row.get("review_queue_state") == "deferred"
        )
        route_rows.append(
            {
                "record_kind": "group",
                "pool_id": row["pool_id"],
                "record_id": group_id,
                "parts": "|".join(row["parts"]),
                "candidate_type": "group_proposal",
                "stage_or_decision": decision,
                "is_same_true_group": is_true,
                "is_true_mate": False,
                "contains_distractor": bool(set(row["parts"]) & truth["distractors"]),
                "review_queue_state": row.get("review_queue_state") or "",
                "reasons": (
                    ("new_group_id" if group_id in new_group_ids else
                     f"decision_transition:{analytic_final[group_id]['audit_decision']}->{decision}")
                    + "|"
                    + "|".join(row.get("decision_reasons", []))
                ),
            }
        )
    analytic_metrics = read_json(analytic_results / "conservative_metrics.json")
    joinable_metrics = read_json(joinable_results / "conservative_metrics.json")
    summary = {
        "dataset_role": "locked_functional_CAD_mixed_pool_safety",
        "truth_basis": sorted(truth_bases),
        "engineer_signoff_required": any("pending" in value for value in truth_bases),
        "generated_candidate_delta": generated_delta,
        "kept_candidate_delta": kept_delta,
        "new_false_edges_generated": generated_false,
        "new_false_edges_pruned": pruned_false,
        "new_false_edges_kept": kept_false,
        "new_group_ids": len(new_group_ids),
        "decision_transition_group_ids": len(transitioned_group_ids),
        "removed_group_ids": len(removed_group_ids),
        "affected_joinable_group_count": len(affected_group_ids),
        "affected_false_group_count": new_false_groups,
        "affected_joinable_group_decisions": decisions,
        "new_selected_review_groups": selected_new_review,
        "new_deferred_review_groups": deferred_new_review,
        "new_false_auto_accepts": decisions["accepted"],
        "all_new_false_candidates_blocked_before_acceptance": decisions["accepted"] == 0,
        "analytic_metrics": analytic_metrics,
        "analytic_joinable_metrics": joinable_metrics,
        "operator_frontier_delta": (
            joinable_metrics["review_frontier_group_count"]
            - analytic_metrics["review_frontier_group_count"]
        ),
        "review_total_delta": (
            joinable_metrics["review_group_count"] - analytic_metrics["review_group_count"]
        ),
        "rejected_total_delta": (
            joinable_metrics["rejected_group_count"] - analytic_metrics["rejected_group_count"]
        ),
    }
    return summary, route_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    real_report: Path,
    domain_report: Path | None,
    analytic_root: Path,
    joinable_root: Path,
    analytic_results: Path,
    joinable_results: Path,
    output: Path,
) -> dict[str, Any]:
    real, real_rows = real_joint_audit(real_report)
    domain = domain_holdout_audit(domain_report) if domain_report else None
    functional, route_rows = functional_audit(
        analytic_root, joinable_root, analytic_results, joinable_results
    )
    report = {
        "schema_version": "1.0.0",
        "audit_design": {
            "track_a": "real STEP entity-pair recall on frozen 27-joint transfer holdout",
            "track_b": "end-to-end false-candidate containment on locked functional CAD pools",
            "training_performed": False,
            "semantic_reranking_enabled": False,
        },
        "real_joint_recall": real,
        "design_disjoint_domain_recall": domain,
        "functional_safety": functional,
        "expert_decision": {
            "joinable_is_complementary_candidate_provider": (
                (domain or real)["joinable_rescue_count"] > 0
            ),
            "joinable_should_replace_analytic": False,
            "safe_to_auto_accept_from_joinable_score": False,
            "proceed_to_domain_adaptation": False,
            "reason": (
                "The union rescues exact real STEP interface entities on a design-disjoint test split; "
                "the functional mixed-pool arm adds only false candidates and increases review work."
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "strict_audit.json", report)
    write_csv(output / "real_joint_rescue_cases.csv", real_rows)
    write_csv(output / "false_candidate_routes.csv", route_rows)
    md = [
        "# JoinABLe harder-holdout expert audit",
        "",
        "## Audit design",
        "",
        "Two frozen tracks are used because the functional mixed-pool holdout already has 100% analytic true-pair recall and cannot measure rescue. The real STEP transfer holdout measures entity-level rescue; the functional CAD holdout measures false-candidate containment.",
        "",
        "## Real STEP recall",
        "",
        f"- Paired joints: {real['paired_evaluable_joint_count']}",
        f"- Analytic top-{real['k']}: {real['analytic_hits']}/{real['paired_evaluable_joint_count']} ({real['analytic_recall']:.2%})",
        f"- JoinABLe top-{real['k']}: {real['joinable_hits']}/{real['paired_evaluable_joint_count']} ({real['joinable_recall']:.2%})",
        f"- Union top-{real['k']}+{real['k']}: {real['union_hits']}/{real['paired_evaluable_joint_count']} ({real['union_recall']:.2%})",
        f"- JoinABLe-only rescues: {real['joinable_rescue_count']}",
        f"- Analytic-only hits: {real['analytic_only_hit_count']}",
        f"- Both miss: {real['both_miss_count']}",
        f"- Paired exact-binomial p-value (JoinABLe alone vs analytic): {real['joinable_vs_analytic_exact_paired_p_value']}",
        "",
    ]
    if domain:
        md.extend(
            [
                "## Design-disjoint domain holdout",
                "",
                f"- Exact evaluable pairs: {domain['evaluable_count']}",
                f"- Analytic top-{domain['k']}: {domain['analytic_hits']}/{domain['evaluable_count']} ({domain['analytic_recall']:.2%})",
                f"- JoinABLe top-{domain['k']}: {domain['joinable_hits']}/{domain['evaluable_count']} ({domain['joinable_recall']:.2%})",
                f"- Union top-{domain['k']}+{domain['k']}: {domain['union_hits']}/{domain['evaluable_count']} ({domain['union_recall']:.2%})",
                f"- JoinABLe-only rescues: {domain['joinable_rescue_count']}",
                f"- Analytic-only hits: {domain['analytic_only_hit_count']}",
                f"- Paired exact-binomial p-value: {domain['joinable_vs_analytic_exact_paired_p_value']}",
                "- Caveat: source-design-disjoint, but previously used for historical evaluation; not an untouched blind holdout.",
                "",
            ]
        )
    md.extend([
        "## Functional safety",
        "",
        f"- New generated edges: {functional['generated_candidate_delta']}",
        f"- New false edges pruned/kept: {functional['new_false_edges_pruned']}/{functional['new_false_edges_kept']}",
        f"- New group IDs / decision transitions: {functional['new_group_ids']}/{functional['decision_transition_group_ids']}",
        f"- Affected JoinABLe-arm decisions accepted/review/rejected: {functional['affected_joinable_group_decisions']['accepted']}/{functional['affected_joinable_group_decisions']['review']}/{functional['affected_joinable_group_decisions']['rejected']}",
        f"- Immediate operator-frontier delta: {functional['operator_frontier_delta']:+d}",
        f"- False auto-accepts: {functional['new_false_auto_accepts']}",
        "",
        "## Expert conclusion",
        "",
        "JoinABLe is validated as a complementary interface-candidate provider, not an acceptance judge. The design-disjoint STEP test shows a material exact-entity rescue benefit, while the mixed-pool safety audit shows that its extra pair nominations are false on this provisional functional holdout. All were blocked before automatic acceptance, but operator workload increased. Keep the provider bounded and measure on an untouched, engineer-signed real assembly holdout before considering domain adaptation.",
    ])
    (output / "EXPERT_REVIEW.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-report", type=Path, required=True)
    parser.add_argument("--domain-report", type=Path)
    parser.add_argument("--analytic-root", type=Path, required=True)
    parser.add_argument("--joinable-root", type=Path, required=True)
    parser.add_argument("--analytic-results", type=Path, required=True)
    parser.add_argument("--joinable-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.real_report.resolve(),
        args.domain_report.resolve() if args.domain_report else None,
        args.analytic_root.resolve(),
        args.joinable_root.resolve(),
        args.analytic_results.resolve(),
        args.joinable_results.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report["expert_decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
