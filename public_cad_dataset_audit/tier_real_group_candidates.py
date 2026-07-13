"""Apply conservative pre-pose gates to real mixed-pool group proposals."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))

from global_optimizer.group_consistency import (  # noqa: E402
    assess_group_consistency,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pair_key(pool_id: str, values: Iterable[str]) -> tuple[str, str, str]:
    first, second = sorted(str(value) for value in values)
    return pool_id, first, second


def graph_connected(
    parts: list[str],
    pairs: Iterable[tuple[str, str]],
) -> bool:
    if len(parts) < 2:
        return False
    adjacency = {part: set() for part in parts}
    for first, second in pairs:
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)
    seen = {parts[0]}
    queue = deque([parts[0]])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(parts)


def maximum_spanning_tree(
    parts: list[str],
    weighted_edges: list[tuple[float, str, str, dict[str, Any]]],
) -> list[tuple[float, str, str, dict[str, Any]]] | None:
    parent = {part: part for part in parts}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    selected = []
    for edge in sorted(
        weighted_edges,
        key=lambda row: (-row[0], row[1], row[2]),
    ):
        score, first, second, payload = edge
        root_a, root_b = find(first), find(second)
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        selected.append((score, first, second, payload))
        if len(selected) == len(parts) - 1:
            return selected
    return None


def physical_evidence_types(rule: dict[str, Any]) -> list[str]:
    """Extract distinct geometric constraints from one analytic interface.

    Model/provider agreement is deliberately excluded.  It is useful
    corroboration, but two algorithms reading the same STEP geometry are not
    two independent physical facts.
    """
    family = str(rule.get("joint_family_candidate") or "unknown")
    evidence = rule.get("score_evidence") or {}
    output = []
    if float(evidence.get("type_compatibility", 0.0) or 0.0) >= 0.9:
        output.append(f"{family}_interface_type_match")
    if (
        float(
            evidence.get("characteristic_size_compatibility", 0.0) or 0.0
        )
        >= 0.8
    ):
        output.append("characteristic_size_match")
    if (
        family == "coaxial"
        and float(evidence.get("radius_compatibility", 0.0) or 0.0) >= 0.8
    ):
        output.append("radius_fit")
    return sorted(set(output))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument("group_proposals", type=Path)
    parser.add_argument("candidate_graph_input", type=Path)
    parser.add_argument("rule_geometry_report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-threshold", type=float, default=0.80)
    parser.add_argument(
        "--hard-reject-geometry-threshold", type=float, default=0.35
    )
    parser.add_argument(
        "--group-consistency-threshold", type=float, default=0.70
    )
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    output_dir = args.output_dir.resolve()
    search = read_json(args.group_proposals.resolve())
    candidate_graph = read_json(args.candidate_graph_input.resolve())
    rule_report = read_json(args.rule_geometry_report.resolve())

    candidate_edge_map = {}
    for pool in candidate_graph.get("pools", []):
        pool_id = str(pool["pool_id"])
        for edge in pool.get("edges", []):
            candidate_edge_map[pair_key(pool_id, edge["part_pair"])] = edge
    rule_map = {
        pair_key(
            str(row["pool_id"]), [row["part_a"], row["part_b"]]
        ): row
        for row in rule_report.get("pairs", [])
    }
    consistency_edge_by_pool = {
        str(pool["pool_id"]): pool.get("candidate_edge_records", [])
        for pool in search.get("pools", [])
    }
    proposals_by_pool = defaultdict(list)
    for proposal in search.get("proposals", []):
        proposals_by_pool[str(proposal["pool_id"])].append(proposal)
    truth_keys = set()
    for pool_dir in sorted(root.glob("pool_*")):
        pool_gt = read_json(pool_dir / "pool_gt.json")
        truth_keys.update(
            (
                pool_dir.name,
                frozenset(str(value) for value in group["part_ids"]),
            )
            for group in pool_gt.get("true_groups", [])
        )

    accepted_geometry = []
    review_geometry = []
    rejected_geometry = []
    truth_tier_counts = defaultdict(int)
    all_tier_counts = defaultdict(int)
    for pool_id, pool_proposals in proposals_by_pool.items():
        enriched_pool = []
        for proposal in pool_proposals:
            item = dict(proposal)
            parts = list(item["parts"])
            weighted_rule_edges = []
            rule_failure_pairs = []
            family_counts = defaultdict(int)
            high_rule_pairs = []
            model_success_pairs = []
            fallback_pairs = []
            for first, second in combinations(parts, 2):
                key = pair_key(pool_id, [first, second])
                rule = rule_map.get(key)
                if rule and rule.get("status") == "success":
                    score = float(rule["geometry_score"])
                    weighted_rule_edges.append(
                        (score, first, second, rule)
                    )
                    family_counts[
                        str(
                            rule.get("joint_family_candidate")
                            or "unknown"
                        )
                    ] += 1
                    if score >= args.geometry_threshold:
                        high_rule_pairs.append((first, second))
                else:
                    rule_failure_pairs.append([first, second])
                candidate = candidate_edge_map.get(key)
                if candidate and candidate.get(
                    "active_for_group_search"
                ):
                    if candidate.get("model_status") == "success":
                        model_success_pairs.append((first, second))
                    else:
                        fallback_pairs.append((first, second))
            spanning_tree = maximum_spanning_tree(
                parts, weighted_rule_edges
            )
            geometry_score = (
                min(edge[0] for edge in spanning_tree)
                if spanning_tree
                else None
            )
            high_rule_connected = graph_connected(
                parts, high_rule_pairs
            )
            model_success_connected = graph_connected(
                parts, model_success_pairs
            )
            corroborating_sources = []
            if model_success_connected:
                corroborating_sources.append(
                    "joinable_model_connected_support"
                )
            if high_rule_connected:
                corroborating_sources.append(
                    "analytic_geometry_connected_support"
                )
            spanning_tree_evidence = [
                physical_evidence_types(rule)
                for _, _, _, rule in (spanning_tree or [])
            ]
            independent_evidence_types = sorted(
                {
                    evidence_type
                    for row in spanning_tree_evidence
                    for evidence_type in row
                }
            )
            minimum_evidence_per_tree_edge = min(
                (len(row) for row in spanning_tree_evidence),
                default=0,
            )
            weak_planar_pair = (
                len(parts) == 2
                and family_counts
                and set(family_counts) <= {"planar"}
            )
            strong_cylindrical_pair = False
            if len(parts) == 2 and spanning_tree:
                evidence = (
                    spanning_tree[0][3].get("score_evidence") or {}
                )
                strong_cylindrical_pair = (
                    spanning_tree[0][3].get(
                        "joint_family_candidate"
                    )
                    == "coaxial"
                    and float(
                        evidence.get("radius_compatibility", 0.0)
                        or 0.0
                    )
                    >= 0.8
                    and float(
                        evidence.get("type_compatibility", 0.0)
                        or 0.0
                    )
                    >= 0.9
                )
            weak_single_interface = (
                bool(fallback_pairs)
                or geometry_score is None
                or not high_rule_connected
                or minimum_evidence_per_tree_edge < 2
                or (weak_planar_pair and not strong_cylindrical_pair)
            )
            item["geometry_score"] = (
                round(geometry_score, 8)
                if geometry_score is not None
                else None
            )
            item["geometry_score_available"] = geometry_score is not None
            item["geometry_score_semantics"] = (
                "Minimum analytic compatibility on the maximum spanning "
                "tree; not a calibrated assembly probability."
            )
            item["geometry_evidence"] = {
                "evidence_taxonomy_version": "2.0.0",
                "independent_evidence_types": independent_evidence_types,
                "independent_evidence_count": (
                    minimum_evidence_per_tree_edge
                ),
                "minimum_independent_evidence_per_tree_edge": (
                    minimum_evidence_per_tree_edge
                ),
                "corroborating_sources": corroborating_sources,
                "corroborating_source_count": len(corroborating_sources),
                "provider_agreement_counts_as_independent_evidence": False,
                "high_rule_graph_connected": high_rule_connected,
                "joinable_model_graph_connected": (
                    model_success_connected
                ),
                "complexity_fallback_pairs": [
                    list(pair) for pair in fallback_pairs
                ],
                "rule_failure_pairs": rule_failure_pairs,
                "interface_family_counts": dict(family_counts),
                "weak_single_interface_match": weak_single_interface,
                "strong_cylindrical_pair": strong_cylindrical_pair,
                "spanning_tree": [
                    {
                        "part_pair": [first, second],
                        "geometry_score": score,
                        "joint_family_candidate": rule.get(
                            "joint_family_candidate"
                        ),
                        "score_evidence": rule.get("score_evidence"),
                        "independent_physical_evidence": (
                            physical_evidence_types(rule)
                        ),
                    }
                    for score, first, second, rule in (
                        spanning_tree or []
                    )
                ],
            }
            enriched_pool.append(item)

        consistency_edges = consistency_edge_by_pool.get(pool_id, [])
        for item in enriched_pool:
            consistency_input = dict(item)
            consistency_input["geometry_score"] = float(
                item["geometry_score"]
                if item["geometry_score"] is not None
                else 0.0
            )
            consistency = assess_group_consistency(
                consistency_input,
                consistency_edges,
                enriched_pool,
            )
            item["consistency"] = consistency
            reasons = []
            geometry_score = item["geometry_score"]
            evidence = item["geometry_evidence"]
            if (
                geometry_score is not None
                and geometry_score
                < args.hard_reject_geometry_threshold
            ):
                tier = "rejected"
                reasons.append("geometry_score_below_hard_reject_threshold")
            else:
                if geometry_score is None:
                    reasons.append("geometry_score_unavailable")
                elif geometry_score < args.geometry_threshold:
                    reasons.append("geometry_score_below_accept_threshold")
                if evidence["independent_evidence_count"] < 2:
                    reasons.append("insufficient_independent_sources")
                if evidence["weak_single_interface_match"]:
                    reasons.append("weak_single_interface_or_fallback")
                if (
                    consistency["group_consistency_score"]
                    < args.group_consistency_threshold
                ):
                    reasons.append(
                        "group_consistency_below_accept_threshold"
                    )
                if consistency["has_global_conflict"]:
                    reasons.append("near_tied_global_conflict")
                if consistency["blocks_larger_better_group"]:
                    reasons.append("blocks_near_tied_larger_group")
                if reasons:
                    tier = "review"
                else:
                    tier = "accepted_for_pose_validation"
                    reasons.append(
                        "passed_conservative_pre_pose_geometry_gate"
                    )
            item["pre_pose_tier"] = tier
            item["decision_reasons"] = sorted(set(reasons))
            item["review_required"] = tier != "accepted_for_pose_validation"
            all_tier_counts[tier] += 1
            if (pool_id, frozenset(item["parts"])) in truth_keys:
                truth_tier_counts[tier] += 1
            {
                "accepted_for_pose_validation": accepted_geometry,
                "review": review_geometry,
                "rejected": rejected_geometry,
            }[tier].append(item)

    def sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            float(item.get("geometry_score") or 0.0),
            float(
                (item.get("consistency") or {}).get(
                    "group_consistency_score", 0.0
                )
            ),
            float(item.get("candidate_priority_score") or 0.0),
            -int(item.get("group_size") or 0),
            str(item["group_id"]),
        )

    for collection in (
        accepted_geometry,
        review_geometry,
        rejected_geometry,
    ):
        collection.sort(key=sort_key, reverse=True)

    manifest = read_json(root / "mixed_pool_manifest.json")
    unresolved = []
    accepted_parts_by_pool = defaultdict(set)
    for item in accepted_geometry:
        accepted_parts_by_pool[str(item["pool_id"])].update(
            item["parts"]
        )
    for pool in manifest.get("pools", []):
        pool_id = str(pool["pool_id"])
        pool_input = read_json(root / pool_id / "pool_input.json")
        review_by_part = defaultdict(list)
        for item in review_geometry:
            if item["pool_id"] != pool_id:
                continue
            for part in item["parts"]:
                review_by_part[part].append(item["group_id"])
        for part in pool_input.get("parts", []):
            part_id = str(part["part_id"])
            if part_id in accepted_parts_by_pool[pool_id]:
                continue
            unresolved.append(
                {
                    "pool_id": pool_id,
                    "part_id": part_id,
                    "reason": (
                        "only_review_candidates_available"
                        if review_by_part[part_id]
                        else "no_pre_pose_candidate_available"
                    ),
                    "review_candidate_ids": sorted(
                        review_by_part[part_id]
                    )[:20],
                }
            )
    metrics = {
        "schema_version": "1.0.0",
        "thresholds": {
            "geometry_accept": args.geometry_threshold,
            "geometry_hard_reject": (
                args.hard_reject_geometry_threshold
            ),
            "group_consistency_accept": (
                args.group_consistency_threshold
            ),
            "minimum_independent_physical_evidence_per_tree_edge": 2,
            "provider_agreement_counts_as_independent_evidence": False,
        },
        "proposal_count": len(search.get("proposals", [])),
        "tier_counts": dict(all_tier_counts),
        "true_group_tier_counts": dict(truth_tier_counts),
        "pre_pose_accepted_count": len(accepted_geometry),
        "review_count": len(review_geometry),
        "rejected_count": len(rejected_geometry),
        "preliminary_unresolved_part_count": len(unresolved),
        "final_auto_accept_count": 0,
        "final_auto_accept_disabled_until_pose": True,
        "rule_geometry_pair_success_count": rule_report.get(
            "pair_success_count"
        ),
        "rule_geometry_pair_failure_count": rule_report.get(
            "pair_failure_count"
        ),
        "failure_reasons": [],
        "unavailable_fields": [
            "collision_result",
            "pose_status",
            "functional_semantic_validity",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "accepted_geometry_candidates.json",
        accepted_geometry,
    )
    write_json(
        output_dir / "review_geometry_candidates.json",
        review_geometry,
    )
    write_json(
        output_dir / "rejected_geometry_candidates.json",
        rejected_geometry,
    )
    write_json(output_dir / "unresolved_parts_pre_pose.json", unresolved)
    write_json(output_dir / "geometry_tiering_metrics.json", metrics)
    write_json(
        output_dir / "geometry_tiering_truth_audit.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "evaluation_only",
            "true_group_tier_counts": dict(truth_tier_counts),
            "truth_group_count": len(truth_keys),
        },
    )
    report_lines = [
        "# Step 6 System Review",
        "",
        "## Independent reviewer conclusion",
        "",
        f"- Pre-pose eligible: {len(accepted_geometry)}",
        f"- Review: {len(review_geometry)}",
        f"- Rejected: {len(rejected_geometry)}",
        f"- Preliminary unresolved part instances: {len(unresolved)}",
        f"- True groups by tier: {dict(truth_tier_counts)}",
        "- Analytic geometry scores many cross-assembly pairs highly; it is "
        "not a grouping classifier.",
        "- JoinABLe and analytic-rule agreement is corroboration only; it is "
        "not counted as independent physical evidence.",
        "- No proposal is finally auto-accepted before pose/collision "
        "validation.",
        "",
        "## System decision",
        "",
        "Proceed with a bounded, score- and size-diverse pose queue drawn "
        "from pre-pose eligible and review candidates. Do not validate all "
        "1191 proposals and do not reinterpret missing geometry as failure.",
    ]
    (output_dir / "geometry_tiering_system_review.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(
        f"Pre-pose tiers: eligible={len(accepted_geometry)}, "
        f"review={len(review_geometry)}, rejected={len(rejected_geometry)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
