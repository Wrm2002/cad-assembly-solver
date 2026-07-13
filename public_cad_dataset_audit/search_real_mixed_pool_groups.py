"""Generate bounded 2-5 part proposals from the real JoinABLe candidate graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict, deque
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_pair(values: Iterable[str]) -> tuple[str, str]:
    first, second = sorted(str(value) for value in values)
    return first, second


def edge_priority(edge: dict[str, Any]) -> float:
    lift = edge.get("joinable_uniform_lift")
    threshold = float(edge.get("joinable_threshold") or 1.0)
    if lift is None:
        return 0.25
    ratio = max(float(lift), 1e-12) / max(threshold, 1e-12)
    return ratio / (1.0 + ratio)


def connected(
    parts: tuple[str, ...],
    edge_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    allowed = set(parts)
    adjacency = {part: set() for part in parts}
    for pair in combinations(parts, 2):
        canonical = canonical_pair(pair)
        edge = edge_by_pair.get(canonical)
        if not edge or not edge.get("active_for_group_search"):
            continue
        adjacency[canonical[0]].add(canonical[1])
        adjacency[canonical[1]].add(canonical[0])
    seen = {parts[0]}
    queue = deque([parts[0]])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == allowed


def proposal_from_parts(
    pool_id: str,
    parts: tuple[str, ...],
    edge_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    internal_source_edges = []
    translated_edges = []
    degree = defaultdict(set)
    for pair in combinations(parts, 2):
        canonical = canonical_pair(pair)
        edge = edge_by_pair.get(canonical)
        if not edge or not edge.get("active_for_group_search"):
            continue
        candidate_id = f"{pool_id}:{canonical[0]}:{canonical[1]}"
        priority = edge_priority(edge)
        families = sorted(
            {
                candidate.get("joint_family_candidate", "unknown")
                for candidate in edge.get(
                    "top_interface_candidates", []
                )
            }
        )
        translated = {
            "candidate_id": candidate_id,
            "parts": list(canonical),
            "candidate_type": "joinable_interface_rank",
            "score": priority,
            "audit_reason": {
                "joinable_uniform_lift": edge.get(
                    "joinable_uniform_lift"
                ),
                "joinable_threshold": edge.get("joinable_threshold"),
                "model_status": edge.get("model_status"),
                "interface_families_in_top_5": families,
                "complexity_fallback": (
                    edge.get("candidate_state")
                    == "review_complexity_fallback"
                ),
            },
        }
        internal_source_edges.append(edge)
        translated_edges.append(translated)
        degree[canonical[0]].add(canonical[1])
        degree[canonical[1]].add(canonical[0])
    possible_edges = len(parts) * (len(parts) - 1) // 2
    density = (
        len(translated_edges) / possible_edges if possible_edges else 0.0
    )
    mean_priority = (
        sum(float(edge["score"]) for edge in translated_edges)
        / len(translated_edges)
        if translated_edges
        else 0.0
    )
    fallback_fraction = (
        sum(
            bool(edge["audit_reason"]["complexity_fallback"])
            for edge in translated_edges
        )
        / len(translated_edges)
        if translated_edges
        else 1.0
    )
    maximum_degree = max(
        (len(neighbors) for neighbors in degree.values()), default=0
    )
    central_score = (
        1.0
        if len(parts) <= 2 or maximum_degree >= 2
        else 0.0
    )
    priority_score = (
        0.45 * mean_priority
        + 0.25 * density
        + 0.20
        + 0.10 * central_score
        - 0.20 * fallback_fraction
    )
    priority_score = max(0.0, min(1.0, priority_score))
    stable_id = hashlib.sha1(
        f"{pool_id}:{'|'.join(parts)}".encode("utf-8")
    ).hexdigest()[:12]
    proposal = {
        "group_id": f"{pool_id}:G_{stable_id}",
        "pool_id": pool_id,
        "parts": list(parts),
        "group_size": len(parts),
        "candidate_edges": [
            edge["candidate_id"] for edge in translated_edges
        ],
        "candidate_priority_score": round(priority_score, 8),
        "score_semantics": (
            "Bounded JoinABLe candidate priority; not geometry probability "
            "and not acceptance evidence."
        ),
        "active_internal_edge_count": len(translated_edges),
        "possible_internal_edge_count": possible_edges,
        "active_edge_density": round(density, 8),
        "mean_edge_priority": round(mean_priority, 8),
        "complexity_fallback_fraction": round(
            fallback_fraction, 8
        ),
        "has_central_part_structure": bool(central_score),
        "geometry_score_available": False,
        "review_required": True,
        "failure_reasons": [],
        "unavailable_fields": [
            "calibrated_geometry_score",
            "collision_result",
            "pose_status",
            "functional_semantic_validity",
        ],
    }
    return proposal, translated_edges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument("candidate_graph_input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-group-size", type=int, default=5)
    parser.add_argument("--maximum-per-size", type=int, default=50)
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    output_dir = args.output_dir.resolve()
    candidate_graph = read_json(args.candidate_graph_input.resolve())
    pool_edges = {
        str(pool["pool_id"]): pool.get("edges", [])
        for pool in candidate_graph.get("pools", [])
    }
    manifest = read_json(root / "mixed_pool_manifest.json")
    all_proposals = []
    pool_reports = []
    exact_truth_total = 0
    exact_truth_generated = 0
    exact_truth_kept = 0
    for pool_record in manifest.get("pools", []):
        pool_id = str(pool_record["pool_id"])
        pool_input = read_json(root / pool_id / "pool_input.json")
        pool_gt = read_json(root / pool_id / "pool_gt.json")
        part_ids = sorted(
            str(part["part_id"])
            for part in pool_input.get("parts", [])
        )
        edge_by_pair = {
            canonical_pair(edge["part_pair"]): edge
            for edge in pool_edges.get(pool_id, [])
        }
        translated_by_id = {}
        raw_by_size = defaultdict(list)
        for size in range(2, min(args.maximum_group_size, len(part_ids)) + 1):
            for parts in combinations(part_ids, size):
                if not connected(parts, edge_by_pair):
                    continue
                proposal, translated = proposal_from_parts(
                    pool_id, parts, edge_by_pair
                )
                raw_by_size[size].append(proposal)
                for edge in translated:
                    translated_by_id[edge["candidate_id"]] = edge
        for size in raw_by_size:
            raw_by_size[size].sort(
                key=lambda row: (
                    -float(row["candidate_priority_score"]),
                    tuple(row["parts"]),
                )
            )
            for rank, proposal in enumerate(
                raw_by_size[size], start=1
            ):
                proposal["rank_within_group_size_before_bound"] = rank

        kept = []
        for size in sorted(raw_by_size):
            kept.extend(raw_by_size[size][: args.maximum_per_size])
        translated_edges = sorted(
            translated_by_id.values(),
            key=lambda row: row["candidate_id"],
        )
        for proposal in kept:
            consistency = assess_group_consistency(
                proposal,
                translated_edges,
                kept,
            )
            proposal["consistency"] = consistency
            proposal["review_required"] = True
            if proposal["complexity_fallback_fraction"] > 0:
                proposal.setdefault("decision_reasons", []).append(
                    "contains_complexity_review_fallback_edge"
                )
            if consistency["review_required"]:
                proposal.setdefault("decision_reasons", []).append(
                    "group_consistency_requires_review"
                )
            proposal.setdefault("decision_reasons", []).append(
                "geometry_and_pose_not_yet_validated"
            )

        truth_sets = {
            frozenset(str(value) for value in group["part_ids"]): str(
                group["group_id"]
            )
            for group in pool_gt.get("true_groups", [])
        }
        exact_truth_total += len(truth_sets)
        raw_lookup = {
            frozenset(proposal["parts"]): proposal
            for proposals in raw_by_size.values()
            for proposal in proposals
        }
        kept_lookup = {
            frozenset(proposal["parts"]): proposal for proposal in kept
        }
        truth_audit = []
        for truth_parts, truth_group_id in sorted(
            truth_sets.items(), key=lambda item: sorted(item[0])
        ):
            raw = raw_lookup.get(truth_parts)
            kept_row = kept_lookup.get(truth_parts)
            exact_truth_generated += int(raw is not None)
            exact_truth_kept += int(kept_row is not None)
            truth_audit.append(
                {
                    "true_group_id": truth_group_id,
                    "part_ids": sorted(truth_parts),
                    "group_size": len(truth_parts),
                    "generated_before_bound": raw is not None,
                    "kept_after_bound": kept_row is not None,
                    "rank_within_group_size": (
                        raw.get("rank_within_group_size_before_bound")
                        if raw
                        else None
                    ),
                    "candidate_priority_score": (
                        raw.get("candidate_priority_score") if raw else None
                    ),
                    "missing_reason": (
                        None
                        if raw
                        else "true_group_not_connected_in_active_candidate_graph"
                    ),
                }
            )
        all_proposals.extend(kept)
        pool_reports.append(
            {
                "pool_id": pool_id,
                "part_count": len(part_ids),
                "active_edge_count": sum(
                    edge.get("active_for_group_search", False)
                    for edge in edge_by_pair.values()
                ),
                "raw_proposal_count_by_size": {
                    str(size): len(rows)
                    for size, rows in sorted(raw_by_size.items())
                },
                "kept_proposal_count_by_size": dict(
                    sorted(
                        Counter(
                            str(proposal["group_size"])
                            for proposal in kept
                        ).items()
                    )
                ),
                "true_group_count": len(truth_sets),
                "truth_audit": truth_audit,
                "candidate_edge_records": translated_edges,
                "failure_reasons": [],
                "unavailable_fields": [
                    "calibrated_geometry_score",
                    "pose_status",
                    "functional_semantic_validity",
                ],
            }
        )
    truth_audit_output = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only",
        "dataset_id": candidate_graph.get("dataset_id"),
        "exact_truth_group_count": exact_truth_total,
        "exact_truth_generated_before_bound": exact_truth_generated,
        "exact_truth_kept_after_bound": exact_truth_kept,
        "exact_truth_recall_before_bound": (
            exact_truth_generated / exact_truth_total
            if exact_truth_total
            else None
        ),
        "exact_truth_recall_after_bound": (
            exact_truth_kept / exact_truth_total
            if exact_truth_total
            else None
        ),
        "pools": [
            {
                "pool_id": pool["pool_id"],
                "true_group_count": pool["true_group_count"],
                "truth_audit": pool["truth_audit"],
            }
            for pool in pool_reports
        ],
    }
    output = {
        "schema_version": "1.0.0",
        "artifact_role": "production_candidates",
        "dataset_id": candidate_graph.get("dataset_id"),
        "search_policy": {
            "group_size_range": [
                2,
                args.maximum_group_size,
            ],
            "connected_active_subgraphs_only": True,
            "maximum_per_group_size_per_pool": args.maximum_per_size,
            "force_complete_partition": False,
            "minimum_vertex_cover_used": False,
            "reinforcement_learning_used": False,
        },
        "proposal_count": len(all_proposals),
        "proposals": all_proposals,
        "pools": [
            {
                key: value
                for key, value in pool.items()
                if key not in {"true_group_count", "truth_audit"}
            }
            for pool in pool_reports
        ],
        "failure_reasons": [],
        "unavailable_fields": [
            "calibrated_geometry_score",
            "collision_result",
            "pose_status",
            "functional_semantic_validity",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "group_proposals.json", output)
    write_json(
        output_dir / "group_proposal_truth_audit.json",
        truth_audit_output,
    )
    write_json(
        output_dir / "group_search_metrics.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "evaluation_only",
            "search_policy": output["search_policy"],
            "proposal_count": output["proposal_count"],
            "exact_truth_group_count": truth_audit_output[
                "exact_truth_group_count"
            ],
            "exact_truth_generated_before_bound": truth_audit_output[
                "exact_truth_generated_before_bound"
            ],
            "exact_truth_kept_after_bound": truth_audit_output[
                "exact_truth_kept_after_bound"
            ],
            "exact_truth_recall_before_bound": truth_audit_output[
                "exact_truth_recall_before_bound"
            ],
            "exact_truth_recall_after_bound": truth_audit_output[
                "exact_truth_recall_after_bound"
            ],
            "failure_reasons": output["failure_reasons"],
            "unavailable_fields": output["unavailable_fields"],
        }
        | {
            "pools": [
                {
                    key: pool[key]
                    for key in (
                        "pool_id",
                        "part_count",
                        "active_edge_count",
                        "raw_proposal_count_by_size",
                        "kept_proposal_count_by_size",
                        "true_group_count",
                        "truth_audit",
                    )
                }
                for pool in pool_reports
            ]
        },
    )
    review_lines = [
        "# Step 5 System Review",
        "",
        "## Independent reviewer conclusion",
        "",
        f"- Generated {len(all_proposals)} bounded connected proposals.",
        (
            f"- Exact true-group recall before bounding: "
            f"{truth_audit_output['exact_truth_recall_before_bound']:.2%}."
        ),
        (
            f"- Exact true-group recall after bounding: "
            f"{truth_audit_output['exact_truth_recall_after_bound']:.2%}."
        ),
        "- No complete partition is forced; parts may remain unresolved.",
        "- Candidate priority is derived from JoinABLe concentration and graph "
        "connectivity. It is not renamed as a calibrated geometry score.",
        "- Every proposal remains review-required until geometry, pose, "
        "collision, and conflict gates are available.",
        "",
        "## System decision",
        "",
        (
            "Proceed to three-tier gating only if exact truth recall after "
            "bounding remains 100%. Otherwise increase the bounded review "
            "frontier, not the neural model or global architecture."
        ),
    ]
    (output_dir / "group_search_system_review.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8"
    )
    print(
        f"Group proposals={len(all_proposals)}, "
        f"truth recall={truth_audit_output['exact_truth_recall_after_bound']:.3f}"
    )
    return (
        0
        if truth_audit_output["exact_truth_recall_after_bound"] == 1.0
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
