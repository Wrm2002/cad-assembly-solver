"""Run the conservative functional proposal pipeline and its failure audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bounded_expansion import generate_bounded_proposals
from pair_edge import build_pair_edges, canonical_pair
from proposal_postprocess import (
    attach_subset_superset_links,
    build_clustered_review_queue,
    cluster_proposals,
)
from role_estimator import estimate_roles


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _key(parts: list[str]) -> tuple[str, ...]:
    return tuple(sorted(parts))


def _truth_rows(pool: Path) -> list[dict[str, Any]]:
    return _load(pool / "pool_gt.json").get("true_groups", [])


def _load_candidates_with_joinable(
    pool: Path, index_name: str, *, include_joinable: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if index_name == "index_union_ablation":
        rows = []
        for name in ("index_analytic_ablation", "index_joinable_ablation"):
            rows.extend(_load(pool / name / "pruned_candidates.json"))
        candidates = list(
            {
                str(row["candidate_id"]): row for row in rows
            }.values()
        )
    else:
        candidates = _load(pool / index_name / "pruned_candidates.json")
    if not include_joinable:
        return candidates, {
            "available": False,
            "disabled_for_ablation": True,
            "synthetic_edge_count": 0,
        }
    audit_paths = (
        pool / index_name / "joinable_candidate_provider_audit.json",
        pool / "index_joinable_ablation" / "joinable_candidate_provider_audit.json",
    )
    audit_path = next((path for path in audit_paths if path.is_file()), None)
    if audit_path is None:
        return candidates, {"available": False, "synthetic_edge_count": 0}
    audit = _load(audit_path)
    learned_rows = []
    for position, selected in enumerate(audit.get("selected_pairs", []), 1):
        top = (selected.get("top_interface_candidates") or [{}])[0]
        pair_features = selected.get("pair_features", {})
        probability = top.get(
            "softmax_probability", pair_features.get("top_1_probability", 0.0)
        )
        learned_rows.append(
            {
                "candidate_id": f"J_{pool.name}_{position:04d}",
                "parts": selected["parts"],
                "candidate_type": "joinable_interface_rank",
                "geometry_score": 0.0,
                "provider_evidence": {
                    "pretrained_joinable": {
                        "rank": 1,
                        "softmax_probability": float(probability or 0.0),
                        "uniform_lift": pair_features.get("top_1_uniform_lift"),
                        "source_audit": str(audit_path),
                    }
                },
                "audit_reason": {
                    "creates_physical_evidence": False,
                    "can_auto_accept": False,
                },
            }
        )
    return candidates + learned_rows, {
        "available": True,
        "source": str(audit_path),
        "synthetic_edge_count": len(learned_rows),
        "note": "Provider rows preserve source/rank only and create no physical evidence.",
    }


def _existing_pose_status(pool: Path, group_id: str) -> str:
    path = pool / "validation" / group_id / "validation_result.json"
    if not path.is_file():
        return "not_run"
    row = _load(path)
    status = str(row.get("status", "unknown"))
    if status == "success" and int(row.get("severe_penetration_count", 0)) == 0:
        return "valid"
    if status == "failed":
        return "failed"
    return "uncertain"


def _required_pair_status(
    truth: dict[str, Any], pair_edges: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    available = {canonical_pair(row["parts"]) for row in pair_edges}
    required = {
        canonical_pair((mate["part_a"], mate["part_b"]))
        for mate in truth.get("true_mates", [])
    }
    present = ["|".join(pair) for pair in sorted(required & available)]
    missing = ["|".join(pair) for pair in sorted(required - available)]
    return present, missing


def run_pool(
    pool: Path,
    output: Path,
    *,
    queue_size: int,
    index_name: str,
    include_joinable: bool,
) -> dict[str, Any]:
    features = _load(pool / "index" / "part_features.json")
    candidates, joinable_audit = _load_candidates_with_joinable(
        pool, index_name, include_joinable=include_joinable
    )
    pair_edges = build_pair_edges(candidates)
    roles = estimate_roles(features, pair_edges)
    proposals, expansion_audit = generate_bounded_proposals(
        features, pair_edges, roles
    )
    proposals = attach_subset_superset_links(proposals)
    proposals, clusters = cluster_proposals(proposals)
    queue, deferred = build_clustered_review_queue(
        proposals, maximum=queue_size
    )

    _write(output / "pair_edges.json", pair_edges)
    _write(output / "joinable_source_audit.json", joinable_audit)
    _write(output / "role_hypotheses.json", roles)
    _write(output / "bounded_expansion_audit.json", expansion_audit)
    _write(
        output / "group_proposals.json",
        [row.model_dump(mode="json") for row in proposals],
    )
    _write(output / "proposal_clusters.json", clusters)
    _write(output / "review_frontier.json", queue)
    _write(output / "deferred_proposals.json", deferred)

    proposal_by_key = {_key(row.parts): row for row in proposals}
    queue_by_key = {_key(row["parts"]): row for row in queue}
    truth_keys = {_key(row["parts"]) for row in _truth_rows(pool)}
    diagnostics = []
    for truth in _truth_rows(pool):
        key = _key(truth["parts"])
        proposal = proposal_by_key.get(key)
        frontier = queue_by_key.get(key)
        present_pairs, missing_pairs = _required_pair_status(truth, pair_edges)
        if missing_pairs:
            failure = "candidate_recall_failure"
        elif proposal is None:
            failure = "proposal_generation_failure"
        elif frontier is None:
            failure = "proposal_ranking_failure"
        else:
            pose = _existing_pose_status(pool, proposal.group_id)
            failure = "none" if pose == "valid" else f"pose_{pose}"
        diagnostics.append(
            {
                "pool_id": pool.name,
                "split": _load(pool / "pool_gt.json").get("split", "unknown"),
                "true_group_id": truth["group_id"],
                "assembly_family": truth["assembly_family"],
                "parts": list(key),
                "required_pair_present": present_pairs,
                "required_pair_missing": missing_pairs,
                "proposal_generated": proposal is not None,
                "proposal_id": proposal.group_id if proposal else None,
                "proposal_cluster_id": (
                    proposal.proposal_cluster_id if proposal else None
                ),
                "review_frontier_recalled": frontier is not None,
                "review_rank": frontier.get("review_rank") if frontier else None,
                "pose_status": (
                    _existing_pose_status(pool, proposal.group_id)
                    if proposal else "not_run"
                ),
                "failure_stage": failure,
            }
        )

    hard_negatives = {
        _key(row["parts"]): row
        for row in _load(pool / "pool_gt.json").get(
            "functional_negative_groups", []
        )
    }
    hard_negative_frontier = [
        {
            "pool_id": pool.name,
            "group_id": row["group_id"],
            "parts": row["parts"],
            "negative_type": hard_negatives[_key(row["parts"])]["negative_type"],
            "review_rank": row["review_rank"],
            "decision": "review",
        }
        for row in queue
        if _key(row["parts"]) in hard_negatives
    ]
    return {
        "pool_id": pool.name,
        "split": _load(pool / "pool_gt.json").get("split", "unknown"),
        "part_count": len(features),
        "pair_edge_count": len(pair_edges),
        "joinable_provider_available": joinable_audit["available"],
        "proposal_count": len(proposals),
        "cluster_count": len(clusters),
        "review_frontier_count": len(queue),
        "true_group_count": len(truth_keys),
        "proposal_true_group_count": len(truth_keys & set(proposal_by_key)),
        "frontier_true_group_count": len(truth_keys & set(queue_by_key)),
        "frontier_false_group_count": len(queue) - len(truth_keys & set(queue_by_key)),
        "hard_negative_frontier": hard_negative_frontier,
        "diagnostics": diagnostics,
    }


def _aggregate(
    pools_root: Path,
    output: Path,
    pool_rows: list[dict[str, Any]],
    queue_size: int,
) -> dict[str, Any]:
    diagnostics = [row for pool in pool_rows for row in pool["diagnostics"]]
    total_truth = sum(row["true_group_count"] for row in pool_rows)
    total_proposals = sum(row["proposal_count"] for row in pool_rows)
    total_frontier = sum(row["review_frontier_count"] for row in pool_rows)
    proposal_true = sum(row["proposal_true_group_count"] for row in pool_rows)
    frontier_true = sum(row["frontier_true_group_count"] for row in pool_rows)
    metrics = {
        "schema_version": "1.0.0",
        "benchmark": str(pools_root),
        "truth_basis": "functional_validity",
        "evaluation_semantics_used_for_inference": False,
        "semantic_reranking_enabled": False,
        "pool_count": len(pool_rows),
        "true_group_count": total_truth,
        "proposal_count": total_proposals,
        "proposal_true_group_count": proposal_true,
        "proposal_true_group_recall": (
            proposal_true / total_truth if total_truth else None
        ),
        "review_frontier_limit_per_pool": queue_size,
        "review_frontier_count": total_frontier,
        "review_frontier_true_group_count": frontier_true,
        "review_frontier_recall": (
            frontier_true / total_truth if total_truth else None
        ),
        "review_frontier_precision": (
            frontier_true / total_frontier if total_frontier else None
        ),
        "accepted_group_count": 0,
        "false_positive_count": 0,
        "auto_accept_precision": None,
        "review_group_count": total_frontier,
        "unresolved_parts_count": sum(row["part_count"] for row in pool_rows),
        "failure_stage_counts": dict(
            sorted(Counter(row["failure_stage"] for row in diagnostics).items())
        ),
        "hard_negative_review_count": sum(
            len(row["hard_negative_frontier"]) for row in pool_rows
        ),
        "baseline_comparison": {
            "before": {
                "proposal_count": 9668,
                "review_frontier_count": 72,
                "review_frontier_true_group_count": 2,
                "review_frontier_recall": 2 / 9,
                "review_frontier_precision": 2 / 72,
                "false_positive_count": 0,
            }
        },
        "pools": [
            {key: value for key, value in row.items() if key != "diagnostics"}
            for row in pool_rows
        ],
    }
    metrics["baseline_comparison"]["after"] = {
        key: metrics[key]
        for key in (
            "proposal_count",
            "review_frontier_count",
            "review_frontier_true_group_count",
            "review_frontier_recall",
            "review_frontier_precision",
            "false_positive_count",
        )
    }
    _write(output / "conservative_metrics.json", metrics)
    _write(output / "failure_diagnosis.json", diagnostics)
    _write(
        output / "hard_negative_review_audit.json",
        [item for row in pool_rows for item in row["hard_negative_frontier"]],
    )
    with (output / "failure_diagnosis.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "pool_id",
            "split",
            "true_group_id",
            "assembly_family",
            "parts",
            "required_pair_present",
            "required_pair_missing",
            "proposal_generated",
            "proposal_id",
            "proposal_cluster_id",
            "review_frontier_recalled",
            "review_rank",
            "pose_status",
            "failure_stage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics:
            item = dict(row)
            for key in ("parts", "required_pair_present", "required_pair_missing"):
                item[key] = "|".join(item[key])
            writer.writerow({key: item.get(key) for key in fields})
    lines = [
        "# Mixed-pool Functional Grouping Experiment",
        "",
        "Inference used geometry and pair-provider tags only; synthetic role/family truth was evaluation-only.",
        "",
        f"- Pools: {len(pool_rows)}",
        f"- Proposals: {total_proposals} (baseline 9668)",
        f"- Proposal true-group recall: {proposal_true}/{total_truth} ({proposal_true / total_truth:.2%})",
        f"- Review frontier: {frontier_true}/{total_frontier} true ({frontier_true / total_frontier:.2%} precision)",
        f"- Review frontier recall: {frontier_true}/{total_truth} ({frontier_true / total_truth:.2%})",
        "- Auto accepted: 0",
        "- False auto accepted: 0",
        "- Semantic reranking: disabled",
        "",
        "## Failure diagnosis",
        "",
    ]
    for stage, count in metrics["failure_stage_counts"].items():
        lines.append(f"- {stage}: {count}")
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            "This experiment evaluates proposal organization and review ordering. Pose success remains necessary but not sufficient for later automatic acceptance.",
        ]
    )
    (output / "failure_diagnosis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return metrics


def run(
    pools_root: Path,
    output: Path,
    *,
    queue_size: int = 20,
    index_name: str = "index",
    include_joinable: bool = True,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for pool in sorted(
        row for row in pools_root.iterdir()
        if row.is_dir() and (row / "pool_gt.json").is_file()
    ):
        rows.append(
            run_pool(
                pool,
                output / "pools" / pool.name,
                queue_size=queue_size,
                index_name=index_name,
                include_joinable=include_joinable,
            )
        )
    return _aggregate(pools_root, output, rows, queue_size)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pools-root",
        type=Path,
        default=here / "data" / "functional_mixed_pools_topology_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "data" / "functional_grouping_experiment_v2",
    )
    parser.add_argument("--queue-size", type=int, default=20)
    parser.add_argument("--index-name", default="index")
    parser.add_argument("--disable-joinable", action="store_true")
    args = parser.parse_args()
    metrics = run(
        args.pools_root.resolve(),
        args.output.resolve(),
        queue_size=args.queue_size,
        index_name=args.index_name,
        include_joinable=not args.disable_joinable,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
