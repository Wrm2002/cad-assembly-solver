"""Compute group-level and review-frontier recall for the functional benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(parts: list[str]) -> tuple[str, ...]:
    return tuple(sorted(parts))


def run(
    pools_root: str | Path,
    results_root: str | Path,
    *,
    cutoffs: tuple[int, ...] = (24, 50, 100, 500, 2000),
) -> dict[str, Any]:
    pools_root = Path(pools_root).resolve()
    results_root = Path(results_root).resolve()
    review = _load(results_root / "final_review_groups.json")
    accepted = _load(results_root / "final_accepted_groups.json")
    rejected = _load(results_root / "final_rejected_groups.json")
    pose = _load(results_root / "true_group_pose_audit.json")

    decisions: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for decision, rows in (
        ("accepted", accepted),
        ("review", review),
        ("rejected", rejected),
    ):
        for row in rows:
            decisions[(row["pool_id"], _key(row["parts"]))] = {
                "decision": decision,
                "review_queue_state": row.get("review_queue_state"),
                "review_rank": (
                    row.get("review_ranking") or {}
                ).get("review_rank"),
                "candidate_id": row["group_id"],
            }

    records: list[dict[str, Any]] = []
    pool_summaries = []
    for pool in sorted(
        path
        for path in pools_root.iterdir()
        if path.is_dir() and (path / "pool_gt.json").is_file()
    ):
        gt = _load(pool / "pool_gt.json")
        proposals = _load(pool / "grouping" / "group_proposals.json")
        proposal_keys = {_key(item["parts"]) for item in proposals}
        pool_truth_rows = []
        for group in gt["true_groups"]:
            key = _key(group["parts"])
            decision = decisions.get((pool.name, key), {})
            row = {
                "pool_id": pool.name,
                "true_group_id": group["group_id"],
                "assembly_family": group["assembly_family"],
                "parts": group["parts"],
                "proposal_generated": key in proposal_keys,
                "final_decision": decision.get(
                    "decision", "missing_proposal"
                ),
                "review_queue_state": decision.get("review_queue_state"),
                "review_rank": decision.get("review_rank"),
                "candidate_id": decision.get("candidate_id"),
            }
            records.append(row)
            pool_truth_rows.append(row)
        pool_summaries.append(
            {
                "pool_id": pool.name,
                "true_group_count": len(pool_truth_rows),
                "proposal_recalled_count": sum(
                    row["proposal_generated"] for row in pool_truth_rows
                ),
                "frontier_recalled_count": sum(
                    row["review_queue_state"] == "selected"
                    for row in pool_truth_rows
                ),
            }
        )

    truth_count = len(records)
    selected_review_count = sum(
        row.get("review_queue_state") == "selected" for row in review
    )
    group_recall_at_k = {}
    for cutoff in cutoffs:
        recalled = sum(
            row["final_decision"] == "accepted"
            or (
                row["final_decision"] == "review"
                and row["review_rank"] is not None
                and int(row["review_rank"]) <= cutoff
            )
            for row in records
        )
        group_recall_at_k[str(cutoff)] = {
            "recalled_true_groups": recalled,
            "true_group_count": truth_count,
            "group_recall": recalled / truth_count if truth_count else None,
            "cutoff_is_per_pool": True,
        }

    frontier_true = sum(
        row["review_queue_state"] == "selected" for row in records
    )
    summary = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only",
        "truth_basis": "functional_validity",
        "true_group_count": truth_count,
        "group_proposal_recalled_count": sum(
            row["proposal_generated"] for row in records
        ),
        "group_proposal_recall": (
            sum(row["proposal_generated"] for row in records) / truth_count
            if truth_count
            else None
        ),
        "group_recall_at_k": group_recall_at_k,
        "review_frontier_true_group_count": frontier_true,
        "review_frontier_group_count": selected_review_count,
        "review_frontier_recall": (
            frontier_true / truth_count if truth_count else None
        ),
        "review_frontier_precision": (
            frontier_true / selected_review_count
            if selected_review_count
            else None
        ),
        "true_group_pose_recall": pose["true_group_pose_recall"],
        "true_group_pose_valid_count": pose["pose_valid_count"],
        "pool_summaries": pool_summaries,
        "records": records,
    }
    json_path = results_root / "functional_group_metrics.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (results_root / "functional_group_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "pool_id",
            "true_group_id",
            "assembly_family",
            "parts",
            "proposal_generated",
            "candidate_id",
            "final_decision",
            "review_queue_state",
            "review_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            output = {field: row.get(field) for field in fields}
            output["parts"] = "|".join(row["parts"])
            writer.writerow(output)

    lines = [
        "# Functional Group-Level Metrics",
        "",
        f"- Group proposal recall: {summary['group_proposal_recall']:.2%}",
        (
            "- Review frontier recall: "
            f"{summary['review_frontier_recall']:.2%} "
            f"({frontier_true}/{truth_count})"
        ),
        (
            "- Review frontier precision: "
            f"{summary['review_frontier_precision']:.2%}"
        ),
        (
            "- True-group pose recall: "
            f"{summary['true_group_pose_recall']:.2%}"
        ),
        "",
        "| K per pool | Recalled | group_recall@K |",
        "|---:|---:|---:|",
    ]
    for cutoff in cutoffs:
        metric = group_recall_at_k[str(cutoff)]
        lines.append(
            f"| {cutoff} | {metric['recalled_true_groups']}/"
            f"{truth_count} | {metric['group_recall']:.2%} |"
        )
    (results_root / "functional_group_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pools-root",
        default=str(here / "data" / "functional_mixed_pools_v1"),
    )
    parser.add_argument(
        "--results-root",
        default=str(here / "data" / "functional_results"),
    )
    args = parser.parse_args()
    summary = run(args.pools_root, args.results_root)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "group_proposal_recall",
                    "group_recall_at_k",
                    "review_frontier_recall",
                    "review_frontier_precision",
                    "true_group_pose_recall",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
