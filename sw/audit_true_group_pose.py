"""Run an evaluation-only pose audit for every functional ground-truth group.

Ground truth is used only to select audit subjects and compute metrics.  The
solver receives the production proposal and its production candidate edges;
truth placements and semantic fields are never passed to pose reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from conservative_pipeline import _pose_record
from contracts import GroupProposal
from geometry_pipeline import isolated_solve_and_validate_group


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parts_key(parts: list[str]) -> tuple[str, ...]:
    return tuple(sorted(parts))


def run(
    pools_root: str | Path,
    results_root: str | Path,
    pipeline_config_path: str | Path,
) -> dict[str, Any]:
    pools_root = Path(pools_root).resolve()
    results_root = Path(results_root).resolve()
    config_path = Path(pipeline_config_path).resolve()
    config = _load(config_path)
    records: list[dict[str, Any]] = []

    pools = sorted(
        path
        for path in pools_root.iterdir()
        if path.is_dir() and (path / "pool_gt.json").is_file()
    )
    for pool in pools:
        ground_truth = _load(pool / "pool_gt.json")
        proposals = _load(pool / "grouping" / "group_proposals.json")
        proposal_by_parts = {
            _parts_key(item["parts"]): item for item in proposals
        }
        for group in ground_truth["true_groups"]:
            item = proposal_by_parts.get(_parts_key(group["parts"]))
            if item is None:
                records.append(
                    {
                        "pool_id": pool.name,
                        "true_group_id": group["group_id"],
                        "assembly_family": group["assembly_family"],
                        "parts": group["parts"],
                        "proposal_generated": False,
                        "candidate_id": None,
                        "final_pose_status": "missing_proposal",
                        "checked_pose_count": 0,
                        "worker_status": "not_run",
                        "evaluation_only": True,
                    }
                )
                continue
            proposal = GroupProposal.model_validate(item)
            validation = isolated_solve_and_validate_group(
                pool, proposal, config_path, config
            )
            pose = _pose_record(
                pool,
                {
                    **item,
                    "pool_id": pool.name,
                },
                validation,
                checked=True,
            )
            records.append(
                {
                    "pool_id": pool.name,
                    "true_group_id": group["group_id"],
                    "assembly_family": group["assembly_family"],
                    "parts": group["parts"],
                    "proposal_generated": True,
                    "evaluation_only": True,
                    **{
                        key: value
                        for key, value in pose.items()
                        if key not in {"pool_id", "parts"}
                    },
                }
            )

    total = len(records)
    proposed = sum(row["proposal_generated"] for row in records)
    valid = sum(
        row["final_pose_status"] == "valid" for row in records
    )
    failed = sum(
        row["final_pose_status"] == "failed" for row in records
    )
    uncertain = sum(
        row["final_pose_status"] == "uncertain" for row in records
    )
    missing = sum(
        row["final_pose_status"] == "missing_proposal"
        for row in records
    )
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({row["assembly_family"] for row in records}):
        family_rows = [
            row for row in records if row["assembly_family"] == family
        ]
        family_valid = sum(
            row["final_pose_status"] == "valid" for row in family_rows
        )
        by_family[family] = {
            "true_group_count": len(family_rows),
            "pose_valid_count": family_valid,
            "true_group_pose_recall": (
                family_valid / len(family_rows) if family_rows else None
            ),
            "status_counts": {
                status: sum(
                    row["final_pose_status"] == status
                    for row in family_rows
                )
                for status in (
                    "valid",
                    "failed",
                    "uncertain",
                    "missing_proposal",
                )
            },
        }

    summary = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only",
        "truth_not_exposed_to_solver": True,
        "true_group_count": total,
        "proposal_generated_count": proposed,
        "group_proposal_recall": proposed / total if total else None,
        "pose_valid_count": valid,
        "pose_failed_count": failed,
        "pose_uncertain_count": uncertain,
        "missing_proposal_count": missing,
        "true_group_pose_recall": valid / total if total else None,
        "by_family": by_family,
        "records": records,
    }
    _write(results_root / "true_group_pose_audit.json", summary)

    csv_path = results_root / "true_group_pose_audit.csv"
    fields = [
        "pool_id",
        "true_group_id",
        "assembly_family",
        "parts",
        "proposal_generated",
        "candidate_id",
        "checked_pose_count",
        "best_pose_rank",
        "selected_constraint_residual",
        "collision_result",
        "occt_common_volume",
        "worker_status",
        "final_pose_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            output = {key: row.get(key) for key in fields}
            output["parts"] = "|".join(row["parts"])
            writer.writerow(output)

    lines = [
        "# Evaluation-only True-Group Pose Audit",
        "",
        "- Truth fields were used only to select and score audit subjects.",
        "- The production proposal and production candidate edges were passed "
        "to the pose solver.",
        f"- True groups: {total}",
        f"- Group proposal recall: {summary['group_proposal_recall']:.2%}",
        f"- Pose valid: {valid}",
        f"- Pose failed: {failed}",
        f"- Pose uncertain: {uncertain}",
        f"- True-group pose recall: {summary['true_group_pose_recall']:.2%}",
        "",
        "| Pool | Family | Proposal | Pose status | Checked poses |",
        "|---|---|---:|---|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['pool_id']} | {row['assembly_family']} | "
            f"{row['proposal_generated']} | {row['final_pose_status']} | "
            f"{row.get('checked_pose_count', 0)} |"
        )
    (results_root / "true_group_pose_audit.md").write_text(
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
    parser.add_argument(
        "--pipeline-config",
        default=str(here / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    summary = run(
        args.pools_root, args.results_root, args.pipeline_config
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "true_group_count",
                    "group_proposal_recall",
                    "pose_valid_count",
                    "pose_failed_count",
                    "pose_uncertain_count",
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
