"""Evaluation-only regression for topology-aware pose reconstruction.

Ground-truth groups select audit subjects only. Production candidate
generation, matching, pose search, residual checks, and OCCT collision checks
remain unchanged and do not receive roles, family labels, or placements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contracts import GroupProposal
from geometry_pipeline import solve_and_validate_group


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(
    root: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    validation_namespace: str = "validation_topology_audit_v1",
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output_dir).resolve()
    config = _load(Path(config_path).resolve())
    records = []
    for pool in sorted(root.iterdir()):
        truth_path = pool / "pool_gt.json"
        if not pool.is_dir() or not truth_path.is_file():
            continue
        truth = _load(truth_path)
        for group in truth.get("true_groups", []):
            subject_id = (
                f"TOPOLOGY_{pool.name}_{group['group_id']}"
            )
            proposal = GroupProposal(
                group_id=subject_id,
                parts=group["parts"],
                candidate_edges=[],
                geometry_score=1.0,
                connected=True,
                status="evaluation_only_true_group",
                reasons=["evaluation_only_pose_regression"],
            )
            result = solve_and_validate_group(
                pool,
                proposal,
                config,
                output_namespace=validation_namespace,
            )
            metrics = result["metrics"]
            records.append(
                {
                    "pool_id": pool.name,
                    "true_group_id": group["group_id"],
                    "assembly_family": group.get("assembly_family"),
                    "group_size": len(group["parts"]),
                    "parts": group["parts"],
                    "final_pose_status": metrics["final_pose_status"],
                    "checked_pose_count": metrics["checked_pose_count"],
                    "best_pose_rank": metrics["best_pose_rank"],
                    "selected_constraint_residual": metrics.get(
                        "selected_constraint_residual"
                    ),
                    "collision_result": metrics.get("collision_result"),
                    "occt_common_volume": metrics.get("occt_common_volume"),
                    "worker_status": metrics.get("worker_status"),
                    "validation_result": str(
                        pool
                        / validation_namespace
                        / subject_id
                        / "validation_result.json"
                    ),
                }
            )
    valid = sum(
        row["final_pose_status"] == "valid" for row in records
    )
    failed = sum(
        row["final_pose_status"] == "failed" for row in records
    )
    uncertain = sum(
        row["final_pose_status"] == "uncertain" for row in records
    )
    report = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only_pose_regression",
        "truth_usage": (
            "Truth selects audit groups only; no role, family, semantic, or "
            "ground-truth placement is provided to the pose solver."
        ),
        "topology_changes": [
            "oriented planar normals from OCCT face orientation",
            "convex-versus-concave cylindrical surface polarity",
            "distinct repeated-cylinder pose hypotheses",
            "type-diverse per-pair pruning",
            "discounted AABB penalty for satisfied insertion/contact interfaces",
        ],
        "true_group_count": len(records),
        "pose_valid_count": valid,
        "pose_failed_count": failed,
        "pose_uncertain_count": uncertain,
        "true_group_pose_recall": (
            valid / len(records) if records else None
        ),
        "records": records,
    }
    _write(output / "true_group_pose_topology_audit.json", report)
    rows = [
        "# Topology-aware true-group pose regression",
        "",
        "- Artifact role: evaluation only",
        f"- True groups: {len(records)}",
        f"- Valid: {valid}",
        f"- Failed: {failed}",
        f"- Uncertain: {uncertain}",
        (
            f"- Pose recall: {valid / len(records):.2%}"
            if records
            else "- Pose recall: not estimable"
        ),
        "",
        "| Pool | Family | Size | Status | Checked | Best rank | Common volume |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for record in records:
        rows.append(
            "| {pool_id} | {assembly_family} | {group_size} | "
            "{final_pose_status} | {checked_pose_count} | {best_pose_rank} | "
            "{occt_common_volume} |".format(**record)
        )
    (output / "true_group_pose_topology_audit.md").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=str(
            Path(__file__).parent
            / "data"
            / "functional_mixed_pools_v1"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parent
            / "data"
            / "topology_pose_audit_v1"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).parent / "configs" / "pool_pipeline.json"
        ),
    )
    args = parser.parse_args()
    report = run(args.root, args.output, args.config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
