"""Validate a clustered review frontier and apply conservative final gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fingerprint(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    root = Path(__file__).resolve().parent
    for name in (
        "constraints.py",
        "match_scoring.py",
        "match_pruning.py",
        "small_assembly_solver.py",
        "placement_validation.py",
        "geometry_pipeline.py",
    ):
        digest.update(name.encode("utf-8"))
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _pose_record(result: dict[str, Any] | None, worker: dict[str, Any]) -> dict[str, Any]:
    if result is None:
        return {
            "final_pose_status": "uncertain",
            "worker_status": worker["status"],
            "checked_pose_count": 0,
            "best_pose_rank": None,
            "rejection_reason_per_rank": [],
            "selected_constraint_residual": None,
            "collision_result": "not_completed",
            "occt_common_volume": None,
        }
    metrics = result.get("metrics", {})
    physical = bool(metrics.get("physical_pose_valid", False))
    worker_status = str(metrics.get("worker_status", "success"))
    if physical:
        status = "valid"
    elif worker_status != "success":
        status = "uncertain"
    elif str(metrics.get("final_pose_status")) == "failed":
        status = "failed"
    else:
        status = "uncertain"
    return {
        "final_pose_status": status,
        "worker_status": worker_status,
        "checked_pose_count": int(metrics.get("checked_pose_count", 0) or 0),
        "best_pose_rank": metrics.get("best_pose_rank"),
        "rejection_reason_per_rank": metrics.get(
            "rejection_reason_per_rank", []
        ),
        "selected_constraint_residual": metrics.get(
            "selected_constraint_residual"
        ),
        "collision_result": metrics.get("collision_result", "not_run"),
        "occt_common_volume": metrics.get("occt_common_volume"),
    }


def _validate_one(
    pool: Path,
    proposal_file: Path,
    group_id: str,
    config_path: Path,
    output_root: Path,
    fingerprint: str,
    timeout: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    result_path = output_root / group_id / "validation_result.json"
    if result_path.is_file():
        existing = _load(result_path)
        if existing.get("metrics", {}).get("pipeline_fingerprint") == fingerprint:
            return existing, {"status": "cache_hit", "returncode": 0}
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "functional_pose_worker.py"),
                str(pool),
                str(proposal_file),
                group_id,
                "--config",
                str(config_path),
                "--output-root",
                str(output_root),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return None, {
            "status": "timeout",
            "returncode": -9,
            "stdout_tail": (exc.stdout or "")[-1000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }
    if process.returncode == 0 and result_path.is_file():
        return _load(result_path), {
            "status": "success",
            "returncode": 0,
            "stdout_tail": process.stdout[-1000:],
            "stderr_tail": process.stderr[-2000:],
        }
    return None, {
        "status": "crashed_or_failed",
        "returncode": process.returncode,
        "stdout_tail": process.stdout[-1000:],
        "stderr_tail": process.stderr[-2000:],
    }


def _truth_sets(pool: Path) -> set[frozenset[str]]:
    return {
        frozenset(row["parts"])
        for row in _load(pool / "pool_gt.json").get("true_groups", [])
    }


def _decide(
    rows: list[dict[str, Any]],
    *,
    role_template_calibration_passed: bool = False,
) -> tuple[list, list, list]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[row["proposal_cluster_id"]].append(row)
    accepted, review, rejected = [], [], []
    for row in rows:
        pose = row["pose_validation"]
        reasons = []
        if pose["final_pose_status"] == "failed":
            reasons.append("pose_failed")
            decision = "rejected"
        elif pose["final_pose_status"] != "valid":
            reasons.append("pose_uncertain_or_worker_incomplete")
            decision = "review"
        else:
            peers = [
                other for other in by_cluster[row["proposal_cluster_id"]]
                if other["group_id"] != row["group_id"]
                and other["pose_validation"]["final_pose_status"] == "valid"
                and set(other["parts"]) & set(row["parts"])
                and float(other["review_rank_score"])
                >= float(row["review_rank_score"]) - 0.04
            ]
            valid_supersets = [
                other for other in by_cluster[row["proposal_cluster_id"]]
                if set(row["parts"]) < set(other["parts"])
                and other["pose_validation"]["final_pose_status"] == "valid"
                and other["completeness_status"] == "family_complete"
                and int(other["independent_evidence_count"])
                > int(row["independent_evidence_count"])
            ]
            local_gate = (
                role_template_calibration_passed
                and
                row["completeness_status"] == "family_complete"
                and not row["missing_required_relations"]
                and float(row["geometry_score"]) >= 0.80
                and int(row["independent_evidence_count"]) >= 2
                and pose["collision_result"] == "success"
                and len(row["parts"]) <= 5
                and not bool(
                    row["ranking_features"].get(
                        "learned_only_critical_edge", 0.0
                    )
                )
            )
            if valid_supersets:
                reasons.append("demoted_by_pose_valid_more_complete_superset")
                reasons.extend(
                    f"superset={other['group_id']}"
                    for other in valid_supersets[:5]
                )
                decision = "review"
            elif not local_gate:
                if not role_template_calibration_passed:
                    reasons.append(
                        "role_template_calibration_gate_closed"
                    )
                else:
                    reasons.append("local_multi_condition_gate_not_satisfied")
                decision = "review"
            elif peers:
                reasons.append("unresolved_pose_valid_cluster_alternatives")
                reasons.extend(
                    f"conflict={peer['group_id']}" for peer in peers[:5]
                )
                decision = "review"
            else:
                decision = "accepted"
                reasons.append("all_conservative_gates_passed")
        item = {**row, "final_decision": decision, "decision_reasons": reasons}
        {"accepted": accepted, "review": review, "rejected": rejected}[decision].append(item)
    return accepted, review, rejected


def run(
    pools_root: Path,
    proposal_experiment: Path,
    output: Path,
    config_path: Path,
    *,
    timeout: float = 240.0,
) -> dict[str, Any]:
    config = _load(config_path)
    fingerprint = _fingerprint(config)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_parts: dict[str, set[str]] = {}
    for pool in sorted(
        row for row in pools_root.iterdir()
        if row.is_dir() and (row / "pool_gt.json").is_file()
    ):
        proposal_dir = proposal_experiment / "pools" / pool.name
        frontier = _load(proposal_dir / "review_frontier.json")
        proposal_file = proposal_dir / "group_proposals.json"
        validation_root = output / "pose_runs" / pool.name
        all_parts[pool.name] = set(_load(pool / "pool_gt.json").get("parts", []))
        for position, proposal in enumerate(frontier, 1):
            print(
                f"{pool.name}: pose {position}/{len(frontier)} "
                f"{proposal['group_id']} size={len(proposal['parts'])}",
                flush=True,
            )
            result, worker = _validate_one(
                pool,
                proposal_file,
                proposal["group_id"],
                config_path,
                validation_root,
                fingerprint,
                timeout,
            )
            all_rows.append(
                {
                    **proposal,
                    "pool_id": pool.name,
                    "pose_validation": _pose_record(result, worker),
                    "worker_audit": worker,
                }
            )
    role_template_calibration_passed = False
    accepted, review, rejected = _decide(
        all_rows,
        role_template_calibration_passed=role_template_calibration_passed,
    )
    covered: dict[str, set[str]] = defaultdict(set)
    for row in accepted + review:
        covered[row["pool_id"]].update(row["parts"])
    unresolved = [
        {"pool_id": pool_id, "part_id": part, "reason": "not_covered_by_accepted_or_review_frontier"}
        for pool_id, parts in all_parts.items()
        for part in sorted(parts - covered[pool_id])
    ]
    truth = {
        (pool.name, group)
        for pool in pools_root.iterdir()
        if pool.is_dir() and (pool / "pool_gt.json").is_file()
        for group in _truth_sets(pool)
    }
    accepted_true = sum(
        (row["pool_id"], frozenset(row["parts"])) in truth for row in accepted
    )
    false_positive = len(accepted) - accepted_true
    metrics = {
        "schema_version": "1.0.0",
        "truth_basis": "functional_validity_evaluation_only",
        "accepted_group_count": len(accepted),
        "accepted_true_positive_count": accepted_true,
        "auto_accept_precision": accepted_true / len(accepted) if accepted else None,
        "false_positive_count": false_positive,
        "review_group_count": len(review),
        "rejected_group_count": len(rejected),
        "unresolved_parts_count": len(unresolved),
        "pose_valid_count": sum(
            row["pose_validation"]["final_pose_status"] == "valid"
            for row in all_rows
        ),
        "pose_failed_count": sum(
            row["pose_validation"]["final_pose_status"] == "failed"
            for row in all_rows
        ),
        "pose_uncertain_count": sum(
            row["pose_validation"]["final_pose_status"] == "uncertain"
            for row in all_rows
        ),
        "semantic_reranking_enabled": False,
        "role_template_calibration_passed": role_template_calibration_passed,
        "role_template_gate_decision": (
            "closed_after_2_of_2_final_provisional_auto_accepts_were_false_on_locked_holdout"
        ),
        "decision_rule": "multi_condition_gate_with_pose_valid_cluster_conflict_abstention",
    }
    _write(output / "pose_validation_records.json", all_rows)
    _write(output / "final_accepted_groups.json", accepted)
    _write(output / "final_review_groups.json", review)
    _write(output / "final_rejected_groups.json", rejected)
    _write(output / "unresolved_parts.json", unresolved)
    _write(output / "conservative_metrics.json", metrics)
    with (output / "candidate_scores.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fields = [
            "pool_id", "group_id", "parts", "assembly_family",
            "review_rank", "review_rank_score", "geometry_score",
            "independent_evidence_count", "final_pose_status",
            "collision_result", "final_decision", "is_true_group",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        decisions = accepted + review + rejected
        for row in decisions:
            writer.writerow(
                {
                    "pool_id": row["pool_id"],
                    "group_id": row["group_id"],
                    "parts": "|".join(row["parts"]),
                    "assembly_family": row["assembly_family"],
                    "review_rank": row["review_rank"],
                    "review_rank_score": row["review_rank_score"],
                    "geometry_score": row["geometry_score"],
                    "independent_evidence_count": row["independent_evidence_count"],
                    "final_pose_status": row["pose_validation"]["final_pose_status"],
                    "collision_result": row["pose_validation"]["collision_result"],
                    "final_decision": row["final_decision"],
                    "is_true_group": (
                        row["pool_id"], frozenset(row["parts"])
                    ) in truth,
                }
            )
    lines = [
        "# Conservative Frontier Pose Experiment",
        "",
        f"- Accepted: {len(accepted)}",
        f"- Review: {len(review)}",
        f"- Rejected: {len(rejected)}",
        f"- Unresolved parts: {len(unresolved)}",
        f"- False auto accepts: {false_positive}",
        f"- Auto-accept precision: {metrics['auto_accept_precision']}",
        f"- Pose valid / failed / uncertain: {metrics['pose_valid_count']} / {metrics['pose_failed_count']} / {metrics['pose_uncertain_count']}",
        "- Semantic reranking: disabled",
        "",
        "Pose-valid alternatives in the same proposal cluster are routed to review; pose success alone never proves functional membership.",
    ]
    (output / "assembly_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools-root", type=Path, required=True)
    parser.add_argument("--proposal-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=here / "configs" / "pool_pipeline.json"
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()
    metrics = run(
        args.pools_root.resolve(),
        args.proposal_experiment.resolve(),
        args.output.resolve(),
        args.config.resolve(),
        timeout=args.timeout,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
