"""Run bounded, isolated pose validation on the real mixed-pool benchmark.

The production queue is selected without reading labels.  A second,
evaluation-only queue contains known source groups and measures pose-solver
capability; those rows are never eligible for a production decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SW_ROOT = PROJECT_ROOT / "sw"
if str(SW_ROOT) not in sys.path:
    sys.path.insert(0, str(SW_ROOT))

from contracts import GroupProposal  # noqa: E402
from geometry_pipeline import isolated_solve_and_validate_group  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    r"D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "public_cad_dataset_audit"
    / "outputs"
    / "phase7_pose_validation"
)
DEFAULT_WORK_ROOT = Path(
    r"D:\Model_match_public_data\real_pose_validation_work_20260705"
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _part_key(parts: list[str]) -> tuple[str, ...]:
    return tuple(sorted(parts))


def _production_candidate(item: dict[str, Any]) -> dict[str, Any]:
    """Remove evaluation-only labels before a candidate enters production."""
    return {
        key: value
        for key, value in item.items()
        if not key.startswith("evaluation_")
        and key not in {"truth_group_id", "source_assembly_id"}
    }


def _candidate_rank(item: dict[str, Any]) -> tuple[float, ...]:
    consistency = item.get("consistency") or {}
    evidence = item.get("geometry_evidence") or {}
    return (
        float(evidence.get("independent_evidence_count", 0) or 0),
        float(consistency.get("group_consistency_score", 0.0) or 0.0),
        float(item.get("geometry_score", 0.0) or 0.0),
        float(item.get("candidate_priority_score", 0.0) or 0.0),
        float(item.get("active_edge_density", 0.0) or 0.0),
        -float(item.get("group_size", len(item.get("parts", []))) or 0),
    )


def _select_blind_queue(
    candidates: list[dict[str, Any]], maximum_per_pool: int
) -> list[dict[str, Any]]:
    """Select a small size-diverse queue without consulting ground truth."""
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_pool[item["pool_id"]].append(item)
    selected: list[dict[str, Any]] = []
    for pool_id in sorted(by_pool):
        pool_rows = sorted(
            by_pool[pool_id],
            key=lambda row: (_candidate_rank(row), row["group_id"]),
            reverse=True,
        )
        chosen: list[dict[str, Any]] = []
        seen_sizes: set[int] = set()
        for row in pool_rows:
            size = int(row["group_size"])
            if size in seen_sizes:
                continue
            chosen.append(row)
            seen_sizes.add(size)
            if len(chosen) >= maximum_per_pool:
                break
        if len(chosen) < maximum_per_pool:
            chosen_ids = {row["group_id"] for row in chosen}
            chosen.extend(
                row
                for row in pool_rows
                if row["group_id"] not in chosen_ids
            )
        selected.extend(chosen[:maximum_per_pool])
    return selected


def _truth_groups(dataset_root: Path) -> list[dict[str, Any]]:
    rows = []
    for pool in sorted(dataset_root.glob("pool_*")):
        gt = _load(pool / "pool_gt.json")
        for group in gt["true_groups"]:
            rows.append(
                {
                    "pool_id": pool.name,
                    "truth_group_id": group["group_id"],
                    "parts": sorted(group["part_ids"]),
                    "group_size": len(group["part_ids"]),
                }
            )
    return rows


def _job_id(pool_id: str, parts: list[str]) -> str:
    digest = hashlib.sha256(
        f"{pool_id}:{'|'.join(sorted(parts))}".encode("utf-8")
    ).hexdigest()[:12]
    return f"pose_{digest}"


def build_queue(
    dataset_root: Path,
    candidates: list[dict[str, Any]],
    blind_per_pool: int,
) -> list[dict[str, Any]]:
    blind = _select_blind_queue(candidates, blind_per_pool)
    truth = _truth_groups(dataset_root)
    by_key: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    candidate_by_key = {
        (row["pool_id"], _part_key(row["parts"])): row for row in candidates
    }
    for row in blind:
        key = (row["pool_id"], _part_key(row["parts"]))
        by_key[key] = {
            "job_id": _job_id(row["pool_id"], row["parts"]),
            "pool_id": row["pool_id"],
            "parts": sorted(row["parts"]),
            "group_size": len(row["parts"]),
            "queue_modes": ["blind_production"],
            "source_candidate_id": row["group_id"],
            "candidate": _production_candidate(row),
        }
    for group in truth:
        key = (group["pool_id"], _part_key(group["parts"]))
        existing = by_key.get(key)
        if existing:
            existing["queue_modes"].append("evaluation_only_truth_audit")
            continue
        candidate = candidate_by_key.get(key)
        by_key[key] = {
            "job_id": _job_id(group["pool_id"], group["parts"]),
            "pool_id": group["pool_id"],
            "parts": group["parts"],
            "group_size": group["group_size"],
            "queue_modes": ["evaluation_only_truth_audit"],
            "source_candidate_id": (
                candidate["group_id"] if candidate else None
            ),
            "candidate": (
                _production_candidate(candidate) if candidate else None
            ),
        }
    return sorted(
        by_key.values(),
        key=lambda row: (
            row["pool_id"],
            "blind_production" not in row["queue_modes"],
            row["group_size"],
            row["job_id"],
        ),
    )


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_workspaces(
    dataset_root: Path,
    work_root: Path,
    queue: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    work_root.mkdir(parents=True, exist_ok=True)
    config_path = work_root / "bounded_pose_config.json"
    _write(config_path, config)
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in queue:
        by_pool[job["pool_id"]].append(job)
    for pool_id, jobs in by_pool.items():
        source_pool = dataset_root / pool_id
        work_pool = work_root / pool_id
        for part in sorted((source_pool / "parts").glob("*.step")):
            _link_or_copy(part, work_pool / "parts" / part.name)
        proposals = []
        for job in jobs:
            candidate = job.get("candidate") or {}
            proposals.append(
                {
                    "schema_version": "1.0.0",
                    "group_id": job["job_id"],
                    "parts": [f"{part}.step" for part in job["parts"]],
                    "candidate_edges": candidate.get("candidate_edges", []),
                    "geometry_score": min(
                        1.0,
                        max(
                            0.0,
                            float(
                                candidate.get(
                                    "geometry_score",
                                    candidate.get(
                                        "candidate_priority_score", 0.0
                                    ),
                                )
                                or 0.0
                            ),
                        ),
                    ),
                    "connected": True,
                    "status": "review",
                    "reasons": [
                        "bounded_pose_validation_only",
                        "pose_success_is_not_group_acceptance",
                    ],
                }
            )
        _write(work_pool / "grouping" / "group_proposals.json", proposals)
    return config_path


def _pose_record(
    job: dict[str, Any],
    work_pool: Path,
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = (validation or {}).get("metrics", {})
    status = metrics.get("final_pose_status")
    if not validation:
        status = "uncertain"
    elif metrics.get("worker_status") != "success":
        status = "uncertain"
    elif status not in {"valid", "failed", "uncertain"}:
        status = (
            "valid" if metrics.get("physical_pose_valid") else "uncertain"
        )
    run_dir = work_pool / "validation" / job["job_id"]
    return {
        "schema_version": "1.0.0",
        "candidate_id": job["source_candidate_id"] or job["job_id"],
        "pose_job_id": job["job_id"],
        "pool_id": job["pool_id"],
        "parts": job["parts"],
        "group_size": job["group_size"],
        "queue_modes": job["queue_modes"],
        "production_eligible": "blind_production" in job["queue_modes"],
        "evaluation_only": job["queue_modes"]
        == ["evaluation_only_truth_audit"],
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
        "worker_status": metrics.get(
            "worker_status", "not_run" if not validation else "unknown"
        ),
        "final_pose_status": status,
        "validation_result": (
            str(run_dir / "validation_result.json")
            if (run_dir / "validation_result.json").is_file()
            else None
        ),
        "assembly_step": (
            str(run_dir / "assembly.step")
            if (run_dir / "assembly.step").is_file()
            else None
        ),
        "decision_boundary": (
            "A valid pose proves physical feasibility only. "
            "It cannot prove source or functional correctness."
        ),
        "failure_reasons": (
            []
            if validation
            else ["not_run_due_to_bounded_validation_budget"]
        ),
        "unavailable_fields": [
            "functional_semantic_validity",
            "designer_intent",
        ],
    }


def _production_pose_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in record.items()
        if key not in {"queue_modes", "evaluation_only"}
    }
    output["validation_scope"] = "bounded_production"
    return output


def finalize(
    output_root: Path,
    work_root: Path,
    queue: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    truth_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    execution_records = []
    queued_keys = set()
    for job in queue:
        queued_keys.add((job["pool_id"], _part_key(job["parts"])))
        result_path = (
            work_root
            / job["pool_id"]
            / "validation"
            / job["job_id"]
            / "validation_result.json"
        )
        validation = _load(result_path) if result_path.is_file() else None
        execution_records.append(
            _pose_record(job, work_root / job["pool_id"], validation)
        )
    execution_by_key = {
        (record["pool_id"], _part_key(record["parts"])): record
        for record in execution_records
    }
    production_records = [
        _production_pose_record(record)
        for record in execution_records
        if record["production_eligible"]
    ]
    production_keys = {
        (record["pool_id"], _part_key(record["parts"]))
        for record in production_records
    }
    for item in all_candidates:
        key = (item["pool_id"], _part_key(item["parts"]))
        if key in production_keys:
            continue
        production_records.append(
            {
                "schema_version": "1.0.0",
                "candidate_id": item["group_id"],
                "pose_job_id": None,
                "pool_id": item["pool_id"],
                "parts": item["parts"],
                "group_size": item["group_size"],
                "production_eligible": False,
                "validation_scope": "not_selected_for_bounded_production",
                "checked_pose_count": 0,
                "best_pose_rank": None,
                "rejection_reason_per_rank": [],
                "selected_constraint_residual": None,
                "collision_result": "not_run",
                "occt_common_volume": None,
                "worker_status": "not_selected_bounded_queue",
                "final_pose_status": "uncertain",
                "validation_result": None,
                "assembly_step": None,
                "decision_boundary": (
                    "Unselected production candidates remain review/deferred; "
                    "an evaluation-only pose result cannot promote or reject "
                    "them."
                ),
                "failure_reasons": [
                    "not_run_due_to_bounded_validation_budget"
                ],
                "unavailable_fields": [
                    "physical_pose_feasibility",
                    "functional_semantic_validity",
                    "designer_intent",
                ],
            }
        )
    valid = [
        r for r in production_records if r["final_pose_status"] == "valid"
    ]
    failed = [
        r for r in production_records if r["final_pose_status"] == "failed"
    ]
    uncertain = [
        r
        for r in production_records
        if r["final_pose_status"] == "uncertain"
    ]
    _write(output_root / "pose_validated_candidates.json", valid)
    _write(output_root / "pose_failed_candidates.json", failed)
    _write(output_root / "pose_uncertain_candidates.json", uncertain)

    queued_records = [
        r for r in execution_records if r["pose_job_id"]
    ]
    blind = [
        r for r in queued_records if "blind_production" in r["queue_modes"]
    ]
    truth = []
    for group in truth_groups:
        record = execution_by_key.get(
            (group["pool_id"], _part_key(group["parts"]))
        )
        truth.append(
            {
                "pool_id": group["pool_id"],
                "truth_group_id": group["truth_group_id"],
                "parts": group["parts"],
                "group_size": group["group_size"],
                "pose_record": record,
                "final_pose_status": (
                    record["final_pose_status"]
                    if record
                    else "uncertain"
                ),
            }
        )
    checked = [r for r in blind if r["worker_status"] != "not_run"]
    status_counts = Counter(r["final_pose_status"] for r in blind)
    truth_counts = Counter(r["final_pose_status"] for r in truth)
    audit = {
        "schema_version": "1.0.0",
        "artifact_role": "production_pose_validation",
        "policy": {
            "beam_width": 50,
            "maximum_exact_pose_candidates": 20,
            "worker_timeout_seconds": 180,
            "blind_selection_uses_ground_truth": False,
            "group_size_six_or_more": "review_without_search",
            "pose_success_semantics": "physical_feasibility_only",
        },
        "candidate_count": len(all_candidates),
        "queue_job_count": len(blind),
        "completed_job_count": len(checked),
        "blind_production_job_count": len(blind),
        "queue_status_counts": dict(status_counts),
        "records": production_records,
        "failure_reasons": (
            []
            if len(checked) == len(blind)
            else ["pose_queue_incomplete"]
        ),
        "unavailable_fields": [
            "functional_semantic_validity",
            "designer_intent",
        ],
    }
    _write(output_root / "pose_validation_full_audit.json", audit)
    _write(
        output_root / "pose_validation_truth_audit.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "evaluation_only",
            "truth_group_count": len(truth),
            "truth_audit_status_counts": dict(truth_counts),
            "truth_pose_valid_rate": (
                truth_counts["valid"] / len(truth) if truth else None
            ),
            "records": truth,
        },
    )
    report = f"""# Real mixed-pool bounded pose validation

## Scope

- Candidate groups: {len(all_candidates)}
- Bounded queue jobs: {len(queue)}
- Blind production jobs: {len(blind)}
- Evaluation-only truth jobs: {len(truth)}
- Completed jobs: {len(checked)}
- Queue status: {dict(status_counts)}
- Truth-audit status: {dict(truth_counts)}

## Decision boundary

Pose validation answers only whether a physical pose was found within the
bounded search.  It does not prove provenance, designer intent, or functional
correctness.  Ground-truth audit rows are isolated from production decisions.
Candidates outside the bounded queue remain review, not rejected.
"""
    (output_root / "pose_validation_report.md").write_text(
        report, encoding="utf-8"
    )
    system_review = f"""# Step 7 independent systemic review

1. The production queue was chosen without labels and is limited to
   {len(blind)} jobs, preventing a combinatorial OCCT run.
2. The {len(truth)} known groups are an evaluation-only solver audit.  Their
   results cannot promote a production candidate.
3. Beam width is 50 and exact checks are capped at 20; no six-part rescue or
   beam expansion was introduced.
4. Physical validity remains a necessary but insufficient signal.
5. Current blocker: {
        'none'
        if len(checked) == len(blind)
        else f'{len(blind) - len(checked)} production jobs have not completed'
   }.
"""
    (output_root / "pose_validation_system_review.md").write_text(
        system_review, encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--blind-per-pool", type=int, default=2)
    parser.add_argument("--maximum-jobs", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    candidate_dir = (
        PROJECT_ROOT
        / "public_cad_dataset_audit"
        / "outputs"
        / "phase6_candidate_tiers"
    )
    candidates = []
    seen_candidate_ids = set()
    for name in (
        "accepted_geometry_candidates.json",
        "review_geometry_candidates.json",
    ):
        for candidate in _load(candidate_dir / name):
            if candidate["group_id"] in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate["group_id"])
            candidates.append(_production_candidate(candidate))
    truth_groups = _truth_groups(args.dataset_root)
    queue = build_queue(
        args.dataset_root, candidates, args.blind_per_pool
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    production_jobs = [
        job for job in queue if "blind_production" in job["queue_modes"]
    ]
    _write(
        args.output_root / "pose_validation_queue.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "production_queue",
            "selection_policy": (
                "Top size-diverse candidates per pool without labels."
            ),
            "job_count": len(production_jobs),
            "jobs": [
                {
                    **job,
                    "queue_modes": ["blind_production"],
                }
                for job in production_jobs
            ],
        },
    )
    _write(
        args.output_root / "pose_validation_truth_audit_queue.json",
        {
            "schema_version": "1.0.0",
            "artifact_role": "evaluation_only",
            "job_count": len(truth_groups),
            "jobs": [
                {
                    **group,
                    "pose_job_id": _job_id(
                        group["pool_id"], group["parts"]
                    ),
                }
                for group in truth_groups
            ],
        },
    )
    config = _load(SW_ROOT / "configs" / "pool_pipeline.json")
    config["placement_validation"].update(
        {
            "beam_width": 50,
            "target_branching": 3,
            "maximum_exact_pose_candidates": 20,
            "worker_timeout_seconds": 180,
            "maximum_grouping_retries": 0,
        }
    )
    config_path = prepare_workspaces(
        args.dataset_root, args.work_root, queue, config
    )
    if args.prepare_only:
        finalize(
            args.output_root,
            args.work_root,
            queue,
            candidates,
            truth_groups,
        )
        print(json.dumps({"prepared_jobs": len(queue)}, indent=2))
        return 0

    run_queue = queue[: args.maximum_jobs] if args.maximum_jobs else queue
    for index, job in enumerate(run_queue, start=1):
        work_pool = args.work_root / job["pool_id"]
        proposal = next(
            GroupProposal.model_validate(row)
            for row in _load(
                work_pool / "grouping" / "group_proposals.json"
            )
            if row["group_id"] == job["job_id"]
        )
        print(
            f"[{index}/{len(run_queue)}] {job['pool_id']} "
            f"{job['job_id']} size={job['group_size']} "
            f"modes={','.join(job['queue_modes'])}",
            flush=True,
        )
        isolated_solve_and_validate_group(
            work_pool, proposal, config_path, config
        )
        finalize(
            args.output_root,
            args.work_root,
            queue,
            candidates,
            truth_groups,
        )
    audit = finalize(
        args.output_root,
        args.work_root,
        queue,
        candidates,
        truth_groups,
    )
    print(
        json.dumps(
            {
                "queue_jobs": audit["queue_job_count"],
                "completed_jobs": audit["completed_job_count"],
                "queue_status_counts": audit["queue_status_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
