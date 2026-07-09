"""Geometry-only grouping, reliable pose solving, and auditable validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from constraints import match_features
from contracts import AgentEvent, GroupProposal, ValidationResult
from features import extract_features
from global_grouping import assign_groups, evaluate_groups, run as run_grouping
from match_pruning import prune_match_graph, write_pruning_logs
from match_scoring import score_matches
from placement_validation import (
    constraint_residual,
    exact_shape_collisions,
    validate_assembly,
)
from small_assembly_solver import solve_small_assembly


def _write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _pipeline_fingerprint(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    )
    root = Path(__file__).parent
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


def _selected_evidence(solution: dict[str, Any]) -> list[dict[str, Any]]:
    unique = {}
    for selected in solution["selected_mates"]:
        for match in selected["evidence"]:
            key = (
                tuple(sorted(match["parts"])),
                match["type"],
                match.get("feat_a_idx"),
                match.get("feat_b_idx"),
            )
            unique[key] = match
    return list(unique.values())


def solve_and_validate_group(
    pool: Path,
    proposal: GroupProposal,
    config: dict[str, Any],
    *,
    output_namespace: str = "validation",
) -> dict[str, Any]:
    run_dir = pool / output_namespace / proposal.group_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for part in proposal.parts:
        shutil.copy2(pool / "parts" / part, run_dir / part)

    features = {
        part: extract_features(str(run_dir / part))
        for part in proposal.parts
    }
    candidate_config = config["candidate_generation"]
    pose_thresholds = {
        **candidate_config["detailed_thresholds"],
        "preserve_planar_face_hypotheses": bool(
            config["placement_validation"][
                "preserve_planar_face_hypotheses"
            ]
        ),
        "preserve_cylindrical_face_hypotheses": bool(
            config["placement_validation"].get(
                "preserve_cylindrical_face_hypotheses",
                False,
            )
        ),
    }
    raw = match_features(
        features,
        thresholds=pose_thresholds,
    )
    scored = score_matches(raw, features)
    kept, removed = prune_match_graph(
        scored,
        min_score=float(candidate_config["minimum_score"]),
        top_k_pair=int(candidate_config["top_k_per_pair"]),
        max_neighbors=int(candidate_config["maximum_neighbors"]),
        planar_hypotheses_per_pair=int(
            config["placement_validation"]["planar_hypotheses_per_pair"]
        ),
    )
    kept_path, _ = write_pruning_logs(run_dir, kept, removed)

    validation_config = config["placement_validation"]
    role_priority = {
        "base": 3.0,
        "housing": 3.0,
        "hub": 3.0,
        "shaft": 2.5,
        "bearing": 2.0,
        "cover": 2.0,
        "end_cover": 1.0,
        "locating_pin": 0.0,
        "key": 0.0,
        "axial_retainer": 0.0,
        "bearing_retainer": 0.0,
    }
    placement_priority = {}
    role_parts = {}
    for role, value in proposal.role_assignment.items():
        parts = value if isinstance(value, list) else [value]
        role_parts[str(role)] = [str(part) for part in parts if part]
        for part in parts:
            if part:
                placement_priority[str(part)] = max(
                    placement_priority.get(str(part), 0.0),
                    role_priority.get(str(role), 0.0),
                )
    preferred_axial_pairs = set()
    for left_role, right_role in (
        ("shaft", "hub"),
        ("housing", "bearing"),
        ("bearing", "shaft"),
    ):
        for left in role_parts.get(left_role, []):
            for right in role_parts.get(right_role, []):
                preferred_axial_pairs.add(tuple(sorted((left, right))))
    solution = solve_small_assembly(
        features,
        kept,
        beam_width=int(validation_config["beam_width"]),
        target_branching=int(validation_config["target_branching"]),
        placement_priority=placement_priority,
        preferred_axial_pairs=preferred_axial_pairs,
    )
    pose_candidates = solution.get("pose_candidates") or [
        {
            "placements": solution["placements"],
            "components": solution["components"],
            "selected_mates": solution["selected_mates"],
            "score": solution["score"],
            "penalty": solution["penalty"],
            "total_score": solution["total_score"],
            "penalty_details": solution["penalty_details"],
        }
    ]
    pose_audit = []
    selected_rank = None
    selected_exact = None
    selected_pose = pose_candidates[0]
    for rank, candidate in enumerate(
        pose_candidates[
            : int(validation_config["maximum_exact_pose_candidates"])
        ],
        start=1,
    ):
        candidate_matches = _selected_evidence(candidate)
        residual_items = [
            constraint_residual(
                match, features, candidate["placements"]
            )
            for match in candidate_matches
        ]
        residual_values = [
            float(item["residual"])
            for item in residual_items
            if item.get("valid")
        ]
        maximum_residual = max(residual_values, default=None)
        audit_item = {
            "rank": rank,
            "search_total_score": candidate["total_score"],
            "max_constraint_residual": maximum_residual,
            "exact_collision_check_status": "not_run",
            "exact_collision_count": None,
            "accepted": False,
        }
        if (
            maximum_residual is not None
            and maximum_residual
            > float(validation_config["residual_threshold"])
        ):
            audit_item["rejection_reason"] = "constraint_residual"
            pose_audit.append(audit_item)
            continue
        exact_candidate = exact_shape_collisions(
            run_dir,
            candidate["components"],
            minimum_volume=float(
                validation_config["minimum_intersection_volume_mm3"]
            ),
            minimum_part_ratio=float(
                validation_config["minimum_intersection_part_ratio"]
            ),
        )
        audit_item["exact_collision_check_status"] = exact_candidate["status"]
        audit_item["exact_collision_count"] = len(
            exact_candidate["collisions"]
        )
        if (
            exact_candidate["status"] == "success"
            and not exact_candidate["collisions"]
        ):
            audit_item["accepted"] = True
            pose_audit.append(audit_item)
            selected_rank = rank
            selected_pose = candidate
            selected_exact = exact_candidate
            break
        audit_item["rejection_reason"] = (
            "exact_collision_check_failed"
            if exact_candidate["status"] != "success"
            else "solid_penetration"
        )
        pose_audit.append(audit_item)

    for key in (
        "placements",
        "components",
        "selected_mates",
        "score",
        "penalty",
        "total_score",
        "penalty_details",
    ):
        solution[key] = selected_pose[key]
    solution["selected_pose_candidate_rank"] = selected_rank
    solution["pose_candidate_audit"] = pose_audit
    manifest = {
        "__description": "Reliable geometry-only group solution",
        "assembly_name": proposal.group_id,
        "global_units": "mm",
        "components": solution["components"],
    }
    _write(run_dir / "assembly_manifest.json", manifest)
    _write(run_dir / "search_report.json", solution)

    selected_matches = _selected_evidence(solution)
    selected_path = run_dir / "selected_matches.json"
    _write(selected_path, selected_matches)
    residual = validate_assembly(
        run_dir,
        selected_path,
        residual_threshold=float(validation_config["residual_threshold"]),
    )
    exact = selected_exact or exact_shape_collisions(
        run_dir,
        solution["components"],
        minimum_volume=float(
            validation_config["minimum_intersection_volume_mm3"]
        ),
        minimum_part_ratio=float(
            validation_config["minimum_intersection_part_ratio"]
        ),
    )
    build_error = None
    try:
        from build_assembly import build_assembly

        build_assembly(
            str(run_dir / "assembly_manifest.json"),
            str(run_dir / "assembly.step"),
        )
    except Exception as exc:
        build_error = str(exc)

    warnings = list(residual["warnings"])
    if exact["status"] != "success":
        warnings.append(
            f"exact collision check status={exact['status']}: "
            + "; ".join(exact["errors"])
        )
    if exact["collisions"]:
        warnings.append(
            f"{len(exact['collisions'])} exact solid penetration(s) detected"
        )
    if build_error:
        warnings.append(f"assembly STEP build/re-read failed: {build_error}")
    physical_pose_valid = (
        solution["status"] == "success"
        and residual["status"] == "success"
        and exact["status"] == "success"
        and build_error is None
        and len(exact["collisions"])
        <= int(validation_config["maximum_severe_penetrations"])
    )
    all_checked_reasons = [
        item.get("rejection_reason")
        for item in pose_audit
        if not item.get("accepted")
    ]
    confirmed_failure = bool(all_checked_reasons) and all(
        reason in {"constraint_residual", "solid_penetration"}
        for reason in all_checked_reasons
    ) and (
        int(solution.get("complete_pose_candidate_count", 0)) > 0
    ) and (
        int(solution.get("complete_pose_candidate_count", 0))
        <= len(pose_audit)
    ) and solution["status"] == "success"
    final_pose_status = (
        "valid"
        if physical_pose_valid
        else ("failed" if confirmed_failure else "uncertain")
    )
    common_volumes = [
        float(
            item.get(
                "intersection_volume_mm3",
                item.get("intersection_volume", 0.0),
            )
            or 0.0
        )
        for item in exact.get("collisions", [])
    ]
    status = "success" if physical_pose_valid else (
        "partial_success"
        if solution["status"] != "failed"
        else "failed"
    )
    result = ValidationResult(
        subject_id=proposal.group_id,
        status=status,
        num_parts=len(proposal.parts),
        solved_parts=sorted(
            set(proposal.parts) - set(solution["unsolved_parts"])
        ),
        unsolved_parts=sorted(solution["unsolved_parts"]),
        max_constraint_residual=residual["max_constraint_residual"],
        collision_count=len(exact["collisions"]),
        severe_penetration_count=len(exact["collisions"]),
        warnings=warnings,
        metrics={
            # Legacy field retained for compatibility. It means only that a
            # physical pose passed; it is not source-assembly acceptance.
            "accepted": physical_pose_valid,
            "physical_pose_valid": physical_pose_valid,
            "final_pose_status": final_pose_status,
            "worker_status": "success",
            "checked_pose_count": len(pose_audit),
            "best_pose_rank": selected_rank,
            "rejection_reason_per_rank": [
                {
                    "rank": item["rank"],
                    "reason": item.get("rejection_reason"),
                }
                for item in pose_audit
                if item.get("rejection_reason")
            ],
            "selected_constraint_residual": residual[
                "max_constraint_residual"
            ],
            "collision_result": exact["status"],
            "occt_common_volume": max(common_volumes, default=0.0),
            "search_total_score": solution["total_score"],
            "expanded_states": solution["expanded_states"],
            "complete_pose_candidate_count": solution[
                "complete_pose_candidate_count"
            ],
            "selected_pose_candidate_rank": selected_rank,
            "exact_pose_candidates_checked": len(pose_audit),
            "residual_validation_status": residual["status"],
            "exact_collision_check_status": exact["status"],
            "exact_collisions": exact["collisions"],
            "assembly_step_build_status": (
                "success" if build_error is None else "failed"
            ),
            "candidate_count": len(scored),
            "kept_match_count": len(kept),
            "pipeline_fingerprint": _pipeline_fingerprint(config),
        },
    ).model_dump(mode="json")
    _write(run_dir / "validation_result.json", result)
    return result


def _worker_failure(
    proposal: GroupProposal,
    returncode: int,
    stdout: str,
    stderr: str,
    pipeline_fingerprint: str,
) -> dict[str, Any]:
    return ValidationResult(
        subject_id=proposal.group_id,
        status="failed",
        num_parts=len(proposal.parts),
        solved_parts=[],
        unsolved_parts=sorted(proposal.parts),
        max_constraint_residual=None,
        collision_count=0,
        severe_penetration_count=0,
        warnings=[
            "isolated CAD validation worker failed; group rejected",
            f"worker_returncode={returncode}",
        ],
        metrics={
            "accepted": False,
            "physical_pose_valid": False,
            "final_pose_status": "uncertain",
            "worker_status": "crashed_or_failed",
            "worker_returncode": returncode,
            "checked_pose_count": 0,
            "best_pose_rank": None,
            "rejection_reason_per_rank": [],
            "selected_constraint_residual": None,
            "collision_result": "worker_failed",
            "occt_common_volume": None,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
            "pipeline_fingerprint": pipeline_fingerprint,
        },
    ).model_dump(mode="json")


def isolated_solve_and_validate_group(
    pool: Path,
    proposal: GroupProposal,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    pipeline_fingerprint = _pipeline_fingerprint(config)
    result_path = pool / "validation" / proposal.group_id / "validation_result.json"
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("metrics", {}).get("pipeline_fingerprint")
                == pipeline_fingerprint
            ):
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    timeout_seconds = float(
        config.get("placement_validation", {}).get(
            "worker_timeout_seconds", 240.0
        )
    )
    try:
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "group_validation_worker.py"),
                str(pool),
                proposal.group_id,
                "--config",
                str(config_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        failure = _worker_failure(
            proposal,
            -9,
            stdout,
            (
                f"CAD validation worker timed out after "
                f"{timeout_seconds:.1f} seconds.\n{stderr}"
            ),
            pipeline_fingerprint,
        )
        failure["warnings"] = [
            "isolated CAD validation worker timed out; result is uncertain",
            f"worker_timeout_seconds={timeout_seconds:.1f}",
        ]
        failure["metrics"]["worker_status"] = "timeout"
        failure["metrics"]["final_pose_status"] = "uncertain"
        failure["metrics"]["collision_result"] = "not_completed"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        _write(result_path, failure)
        return failure
    if process.returncode == 0 and result_path.is_file():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    failure = _worker_failure(
        proposal,
        process.returncode,
        process.stdout,
        process.stderr,
        pipeline_fingerprint,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write(result_path, failure)
    return failure


def run_pool(
    pool_dir: str | Path,
    config_path: str | Path,
    *,
    utility_overrides: dict[str, float] | None = None,
    output_namespace: str = "validation",
) -> dict[str, Any]:
    pool = Path(pool_dir).resolve()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    initial = run_grouping(pool, config_path)
    part_ids = sorted(
        path.name
        for path in (pool / "parts").iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )
    gt_path = pool / "pool_gt.json"
    gt = (
        json.loads(gt_path.read_text(encoding="utf-8"))
        if gt_path.is_file()
        else None
    )
    proposals = [
        GroupProposal.model_validate(item)
        for item in json.loads(
            (pool / "grouping" / "group_proposals.json").read_text(
                encoding="utf-8"
            )
        )
    ]
    available = list(proposals)
    cache = {}
    rejected_ids = set()
    attempts = []
    events = []
    maximum_attempts = int(
        config["placement_validation"]["maximum_grouping_retries"]
    ) + 1
    final_assignment = initial

    for attempt_number in range(maximum_attempts):
        assignment = assign_groups(
            part_ids,
            available,
            config["global_grouping"],
            utility_overrides,
        )
        selected = [
            item for item in assignment["selected_groups"]
            if len(item["parts"]) > 1
        ]
        rejected_this_attempt = []
        validations = []
        for item in selected:
            proposal = next(
                proposal
                for proposal in available
                if proposal.group_id == item["group_id"]
            )
            if proposal.group_id not in cache:
                cache[proposal.group_id] = isolated_solve_and_validate_group(
                    pool, proposal, config_path, config
                )
            validation = cache[proposal.group_id]
            validations.append(validation)
            if not validation["metrics"]["accepted"]:
                rejected_this_attempt.append(proposal.group_id)
                rejected_ids.add(proposal.group_id)

        attempts.append(
            {
                "attempt": attempt_number,
                "selected_group_ids": [item["group_id"] for item in selected],
                "rejected_group_ids": rejected_this_attempt,
                "validations": validations,
            }
        )
        event_id = hashlib.sha256(
            f"{pool.name}:{attempt_number}".encode("utf-8")
        ).hexdigest()[:16]
        events.append(
            AgentEvent(
                event_id=f"E_{event_id}",
                timestamp=datetime.now(timezone.utc),
                run_id=f"geometry_{pool.name}",
                sequence=attempt_number,
                state="validate_group_partition",
                action="solve_validate_and_retry",
                tool="geometry_pipeline.solve_and_validate_group",
                parameters={
                    "selected_group_ids": [
                        item["group_id"] for item in selected
                    ]
                },
                outcome=(
                    "accepted" if not rejected_this_attempt else "retry"
                ),
                evidence_refs=[
                    f"validation/{item['group_id']}/validation_result.json"
                    for item in selected
                ],
                retry_count=attempt_number,
                message=(
                    "All selected groups passed validation."
                    if not rejected_this_attempt
                    else "Rejected groups were removed before reassignment."
                ),
            ).model_dump(mode="json")
        )
        if not rejected_this_attempt:
            final_assignment = assignment
            break
        available = [
            proposal for proposal in available
            if proposal.group_id not in rejected_ids
        ]
        final_assignment = assignment

    if attempts[-1]["rejected_group_ids"]:
        # On retry exhaustion, return only groups already proven acceptable.
        # It is safer to expose remaining parts as singletons than to present
        # an unvalidated partition as a successful assembly.
        proven = [
            proposal
            for proposal in proposals
            if cache.get(proposal.group_id, {})
            .get("metrics", {})
            .get("accepted", False)
        ]
        final_assignment = assign_groups(
            part_ids,
            proven,
            config["global_grouping"],
            utility_overrides,
        )

    final_assignment["schema_version"] = "1.0.0"
    final_assignment["pool_id"] = pool.name
    metrics = (
        evaluate_groups(final_assignment["selected_groups"], gt)
        if gt is not None
        else None
    )
    if metrics is not None:
        final_assignment["metrics"] = metrics
    final_assignment["evaluation"] = {
        "available": gt is not None,
        "source": "pool_gt.json" if gt is not None else None,
        "reason": (
            None
            if gt is not None
            else "pool_gt.json absent; inference completed without labels"
        ),
    }
    output = pool / output_namespace
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "validated_group_assignment.json", final_assignment)
    report = {
        "schema_version": "1.0.0",
        "pool_id": pool.name,
        "attempt_count": len(attempts),
        "converged": not attempts[-1]["rejected_group_ids"],
        "rejected_proposal_ids": sorted(rejected_ids),
        "attempts": attempts,
        "metrics": metrics,
        "assignment_mode": (
            "semantic_tiebreak"
            if utility_overrides
            else "geometry_only"
        ),
    }
    _write(output / "validation_summary.json", report)
    _write(output / "agent_events.json", events)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    result = run_pool(args.pool_dir, args.config)
    print(
        json.dumps(
            {
                "pool_id": result["pool_id"],
                "converged": result["converged"],
                "attempt_count": result["attempt_count"],
                "metrics": result["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
