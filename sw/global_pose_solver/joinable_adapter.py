"""Adapters between JoinABLe E2E pose reports and bounded global pose search.

Part identifiers are caller-provided bookkeeping keys only.  They are never
used as a geometric or semantic feature; all pose alternatives come from the
stored rigid transforms and their audited pair validation status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from pose_search import matrix_to_placement


def _transform_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Dimensionless SE(3) separation used only for frontier diversity."""

    a = np.asarray(left["T_rel"], dtype=float)
    b = np.asarray(right["T_rel"], dtype=float)
    relative = np.linalg.inv(a) @ b
    translation_mm = float(np.linalg.norm(relative[:3, 3]))
    rotation_degrees = float(np.degrees(
        np.linalg.norm(Rotation.from_matrix(relative[:3, :3]).as_rotvec())
    ))
    # These are diversity scales, not a physical compatibility threshold.
    return translation_mm / 20.0 + rotation_degrees / 45.0


def _select_diverse_candidates(
    candidates: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Retain score anchor plus geometrically distinct Pose alternatives."""

    if len(candidates) <= maximum:
        return candidates
    ordered = sorted(
        candidates, key=lambda row: (-float(row["prior"]), row["candidate_id"])
    )
    selected = [ordered[0]]
    while len(selected) < maximum:
        remaining = [row for row in ordered if row not in selected]
        if not remaining:
            break
        # Farthest-first prevents a single high-contact local optimum from
        # consuming all slots.  Prior breaks ties but never removes diversity.
        next_row = max(
            remaining,
            key=lambda row: (
                min(_transform_distance(row, kept) for kept in selected),
                float(row["prior"]),
                row["candidate_id"],
            ),
        )
        selected.append(next_row)
    return selected


def load_joinable_pose_pool(
    source: str,
    target: str,
    result_path: str | Path,
    *,
    maximum_candidates: int = 8,
    include_not_checked: bool = True,
    include_manifold_initials: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read proper-rigid pair hypotheses from one JoinABLe E2E JSON report.

    A large multi-solid STEP may be safe to run through B-Rep inference but too
    expensive to mesh for the released SDF search.  In that case the audited
    constraint-manifold ``initial_pose_b_in_a`` rows remain legitimate pose
    *proposals*.  They are admitted with ``pair_exact_status=not_checked`` and
    can never become acceptance evidence by themselves.
    """

    path = Path(result_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("pose_search", {}).get("results", [])
    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[float, ...]] = set()
    for index, row in enumerate(rows):
        exact_record = row.get("exact_collision") or {}
        exact = exact_record.get("status", "not_checked")
        if exact == "success" and exact_record.get("collisions"):
            excluded.append({
                "result_index": index,
                "reason": "pair_exact_collision",
                "collision_count": len(exact_record.get("collisions") or []),
            })
            continue
        if exact not in ({"success", "not_checked"} if include_not_checked else {"success"}):
            excluded.append({
                "result_index": index,
                "reason": f"pair_exact_status:{exact}",
            })
            continue
        try:
            transform = np.asarray(row["transform"], dtype=float)
            if transform.shape != (4, 4):
                raise ValueError("not_4x4")
            determinant = float(np.linalg.det(transform[:3, :3]))
            if not np.isfinite(transform).all() or abs(determinant - 1.0) > 1e-4:
                raise ValueError("not_proper_rigid")
        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({
                "result_index": index,
                "reason": f"invalid_transform:{exc}",
            })
            continue
        signature = tuple(np.round(transform.reshape(-1), 5).tolist())
        if signature in seen:
            continue
        seen.add(signature)
        evaluation = row.get("evaluation") or {}
        # Contact is a weak ranking prior only; the global solver never uses
        # it to auto-accept a hypothesis.
        try:
            prior = float(evaluation.get("contact", 0.0))
        except (TypeError, ValueError):
            prior = 0.0
        retained.append({
            "candidate_id": f"{path.stem}:pose:{index:03d}",
            "source": source,
            "target": target,
            "T_rel": transform.tolist(),
            "prior": prior,
            "rank": row.get("prediction_rank"),
            "pair_exact_status": exact,
            "candidate_origin": row.get("candidate_origin"),
            "prediction_score": row.get("prediction_score"),
            "offset": row.get("offset"),
            "rotation_degrees": row.get("rotation_degrees"),
            "axis_flip": row.get("axis_flip"),
        })
    manifold_rows = (data.get("joint_hypotheses") or {}).get("rows") or []
    manifold_input_count = len(manifold_rows)
    manifold_retained_count = 0
    if include_manifold_initials:
        ordered_manifolds = sorted(
            enumerate(manifold_rows),
            key=lambda item: (
                int(item[1].get("rank", 1_000_000) or 1_000_000),
                item[0],
            ),
        )
        # Bound parsing before farthest-first selection.  Multiple phase and
        # polarity rows are useful, but an unbounded periodic manifold is not.
        for index, row in ordered_manifolds[: max(16, maximum_candidates * 12)]:
            try:
                transform = np.asarray(row["initial_pose_b_in_a"], dtype=float)
                if transform.shape != (4, 4):
                    raise ValueError("not_4x4")
                determinant = float(np.linalg.det(transform[:3, :3]))
                if (
                    not np.isfinite(transform).all()
                    or abs(determinant - 1.0) > 1e-4
                ):
                    raise ValueError("not_proper_rigid")
            except (KeyError, TypeError, ValueError) as exc:
                excluded.append({
                    "manifold_index": index,
                    "reason": f"invalid_manifold_initial:{exc}",
                })
                continue
            signature = tuple(np.round(transform.reshape(-1), 5).tolist())
            if signature in seen:
                continue
            seen.add(signature)
            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            retained.append({
                "candidate_id": f"{path.stem}:manifold:{index:04d}",
                "source": source,
                "target": target,
                "T_rel": transform.tolist(),
                "prior": confidence,
                "rank": row.get("rank"),
                "pair_exact_status": "not_checked",
                "candidate_origin": "joinable_constraint_manifold_initial",
                "proposal_only": True,
                "can_auto_accept": False,
                "manifold_type": row.get("manifold_type"),
                "entity_a": row.get("entity_a"),
                "entity_b": row.get("entity_b"),
                "polarity": row.get("polarity"),
                "phase_degrees": row.get("phase_degrees"),
            })
            manifold_retained_count += 1
    retained = _select_diverse_candidates(retained, maximum_candidates)
    return {
        "source": source,
        "target": target,
        "candidates": retained,
    }, {
        "source": source,
        "target": target,
        "result_path": str(path.resolve()),
        "input_result_count": len(rows),
        "input_manifold_count": manifold_input_count,
        "parsed_manifold_initial_count": manifold_retained_count,
        "retained_count": len(retained),
        "excluded": excluded,
        "include_not_checked": include_not_checked,
        "include_manifold_initials": include_manifold_initials,
    }


def make_occt_exact_validator(
    part_sources: dict[str, str | Path],
):
    """Build an OCCT top-N validator for parts stored in one directory.

    If sources do not have a common directory, the returned validator emits an
    explicit ``uncertain`` result instead of moving/copying CAD inputs.
    """

    resolved = {part: Path(source).resolve() for part, source in part_sources.items()}
    parents = {path.parent for path in resolved.values()}
    if len(parents) != 1:
        reason = "exact_validator_requires_step_sources_in_one_directory"

        def unavailable(_: dict[str, np.ndarray]) -> dict[str, Any]:
            return {"status": "uncertain", "reason": reason}

        return unavailable
    folder = next(iter(parents))

    def validator(poses: dict[str, np.ndarray]) -> dict[str, Any]:
        try:
            from placement_validation import exact_shape_collisions
            components = [
                {
                    "source": resolved[part].name,
                    "placement": matrix_to_placement(poses[part]),
                }
                for part in sorted(resolved)
            ]
            result = exact_shape_collisions(folder, components)
        except Exception as exc:
            return {"status": "uncertain", "reason": f"occt_validator_error:{type(exc).__name__}:{exc}"}
        if result.get("status") != "success":
            return {
                "status": "uncertain",
                "reason": "occt_validation_incomplete",
                "occt": result,
            }
        return {
            "status": "failed" if result.get("collisions") else "valid",
            "occt": result,
        }

    return validator


def build_pools_from_joinable_reports(
    records: Iterable[dict[str, Any]],
    *,
    maximum_candidates_per_pair: int = 8,
    include_not_checked: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for record in records:
        pool, audit = load_joinable_pose_pool(
            str(record["source"]),
            str(record["target"]),
            record["result_path"],
            maximum_candidates=maximum_candidates_per_pair,
            include_not_checked=include_not_checked,
        )
        pools.append(pool)
        audits.append(audit)
    return pools, audits
