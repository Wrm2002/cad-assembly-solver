"""Bounded discrete--continuous global pose recovery for small assemblies.

The solver is deliberately an *evidence combiner*, not a semantic assembly
oracle.  It selects bounded alternatives from pair-pose pools, optimizes the
resulting SE(3) pose graph, tests independent cycle closures, and optionally
hands only the top hypotheses to an exact CAD validator.  It never reads paths,
case names, source IDs, labels, or part roles.

The intended scope is a known 3--5 part assembly.  A tree with no independent
closure remains review-only even if its residual is exactly zero: a spanning
tree can always fit its own selected measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import least_squares

from .se3_manifold import relative_error, se3_exp, se3_inv, se3_log


_EPS = 1e-12


@dataclass(frozen=True)
class _Candidate:
    source: str
    target: str
    transform: np.ndarray
    candidate_id: str
    prior: float
    source_rank: int | None
    provenance: dict[str, Any]


def _as_transform(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not bool(np.all(np.isfinite(matrix))):
        raise ValueError("candidate_transform_must_be_finite_4x4")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("candidate_transform_has_invalid_homogeneous_row")
    determinant = float(np.linalg.det(matrix[:3, :3]))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > 1e-4:
        raise ValueError("candidate_transform_must_be_proper_rigid")
    return matrix.copy()


def _pair_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((str(source), str(target))))


def _finite_prior(row: dict[str, Any]) -> float:
    """Read a ranking prior without making it an acceptance criterion."""

    for field in ("prior", "geometry_score", "score", "confidence"):
        try:
            value = float(row.get(field))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    rank = row.get("rank")
    try:
        return 1.0 / max(1.0, float(rank))
    except (TypeError, ValueError):
        return 0.0


def _normalise_pools(
    part_ids: Iterable[str], candidate_pools: Iterable[dict[str, Any]]
) -> dict[tuple[str, str], list[_Candidate]]:
    allowed = {str(part) for part in part_ids}
    pools: dict[tuple[str, str], list[_Candidate]] = {}
    for pool_index, pool in enumerate(candidate_pools):
        source = str(pool.get("source", pool.get("src", "")))
        target = str(pool.get("target", pool.get("dst", "")))
        if source not in allowed or target not in allowed or source == target:
            continue
        raw_candidates = pool.get("candidates")
        if raw_candidates is None:
            raw_candidates = [pool]
        if not isinstance(raw_candidates, list):
            continue
        key = _pair_key(source, target)
        for item_index, raw in enumerate(raw_candidates):
            if not isinstance(raw, dict):
                continue
            try:
                transform = _as_transform(
                    raw.get("T_rel", raw.get("transform"))
                )
            except (TypeError, ValueError):
                continue
            # A candidate in reversed pool orientation is converted once into
            # the pool convention.  This avoids treating graph direction as a
            # separate physical hypothesis.
            raw_source = str(raw.get("source", raw.get("src", source)))
            raw_target = str(raw.get("target", raw.get("dst", target)))
            if raw_source == target and raw_target == source:
                transform = se3_inv(transform)
            elif raw_source != source or raw_target != target:
                continue
            candidate_id = str(raw.get(
                "candidate_id", f"pool{pool_index}:candidate{item_index}"
            ))
            rank = raw.get("rank")
            try:
                source_rank = int(rank) if rank is not None else None
            except (TypeError, ValueError):
                source_rank = None
            provenance = {
                key: value for key, value in raw.items()
                if key not in {"T_rel", "transform"}
            }
            pools.setdefault(key, []).append(_Candidate(
                source=source,
                target=target,
                transform=transform,
                candidate_id=candidate_id,
                prior=_finite_prior(raw),
                source_rank=source_rank,
                provenance=provenance,
            ))
    return pools


def _deduplicate_pool(
    values: list[_Candidate], maximum: int
) -> list[_Candidate]:
    """Keep a small diverse transform frontier without case-specific bins."""

    ordered = sorted(values, key=lambda row: (-row.prior, row.candidate_id))
    result: list[_Candidate] = []
    seen: set[tuple[float, ...]] = set()
    for row in ordered:
        signature = tuple(np.round(se3_log(row.transform), 4).tolist())
        if signature in seen:
            continue
        seen.add(signature)
        result.append(row)
        if len(result) >= maximum:
            break
    return result


def _connected(part_ids: list[str], edges: Iterable[tuple[str, str]]) -> bool:
    if not part_ids:
        return False
    parent = {part: part for part in part_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def join(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source, target in edges:
        join(source, target)
    return len({find(part) for part in part_ids}) == 1


def _topologies(
    part_ids: list[str], pools: dict[tuple[str, str], list[_Candidate]], maximum: int
) -> list[tuple[tuple[str, str], ...]]:
    """Enumerate a bounded high-prior spanning-tree frontier."""

    keys = sorted(pools)
    if len(part_ids) == 1:
        return [tuple()]
    scored: list[tuple[float, tuple[tuple[str, str], ...]]] = []
    for choice in itertools.combinations(keys, len(part_ids) - 1):
        if not _connected(part_ids, choice):
            continue
        quality = sum(max(row.prior for row in pools[key]) for key in choice)
        scored.append((quality, choice))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [choice for _, choice in scored[:maximum]]


def _bounded_assignments(
    candidate_lists: list[list[_Candidate]], maximum: int
) -> list[tuple[_Candidate, ...]]:
    """Keep a high-prior anchor plus a geometrically diverse combination set."""

    assignments = list(itertools.product(*candidate_lists))
    assignments.sort(key=lambda rows: (
        -sum(row.prior for row in rows),
        tuple(row.candidate_id for row in rows),
    ))
    if len(assignments) <= maximum:
        return assignments
    selected = [assignments[0]]
    selected_ids = {
        tuple(candidate.candidate_id for candidate in assignments[0])
    }

    def distance(left: tuple[_Candidate, ...], right: tuple[_Candidate, ...]) -> float:
        total = 0.0
        for candidate_left, candidate_right in zip(left, right):
            delta = se3_log(se3_inv(candidate_left.transform) @ candidate_right.transform)
            total += float(np.linalg.norm(delta[:3])) / 20.0
            total += math.degrees(float(np.linalg.norm(delta[3:]))) / 45.0
        return total

    while len(selected) < maximum:
        remaining = [
            row for row in assignments
            if tuple(candidate.candidate_id for candidate in row) not in selected_ids
        ]
        if not remaining:
            break
        next_row = max(
            remaining,
            key=lambda row: (
                min(distance(row, kept) for kept in selected),
                sum(candidate.prior for candidate in row),
                tuple(candidate.candidate_id for candidate in row),
            ),
        )
        selected.append(next_row)
        selected_ids.add(tuple(candidate.candidate_id for candidate in next_row))
    return selected


def _tree_initial_poses(
    part_ids: list[str], anchor_id: str, edges: list[_Candidate]
) -> dict[str, np.ndarray]:
    poses = {anchor_id: np.eye(4)}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.source in poses and edge.target not in poses:
                poses[edge.target] = poses[edge.source] @ edge.transform
                changed = True
            elif edge.target in poses and edge.source not in poses:
                poses[edge.source] = poses[edge.target] @ se3_inv(edge.transform)
                changed = True
    for part in part_ids:
        poses.setdefault(part, np.eye(4))
    return poses


def _optimise(
    part_ids: list[str],
    anchor_id: str,
    edges: list[_Candidate],
    initial_poses: dict[str, np.ndarray],
    *,
    translation_scale_mm: float,
    rotation_scale_degrees: float,
    max_nfev: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    index = {part: offset for offset, part in enumerate(part_ids)}
    anchor_index = index[anchor_id]
    non_anchor = [part for part in part_ids if part != anchor_id]
    rotation_scale_rad = math.radians(rotation_scale_degrees)
    if translation_scale_mm <= 0 or rotation_scale_rad <= 0:
        raise ValueError("residual_scales_must_be_positive")

    x0 = np.concatenate([se3_log(initial_poses[part]) for part in non_anchor])

    def state(value: np.ndarray) -> dict[str, np.ndarray]:
        result = {anchor_id: np.eye(4)}
        for offset, part in enumerate(non_anchor):
            result[part] = se3_exp(value[offset * 6:(offset + 1) * 6])
        return result

    # Scores supply a bounded tie-breaking weight only.  The relative
    # transform residual is the constraint; a high JoinABLe score cannot hide
    # a contradictory cycle.
    priors = np.asarray([edge.prior for edge in edges], dtype=float)
    finite = priors[np.isfinite(priors)]
    base = float(np.median(np.abs(finite))) if len(finite) else 1.0
    base = max(base, 1.0)

    def residual(value: np.ndarray) -> np.ndarray:
        poses = state(value)
        rows: list[float] = []
        for edge in edges:
            error = relative_error(
                edge.transform, poses[edge.source], poses[edge.target]
            )
            # Translation is millimetres and rotation radians.  Normalising
            # them makes the robust loss meaningful across CAD scales.
            scaled = np.concatenate((
                error[:3] / translation_scale_mm,
                error[3:] / rotation_scale_rad,
            ))
            weight = math.sqrt(max(0.25, min(4.0, 1.0 + edge.prior / base)))
            rows.extend((weight * scaled).tolist())
        return np.asarray(rows, dtype=float)

    solved = least_squares(
        residual,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    poses = state(solved.x)
    rows = []
    for edge in edges:
        error = relative_error(edge.transform, poses[edge.source], poses[edge.target])
        rows.append({
            "candidate_id": edge.candidate_id,
            "source": edge.source,
            "target": edge.target,
            "translation_residual_mm": float(np.linalg.norm(error[:3])),
            "rotation_residual_degrees": float(math.degrees(np.linalg.norm(error[3:]))),
        })
    return poses, {
        "status": "converged" if solved.success else "failed",
        "message": str(solved.message),
        "nfev": int(solved.nfev),
        "cost": float(solved.cost),
        "edge_residuals": rows,
    }


def _best_cycle_edges(
    poses: dict[str, np.ndarray],
    pools: dict[tuple[str, str], list[_Candidate]],
    tree_keys: set[tuple[str, str]],
    *,
    translation_threshold_mm: float,
    rotation_threshold_degrees: float,
) -> tuple[list[_Candidate], list[dict[str, Any]]]:
    selected: list[_Candidate] = []
    audits: list[dict[str, Any]] = []
    for key, candidates in sorted(pools.items()):
        if key in tree_keys:
            continue
        best: tuple[tuple[float, float, float, str], _Candidate, np.ndarray] | None = None
        for candidate in candidates:
            error = relative_error(
                candidate.transform, poses[candidate.source], poses[candidate.target]
            )
            translation = float(np.linalg.norm(error[:3]))
            rotation = float(math.degrees(np.linalg.norm(error[3:])))
            quality = (
                max(translation / translation_threshold_mm,
                    rotation / rotation_threshold_degrees),
                translation,
                rotation,
                candidate.candidate_id,
            )
            if best is None or quality < best[0]:
                best = quality, candidate, error
        if best is None:
            continue
        quality, candidate, _ = best
        consistent = quality[0] <= 1.0
        audit = {
            "pair": list(key),
            "best_candidate_id": candidate.candidate_id,
            "translation_residual_mm": quality[1],
            "rotation_residual_degrees": quality[2],
            "consistent": consistent,
        }
        audits.append(audit)
        if consistent:
            selected.append(candidate)
    return selected, audits


def _serialise_candidate(candidate: _Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "target": candidate.target,
        "prior": candidate.prior,
        "source_rank": candidate.source_rank,
        "provenance": candidate.provenance,
    }


def _serialise_poses(poses: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {part: matrix.tolist() for part, matrix in poses.items()}


def solve_bounded_global_pose(
    part_ids: Iterable[str],
    candidate_pools: Iterable[dict[str, Any]],
    *,
    anchor_id: str | None = None,
    max_candidates_per_pair: int = 3,
    max_topologies: int = 8,
    max_hypotheses: int = 96,
    translation_scale_mm: float = 2.0,
    rotation_scale_degrees: float = 5.0,
    cycle_translation_threshold_mm: float = 2.0,
    cycle_rotation_threshold_degrees: float = 5.0,
    max_nfev: int = 200,
    exact_validator: Callable[[dict[str, np.ndarray]], dict[str, Any]] | None = None,
    validate_top_n: int = 0,
) -> dict[str, Any]:
    """Solve bounded pose hypotheses and emit conservative audit records.

    ``exact_validator`` is intentionally an injected callback.  This keeps the
    discrete/continuous solver independent from a specific STEP loader while
    allowing callers to run OCCT Boolean collision checks only on top-N rows.
    Its return must state ``status`` as ``valid``, ``failed`` or ``uncertain``;
    unknown/failed validation never becomes an automatic acceptance.
    """

    started = time.monotonic()
    parts = [str(part) for part in part_ids]
    if not 1 <= len(parts) <= 5 or len(set(parts)) != len(parts):
        raise ValueError("solver_requires_1_to_5_unique_part_ids")
    anchor = str(anchor_id or parts[0])
    if anchor not in parts:
        raise ValueError("anchor_id_not_in_part_ids")
    if min(max_candidates_per_pair, max_topologies, max_hypotheses, max_nfev) < 1:
        raise ValueError("bounded_search_limits_must_be_positive")

    raw_pools = _normalise_pools(parts, candidate_pools)
    pools = {
        key: _deduplicate_pool(values, max_candidates_per_pair)
        for key, values in raw_pools.items()
        if values
    }
    topology_frontier = _topologies(parts, pools, max_topologies)
    if not topology_frontier:
        return {
            "schema_version": "bounded_global_pose.v1",
            "status": "insufficient_connectivity",
            "accepted": False,
            "review_required": True,
            "reason": "No connected spanning topology can be formed from candidate pools.",
            "part_ids": parts,
            "candidate_pool_count": len(pools),
            "hypotheses": [],
        }

    hypotheses: list[dict[str, Any]] = []
    # Distribute the global search budget across topologies.  Previously a
    # lexicographic product could consume the entire budget on the first tree,
    # silently preventing alternate valid connection structures from being
    # evaluated.
    per_topology_limit = max(1, math.ceil(max_hypotheses / len(topology_frontier)))
    for topology_index, topology in enumerate(topology_frontier):
        candidate_lists = [pools[key] for key in topology]
        assignments = _bounded_assignments(candidate_lists, per_topology_limit)
        for assignment in assignments:
            tree_edges = list(assignment)
            initial = _tree_initial_poses(parts, anchor, tree_edges)
            poses, first_pass = _optimise(
                parts, anchor, tree_edges, initial,
                translation_scale_mm=translation_scale_mm,
                rotation_scale_degrees=rotation_scale_degrees,
                max_nfev=max_nfev,
            )
            cycle_edges, cycle_audit = _best_cycle_edges(
                poses, pools, set(topology),
                translation_threshold_mm=cycle_translation_threshold_mm,
                rotation_threshold_degrees=cycle_rotation_threshold_degrees,
            )
            all_edges = tree_edges + cycle_edges
            if cycle_edges:
                poses, final_pass = _optimise(
                    parts, anchor, all_edges, poses,
                    translation_scale_mm=translation_scale_mm,
                    rotation_scale_degrees=rotation_scale_degrees,
                    max_nfev=max_nfev,
                )
            else:
                final_pass = first_pass
            residual_rows = final_pass["edge_residuals"]
            max_translation = max(
                (row["translation_residual_mm"] for row in residual_rows), default=0.0
            )
            max_rotation = max(
                (row["rotation_residual_degrees"] for row in residual_rows), default=0.0
            )
            all_cycles_consistent = bool(cycle_audit) and all(
                row["consistent"] for row in cycle_audit
            )
            hypotheses.append({
                "hypothesis_id": f"topology_{topology_index:02d}_choice_{len(hypotheses):03d}",
                "tree_pairs": [list(key) for key in topology],
                "tree_candidates": [_serialise_candidate(row) for row in tree_edges],
                "selected_cycle_candidates": [_serialise_candidate(row) for row in cycle_edges],
                "cycle_audit": cycle_audit,
                "independent_cycle_count": len(cycle_audit),
                "all_independent_cycles_consistent": all_cycles_consistent,
                "optimizer": final_pass,
                "max_translation_residual_mm": max_translation,
                "max_rotation_residual_degrees": max_rotation,
                "part_poses": _serialise_poses(poses),
                "exact_validation": {"status": "not_checked"},
                # A fit to pair constraints alone never establishes source or
                # functional correctness.  This remains review-only until an
                # outer conservative gate combines exact validation and any
                # independent semantic evidence.
                "accepted": False,
                "review_required": True,
            })

    def ranking(row: dict[str, Any]) -> tuple[Any, ...]:
        validation = row["exact_validation"].get("status")
        physical_rank = {"valid": 0, "not_checked": 1, "uncertain": 2, "failed": 3}.get(
            validation, 4
        )
        cycle_rank = 0 if row["all_independent_cycles_consistent"] else 1
        return (
            physical_rank,
            cycle_rank,
            row["max_translation_residual_mm"],
            row["max_rotation_residual_degrees"],
            row["hypothesis_id"],
        )

    hypotheses.sort(key=ranking)
    if exact_validator is not None:
        for row in hypotheses[:max(0, int(validate_top_n))]:
            matrix_poses = {
                part: np.asarray(matrix, dtype=float)
                for part, matrix in row["part_poses"].items()
            }
            try:
                validation = dict(exact_validator(matrix_poses))
            except Exception as exc:  # Validator failure is evidence, not success.
                validation = {"status": "uncertain", "reason": f"validator_error:{type(exc).__name__}:{exc}"}
            if validation.get("status") not in {"valid", "failed", "uncertain"}:
                validation = {
                    "status": "uncertain",
                    "reason": "validator_must_return_valid_failed_or_uncertain",
                    "validator_output": validation,
                }
            row["exact_validation"] = validation
        hypotheses.sort(key=ranking)

    status = "review_required"
    if not hypotheses:
        status = "no_bounded_hypothesis"
    return {
        "schema_version": "bounded_global_pose.v1",
        "status": status,
        "accepted": False,
        "review_required": True,
        "reason": (
            "Global pose candidates are geometrically optimized but require "
            "outer conservative acceptance gates; pose fit is not semantic proof."
        ),
        "part_ids": parts,
        "anchor_id": anchor,
        "candidate_pool_count": len(pools),
        "topology_count": len(topology_frontier),
        "hypothesis_count": len(hypotheses),
        "limits": {
            "max_candidates_per_pair": max_candidates_per_pair,
            "max_topologies": max_topologies,
            "max_hypotheses": max_hypotheses,
            "max_nfev": max_nfev,
        },
        "hypotheses": hypotheses,
        "runtime_ms": (time.monotonic() - started) * 1000.0,
    }
