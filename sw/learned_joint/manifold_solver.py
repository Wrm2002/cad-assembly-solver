"""Bounded discrete + continuous multi-part constraint-manifold solver.

Every active edge selects one learned local-interface hypothesis.  The
continuous stage optimizes all non-anchor poses jointly on SE(3), but only the
constrained components of each local-frame error are penalized.  Consequently
an axial or planar relation remains a manifold rather than a hard-coded 4x4
answer.  Optional geometry residuals and an exact validator are injected by
the caller so this module never needs names, case IDs, or mechanical labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.optimize import least_squares

from global_pose_solver.se3_manifold import se3_exp, se3_inv, se3_log
from learned_joint.precision_pose_validator import filter_precision_evidence


@dataclass(frozen=True)
class _Factor:
    source: str
    target: str
    candidate_id: str
    frame_a: np.ndarray
    frame_b: np.ndarray
    initial: np.ndarray
    free_mask: np.ndarray
    confidence: float
    manifold_type: str
    provenance: dict[str, Any]

    @property
    def constrained_mask(self) -> np.ndarray:
        return 1.0 - self.free_mask


def _proper_matrix(value: Any, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field}_must_be_finite_4x4")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{field}_invalid_homogeneous_row")
    # A 3x3 determinant is trivial to evaluate directly.  Besides avoiding a
    # needless LAPACK dispatch during pool parsing, this keeps the validation
    # path stable in the local CAD runtime where ``np.linalg.det`` has caused
    # a native-library abort for otherwise valid 3x3 arrays.
    rotation = matrix[:3, :3]
    determinant = float(np.dot(rotation[0], np.cross(rotation[1], rotation[2])))
    if abs(determinant - 1.0) > 1e-4:
        raise ValueError(f"{field}_must_be_proper_rigid")
    return matrix.copy()


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


def _parse_factor(row: dict[str, Any], index: int) -> _Factor:
    source, target = str(row.get("source", row.get("src", ""))), str(row.get("target", row.get("dst", "")))
    free = np.asarray(row.get("free_dof_mask"), dtype=float)
    if free.shape != (6,) or not np.all(np.isin(free, [0, 1])):
        raise ValueError("free_dof_mask_must_have_six_binary_values")
    return _Factor(
        source=source,
        target=target,
        candidate_id=str(row.get("candidate_id", f"candidate_{index:06d}")),
        frame_a=_proper_matrix(row.get("frame_a"), "frame_a"),
        frame_b=_proper_matrix(row.get("frame_b"), "frame_b"),
        initial=_proper_matrix(row.get("initial_pose_b_in_a"), "initial_pose_b_in_a"),
        free_mask=free,
        confidence=float(row.get("confidence", 0.0)),
        manifold_type=str(row.get("manifold_type", "direction_coincidence")),
        provenance=dict(row.get("provenance") or {}),
    )


def _normalise_pools(parts: list[str], values: Iterable[dict[str, Any]], maximum: int) -> dict[tuple[str, str], list[_Factor]]:
    allowed = set(parts)
    pools: dict[tuple[str, str], list[_Factor]] = {}
    index = 0
    for value in values:
        rows = value.get("candidates") if isinstance(value, dict) else None
        rows = rows if isinstance(rows, list) else [value]
        parent_source = str(value.get("source", value.get("src", "")))
        parent_target = str(value.get("target", value.get("dst", "")))
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            candidate.setdefault("source", parent_source)
            candidate.setdefault("target", parent_target)
            try:
                factor = _parse_factor(candidate, index)
            except (TypeError, ValueError):
                index += 1
                continue
            index += 1
            if factor.source not in allowed or factor.target not in allowed or factor.source == factor.target:
                continue
            pools.setdefault(_pair_key(factor.source, factor.target), []).append(factor)
    for key, rows in tuple(pools.items()):
        rows.sort(key=lambda row: (
            not bool(row.provenance.get("pair_exact_collision_free")),
            -row.confidence,
            row.candidate_id,
        ))
        deduplicated = []
        seen = set()
        for row in rows:
            learned_initial = bool(row.provenance.get("learned_pose_initial"))
            signature = (
                row.manifold_type,
                tuple(row.free_mask.tolist()),
                tuple(np.round(row.frame_a.reshape(-1), 5)),
                tuple(np.round(row.frame_b.reshape(-1), 5)),
                # Analytic manifold rows with identical frames are genuinely
                # redundant.  A learned sidecar row also carries a distinct
                # full-Pose seed/soft prior, so omitting its initial transform
                # here used to erase every learned proposal before search.
                tuple(np.round(row.initial.reshape(-1), 5)) if learned_initial else (),
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduplicated.append(row)
            if len(deduplicated) >= maximum:
                break
        pools[key] = deduplicated
    return {key: rows for key, rows in pools.items() if rows}


def _connected(parts: list[str], keys: Iterable[tuple[str, str]]) -> bool:
    if not parts:
        return False
    parent = {part: part for part in parts}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for source, target in keys:
        a, b = find(source), find(target)
        if a != b:
            parent[b] = a
    return len({find(part) for part in parts}) == 1


def _topologies(parts: list[str], pools: dict[tuple[str, str], list[_Factor]], maximum: int) -> list[tuple[tuple[str, str], ...]]:
    if len(parts) == 1:
        return [tuple()]
    rows = []
    for keys in itertools.combinations(sorted(pools), len(parts) - 1):
        if _connected(parts, keys):
            rows.append((sum(max(row.confidence for row in pools[key]) for key in keys), keys))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [keys for _, keys in rows[:maximum]]


def _initial_poses(parts: list[str], anchor: str, factors: Iterable[_Factor]) -> dict[str, np.ndarray]:
    poses = {anchor: np.eye(4)}
    changed = True
    while changed:
        changed = False
        for factor in factors:
            if factor.source in poses and factor.target not in poses:
                poses[factor.target] = poses[factor.source] @ factor.initial
                changed = True
            elif factor.target in poses and factor.source not in poses:
                poses[factor.source] = poses[factor.target] @ se3_inv(factor.initial)
                changed = True
    for part in parts:
        poses.setdefault(part, np.eye(4))
    return poses


def _bounded_assignments(candidate_lists: list[list[_Factor]], maximum: int) -> list[tuple[_Factor, ...]]:
    """Keep high-prior and geometrically diverse discrete combinations."""

    total = math.prod(len(values) for values in candidate_lists)
    pool_limit = max(256, maximum * 12)
    if total <= pool_limit:
        assignments = list(itertools.product(*candidate_lists))
    else:
        index_rows: set[tuple[int, ...]] = set()
        dimensions = len(candidate_lists)

        def add(indices: tuple[int, ...]) -> None:
            if all(0 <= value < len(candidate_lists[axis]) for axis, value in enumerate(indices)):
                index_rows.add(indices)

        add((0,) * dimensions)
        # Single-edge sweeps ensure that a lower-ranked offset/polarity on one
        # connection survives even when the other edges keep their best seed.
        for axis, values in enumerate(candidate_lists):
            for index in range(1, len(values)):
                row = [0] * dimensions
                row[axis] = index
                add(tuple(row))
        # Deterministic mixed corners cover complementary choices across
        # edges without materialising the full Cartesian product.
        longest = max(len(values) for values in candidate_lists)
        for step in range(longest * 3):
            add(tuple(
                (step * (2 * axis + 1) + axis) % len(candidate_lists[axis])
                for axis in range(dimensions)
            ))

        # Add k-best confidence combinations with a standard monotone heap.
        start = (0,) * dimensions

        def priority(indices: tuple[int, ...]) -> float:
            return -sum(
                candidate_lists[axis][value].confidence
                for axis, value in enumerate(indices)
            )

        heap = [(priority(start), start)]
        heap_seen = {start}
        while heap and len(index_rows) < pool_limit:
            _, indices = heapq.heappop(heap)
            add(indices)
            for axis in range(dimensions):
                if indices[axis] + 1 >= len(candidate_lists[axis]):
                    continue
                neighbor = list(indices)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if neighbor not in heap_seen:
                    heap_seen.add(neighbor)
                    heapq.heappush(heap, (priority(neighbor), neighbor))
        assignments = [
            tuple(candidate_lists[axis][value] for axis, value in enumerate(indices))
            for indices in index_rows
        ]
    assignments.sort(key=lambda rows: (
        -sum(row.confidence for row in rows),
        tuple(row.candidate_id for row in rows),
    ))
    if len(assignments) <= maximum:
        return assignments
    selected = [assignments[0]]
    selected_keys = {tuple(row.candidate_id for row in assignments[0])}

    locations = {
        factor.candidate_id: (axis, index)
        for axis, values in enumerate(candidate_lists)
        for index, factor in enumerate(values)
    }
    pair_distances = []
    for values in candidate_lists:
        matrix = np.zeros((len(values), len(values)), dtype=float)
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                delta = se3_log(
                    se3_inv(values[left_index].initial) @ values[right_index].initial
                )
                value = (
                    1.0
                    + float(np.linalg.norm(delta[:3])) / 20.0
                    + math.degrees(float(np.linalg.norm(delta[3:]))) / 45.0
                )
                matrix[left_index, right_index] = value
                matrix[right_index, left_index] = value
        pair_distances.append(matrix)

    def distance(left: tuple[_Factor, ...], right: tuple[_Factor, ...]) -> float:
        total = 0.0
        for axis, (a, b) in enumerate(zip(left, right)):
            _, a_index = locations[a.candidate_id]
            _, b_index = locations[b.candidate_id]
            total += float(pair_distances[axis][a_index, b_index])
        return total

    remaining = {
        tuple(factor.candidate_id for factor in row): row
        for row in assignments
        if tuple(factor.candidate_id for factor in row) not in selected_keys
    }
    minimum_distance = {
        key: distance(row, selected[0]) for key, row in remaining.items()
    }
    while len(selected) < maximum and remaining:
        chosen_key, chosen = max(remaining.items(), key=lambda item: (
            minimum_distance[item[0]],
            sum(factor.confidence for factor in item[1]),
            item[0],
        ))
        selected.append(chosen)
        selected_keys.add(chosen_key)
        remaining.pop(chosen_key)
        minimum_distance.pop(chosen_key)
        for key, row in remaining.items():
            minimum_distance[key] = min(
                minimum_distance[key], distance(row, chosen)
            )
    return selected


def _closure_ranked_assignments(
    candidate_lists: list[list[_Factor]],
    maximum: int,
    *,
    parts: list[str],
    anchor: str,
    pools: dict[tuple[str, str], list[_Factor]],
    topology: tuple[tuple[str, str], ...],
    translation_scale_mm: float,
    rotation_scale_degrees: float,
    enumeration_limit: int,
) -> list[tuple[_Factor, ...]] | None:
    """Select a bounded frontier by cheap multi-edge closure before solving.

    Repeated-hole and slot interfaces often create many symmetry-equivalent
    rigid candidates.  Sampling their Cartesian product by confidence alone
    can miss the only combination that agrees with the remaining pair edges.
    When the exact discrete product is still small, evaluate local-frame
    closure analytically and send only the best ``maximum`` combinations to
    the expensive continuous/mesh stage.  This changes selection quality,
    not the number of optimized hypotheses.
    """

    total = math.prod(len(values) for values in candidate_lists)
    tree_keys = set(topology)
    cycle_pools = []
    for key, rows in sorted(pools.items()):
        if key in tree_keys:
            continue
        # Closure preselection is most useful with the strongest available
        # interface representation.  If a full compound factor exists for a
        # pair, weaker single-axis/plane factors cannot disambiguate symmetry
        # and only add correlated work.
        maximum_strength = max(
            (int(np.sum(row.constrained_mask)) for row in rows), default=0
        )
        cycle_pools.append([
            row for row in rows
            if int(np.sum(row.constrained_mask)) == maximum_strength
        ])
    if (
        total < 1
        or total > max(0, int(enumeration_limit))
        or not cycle_pools
    ):
        return None

    scored: list[tuple[tuple[Any, ...], tuple[_Factor, ...]]] = []
    for assignment in itertools.product(*candidate_lists):
        poses = _initial_poses(parts, anchor, assignment)
        compatible_strength = 0.0
        closure_error = 0.0
        compatible_pair_count = 0
        for candidates in cycle_pools:
            values = []
            for factor in candidates:
                strength_count = int(np.sum(factor.constrained_mask))
                if strength_count == 6:
                    world_a = poses[factor.source] @ factor.frame_a
                    world_b = poses[factor.target] @ factor.frame_b
                    delta = se3_inv(world_a) @ world_b
                    cosine = max(
                        -1.0,
                        min(1.0, (float(np.trace(delta[:3, :3])) - 1.0) * 0.5),
                    )
                    normalized = (
                        float(np.linalg.norm(delta[:3, 3]))
                        / translation_scale_mm
                        + math.degrees(math.acos(cosine))
                        / rotation_scale_degrees
                    )
                else:
                    error = factor_error(factor, poses)
                    normalized = (
                        float(np.linalg.norm(error[:3]))
                        / translation_scale_mm
                        + math.degrees(float(np.linalg.norm(error[3:])))
                        / rotation_scale_degrees
                    )
                strength = float(strength_count) / 6.0
                values.append((normalized, strength, factor.candidate_id))
            consistent = [value for value in values if value[0] <= 1.0]
            if consistent:
                # Prefer a compatible factor that constrains more of SE(3).
                # A full pattern/slot bundle is more discriminating than a
                # trivially compatible single axis or plane.
                normalized, strength, _ = min(
                    consistent, key=lambda value: (-value[1], value[0], value[2])
                )
                compatible_pair_count += 1
                compatible_strength += strength
                closure_error += normalized
            elif values:
                normalized, _, _ = min(values)
                closure_error += 1.0 + min(normalized, 100.0)
        prior = float(sum(factor.confidence for factor in assignment))
        pair_exact_failed = sum(
            factor.provenance.get("pair_exact_status") == "failed"
            for factor in assignment
        )
        pair_exact_valid = sum(
            bool(factor.provenance.get("pair_exact_collision_free"))
            for factor in assignment
        )
        candidate_ids = tuple(factor.candidate_id for factor in assignment)
        key = (
            pair_exact_failed,
            -pair_exact_valid,
            -compatible_pair_count,
            -compatible_strength,
            closure_error,
            -prior,
            candidate_ids,
        )
        scored.append((key, assignment))
    scored.sort(key=lambda item: item[0])
    return [assignment for _, assignment in scored[: max(1, int(maximum))]]


def factor_error(factor: _Factor, poses: dict[str, np.ndarray]) -> np.ndarray:
    """Projected local-frame error; free dimensions are identically zero."""

    world_a = poses[factor.source] @ factor.frame_a
    world_b = poses[factor.target] @ factor.frame_b
    return factor.constrained_mask * se3_log(se3_inv(world_a) @ world_b)


def learned_pose_prior_error(factor: _Factor, poses: dict[str, np.ndarray]) -> np.ndarray:
    """Soft residual for a learned full relative pose.

    This is deliberately separate from the interface-frame factor.  It is used
    only when a learned pose head supplied ``initial``; contact, cycle closure,
    and exact OCCT validation can still reject or move the candidate.  Free
    joint dimensions are never artificially fixed by this term.
    """

    actual = se3_inv(poses[factor.source]) @ poses[factor.target]
    return factor.constrained_mask * se3_log(se3_inv(factor.initial) @ actual)


def _optimise(
    parts: list[str],
    anchor: str,
    factors: list[_Factor],
    initial: dict[str, np.ndarray],
    *,
    translation_scale_mm: float,
    rotation_scale_degrees: float,
    max_nfev: int,
    translation_bound_mm: float,
    rotation_bound_degrees: float,
    geometry_residual_provider: Callable[..., np.ndarray] | None,
    learned_pose_prior_weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    movable = [part for part in parts if part != anchor]
    x0 = np.zeros(6 * len(movable), dtype=float)
    rotation_scale = math.radians(rotation_scale_degrees)
    rotation_bound = math.radians(rotation_bound_degrees)

    def materialise(value: np.ndarray) -> dict[str, np.ndarray]:
        poses = {anchor: np.eye(4)}
        for offset, part in enumerate(movable):
            poses[part] = se3_exp(value[offset * 6:(offset + 1) * 6]) @ initial[part]
        return poses

    def residual(value: np.ndarray) -> np.ndarray:
        poses = materialise(value)
        rows = []
        for factor in factors:
            error = factor_error(factor, poses)
            weight = math.sqrt(max(0.25, min(4.0, 0.5 + factor.confidence)))
            rows.extend((weight * np.concatenate((
                error[:3] / translation_scale_mm,
                error[3:] / rotation_scale,
            ))).tolist())
            if (
                learned_pose_prior_weight > 0.0
                and bool(factor.provenance.get("learned_pose_initial"))
            ):
                prior = learned_pose_prior_error(factor, poses)
                learned_score = float(factor.provenance.get("learned_pose_score", 0.0))
                learned_confidence = 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, learned_score))))
                prior_weight = math.sqrt(float(learned_pose_prior_weight) * (0.25 + learned_confidence))
                rows.extend((prior_weight * np.concatenate((
                    prior[:3] / translation_scale_mm,
                    prior[3:] / rotation_scale,
                ))).tolist())
        if geometry_residual_provider is not None:
            extra = np.asarray(
                geometry_residual_provider(poses, factors), dtype=float
            ).reshape(-1)
            if not np.all(np.isfinite(extra)):
                raise ValueError("geometry_residual_provider_returned_nonfinite_values")
            rows.extend(extra.tolist())
        return np.asarray(rows, dtype=float)

    if not movable:
        poses = materialise(x0)
        return poses, {"status": "converged", "nfev": 0, "cost": 0.0, "message": "anchor_only"}
    result = least_squares(
        residual,
        x0,
        method="trf",
        # The dense exact solver dispatches to a native SVD backend that is
        # unstable in the current Windows scientific runtime.  LSMR solves
        # the same trust-region subproblem iteratively and is also the better
        # scaling choice once learned sidecar seeds enlarge the residual set.
        tr_solver="lsmr",
        bounds=(
            np.tile(
                [-translation_bound_mm] * 3 + [-rotation_bound] * 3,
                len(movable),
            ),
            np.tile(
                [translation_bound_mm] * 3 + [rotation_bound] * 3,
                len(movable),
            ),
        ),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )
    return materialise(result.x), {
        "status": "converged" if result.success else "uncertain",
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "message": str(result.message),
    }


def _factor_audit(factor: _Factor, poses: dict[str, np.ndarray]) -> dict[str, Any]:
    error = factor_error(factor, poses)
    result = {
        "candidate_id": factor.candidate_id,
        "source": factor.source,
        "target": factor.target,
        "manifold_type": factor.manifold_type,
        "free_dof_mask": factor.free_mask.astype(int).tolist(),
        "projected_translation_residual_mm": float(np.linalg.norm(error[:3])),
        "projected_rotation_residual_degrees": float(math.degrees(np.linalg.norm(error[3:]))),
    }
    precision_evidence = filter_precision_evidence(
        factor.manifold_type, factor.provenance
    )
    if precision_evidence:
        # Strictly filtered numeric/boolean geometry only.  In particular,
        # never serialise names, roles, family/case labels, or free-form text.
        result["precision_evidence"] = precision_evidence
    return result


def _best_cycle_factors(
    poses: dict[str, np.ndarray],
    pools: dict[tuple[str, str], list[_Factor]],
    tree_keys: set[tuple[str, str]],
    translation_scale_mm: float,
    rotation_scale_degrees: float,
) -> tuple[list[_Factor], list[dict[str, Any]]]:
    selected, audits = [], []
    for key, candidates in sorted(pools.items()):
        if key in tree_keys:
            continue
        scored = []
        for factor in candidates:
            audit = _factor_audit(factor, poses)
            score = (
                audit["projected_translation_residual_mm"] / translation_scale_mm
                + audit["projected_rotation_residual_degrees"] / rotation_scale_degrees
            )
            scored.append((score, -factor.confidence, factor.candidate_id, factor, audit))
        if scored:
            score, _, _, factor, audit = min(scored, key=lambda row: row[:3])
            audit["consistent"] = bool(score <= 1.0)
            audit["normalized_closure_error"] = float(score)
            if audit["consistent"]:
                selected.append(factor)
            audits.append(audit)
    return selected, audits


def _serialise_poses(poses: dict[str, np.ndarray]) -> dict[str, list[list[float]]]:
    return {part: matrix.tolist() for part, matrix in poses.items()}


def solve_manifold_pose_graph(
    part_ids: Iterable[str],
    candidate_pools: Iterable[dict[str, Any]],
    *,
    anchor_id: str | None = None,
    max_candidates_per_pair: int = 4,
    max_topologies: int = 8,
    max_hypotheses: int = 96,
    translation_scale_mm: float = 2.0,
    rotation_scale_degrees: float = 5.0,
    max_nfev: int = 250,
    translation_bound_mm: float = 250.0,
    rotation_bound_degrees: float = 180.0,
    geometry_residual_provider: Callable[..., np.ndarray] | None = None,
    learned_pose_prior_weight: float = 0.0,
    exact_validator: Callable[[dict[str, np.ndarray]], dict[str, Any]] | None = None,
    validate_top_n: int = 0,
    closure_enumeration_limit: int = 50000,
) -> dict[str, Any]:
    """Jointly solve a known 2--5 part group from top-k interface manifolds."""

    started = time.monotonic()
    parts = [str(part) for part in part_ids]
    if not 1 <= len(parts) <= 5 or len(set(parts)) != len(parts):
        raise ValueError("solver_requires_1_to_5_unique_parts")
    anchor = str(anchor_id or parts[0])
    if anchor not in parts:
        raise ValueError("anchor_not_in_parts")
    if min(max_candidates_per_pair, max_topologies, max_hypotheses, max_nfev) < 1:
        raise ValueError("search_limits_must_be_positive")
    if translation_bound_mm <= 0 or rotation_bound_degrees <= 0:
        raise ValueError("continuous_search_bounds_must_be_positive")
    if learned_pose_prior_weight < 0:
        raise ValueError("learned_pose_prior_weight_must_be_nonnegative")
    if closure_enumeration_limit < 0:
        raise ValueError("closure_enumeration_limit_must_be_nonnegative")
    pools = _normalise_pools(parts, candidate_pools, max_candidates_per_pair)
    topologies = _topologies(parts, pools, max_topologies)
    if not topologies:
        return {
            "schema_version": "constraint_manifold_global_pose.v1",
            "status": "insufficient_connectivity",
            "accepted": False,
            "review_required": True,
            "hypotheses": [],
            "reason": "No connected topology exists in the learned top-k pools.",
        }

    hypotheses = []
    per_topology = max(1, math.ceil(max_hypotheses / len(topologies)))
    for topology_index, topology in enumerate(topologies):
        candidate_lists = [pools[key] for key in topology]
        assignments = _closure_ranked_assignments(
            candidate_lists,
            per_topology,
            parts=parts,
            anchor=anchor,
            pools=pools,
            topology=topology,
            translation_scale_mm=translation_scale_mm,
            rotation_scale_degrees=rotation_scale_degrees,
            enumeration_limit=closure_enumeration_limit,
        )
        if assignments is None:
            assignments = _bounded_assignments(candidate_lists, per_topology)
        for choice_index, assignment in enumerate(assignments):
            tree = list(assignment)
            initial = _initial_poses(parts, anchor, tree)
            poses, first = _optimise(
                parts,
                anchor,
                tree,
                initial,
                translation_scale_mm=translation_scale_mm,
                rotation_scale_degrees=rotation_scale_degrees,
                max_nfev=max_nfev,
                translation_bound_mm=translation_bound_mm,
                rotation_bound_degrees=rotation_bound_degrees,
                geometry_residual_provider=geometry_residual_provider,
                learned_pose_prior_weight=learned_pose_prior_weight,
            )
            cycles, cycle_before = _best_cycle_factors(
                poses,
                pools,
                set(topology),
                translation_scale_mm,
                rotation_scale_degrees,
            )
            active = tree + cycles
            if cycles:
                poses, final = _optimise(
                    parts,
                    anchor,
                    active,
                    poses,
                    translation_scale_mm=translation_scale_mm,
                    rotation_scale_degrees=rotation_scale_degrees,
                    max_nfev=max_nfev,
                    translation_bound_mm=translation_bound_mm,
                    rotation_bound_degrees=rotation_bound_degrees,
                    geometry_residual_provider=geometry_residual_provider,
                    learned_pose_prior_weight=learned_pose_prior_weight,
                )
            else:
                final = first
            factor_rows = [_factor_audit(factor, poses) for factor in active]
            underconstrained = sorted({
                dof
                for factor in active
                for dof, free in zip(("tx", "ty", "tz", "rx", "ry", "rz"), factor.free_mask)
                if free
            })
            hypotheses.append({
                "hypothesis_id": f"topology_{topology_index:02d}_choice_{choice_index:03d}",
                "topology_pairs": [list(key) for key in topology],
                "tree_candidate_ids": [factor.candidate_id for factor in tree],
                "cycle_candidate_ids": [factor.candidate_id for factor in cycles],
                "consistent_cycle_count": len(cycles),
                "cycle_residuals_before_joint_optimization": cycle_before,
                "factor_residuals": factor_rows,
                "optimizer": final,
                "prior_sum": float(sum(factor.confidence for factor in active)),
                "part_poses": _serialise_poses(poses),
                "geometry_residual_audit": (
                    geometry_residual_provider.audit(poses, active)
                    if geometry_residual_provider is not None
                    and hasattr(geometry_residual_provider, "audit")
                    else None
                ),
                "unresolved_manifold_dofs": underconstrained,
                "exact_validation": {"status": "not_checked"},
                "accepted": False,
                "review_required": True,
            })
    hypotheses.sort(key=lambda row: (
        -row["consistent_cycle_count"],
        row["optimizer"]["cost"],
        -row["prior_sum"],
        row["hypothesis_id"],
    ))
    if exact_validator is not None:
        for row in hypotheses[: max(0, int(validate_top_n))]:
            poses = {part: np.asarray(matrix, dtype=float) for part, matrix in row["part_poses"].items()}
            try:
                result = exact_validator(poses)
                row["exact_validation"] = result if isinstance(result, dict) else {"status": "uncertain", "reason": "invalid_validator_result"}
            except Exception as exc:
                row["exact_validation"] = {"status": "uncertain", "reason": f"{type(exc).__name__}:{exc}"}
    hypotheses.sort(key=lambda row: (
        {"valid": 0, "not_checked": 1, "uncertain": 2, "failed": 3}.get(row["exact_validation"].get("status"), 4),
        -row["consistent_cycle_count"],
        row["optimizer"]["cost"],
        -row["prior_sum"],
    ))
    return {
        "schema_version": "constraint_manifold_global_pose.v1",
        "status": "review_required",
        "accepted": False,
        "review_required": True,
        "part_ids": parts,
        "anchor_id": anchor,
        "candidate_pool_count": len(pools),
        "topology_count": len(topologies),
        "hypothesis_count": len(hypotheses),
        "hypotheses": hypotheses,
        "runtime_seconds": time.monotonic() - started,
        "closure_enumeration_limit": int(closure_enumeration_limit),
        "acceptance_boundary": (
            "Constraint closure and pose feasibility are necessary geometric evidence; "
            "they do not independently establish functional or semantic correctness."
        ),
    }
