"""Compose pair-pose seeds into an auditable multi-part group pose."""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import matrix_to_placement


@dataclass(frozen=True)
class PairPoseSeed:
    """Transform mapping ``part_b`` local coordinates into ``part_a``."""

    part_a: str
    part_b: str
    transform_b_to_a: tuple[tuple[float, float, float, float], ...]
    source: str
    score: float | None = None

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.asarray(self.transform_b_to_a, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("pair pose transform must be 4x4")
        return matrix


def _part_name(value: str | Path) -> str:
    return Path(str(value)).name


def load_joinable_pair_pose(path: str | Path) -> PairPoseSeed | None:
    """Load the best exact-collision-free pose from an E2E v2 report."""

    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    pose = (payload.get("pose_search") or {}).get(
        "best_exact_collision_free"
    )
    if not pose:
        return None
    matrix = np.asarray(pose.get("transform"), dtype=float)
    if matrix.shape != (4, 4) or abs(np.linalg.det(matrix[:3, :3]) - 1.0) > 1e-5:
        return None
    return PairPoseSeed(
        part_a=_part_name(payload["part_a_fixed"]),
        part_b=_part_name(payload["part_b_moving"]),
        transform_b_to_a=tuple(tuple(float(value) for value in row) for row in matrix),
        source=str(report_path),
        score=float((pose.get("evaluation") or {}).get("cost", 0.0)),
    )


def load_joinable_pair_pose_candidates(
    path: str | Path,
    *,
    limit: int = 8,
    include_manifold_initials: bool = True,
) -> list[PairPoseSeed]:
    """Load bounded pair-pose proposals from one JoinABLe report.

    Exact collision-free search results retain priority.  A ``--no-search``
    report may contain only analytic constraint-manifold initial poses; those
    are admitted as *proposals* so the legacy group solver can perform its own
    closure and exact validation instead of silently treating the pair as
    missing.  The source suffix makes that weaker provenance auditable.
    """

    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = (payload.get("pose_search") or {}).get("results") or []
    candidates = []
    for index, row in enumerate(rows):
        exact = row.get("exact_collision") or {}
        if exact.get("status") != "success" or exact.get("collisions"):
            continue
        matrix = np.asarray(row.get("transform"), dtype=float)
        if (
            matrix.shape != (4, 4)
            or abs(np.linalg.det(matrix[:3, :3]) - 1.0) > 1e-5
        ):
            continue
        candidates.append(PairPoseSeed(
            part_a=_part_name(payload["part_a_fixed"]),
            part_b=_part_name(payload["part_b_moving"]),
            transform_b_to_a=tuple(
                tuple(float(value) for value in values) for values in matrix
            ),
            source=f"{report_path}#pose_result={index}",
            score=float((row.get("evaluation") or {}).get("cost", 0.0)),
        ))
        if len(candidates) >= max(1, int(limit)):
            break
    if include_manifold_initials and len(candidates) < max(1, int(limit)):
        manifold_rows = (payload.get("joint_hypotheses") or {}).get("rows") or []
        for index, row in enumerate(manifold_rows):
            matrix = np.asarray(row.get("initial_pose_b_in_a"), dtype=float)
            if (
                matrix.shape != (4, 4)
                or not np.all(np.isfinite(matrix))
                or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
                or abs(np.linalg.det(matrix[:3, :3]) - 1.0) > 1e-5
            ):
                continue
            confidence = float(row.get("confidence", 0.0) or 0.0)
            candidates.append(PairPoseSeed(
                part_a=_part_name(payload["part_a_fixed"]),
                part_b=_part_name(payload["part_b_moving"]),
                transform_b_to_a=tuple(
                    tuple(float(value) for value in values) for values in matrix
                ),
                source=(
                    f"{report_path}#manifold_initial={index};"
                    "proposal_only=true;exact_collision=not_checked"
                ),
                # Pair seed scores are costs (lower is preferred).
                score=float(1.0 - max(0.0, min(1.0, confidence))),
            ))
            if len(candidates) >= max(1, int(limit)):
                break
    return candidates


def load_joinable_pair_pose_directory(
    root: str | Path,
) -> list[PairPoseSeed]:
    root = Path(root).resolve()
    paths = (
        [root] if root.is_file()
        else sorted(root.rglob("joinable_e2e_result.json"))
    )
    seeds = []
    for path in paths:
        try:
            seed = load_joinable_pair_pose(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if seed is not None:
            seeds.append(seed)
    return seeds


def load_joinable_pair_pose_candidate_directory(
    root: str | Path,
    *,
    limit_per_pair: int = 8,
    include_manifold_initials: bool = True,
) -> list[PairPoseSeed]:
    root = Path(root).resolve()
    paths = (
        [root] if root.is_file()
        else sorted(root.rglob("joinable_e2e_result.json"))
    )
    seeds = []
    for path in paths:
        try:
            seeds.extend(load_joinable_pair_pose_candidates(
                path,
                limit=limit_per_pair,
                include_manifold_initials=include_manifold_initials,
            ))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return seeds


def _pose_disagreement(
    existing: np.ndarray, proposed: np.ndarray
) -> dict[str, float]:
    relative = np.linalg.inv(existing) @ proposed
    return {
        "translation_mm": float(np.linalg.norm(relative[:3, 3])),
        "rotation_degrees": float(
            np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude())
        ),
    }


def compose_group_pose(
    parts: Iterable[str],
    pair_seeds: Iterable[PairPoseSeed],
    *,
    reference_part: str | None = None,
    translation_tolerance_mm: float = 1.0,
    rotation_tolerance_degrees: float = 1.0,
) -> dict[str, Any]:
    """Propagate pair transforms through the known assembly graph.

    Pair poses remain proposals.  Cycle disagreement is recorded and routes the
    result to review instead of being averaged away.
    """

    part_list = [str(part) for part in parts]
    valid = set(part_list)
    adjacency: dict[str, list[tuple[str, PairPoseSeed, bool]]] = {
        part: [] for part in part_list
    }
    usable = []
    for seed in pair_seeds:
        if seed.part_a not in valid or seed.part_b not in valid:
            continue
        seed.matrix  # validate eagerly
        adjacency[seed.part_a].append((seed.part_b, seed, True))
        adjacency[seed.part_b].append((seed.part_a, seed, False))
        usable.append(seed)
    if reference_part is None:
        reference_part = max(
            part_list,
            key=lambda part: (len(adjacency[part]), part),
        )
    if reference_part not in valid:
        raise ValueError("reference part is not in the group")

    world: dict[str, np.ndarray] = {reference_part: np.eye(4)}
    queue = [reference_part]
    cycle_checks = []
    while queue:
        current = queue.pop(0)
        for neighbor, seed, current_is_a in adjacency[current]:
            if current_is_a:
                proposed = world[current] @ seed.matrix
            else:
                proposed = world[current] @ np.linalg.inv(seed.matrix)
            if neighbor not in world:
                world[neighbor] = proposed
                queue.append(neighbor)
                continue
            disagreement = _pose_disagreement(world[neighbor], proposed)
            cycle_checks.append({
                "parts": [seed.part_a, seed.part_b],
                "source": seed.source,
                **disagreement,
                "consistent": (
                    disagreement["translation_mm"] <= translation_tolerance_mm
                    and disagreement["rotation_degrees"] <= rotation_tolerance_degrees
                ),
            })

    unresolved = sorted(valid - set(world))
    inconsistent = [row for row in cycle_checks if not row["consistent"]]
    return {
        "status": (
            "partial" if unresolved
            else "inconsistent" if inconsistent
            else "complete"
        ),
        "reference_part": reference_part,
        "usable_pair_seed_count": len(usable),
        "placements": {
            part: matrix_to_placement(matrix) for part, matrix in world.items()
        },
        "cycle_checks": cycle_checks,
        "inconsistent_cycles": inconsistent,
        "unresolved_parts": unresolved,
        "review_required": bool(unresolved or inconsistent),
    }


def compose_group_pose_hypotheses(
    parts: Iterable[str],
    pair_seeds: Iterable[PairPoseSeed],
    *,
    maximum_candidates_per_pair: int = 4,
    maximum_combinations: int = 64,
) -> list[dict[str, Any]]:
    """Enumerate a bounded set of group poses from alternative pair poses."""

    grouped: dict[tuple[str, str], list[PairPoseSeed]] = {}
    for seed in pair_seeds:
        key = tuple(sorted((seed.part_a, seed.part_b)))
        grouped.setdefault(key, []).append(seed)
    choices = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda row: (
                float("inf") if row.score is None else row.score,
                row.source,
            ),
        )[: max(1, int(maximum_candidates_per_pair))]
        choices.append(rows)
    if not choices:
        return []
    results = []
    for combination_index, combination in enumerate(
        itertools.islice(
            itertools.product(*choices), max(1, int(maximum_combinations))
        )
    ):
        composed = compose_group_pose(parts, combination)
        composed["combination_index"] = combination_index
        composed["pair_seed_sources"] = [row.source for row in combination]
        composed["pair_seed_score_sum"] = sum(
            float(row.score or 0.0) for row in combination
        )
        results.append(composed)
    return results
