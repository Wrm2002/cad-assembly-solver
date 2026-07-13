"""Compatibility utilities backed by the canonical JoinABLe pose core.

Use :mod:`joinable_e2e` for released-checkpoint top-k pose proposals and
``known_group_assembly.py`` for multi-part closure/exact validation.  This file
retains the earlier helper API so existing notebooks and scripts do not keep a
second, inconsistent SDF implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize
import trimesh

from pose_search import JointAxisSeed, JoinablePoseSearch, placement_to_matrix
from pose_search.transforms import (
    rotation_about_axis_matrix,
    transform_points,
    translation_along_axis_matrix,
)


def step_to_mesh(step_path: str | Path) -> trimesh.Trimesh:
    from joinable_e2e import step_to_stl

    step_path = Path(step_path).resolve()
    stl = step_to_stl(step_path, step_path.parent / ".joinable_mesh_cache")
    mesh = trimesh.load(str(stl), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def validate_placement(
    insert_mesh: trimesh.Trimesh,
    receiver_mesh: trimesh.Trimesh,
    transform_4x4: np.ndarray,
    num_samples: int = 4096,
    seed: int = 42,
) -> dict[str, Any]:
    searcher = JoinablePoseSearch(
        receiver_mesh,
        insert_mesh,
        sample_count=num_samples,
        budget=1,
        seed=seed,
    )
    evaluation = searcher.evaluate(np.asarray(transform_4x4, dtype=float))
    if evaluation.overlap > 0.1:
        status = "overlap"
    elif evaluation.contact > 0.0:
        status = "good"
    elif evaluation.closest_distance <= 2.0 * searcher.contact_tolerance:
        status = "clearance"
    else:
        status = "no_contact"
    return {
        "overlap": evaluation.overlap,
        "contact": evaluation.contact,
        "cost": evaluation.cost,
        "closest_distance": evaluation.closest_distance,
        "status": status,
        "method": "canonical_joinable_sdf_objective",
    }


def refine_rotation(
    insert_mesh: trimesh.Trimesh,
    receiver_mesh: trimesh.Trimesh,
    joint_axis: tuple[list[float], list[float]],
    num_samples: int = 4096,
    budget: int = 50,
) -> tuple[float, float, float, float]:
    origin, direction = joint_axis
    seed = JointAxisSeed(
        moving_origin=tuple(origin),
        moving_direction=tuple(direction),
        fixed_origin=tuple(origin),
        fixed_direction=tuple(direction),
    )
    result = JoinablePoseSearch(
        receiver_mesh,
        insert_mesh,
        sample_count=num_samples,
        budget=budget,
    ).search([seed], top_k=1)[0]
    return (
        result.rotation_degrees,
        result.evaluation.overlap,
        result.evaluation.contact,
        result.evaluation.cost,
    )


def maximize_contact_along_axis(
    insert_mesh: trimesh.Trimesh,
    receiver_mesh: trimesh.Trimesh,
    initial_transform_4x4: np.ndarray,
    axis_origin: list[float],
    axis_direction: list[float],
    search_range: tuple[float, float] = (-500.0, 500.0),
    num_samples: int = 4096,
    budget: int = 50,
) -> tuple[float, np.ndarray, float, float]:
    searcher = JoinablePoseSearch(
        receiver_mesh,
        insert_mesh,
        sample_count=num_samples,
        budget=budget,
    )
    initial = np.asarray(initial_transform_4x4, dtype=float)

    def transform(offset: float) -> np.ndarray:
        return translation_along_axis_matrix(
            axis_direction, float(offset)
        ) @ initial

    result = scipy.optimize.minimize_scalar(
        lambda value: searcher.evaluate(transform(float(value))).cost,
        bounds=(float(search_range[0]), float(search_range[1])),
        method="bounded",
        options={"maxiter": budget, "xatol": 0.05},
    )
    final = transform(float(result.x))
    evaluation = searcher.evaluate(final)
    return (
        float(result.x),
        final,
        evaluation.overlap,
        evaluation.contact,
    )


def solve_with_joinable_validation(
    case_dir: str | Path,
    refine: bool = True,
    num_samples: int = 4096,
) -> dict[str, Any]:
    """Validate legacy stop-plane placements against its reference part.

    This remains a diagnostic compatibility path.  It cannot replace group
    collision validation because it does not test insert-to-insert collisions.
    """

    from stop_plane_solver import solve_stop_plane

    case_dir = Path(case_dir).resolve()
    result = solve_stop_plane(case_dir)
    placements = result["placements"]
    receiver = result["receiver"]
    meshes = {
        path.name: step_to_mesh(path)
        for path in case_dir.iterdir()
        if path.suffix.lower() in {".step", ".stp"}
        and not path.name.lower().startswith("assembly")
    }
    receiver_mesh = meshes[receiver]
    validation = {}
    for part_name, placement in placements.items():
        if part_name == receiver or part_name not in meshes:
            continue
        validation[part_name] = validate_placement(
            meshes[part_name],
            receiver_mesh,
            placement_to_matrix(placement),
            num_samples=num_samples,
        )
    result["validation"] = validation
    result["validation_scope"] = (
        "pairwise_against_reference_only; use known_group_assembly for final gate"
    )
    return result


__all__ = [
    "step_to_mesh",
    "validate_placement",
    "refine_rotation",
    "maximize_contact_along_axis",
    "solve_with_joinable_validation",
    "rotation_about_axis_matrix",
    "transform_points",
]
