"""Compatibility wrapper for the unit-safe JoinABLe-style pose search.

Historically this module contained a KD-tree proxy whose offset limit was never
assigned and whose physical offset was multiplied by 1500.  Keep the public
``SearchSimplex`` API for existing callers, while delegating all geometry and
optimization to :mod:`pose_search`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from pose_search import JointAxisSeed, JoinablePoseSearch


def load_stl_mesh(stl_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


class SearchSimplex:
    """Backward-compatible single-axis interface.

    New code should pass distinct moving and fixed axes directly to
    ``JoinablePoseSearch``.  Legacy callers only provide one world axis, so the
    wrapper treats the moving mesh as already axis-aligned and searches the
    residual offset/rotation/sign ambiguity.
    """

    def __init__(
        self,
        body_one_stl: str | Path,
        body_two_stl: str | Path,
        joint_origin: list[float],
        joint_direction: list[float],
        num_surface_samples: int = 2000,
        budget: int = 80,
        contact_weight: float = 10.0,
        overlap_weight: float = 1.0,
    ) -> None:
        # contact_weight/overlap_weight are retained for API compatibility.
        # The official JoinABLe objective fixes these semantics at overlap -
        # 10*contact below the severe-overlap threshold.
        self.searcher = JoinablePoseSearch(
            body_one_stl,
            body_two_stl,
            sample_count=num_surface_samples,
            budget=budget,
            objective="default",
        )
        origin = tuple(float(value) for value in joint_origin)
        direction = tuple(float(value) for value in joint_direction)
        self.seed = JointAxisSeed(
            moving_origin=origin,
            moving_direction=direction,
            fixed_origin=origin,
            fixed_direction=direction,
        )

    def search(self) -> dict[str, Any]:
        result = self.searcher.search([self.seed], top_k=1)[0]
        return {
            "evaluation": result.evaluation.cost,
            "offset": result.offset,
            "rotation_deg": result.rotation_degrees,
            "flip": result.axis_flip,
            "overlap": result.evaluation.overlap,
            "contact": result.evaluation.contact,
            "mean_gap": result.evaluation.closest_distance,
            "transform": result.transform,
            "transform_determinant": result.transform_determinant,
            "optimizer_success": result.optimizer_success,
            "optimizer_message": result.optimizer_message,
            "function_evaluations": result.function_evaluations,
            "offset_limit": result.offset_limit,
            "objective": "joinable_default_sdf",
        }


def searchsimplex_for_pair(
    ref_stl: str | Path,
    tgt_stl: str | Path,
    joint_origin: list[float],
    joint_direction: list[float],
    **kwargs: Any,
) -> dict[str, Any]:
    return SearchSimplex(
        ref_stl,
        tgt_stl,
        joint_origin,
        joint_direction,
        **kwargs,
    ).search()
