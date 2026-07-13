"""Continuous multi-body SDF contact/overlap refinement for a pose hypothesis.

This is deliberately downstream of discrete candidate selection.  Pair poses
provide an initial global placement; the refiner jointly perturbs all
non-anchor SE(3) poses to preserve contact on selected direct edges while
penalising penetration between every other pair.  Exact OCCT remains the final
authority because sampled SDF is only a smooth search surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.optimize
import trimesh

from pose_search.joinable_search import JoinablePoseSearch

from .se3_manifold import se3_exp, se3_inv


def _load_mesh(value: trimesh.Trimesh | str | Path) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        mesh = value.copy()
    else:
        loaded = trimesh.load(str(value), force="mesh", process=False)
        mesh = (
            trimesh.util.concatenate(tuple(loaded.geometry.values()))
            if isinstance(loaded, trimesh.Scene) else loaded
        )
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("contact_refinement_requires_nonempty_meshes")
    return mesh


def _relative(poses: dict[str, np.ndarray], source: str, target: str) -> np.ndarray:
    return se3_inv(poses[source]) @ poses[target]


@dataclass(frozen=True)
class _Scorer:
    source: str
    target: str
    direct_edge: bool
    evaluator: JoinablePoseSearch


class MultiBodyContactRefiner:
    """Bounded joint pose refinement using the JoinABLe SDF objective."""

    def __init__(
        self,
        meshes: dict[str, trimesh.Trimesh | str | Path],
        direct_edges: Iterable[tuple[str, str]],
        *,
        sample_count: int = 384,
        seed: int = 42,
        direct_contact_weight: float = 1.0,
        nonedge_overlap_weight: float = 8.0,
    ) -> None:
        self.parts = sorted(meshes)
        if not 2 <= len(self.parts) <= 5:
            raise ValueError("contact_refinement_supports_2_to_5_parts")
        self.meshes = {part: _load_mesh(value) for part, value in meshes.items()}
        direct = {tuple(sorted(map(str, edge))) for edge in direct_edges}
        self.direct_contact_weight = float(direct_contact_weight)
        self.nonedge_overlap_weight = float(nonedge_overlap_weight)
        self.scorers: list[_Scorer] = []
        for offset, source in enumerate(self.parts):
            for target in self.parts[offset + 1:]:
                key = tuple(sorted((source, target)))
                is_direct = key in direct
                self.scorers.append(_Scorer(
                    source=source,
                    target=target,
                    direct_edge=is_direct,
                    evaluator=JoinablePoseSearch(
                        self.meshes[source], self.meshes[target],
                        sample_count=sample_count,
                        budget=1,
                        # ``default`` assigns zero cost to separated bodies;
                        # selected direct edges instead need the paper's
                        # distance-aware smooth objective to preserve contact.
                        objective="smooth" if is_direct else "default",
                        seed=seed + offset * 31 + len(self.scorers),
                    ),
                ))
        self.default_translation_bound_mm = max(
            float(np.linalg.norm(mesh.extents)) for mesh in self.meshes.values()
        )

    def evaluate(self, poses: dict[str, np.ndarray]) -> dict[str, Any]:
        rows = []
        objective = 0.0
        for scorer in self.scorers:
            result = scorer.evaluator.evaluate(
                _relative(poses, scorer.source, scorer.target)
            )
            if scorer.direct_edge:
                contribution = self.direct_contact_weight * float(result.cost)
            else:
                contribution = self.nonedge_overlap_weight * float(result.overlap)
            objective += contribution
            rows.append({
                "source": scorer.source,
                "target": scorer.target,
                "direct_edge": scorer.direct_edge,
                "cost": float(result.cost),
                "overlap": float(result.overlap),
                "contact": float(result.contact),
                "closest_distance": float(result.closest_distance),
                "objective_contribution": contribution,
            })
        return {"objective": objective, "pair_scores": rows}

    def refine(
        self,
        initial_poses: dict[str, Any],
        *,
        anchor_id: str | None = None,
        translation_bound_mm: float | None = None,
        rotation_bound_degrees: float = 180.0,
        maxiter: int = 120,
    ) -> dict[str, Any]:
        anchor = str(anchor_id or self.parts[0])
        if anchor not in self.meshes:
            raise ValueError("anchor_id_not_in_meshes")
        initial = {
            part: np.asarray(initial_poses[part], dtype=float)
            for part in self.parts
        }
        if any(matrix.shape != (4, 4) for matrix in initial.values()):
            raise ValueError("initial_poses_must_contain_4x4_matrices")
        movable = [part for part in self.parts if part != anchor]
        bound_translation = float(
            translation_bound_mm
            if translation_bound_mm is not None
            else max(1.0, self.default_translation_bound_mm)
        )
        bound_rotation = math.radians(float(rotation_bound_degrees))
        if bound_translation <= 0 or bound_rotation <= 0:
            raise ValueError("refinement_bounds_must_be_positive")

        def materialise(values: np.ndarray) -> dict[str, np.ndarray]:
            poses = {anchor: initial[anchor]}
            for index, part in enumerate(movable):
                delta = values[index * 6:(index + 1) * 6]
                poses[part] = se3_exp(delta) @ initial[part]
            return poses

        def objective(values: np.ndarray) -> float:
            score = self.evaluate(materialise(values))["objective"]
            # Tiny normalised trust region discourages arbitrary drift where
            # an interface has an unconstrained DOF, without turning the pair
            # pose into a hard six-dimensional decode.
            state = values.reshape(-1, 6)
            regulariser = 0.002 * float(np.sum(
                (state[:, :3] / bound_translation) ** 2
            ) + np.sum((state[:, 3:] / bound_rotation) ** 2))
            return score + regulariser

        x0 = np.zeros(6 * len(movable), dtype=float)
        bounds = [
            value
            for _ in movable
            for value in (
                (-bound_translation, bound_translation),
                (-bound_translation, bound_translation),
                (-bound_translation, bound_translation),
                (-bound_rotation, bound_rotation),
                (-bound_rotation, bound_rotation),
                (-bound_rotation, bound_rotation),
            )
        ]
        before = self.evaluate(materialise(x0))
        result = scipy.optimize.minimize(
            objective, x0, method="Powell", bounds=bounds,
            options={"maxiter": int(maxiter), "xtol": 1e-3, "ftol": 1e-4},
        )
        poses = materialise(np.asarray(result.x, dtype=float))
        after = self.evaluate(poses)
        return {
            "schema_version": "multibody_contact_refinement.v1",
            "status": "converged" if result.success else "uncertain",
            "optimizer_message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "initial_objective": before["objective"],
            "final_objective": after["objective"],
            "initial_pair_scores": before["pair_scores"],
            "final_pair_scores": after["pair_scores"],
            "part_poses": {part: matrix.tolist() for part, matrix in poses.items()},
            "accepted": False,
            "review_required": True,
            "reason": (
                "Sampled SDF contact/overlap is a continuous search surrogate; "
                "exact OCCT and outer acceptance gates remain required."
            ),
        }
