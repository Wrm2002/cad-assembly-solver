"""Faithful, unit-safe reproduction of JoinABLe's pair pose search.

The released method searches each top-k predicted entity pair by:

1. deriving one joint axis on each body;
2. aligning the two axes;
3. optimizing axial offset and axial rotation with Nelder-Mead;
4. enumerating the joint-axis sign/flip ambiguity;
5. minimizing normalized overlap minus normalized contact.

This implementation keeps those semantics, but works in the STEP model's
physical units and emits only proper rigid transforms.  It is a pose proposal
generator; exact OCCT validation is still required before acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.optimize
import trimesh
from pysdf import SDF

from .transforms import joint_parameter_matrix, transform_points, unit


@dataclass(frozen=True)
class JointAxisSeed:
    moving_origin: tuple[float, float, float]
    moving_direction: tuple[float, float, float]
    fixed_origin: tuple[float, float, float]
    fixed_direction: tuple[float, float, float]
    prediction_rank: int = 1
    prediction_score: float | None = None
    entity_a: str | None = None
    entity_b: str | None = None
    rotation_seed_degrees: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class PoseEvaluation:
    cost: float
    overlap: float
    contact: float
    closest_distance: float
    overlap_moving_fraction: float
    contact_moving_fraction: float


@dataclass(frozen=True)
class PoseSearchResult:
    candidate_origin: str
    prediction_rank: int
    prediction_score: float | None
    offset: float
    rotation_degrees: float
    rotation_seed_degrees: float
    axis_flip: bool
    transform: list[list[float]]
    transform_determinant: float
    evaluation: PoseEvaluation
    optimizer_success: bool
    optimizer_message: str
    function_evaluations: int
    offset_limit: float
    rotation_skipped: bool
    entity_a: str | None = None
    entity_b: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluation"] = asdict(self.evaluation)
        return payload


def _as_mesh(mesh_or_path: trimesh.Trimesh | str | Path) -> trimesh.Trimesh:
    if isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path.copy()
    else:
        loaded = trimesh.load(str(mesh_or_path), force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("pose search requires a non-empty triangle mesh")
    return mesh


def _surface_area(mesh: trimesh.Trimesh) -> float:
    value = float(mesh.area)
    return value if np.isfinite(value) and value > 1e-12 else 1.0


def _volume(mesh: trimesh.Trimesh) -> float:
    value = abs(float(mesh.volume))
    if np.isfinite(value) and value > 1e-12:
        return value
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 1e-6)
    return float(np.prod(extents))


def _volume_samples(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> np.ndarray:
    # trimesh's volume sampler uses NumPy's legacy global RNG.  Preserve the
    # deterministic contract by temporarily seeding it and restoring state.
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        if mesh.is_watertight:
            samples = trimesh.sample.volume_mesh(mesh, count)
            if len(samples) >= count:
                return np.asarray(samples[:count], dtype=float)
    except Exception:
        pass
    finally:
        np.random.set_state(state)

    # Robust fallback: rejection sample the mesh AABB with the mesh SDF.
    lo, hi = np.asarray(mesh.bounds, dtype=float)
    sdf = SDF(mesh.vertices, mesh.faces)
    accepted: list[np.ndarray] = []
    accepted_count = 0
    for _ in range(12):
        points = rng.uniform(lo, hi, size=(max(count * 2, 256), 3))
        inside = points[sdf(points) > 0.0]
        if len(inside):
            accepted.append(inside)
            accepted_count += len(inside)
        if accepted_count >= count:
            return np.concatenate(accepted, axis=0)[:count]

    # Thin/open meshes have no reliable volume.  Move surface samples slightly
    # inward; overlap remains approximate and is explicitly audited as such.
    points, face_indices = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_indices]
    tolerance = max(float(np.max(mesh.extents)) * 1e-5, 1e-5)
    return np.asarray(points - normals * tolerance, dtype=float)


def _surface_samples(
    mesh: trimesh.Trimesh, count: int, rng: np.random.Generator
) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, count)
        return np.asarray(points, dtype=float)
    finally:
        np.random.set_state(state)


class JoinablePoseSearch:
    """Search top-k joint-axis seeds with JoinABLe's geometric objective."""

    def __init__(
        self,
        fixed_mesh: trimesh.Trimesh | str | Path,
        moving_mesh: trimesh.Trimesh | str | Path,
        *,
        sample_count: int = 2048,
        contact_tolerance: float | None = None,
        overlap_tolerance: float | None = None,
        budget: int = 80,
        objective: str = "default",
        seed: int = 42,
        rotation_seeds_degrees: Sequence[float] | None = None,
    ) -> None:
        if objective not in {"default", "smooth"}:
            raise ValueError("objective must be 'default' or 'smooth'")
        self.fixed_mesh = _as_mesh(fixed_mesh)
        self.moving_mesh = _as_mesh(moving_mesh)
        self.sample_count = max(256, int(sample_count))
        self.budget = max(1, int(budget))
        self.objective = objective
        provided_rotations = rotation_seeds_degrees or (0.0,)
        self.rotation_seeds_degrees = tuple(sorted({
            round(((float(value) + 180.0) % 360.0) - 180.0, 8)
            for value in (*provided_rotations, 0.0)
        }))
        diagonal = max(
            float(np.linalg.norm(self.fixed_mesh.extents)),
            float(np.linalg.norm(self.moving_mesh.extents)),
            1e-6,
        )
        # Official data is normalized and uses 0.01.  Convert that intent to a
        # scale-aware physical tolerance with conservative absolute bounds.
        default_tolerance = float(np.clip(diagonal * 1e-3, 0.01, 0.5))
        self.contact_tolerance = float(
            contact_tolerance
            if contact_tolerance is not None else default_tolerance
        )
        self.overlap_tolerance = float(
            overlap_tolerance
            if overlap_tolerance is not None else default_tolerance
        )
        rng = np.random.default_rng(seed)
        self.volume_samples = _volume_samples(
            self.moving_mesh, self.sample_count, rng
        )
        self.surface_samples = _surface_samples(
            self.moving_mesh, self.sample_count, rng
        )
        self.sdf = SDF(self.fixed_mesh.vertices, self.fixed_mesh.faces)
        self.moving_volume = _volume(self.moving_mesh)
        self.fixed_volume = _volume(self.fixed_mesh)
        self.moving_area = _surface_area(self.moving_mesh)
        self.fixed_area = _surface_area(self.fixed_mesh)

    def _offset_limit(self, seed: JointAxisSeed) -> float:
        direction = unit(seed.fixed_direction)

        def span(mesh: trimesh.Trimesh) -> float:
            values = np.asarray(mesh.vertices, dtype=float) @ direction
            return float(np.ptp(values)) if len(values) else 0.0

        value = span(self.fixed_mesh) + span(self.moving_mesh)
        if not np.isfinite(value) or value <= 1e-6:
            value = max(
                float(np.max(self.fixed_mesh.extents)),
                float(np.max(self.moving_mesh.extents)),
                1.0,
            )
        return value

    def evaluate(self, transform: np.ndarray) -> PoseEvaluation:
        volume = transform_points(self.volume_samples, transform)
        surface = transform_points(self.surface_samples, transform)
        volume_sdf = np.asarray(self.sdf(volume), dtype=float)
        surface_sdf = np.asarray(self.sdf(surface), dtype=float)
        overlap_moving = float(
            np.mean(volume_sdf > self.overlap_tolerance)
        )
        contact_moving = float(
            np.mean(np.abs(surface_sdf) < self.contact_tolerance)
        )
        overlap_fixed_equivalent = (
            overlap_moving * self.moving_volume / self.fixed_volume
        )
        contact_fixed_equivalent = (
            contact_moving * self.moving_area / self.fixed_area
        )
        overlap = float(np.clip(
            max(overlap_moving, overlap_fixed_equivalent), 0.0, 1.0
        ))
        contact = float(np.clip(
            max(contact_moving, contact_fixed_equivalent), 0.0, 1.0
        ))
        closest_distance = float(np.min(np.abs(surface_sdf)))
        if self.objective == "default":
            cost = overlap if overlap > 0.1 else overlap - 10.0 * contact
        else:
            distance_scale = max(
                float(np.linalg.norm(self.fixed_mesh.extents)), 1e-6
            )
            distance_term = (
                min(1.0, closest_distance / distance_scale)
                if contact == 0.0 and overlap == 0.0 else 0.0
            )
            usable_contact = 0.0 if overlap > 0.1 else contact
            cost = (
                0.1 * overlap
                + 0.6 * (1.0 - usable_contact)
                + 0.3 * distance_term
            )
        return PoseEvaluation(
            cost=float(cost),
            overlap=overlap,
            contact=contact,
            closest_distance=closest_distance,
            overlap_moving_fraction=overlap_moving,
            contact_moving_fraction=contact_moving,
        )

    def _search_seed(
        self,
        seed: JointAxisSeed,
        axis_flip: bool,
        rotation_seed_degrees: float,
    ) -> PoseSearchResult:
        offset_limit = self._offset_limit(seed)
        # Geometry-only mesh symmetry is not functional equivalence. Never
        # suppress axial rotation merely because the mesh looks symmetric.
        rotation_skipped = False

        def unpack(values: np.ndarray) -> tuple[float, float]:
            offset = float(values[0]) * offset_limit
            rotation = (
                float(rotation_seed_degrees) + float(values[1]) * 360.0
            )
            return offset, rotation

        def cost(values: np.ndarray) -> float:
            offset, rotation = unpack(values)
            transform = joint_parameter_matrix(
                seed.moving_origin,
                seed.moving_direction,
                seed.fixed_origin,
                seed.fixed_direction,
                offset=offset,
                rotation_degrees=rotation,
                axis_flip=axis_flip,
            )
            return self.evaluate(transform).cost

        if rotation_skipped:
            x0 = np.array([0.0], dtype=float)
            bounds = scipy.optimize.Bounds([-1.0], [1.0])
            initial_simplex = np.array([[0.0], [0.05]], dtype=float)
        else:
            x0 = np.array([0.0, 0.0], dtype=float)
            bounds = scipy.optimize.Bounds([-1.0, -0.5], [1.0, 0.5])
            initial_simplex = np.array(
                [[0.0, 0.0], [0.05, 0.0], [0.0, 1.0 / 24.0]],
                dtype=float,
            )
        result = scipy.optimize.minimize(
            cost,
            x0,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": self.budget,
                "xatol": 1e-4,
                "fatol": 1e-5,
                "initial_simplex": initial_simplex,
            },
        )
        offset, rotation = unpack(result.x)
        transform = joint_parameter_matrix(
            seed.moving_origin,
            seed.moving_direction,
            seed.fixed_origin,
            seed.fixed_direction,
            offset=offset,
            rotation_degrees=rotation,
            axis_flip=axis_flip,
        )
        evaluation = self.evaluate(transform)
        return PoseSearchResult(
            candidate_origin="nelder_mead",
            prediction_rank=int(seed.prediction_rank),
            prediction_score=seed.prediction_score,
            offset=offset,
            rotation_degrees=rotation,
            rotation_seed_degrees=float(rotation_seed_degrees),
            axis_flip=bool(axis_flip),
            transform=transform.tolist(),
            transform_determinant=float(np.linalg.det(transform[:3, :3])),
            evaluation=evaluation,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
            function_evaluations=int(result.nfev),
            offset_limit=offset_limit,
            rotation_skipped=rotation_skipped,
            entity_a=seed.entity_a,
            entity_b=seed.entity_b,
        )

    def _alignment_seed(
        self,
        seed: JointAxisSeed,
        axis_flip: bool,
        rotation_seed_degrees: float,
    ) -> PoseSearchResult:
        offset_limit = self._offset_limit(seed)
        transform = joint_parameter_matrix(
            seed.moving_origin,
            seed.moving_direction,
            seed.fixed_origin,
            seed.fixed_direction,
            offset=0.0,
            rotation_degrees=float(rotation_seed_degrees),
            axis_flip=axis_flip,
        )
        return PoseSearchResult(
            candidate_origin="axis_alignment_seed",
            prediction_rank=int(seed.prediction_rank),
            prediction_score=seed.prediction_score,
            offset=0.0,
            rotation_degrees=float(rotation_seed_degrees),
            rotation_seed_degrees=float(rotation_seed_degrees),
            axis_flip=bool(axis_flip),
            transform=transform.tolist(),
            transform_determinant=float(np.linalg.det(transform[:3, :3])),
            evaluation=self.evaluate(transform),
            optimizer_success=True,
            optimizer_message="deterministic pre-optimization seed",
            function_evaluations=1,
            offset_limit=offset_limit,
            rotation_skipped=False,
            entity_a=seed.entity_a,
            entity_b=seed.entity_b,
        )

    def search(
        self,
        seeds: Iterable[JointAxisSeed],
        *,
        top_k: int = 5,
    ) -> list[PoseSearchResult]:
        results: list[PoseSearchResult] = []
        for seed in list(seeds)[: max(1, int(top_k))]:
            for axis_flip in (False, True):
                rotations = (
                    tuple(seed.rotation_seed_degrees)
                    if seed.rotation_seed_degrees else self.rotation_seeds_degrees
                )
                for rotation_seed_degrees in rotations:
                    # Preserve every exact feature/axis alignment seed. A
                    # sampled SDF objective can otherwise move a valid contact
                    # pose slightly into penetration to gain contact samples.
                    results.append(self._alignment_seed(
                        seed, axis_flip, rotation_seed_degrees
                    ))
                    results.append(self._search_seed(
                        seed, axis_flip, rotation_seed_degrees
                    ))
        unique: list[PoseSearchResult] = []
        seen: set[tuple[float, ...]] = set()
        for result in results:
            key = tuple(
                np.round(np.asarray(result.transform, dtype=float), 8).ravel()
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
        return sorted(
            unique,
            key=lambda row: (
                row.evaluation.cost,
                row.evaluation.overlap,
                -row.evaluation.contact,
                row.prediction_rank,
                row.axis_flip,
            ),
        )
