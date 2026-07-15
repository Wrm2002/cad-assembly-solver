"""Generic sampled-SDF contact and penetration residuals for global solving.

This is a smooth-ish search surrogate, not the final collision authority.
Every candidate topology is treated uniformly: selected graph edges should
approach contact without penetration; other part pairs are penalized only for
penetration.  Exact OCCT validation remains a separate top-N gate.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable
import struct

import numpy as np
from scipy.spatial import cKDTree

from global_pose_solver.se3_manifold import se3_inv


class _TriangleMesh:
    """Minimal triangle mesh used by the OCCT-only solver environment.

    ``cad_asm`` has a native crash in ``trimesh.load(...).to_mesh()`` for some
    OCCT-written STL files.  This deliberately small reader/sampler avoids
    that code path; it is enough for the sampled-distance surrogate and leaves
    final collision authority with OCCT.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray | None = None) -> None:
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        triangles = self.vertices[self.faces]
        generated = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        generated /= np.maximum(np.linalg.norm(generated, axis=1, keepdims=True), 1e-12)
        self.face_normals = generated if normals is None else np.asarray(normals, dtype=float)
        self.extents = self.vertices.max(axis=0) - self.vertices.min(axis=0)


def _read_stl(path: Path) -> _TriangleMesh:
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if 84 + 50 * count == len(raw):
            record = np.frombuffer(raw, dtype=np.dtype([
                ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2"),
            ]), offset=84, count=count)
            vertices = record["vertices"].reshape(-1, 3).astype(float)
            faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
            return _TriangleMesh(vertices, faces, record["normal"])
    # OCCT can also emit ASCII STL.  Only ``vertex`` records are relevant.
    values = []
    for line in raw.decode("utf-8", errors="ignore").splitlines():
        words = line.strip().split()
        if len(words) == 4 and words[0].lower() == "vertex":
            try:
                values.append([float(words[1]), float(words[2]), float(words[3])])
            except ValueError:
                pass
    vertices = np.asarray(values, dtype=float)
    if len(vertices) < 3 or len(vertices) % 3:
        raise ValueError(f"invalid_stl_mesh:{path}")
    return _TriangleMesh(vertices, np.arange(len(vertices), dtype=np.int64).reshape(-1, 3))


def _sample_surface(mesh: _TriangleMesh, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    triangles = mesh.vertices[mesh.faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    probability = areas / max(float(areas.sum()), 1e-12)
    selected = rng.choice(len(triangles), size=max(1, int(count)), replace=True, p=probability)
    u, v = rng.random(len(selected)), rng.random(len(selected))
    swap = u + v > 1.0
    u[swap], v[swap] = 1.0 - u[swap], 1.0 - v[swap]
    triangle = triangles[selected]
    points = triangle[:, 0] + u[:, None] * (triangle[:, 1] - triangle[:, 0]) + v[:, None] * (triangle[:, 2] - triangle[:, 0])
    return points, mesh.face_normals[selected]


class _SignedDistanceScorer:
    """Bidirectional sampled signed-distance evidence without custom extensions."""

    def __init__(self, fixed: _TriangleMesh, moving: _TriangleMesh, count: int, seed: int) -> None:
        self.fixed = fixed
        self.moving = moving
        self.rng = np.random.default_rng(seed)
        self.surface_fixed, self.normal_fixed = self._surface(fixed, count)
        self.surface_moving, self.normal_moving = self._surface(moving, count)
        self.tree_fixed = cKDTree(self.surface_fixed)
        self.tree_moving = cKDTree(self.surface_moving)
        diagonal = max(
            float(np.linalg.norm(fixed.extents)),
            float(np.linalg.norm(moving.extents)),
            1e-6,
        )
        sampling_resolution = diagonal / math.sqrt(max(128, int(count)))
        self.contact_tolerance = float(np.clip(
            # Independent triangle samples are not point-correspondent.  A
            # fraction of their expected spacing is therefore required to
            # recognise a true touching boundary; OCCT remains the exact
            # authority for collision/clearance.
            max(diagonal * 1e-3, 0.75 * sampling_resolution),
            0.01,
            0.5,
        ))
        self.distance_scale = diagonal

    def _surface(self, mesh: _TriangleMesh, count: int) -> tuple[np.ndarray, np.ndarray]:
        return _sample_surface(mesh, max(128, int(count)), self.rng)

    @staticmethod
    def _points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        return points @ transform[:3, :3].T + transform[:3, 3]

    def evaluate(self, moving_to_fixed: np.ndarray) -> dict[str, float]:
        moving_in_fixed = self._points(self.surface_moving, moving_to_fixed)
        fixed_to_moving = se3_inv(moving_to_fixed)
        fixed_in_moving = self._points(self.surface_fixed, fixed_to_moving)
        # trimesh defines positive signed distance for points inside a solid.
        distance_moving, index_moving = self.tree_fixed.query(moving_in_fixed, workers=1)
        distance_fixed, index_fixed = self.tree_moving.query(fixed_in_moving, workers=1)
        # Oriented nearest-surface approximation: a point displaced against
        # an outward normal is treated as inside.  The final authority remains
        # OCCT; this term is bounded and stable enough for least-squares search.
        signed_moving = np.einsum(
            "ij,ij->i",
            moving_in_fixed - self.surface_fixed[index_moving],
            self.normal_fixed[index_moving],
        )
        signed_fixed = np.einsum(
            "ij,ij->i",
            fixed_in_moving - self.surface_moving[index_fixed],
            self.normal_moving[index_fixed],
        )
        overlap = float(max(
            np.mean(signed_moving < -self.contact_tolerance),
            np.mean(signed_fixed < -self.contact_tolerance),
        ))
        contact = float(max(
            np.mean(distance_moving <= self.contact_tolerance),
            np.mean(distance_fixed <= self.contact_tolerance),
        ))
        closest = float(min(
            np.min(distance_moving),
            np.min(distance_fixed),
        ))
        keep_moving = max(4, int(math.ceil(0.05 * len(distance_moving))))
        keep_fixed = max(4, int(math.ceil(0.05 * len(distance_fixed))))
        contact_gap = float(min(
            np.mean(np.partition(distance_moving, keep_moving - 1)[:keep_moving]),
            np.mean(np.partition(distance_fixed, keep_fixed - 1)[:keep_fixed]),
        )) / self.distance_scale
        penetration_depth = float(max(
            np.mean(np.maximum(0.0, -signed_moving)),
            np.mean(np.maximum(0.0, -signed_fixed)),
        )) / self.distance_scale
        return {
            "overlap": overlap,
            "contact": contact,
            "closest_distance": closest,
            "contact_gap_normalized": contact_gap,
            "penetration_depth_normalized": penetration_depth,
        }


def _load_mesh(value: Any) -> _TriangleMesh:
    if isinstance(value, (str, Path)):
        mesh = _read_stl(Path(value))
    elif all(hasattr(value, name) for name in ("vertices", "faces", "face_normals")):
        mesh = _TriangleMesh(value.vertices, value.faces, value.face_normals)
    else:
        raise ValueError("mesh_residual_provider_requires_triangle_mesh_or_stl")
    if len(mesh.faces) == 0:
        raise ValueError("mesh_residual_provider_requires_nonempty_meshes")
    return mesh


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))


class MeshContactResidualProvider:
    """Convert multi-body SDF evidence into least-squares residuals."""

    def __init__(
        self,
        meshes: dict[str, Any],
        *,
        sample_count: int = 384,
        seed: int = 73,
        selected_overlap_weight: float = 16.0,
        selected_contact_weight: float = 2.0,
        selected_distance_weight: float = 0.5,
        nonedge_overlap_weight: float = 25.0,
        local_patch_gap_weight: float = 0.0,
        local_patch_normal_weight: float = 0.0,
    ) -> None:
        self.parts = sorted(map(str, meshes))
        if not 2 <= len(self.parts) <= 5:
            raise ValueError("mesh_residual_provider_supports_2_to_5_parts")
        self.meshes = {str(key): _load_mesh(value) for key, value in meshes.items()}
        self.weights = {
            "selected_overlap": float(selected_overlap_weight),
            "selected_contact": float(selected_contact_weight),
            "selected_distance": float(selected_distance_weight),
            "nonedge_overlap": float(nonedge_overlap_weight),
            "local_patch_gap": float(local_patch_gap_weight),
            "local_patch_normal": float(local_patch_normal_weight),
        }
        self.scorers: dict[tuple[str, str], _SignedDistanceScorer] = {}
        for left_index, left in enumerate(self.parts):
            for right in self.parts[left_index + 1:]:
                self.scorers[(left, right)] = _SignedDistanceScorer(
                    self.meshes[left],
                    self.meshes[right],
                    sample_count,
                    seed + len(self.scorers) * 17,
                )
        self.scale = max(
            float(np.linalg.norm(mesh.extents))
            for mesh in self.meshes.values()
        )
        # One stable surface sample set per part is used for factor-local
        # patches.  Selection is driven only by the learned B-Rep frame and
        # manifold type; no mechanical-family or case labels are consulted.
        self.part_surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for index, part in enumerate(self.parts):
            points, normals = _sample_surface(
                self.meshes[part], max(256, int(sample_count)), np.random.default_rng(seed + 1009 + index * 31)
            )
            self.part_surfaces[part] = (
                np.asarray(points, dtype=float),
                np.asarray(normals, dtype=float),
            )

    @staticmethod
    def _active_pairs(factors: Iterable[Any]) -> set[tuple[str, str]]:
        return {
            _pair_key(factor.source, factor.target)
            for factor in factors
        }

    @staticmethod
    def _transform_vectors(vectors: np.ndarray, transform: np.ndarray) -> np.ndarray:
        return vectors @ transform[:3, :3].T

    def _local_patch(
        self,
        part: str,
        frame: np.ndarray,
        manifold_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        points, normals = self.part_surfaces[part]
        origin = np.asarray(frame[:3, 3], dtype=float)
        axis = np.asarray(frame[:3, 2], dtype=float)
        delta = points - origin
        axial = np.abs(delta @ axis)
        radial = np.linalg.norm(delta - np.outer(delta @ axis, axis), axis=1)
        if "plane" in manifold_type:
            # Prefer points on the selected plane, but retain a local
            # tangential neighbourhood instead of a single closest vertex.
            metric = axial + 0.12 * radial
        elif "axis" in manifold_type:
            # A joint-axis origin selects an end-ring/local axial band.  The
            # radial term is deliberately weak so nested cylindrical surfaces
            # are not collapsed onto the axis itself.
            metric = axial + 0.04 * radial
        else:
            metric = np.linalg.norm(delta, axis=1)
        count = min(len(points), max(24, int(math.ceil(0.12 * len(points)))))
        indices = np.argpartition(metric, count - 1)[:count]
        return points[indices], normals[indices]

    def _patch_evidence(self, poses: dict[str, np.ndarray], factor: Any) -> dict[str, float]:
        source_points, source_normals = self._local_patch(
            factor.source, factor.frame_a, factor.manifold_type
        )
        target_points, target_normals = self._local_patch(
            factor.target, factor.frame_b, factor.manifold_type
        )
        source_world = _SignedDistanceScorer._points(source_points, poses[factor.source])
        target_world = _SignedDistanceScorer._points(target_points, poses[factor.target])
        source_normals_world = self._transform_vectors(source_normals, poses[factor.source])
        target_normals_world = self._transform_vectors(target_normals, poses[factor.target])
        tree_source, tree_target = cKDTree(source_world), cKDTree(target_world)
        distance_target, index_source = tree_source.query(target_world, workers=1)
        distance_source, index_target = tree_target.query(source_world, workers=1)
        keep_target = max(6, int(math.ceil(0.25 * len(distance_target))))
        keep_source = max(6, int(math.ceil(0.25 * len(distance_source))))
        low_target = np.partition(distance_target, keep_target - 1)[:keep_target]
        low_source = np.partition(distance_source, keep_source - 1)[:keep_source]
        gap = float(0.5 * (np.mean(low_target) + np.mean(low_source)))
        tolerance = max(0.05, 0.003 * self.scale)
        coverage = float(0.5 * (
            np.mean(distance_target <= tolerance) + np.mean(distance_source <= tolerance)
        ))
        dot_target = np.einsum(
            "ij,ij->i", target_normals_world, source_normals_world[index_source]
        )
        dot_source = np.einsum(
            "ij,ij->i", source_normals_world, target_normals_world[index_target]
        )
        # Mating boundary normals should generally oppose one another.  Use
        # only the closest quartile so unrelated nearby faces do not dominate.
        opposition = float(0.5 * (
            np.mean(np.abs(1.0 + dot_target[np.argpartition(distance_target, keep_target - 1)[:keep_target]]))
            + np.mean(np.abs(1.0 + dot_source[np.argpartition(distance_source, keep_source - 1)[:keep_source]]))
        ))
        return {
            "local_patch_gap": gap,
            "local_patch_gap_normalized": gap / self.scale,
            "local_patch_contact_coverage": coverage,
            "local_patch_normal_opposition_error": opposition,
            "single_point_contact_risk": bool(gap <= tolerance and coverage < 0.08),
        }

    def _rows(self, poses: dict[str, np.ndarray], factors: Iterable[Any]) -> list[dict[str, Any]]:
        factors = list(factors)
        active = self._active_pairs(factors)
        factor_by_pair = {_pair_key(row.source, row.target): row for row in factors}
        rows = []
        for (left, right), scorer in self.scorers.items():
            relative = se3_inv(poses[left]) @ poses[right]
            score = scorer.evaluate(relative)
            selected = (left, right) in active
            if selected:
                factor = factor_by_pair[(left, right)]
                patch_enabled = (
                    self.weights["local_patch_gap"] > 0.0
                    or self.weights["local_patch_normal"] > 0.0
                ) and all(hasattr(factor, name) for name in ("frame_a", "frame_b", "manifold_type"))
                patch = self._patch_evidence(poses, factor) if patch_enabled else {
                    "local_patch_gap": float("nan"),
                    "local_patch_gap_normalized": 0.0,
                    "local_patch_contact_coverage": 0.0,
                    "local_patch_normal_opposition_error": 0.0,
                    "single_point_contact_risk": False,
                }
                residual = [
                    math.sqrt(self.weights["selected_overlap"])
                    * score["penetration_depth_normalized"],
                    math.sqrt(self.weights["selected_contact"])
                    * score["contact_gap_normalized"],
                    math.sqrt(self.weights["selected_distance"])
                    * min(1.0, score["contact_gap_normalized"]),
                    math.sqrt(self.weights["local_patch_gap"])
                    * min(1.0, patch["local_patch_gap_normalized"]),
                    math.sqrt(self.weights["local_patch_normal"])
                    * min(2.0, patch["local_patch_normal_opposition_error"]),
                ]
            else:
                patch = {
                    "local_patch_gap": float("nan"),
                    "local_patch_gap_normalized": 0.0,
                    "local_patch_contact_coverage": 0.0,
                    "local_patch_normal_opposition_error": 0.0,
                    "single_point_contact_risk": False,
                }
                residual = [
                    math.sqrt(self.weights["nonedge_overlap"])
                    * score["penetration_depth_normalized"],
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            rows.append({
                "source": left,
                "target": right,
                "selected_constraint_edge": selected,
                "overlap": float(score["overlap"]),
                "contact": float(score["contact"]),
                "closest_distance": float(score["closest_distance"]),
                "contact_gap_normalized": float(score["contact_gap_normalized"]),
                "penetration_depth_normalized": float(score["penetration_depth_normalized"]),
                "residual": residual,
                **patch,
            })
        return rows

    def __call__(self, poses: dict[str, np.ndarray], factors: Iterable[Any]) -> np.ndarray:
        return np.asarray(
            [value for row in self._rows(poses, factors) for value in row["residual"]],
            dtype=float,
        )

    def audit(self, poses: dict[str, np.ndarray], factors: Iterable[Any]) -> dict[str, Any]:
        rows = self._rows(poses, factors)
        return {
            "schema_version": "mesh_contact_residual_audit.v1",
            "pair_scores": rows,
            "maximum_overlap": max((row["overlap"] for row in rows), default=0.0),
            "minimum_selected_contact": min(
                (row["contact"] for row in rows if row["selected_constraint_edge"]),
                default=0.0,
            ),
            "authority": "oriented_sampled_signed_distance_search_surrogate_only",
        }
