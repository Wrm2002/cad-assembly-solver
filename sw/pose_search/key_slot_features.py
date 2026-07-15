"""Conservative topological key-slot evidence from an audited B-Rep graph.

This module does not use filenames, case identifiers, or a relation-label
vocabulary.  A slot is reported only when its bounded local topology supports
the claim: two opposing side walls, a third bottom wall shared through concave
edges, and finite face areas.  Planar fragments without that evidence produce
``unknown`` or ``not_detected`` rather than a fabricated directional cue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from .transforms import unit


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if abs(wrapped + 180.0) < 1e-9 else wrapped


def _axis_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = unit(np.cross(axis, seed))
    return first, unit(np.cross(axis, first))


def _angle_and_radius(
    point: np.ndarray, origin: np.ndarray, axis: np.ndarray
) -> tuple[float, float]:
    offset = point - origin
    radial = offset - float(np.dot(offset, axis)) * axis
    first, second = _axis_basis(axis)
    return (
        _wrap_degrees(math.degrees(math.atan2(
            float(np.dot(radial, second)), float(np.dot(radial, first))
        ))),
        float(np.linalg.norm(radial)),
    )


@dataclass(frozen=True)
class KeySlotCandidate:
    """A locally closed, concave three-face slot witness."""

    candidate_id: str
    wall_face_ids: tuple[str, str]
    bottom_face_id: str
    concave_edge_ids: tuple[str, ...]
    center_point_mm: tuple[float, float, float]
    wall_normal: tuple[float, float, float]
    center_angle_degrees: float
    radial_distance_mm: float
    wall_separation_mm: float
    evidence_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vector(row: dict[str, Any], field: str) -> np.ndarray | None:
    try:
        value = np.asarray(row.get(field), dtype=float)
    except (TypeError, ValueError):
        return None
    if value.shape != (3,) or not bool(np.all(np.isfinite(value))):
        return None
    return value


def extract_key_slot_evidence(
    source: dict[str, Any],
    axis_origin: Iterable[float],
    axis_direction: Iterable[float],
    *,
    wall_opposition_cosine: float = 0.94,
    maximum_axis_normal_cosine: float = 0.25,
    minimum_wall_separation_mm: float = 0.1,
) -> dict[str, Any]:
    """Extract strict, topology-supported slot candidates around an axis.

    The expected input is the STEP B-Rep graph emitted by
    ``step_to_brep_graph_probe`` version 2.1 or later.  It is intentionally
    strict: a pair of parallel planes is not a slot.  Both walls must connect
    to the same finite bottom face via concave local edges.
    """

    origin = np.asarray(axis_origin, dtype=float)
    axis = unit(np.asarray(axis_direction, dtype=float))
    if origin.shape != (3,) or axis.shape != (3,):
        raise ValueError("axis_origin_and_direction_must_be_3d")

    metadata = source.get("metadata") or {}
    topology_meta = metadata.get("edge_topology_features") or {}
    edge_nodes = [
        row for row in (source.get("nodes") or [])
        if row.get("entity_type") == "edge"
    ]
    has_edge_evidence = bool(topology_meta.get("available")) or any(
        row.get("topology_feature_status") == "success" for row in edge_nodes
    )
    if not has_edge_evidence:
        return {
            "schema_version": "key_slot_evidence.v1",
            "status": "unknown",
            "topology_available": False,
            "candidates": [],
            "reason": (
                "No audited local edge convexity/face-adjacency evidence is "
                "available; planar fragments are not treated as a key-slot."
            ),
        }

    faces = {
        str(row.get("node_id")): row
        for row in (source.get("nodes") or [])
        if row.get("entity_type") == "face"
        and str(row.get("surface_type", "")).lower() == "plane"
    }
    face_to_concave_edges: dict[str, set[str]] = {key: set() for key in faces}
    for edge in edge_nodes:
        if edge.get("topology_feature_status") != "success":
            continue
        if edge.get("convexity") != "concave":
            continue
        adjacent = [str(value) for value in edge.get("adjacent_face_ids") or []]
        if len(adjacent) != 2:
            continue
        for face_id in adjacent:
            if face_id in face_to_concave_edges:
                face_to_concave_edges[face_id].add(str(edge.get("node_id")))

    plane_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for face_id, face in faces.items():
        centroid = _vector(face, "centroid")
        normal = _vector(face, "normal")
        try:
            normal = None if normal is None else unit(normal)
            area = float(face.get("area") or 0.0)
        except (TypeError, ValueError):
            continue
        if centroid is None or normal is None or area <= 1e-9:
            continue
        if abs(float(np.dot(normal, axis))) > maximum_axis_normal_cosine:
            continue
        plane_data[face_id] = centroid, normal, area

    candidates: list[KeySlotCandidate] = []
    face_ids = sorted(plane_data)
    for left_index, left_id in enumerate(face_ids):
        left_center, left_normal, _ = plane_data[left_id]
        for right_id in face_ids[left_index + 1:]:
            right_center, right_normal, _ = plane_data[right_id]
            if float(np.dot(left_normal, right_normal)) > -wall_opposition_cosine:
                continue
            separation = abs(float(np.dot(right_center - left_center, left_normal)))
            if separation < minimum_wall_separation_mm:
                continue
            for bottom_id in face_ids:
                if bottom_id in {left_id, right_id}:
                    continue
                bottom_center, bottom_normal, _ = plane_data[bottom_id]
                # A slot bottom cannot be coplanar/parallel with either wall.
                if (
                    abs(float(np.dot(bottom_normal, left_normal))) > 0.4
                    or abs(float(np.dot(bottom_normal, right_normal))) > 0.4
                ):
                    continue
                left_bottom = face_to_concave_edges[left_id].intersection(
                    face_to_concave_edges[bottom_id]
                )
                right_bottom = face_to_concave_edges[right_id].intersection(
                    face_to_concave_edges[bottom_id]
                )
                if not left_bottom or not right_bottom:
                    continue
                center = (left_center + right_center + bottom_center) / 3.0
                angle, radial_distance = _angle_and_radius(center, origin, axis)
                if radial_distance <= minimum_wall_separation_mm:
                    continue
                concave_edges = tuple(sorted(left_bottom.union(right_bottom)))
                candidates.append(KeySlotCandidate(
                    candidate_id=(
                        f"slot:{left_id}:{right_id}:{bottom_id}"
                    ),
                    wall_face_ids=(left_id, right_id),
                    bottom_face_id=bottom_id,
                    concave_edge_ids=concave_edges,
                    center_point_mm=tuple(float(value) for value in center),
                    wall_normal=tuple(float(value) for value in left_normal),
                    center_angle_degrees=angle,
                    radial_distance_mm=radial_distance,
                    wall_separation_mm=separation,
                    evidence_count=5,
                    reason=(
                        "Two opposing finite planar walls share a nonparallel "
                        "finite bottom through concave edge topology."
                    ),
                ))

    # Different enumeration orders can expose the same three faces only once;
    # preserve a small deterministic audit list rather than score/rank it.
    candidates = sorted(
        candidates,
        key=lambda row: (row.center_angle_degrees, row.candidate_id),
    )
    return {
        "schema_version": "key_slot_evidence.v1",
        "status": "detected" if candidates else "not_detected",
        "topology_available": True,
        "candidates": [row.to_dict() for row in candidates],
        "reason": (
            "Candidates are local geometric/topological evidence only; they do "
            "not establish a functional assembly relation or auto-accept a pose."
            if candidates else
            "Audited topology is available, but no closed concave three-face "
            "slot witness was found."
        ),
    }
