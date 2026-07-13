"""Lift learned B-Rep entity-pair scores into geometric joint manifolds.

The learned decision is *which local entities may form an interface*.  The
analytic B-Rep attached to those entities supplies an origin and direction.
No mechanical family name is decoded: the remaining freedom is represented
as a subspace of se(3), so an axis match is not collapsed into one transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Iterable

import numpy as np


_AXIAL_TOKENS = ("cylinder", "cone", "circle", "ellipse", "line")
_PLANAR_TOKENS = ("plane",)


def _unit(value: Any) -> np.ndarray | None:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not bool(np.all(np.isfinite(array))):
        return None
    norm = float(np.linalg.norm(array))
    return None if norm <= 1e-12 else array / norm


def _orthogonal_axis(z_axis: np.ndarray) -> np.ndarray:
    basis = np.eye(3)[int(np.argmin(np.abs(z_axis)))]
    x_axis = basis - float(np.dot(basis, z_axis)) * z_axis
    return x_axis / np.linalg.norm(x_axis)


def frame_from_entity(entity: dict[str, Any]) -> np.ndarray | None:
    """Construct a deterministic local frame from one analytic B-Rep entity."""

    origin = entity.get("axis_origin") or entity.get("centroid")
    direction = entity.get("axis_direction") or entity.get("normal")
    origin_array = np.asarray(origin, dtype=float)
    z_axis = _unit(direction)
    if origin_array.shape != (3,) or not np.all(np.isfinite(origin_array)) or z_axis is None:
        return None
    witness = _unit(entity.get("local_direction", []))
    if witness is not None:
        witness = witness - float(np.dot(witness, z_axis)) * z_axis
        if float(np.linalg.norm(witness)) > 1e-8:
            x_axis = witness / np.linalg.norm(witness)
        else:
            x_axis = _orthogonal_axis(z_axis)
    else:
        x_axis = _orthogonal_axis(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    frame[:3, 3] = origin_array
    return frame


def _geometry_family(entity: dict[str, Any]) -> str:
    text = str(entity.get("geometry_type", "")).lower()
    if any(token in text for token in _PLANAR_TOKENS):
        return "planar"
    if any(token in text for token in _AXIAL_TOKENS):
        return "axial"
    return "directional"


def _manifold(entity_a: dict[str, Any], entity_b: dict[str, Any]) -> tuple[str, tuple[int, ...], dict[str, Any]]:
    family_a, family_b = _geometry_family(entity_a), _geometry_family(entity_b)
    asymmetric_witness = (
        entity_a.get("local_direction") is not None
        and entity_b.get("local_direction") is not None
        and float(entity_a.get("local_asymmetry_score", 0.0)) >= 0.15
        and float(entity_b.get("local_asymmetry_score", 0.0)) >= 0.15
    )
    if family_a == family_b == "planar":
        return (
            "plane_coincidence",
            (1, 1, 0, 0, 0, 0 if asymmetric_witness else 1),
            {
                "group": "R2" if asymmetric_witness else "SE2",
                "continuous": True,
                "axis": "local_z",
                "local_asymmetric_witness": asymmetric_witness,
            },
        )
    if "axial" in {family_a, family_b}:
        return (
            "axis_coincidence",
            (0, 0, 1, 0, 0, 0 if asymmetric_witness else 1),
            {
                "group": "R" if asymmetric_witness else "R_x_SO2",
                "continuous": True,
                "axis": "local_z",
                "local_asymmetric_witness": asymmetric_witness,
            },
        )
    return (
        "direction_coincidence",
        (0, 0, 1, 0, 0, 0 if asymmetric_witness else 1),
        {
            "group": "R" if asymmetric_witness else "R_x_SO2",
            "continuous": True,
            "axis": "local_z",
            "local_asymmetric_witness": asymmetric_witness,
        },
    )


def _phase_transform(polarity: int, phase_degrees: float) -> np.ndarray:
    phase = math.radians(float(phase_degrees))
    cz, sz = math.cos(phase), math.sin(phase)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    flip = np.diag([1.0, -1.0, -1.0]) if polarity < 0 else np.eye(3)
    result = np.eye(4)
    result[:3, :3] = flip @ rz
    return result


@dataclass(frozen=True)
class JointHypothesis:
    source: str
    target: str
    entity_a: str
    entity_b: str
    manifold_type: str
    frame_a: tuple[tuple[float, ...], ...]
    frame_b: tuple[tuple[float, ...], ...]
    free_dof_mask: tuple[int, int, int, int, int, int]
    initial_pose_b_in_a: tuple[tuple[float, ...], ...]
    confidence: float
    rank: int
    polarity: int
    phase_degrees: float
    symmetry_class: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def constrained_dof_mask(self) -> tuple[int, ...]:
        return tuple(1 - int(value) for value in self.free_dof_mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "manifold_type": self.manifold_type,
            "frame_a": [list(row) for row in self.frame_a],
            "frame_b": [list(row) for row in self.frame_b],
            "free_dof_mask": list(self.free_dof_mask),
            "constrained_dof_mask": list(self.constrained_dof_mask),
            "initial_pose_b_in_a": [list(row) for row in self.initial_pose_b_in_a],
            "confidence": self.confidence,
            "rank": self.rank,
            "polarity": self.polarity,
            "phase_degrees": self.phase_degrees,
            "symmetry_class": self.symmetry_class,
            "provenance": self.provenance,
        }


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _candidate_phases(candidate: dict[str, Any], maximum: int) -> list[float]:
    values = [0.0]
    for row in candidate.get("rotation_hypotheses") or []:
        try:
            value = row.get("rotation_degrees", row) if isinstance(row, dict) else row
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    unique = []
    for value in values:
        canonical = (value + 180.0) % 360.0 - 180.0
        if not any(abs(canonical - kept) < 1e-6 for kept in unique):
            unique.append(canonical)
    return unique[: max(1, int(maximum))]


def build_joint_hypotheses(
    source: str,
    target: str,
    learned_candidates: Iterable[dict[str, Any]],
    *,
    maximum_phases_per_entity_pair: int = 4,
    enumerate_polarity: bool = True,
) -> list[JointHypothesis]:
    """Build a bounded, symmetry-aware manifold frontier from JoinABLe top-k."""

    output = []
    for candidate in learned_candidates:
        entity_a, entity_b = candidate.get("node_a"), candidate.get("node_b")
        if not isinstance(entity_a, dict) or not isinstance(entity_b, dict):
            continue
        frame_a, frame_b = frame_from_entity(entity_a), frame_from_entity(entity_b)
        if frame_a is None or frame_b is None:
            continue
        manifold_type, free_mask, symmetry = _manifold(entity_a, entity_b)
        polarities = (1, -1) if enumerate_polarity else (1,)
        phases = _candidate_phases(candidate, maximum_phases_per_entity_pair)
        if symmetry.get("local_asymmetric_witness") and not any(
            abs(abs(value) - 180.0) < 1e-6 for value in phases
        ):
            phases = (phases + [-180.0])[: max(1, int(maximum_phases_per_entity_pair))]
        for polarity in polarities:
            for phase in phases:
                symmetry_transform = _phase_transform(polarity, phase)
                initial = frame_a @ np.linalg.inv(frame_b @ symmetry_transform)
                if abs(float(np.linalg.det(initial[:3, :3])) - 1.0) > 1e-6:
                    continue
                output.append(JointHypothesis(
                    source=str(source),
                    target=str(target),
                    entity_a=str(entity_a.get("entity_id", entity_a.get("topology_index", ""))),
                    entity_b=str(entity_b.get("entity_id", entity_b.get("topology_index", ""))),
                    manifold_type=manifold_type,
                    frame_a=_matrix_tuple(frame_a),
                    frame_b=_matrix_tuple(frame_b @ symmetry_transform),
                    free_dof_mask=free_mask,
                    initial_pose_b_in_a=_matrix_tuple(initial),
                    confidence=float(candidate.get("probability", candidate.get("confidence", 0.0))),
                    rank=int(candidate.get("rank", len(output) + 1)),
                    polarity=polarity,
                    phase_degrees=phase,
                    symmetry_class={**symmetry, "axis_polarity_enumerated": enumerate_polarity},
                    provenance={
                        "learned_entity_pair": True,
                        "geometry_lift": "analytic_brep_local_frame",
                        "fixed_relative_pose": False,
                    },
                ))
    # For each learned entity pair, cover axis/normal polarity before spending
    # additional slots on phase variants of the same polarity.
    output.sort(key=lambda row: (
        -row.confidence,
        row.rank,
        abs(row.phase_degrees),
        -row.polarity,
    ))
    return output


def attach_pose_initials(
    hypotheses: Iterable[JointHypothesis],
    pose_results: Iterable[dict[str, Any]],
) -> list[JointHypothesis]:
    """Attach pair-search transforms as initial points, never as hard factors."""

    base = list(hypotheses)
    by_entities: dict[tuple[str, str], list[JointHypothesis]] = {}
    for row in base:
        by_entities.setdefault((row.entity_a, row.entity_b), []).append(row)
    enriched = []
    seen = set()
    for index, pose in enumerate(pose_results):
        key = (str(pose.get("entity_a")), str(pose.get("entity_b")))
        matches = by_entities.get(key) or []
        if not matches:
            continue
        try:
            transform = np.asarray(pose.get("transform"), dtype=float)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                continue
            if abs(float(np.linalg.det(transform[:3, :3])) - 1.0) > 1e-4:
                continue
        except (TypeError, ValueError):
            continue
        desired_polarity = -1 if bool(pose.get("axis_flip")) else 1
        desired_phase = float(pose.get("rotation_seed_degrees", 0.0))
        source = min(matches, key=lambda row: (
            row.polarity != desired_polarity,
            abs(((row.phase_degrees - desired_phase + 180.0) % 360.0) - 180.0),
        ))
        exact = pose.get("exact_collision") or {}
        exact_status = str(exact.get("status", "not_checked"))
        collision_free = exact_status == "success" and not exact.get("collisions")
        evaluation = pose.get("evaluation") or {}
        signature = (
            source.entity_a,
            source.entity_b,
            tuple(np.round(transform.reshape(-1), 5)),
        )
        if signature in seen:
            continue
        seen.add(signature)
        provenance = dict(source.provenance)
        provenance.update({
            "pose_search_initial": True,
            "pose_search_result_index": index,
            "pair_search_cost": float(evaluation.get("cost", float("inf"))),
            "pair_search_contact": float(evaluation.get("contact", 0.0)),
            "pair_search_overlap": float(evaluation.get("overlap", 0.0)),
            "pair_exact_status": exact_status,
            "pair_exact_collision_free": collision_free,
            "initial_pose_is_constraint": False,
        })
        enriched.append(replace(
            source,
            initial_pose_b_in_a=_matrix_tuple(transform),
            polarity=desired_polarity,
            phase_degrees=float(pose.get("rotation_degrees", desired_phase)),
            provenance=provenance,
        ))
    # Search initials lead because they contain lifecycle evidence.  Analytic
    # frame alignment stays as a fallback when pair search is unavailable.
    enriched.sort(key=lambda row: (
        not bool(row.provenance.get("pair_exact_collision_free")),
        float(row.provenance.get("pair_search_cost", float("inf"))),
        row.rank,
    ))
    return enriched + base
