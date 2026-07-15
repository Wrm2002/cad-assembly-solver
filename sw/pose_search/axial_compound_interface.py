"""Conservative compound axial-interface recall and validation.

The existing pair matcher treats a coaxial relation and a planar relation as
independent observations.  That is unsafe for two flange-like interfaces: the
useful hypothesis is the *compound* of an axis-line coincidence and contact of
two large end faces.  This module recalls that hypothesis from anonymous
feature summaries and keeps rotational ambiguity explicit.

All geometry is interpreted in each part's own coordinate frame.  In
particular, end-face normals are compared only with the main axis of the same
part; normals from two unplaced parts are never dotted together.  Returned
poses are proposals, not an acceptance decision.  A periodic hole interface
remains review-only until a paired asymmetric witness (for example, an
audited key-slot) selects one member of its phase orbit.

Phase convention
----------------

``fixed_angle = wrap(axis_polarity * moving_angle + phase_degrees)``

``axis_polarity`` is +1 when the moving main-axis direction maps to the fixed
main-axis direction and -1 when it maps to its negative.  Both branches use
proper rotations (determinant +1); no reflection is ever emitted.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .axial_features import (
    build_circular_patterns,
    extract_axial_circular_features,
)
from .key_slot_features import extract_key_slot_evidence
from .transforms import rotation_about_axis_matrix, unit


PHASE_CONVENTION = (
    "fixed_angle = wrap(axis_polarity * moving_angle + phase_degrees)"
)


def _wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if abs(wrapped + 180.0) < 1e-9 else wrapped


def _angular_distance_degrees(left: float, right: float) -> float:
    return abs(_wrap_degrees(float(left) - float(right)))


def correspondence_phase_degrees(
    fixed_angle_degrees: float,
    moving_angle_degrees: float,
    axis_polarity: int,
) -> float:
    """Return the phase that maps a moving axial direction to a fixed one."""

    if axis_polarity not in {-1, 1}:
        raise ValueError("axis_polarity_must_be_plus_or_minus_one")
    return _wrap_degrees(
        float(fixed_angle_degrees)
        - axis_polarity * float(moving_angle_degrees)
    )


def phase_residual_degrees(
    observed_phase_degrees: float,
    phase_orbit_degrees: Iterable[float],
) -> float:
    """Distance to a discrete symmetry orbit.

    An empty orbit represents an unobservable continuous ``SO(2)`` phase, for
    which no numerical residual can be measured.  It therefore returns zero;
    callers must still keep ``phase_observable``/``phase_status`` as a
    separate gate and must not promote an SO(2) proposal to a closed mate.
    """

    orbit = [float(value) for value in phase_orbit_degrees]
    if not orbit:
        return 0.0
    return min(
        _angular_distance_degrees(observed_phase_degrees, value)
        for value in orbit
    )


def _vector(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (3,) or not bool(np.all(np.isfinite(result))):
        return None
    return result


def _first_present(row: dict[str, Any], *fields: str) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None:
            return value
    return None


def _iter_cylinders(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Union sidecar face nodes AND feature-dict cylinders so that a loaded
    # B-Rep sidecar with incomplete analytic parameters (radius/axis None on
    # some faces) cannot silently disable the correct feature-dict cylinders.
    entities: list[dict[str, Any]] = []
    if "nodes" in source:
        entities.extend(
            row for row in (source.get("nodes") or [])
            if str(row.get("surface_type") or "").lower() == "cylinder"
        )
    entities.extend(list(source.get("cylinders") or []))
    seen: set[str] = set()
    for index, row in enumerate(entities):
        origin = _vector(_first_present(row, "axis_origin", "origin", "centroid"))
        direction = _vector(_first_present(row, "axis_direction", "axis"))
        try:
            radius = float(row.get("radius"))
            area = float(row.get("area") or 0.0)
            direction = None if direction is None else unit(direction)
        except (TypeError, ValueError):
            continue
        if origin is None or direction is None or radius <= 1e-9:
            continue
        fid = str(row.get("node_id") or row.get("feature_id") or f"cylinder:{index}")
        if fid in seen:
            continue
        seen.add(fid)
        rows.append({
            "feature_id": fid,
            "origin": origin,
            "direction": direction,
            "radius_mm": radius,
            "area_mm2": max(0.0, area),
        })
    return rows


def _iter_planes(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    if "nodes" in source:
        entities.extend(
            row for row in (source.get("nodes") or [])
            if str(row.get("surface_type") or "").lower() == "plane"
        )
    entities.extend(list(source.get("planes") or []))
    seen: set[str] = set()
    for index, row in enumerate(entities):
        position = _vector(_first_present(row, "centroid", "position"))
        normal = _vector(row.get("normal"))
        try:
            normal = None if normal is None else unit(normal)
            area = float(row.get("area") or 0.0)
        except (TypeError, ValueError):
            continue
        if position is None or normal is None or area <= 1e-9:
            continue
        fid = str(row.get("node_id") or row.get("feature_id") or f"plane:{index}")
        if fid in seen:
            continue
        seen.add(fid)
        rows.append({
            "feature_id": fid,
            "position": position,
            "normal": normal,
            "area_mm2": area,
        })
    return rows


def _serialize_vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in value]


def _main_axis(source: dict[str, Any]) -> dict[str, Any] | None:
    """Select a recall-oriented main cylindrical axis.

    Off-axis bolt holes are normally much smaller than the central bore, hub,
    or pipe cylinder.  Radius is therefore the primary signal and finite face
    area is a deterministic tie breaker.  No part name or relation label is
    consulted.
    """

    cylinders = _iter_cylinders(source)
    if not cylinders:
        return None
    selected = max(
        cylinders,
        key=lambda row: (row["radius_mm"], row["area_mm2"], row["feature_id"]),
    )
    return {
        "feature_id": selected["feature_id"],
        "origin": _serialize_vector(selected["origin"]),
        "direction": _serialize_vector(selected["direction"]),
        "radius_mm": float(selected["radius_mm"]),
        "area_mm2": float(selected["area_mm2"]),
    }


def _large_end_faces(
    source: dict[str, Any],
    axis: dict[str, Any],
    *,
    axis_parallel_cosine: float,
    minimum_relative_area: float,
    minimum_area_mm2: float,
) -> list[dict[str, Any]]:
    """Find large planes normal to this part's own cylindrical axis."""

    direction = unit(np.asarray(axis["direction"], dtype=float))
    aligned: list[dict[str, Any]] = []
    for row in _iter_planes(source):
        signed_alignment = float(np.dot(row["normal"], direction))
        if abs(signed_alignment) < axis_parallel_cosine:
            continue
        aligned.append({
            "feature_id": row["feature_id"],
            "position": _serialize_vector(row["position"]),
            "normal": _serialize_vector(row["normal"]),
            "area_mm2": float(row["area_mm2"]),
            "own_axis_alignment": abs(signed_alignment),
            "normal_sign_along_own_axis": 1 if signed_alignment >= 0.0 else -1,
        })
    if not aligned:
        return []
    maximum_area = max(row["area_mm2"] for row in aligned)
    area_floor = max(float(minimum_area_mm2), maximum_area * minimum_relative_area)
    return sorted(
        [row for row in aligned if row["area_mm2"] >= area_floor],
        key=lambda row: (-row["area_mm2"], row["feature_id"]),
    )


def _axis_basis(direction: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Use the same deterministic right-handed basis as axial_features."""

    axis = unit(np.asarray(direction, dtype=float))
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = unit(np.cross(axis, seed))
    second = unit(np.cross(axis, first))
    return first, second


def _axis_frame(direction: Iterable[float]) -> np.ndarray:
    axis = unit(np.asarray(direction, dtype=float))
    first, second = _axis_basis(axis)
    return np.column_stack((first, second, axis))


def _pattern_descriptors(
    source: dict[str, Any], axis: dict[str, Any]
) -> list[dict[str, Any]]:
    features = extract_axial_circular_features(
        source,
        axis["origin"],
        axis["direction"],
    )
    patterns = build_circular_patterns(features)
    by_id = {row.feature_id: row for row in features}
    output: list[dict[str, Any]] = []
    for pattern in patterns:
        members = [by_id[value] for value in pattern.feature_ids if value in by_id]
        pitch = (
            float(np.mean([row.radial_distance_mm for row in members]))
            if members else 0.0
        )
        output.append({
            "pattern_id": pattern.pattern_id,
            "hole_radius_mm": float(pattern.radius_mm),
            "pitch_radius_mm": pitch,
            "angles_degrees": [float(value) for value in pattern.angles_degrees],
            "periodicity": pattern.periodicity,
            "geometry_symmetric": bool(pattern.geometry_symmetric),
            "feature_ids": list(pattern.feature_ids),
        })
    return output


def _compatible_pattern_pair(
    fixed_patterns: list[dict[str, Any]],
    moving_patterns: list[dict[str, Any]],
    *,
    hole_radius_tolerance_mm: float,
    pitch_radius_relative_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    compatible: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for fixed in fixed_patterns:
        for moving in moving_patterns:
            order = fixed.get("periodicity")
            if order is None or order < 2 or order != moving.get("periodicity"):
                continue
            hole_delta = abs(fixed["hole_radius_mm"] - moving["hole_radius_mm"])
            if hole_delta > hole_radius_tolerance_mm:
                continue
            pitch_max = max(
                fixed["pitch_radius_mm"], moving["pitch_radius_mm"], 1e-9
            )
            pitch_delta = abs(
                fixed["pitch_radius_mm"] - moving["pitch_radius_mm"]
            )
            if pitch_delta / pitch_max > pitch_radius_relative_tolerance:
                continue
            quality = (
                float(order),
                -pitch_delta / pitch_max,
                -hole_delta,
            )
            compatible.append((quality, fixed, moving))
    if not compatible:
        return None
    _, fixed, moving = max(compatible, key=lambda row: row[0])
    return fixed, moving


def _pattern_matching_fraction(
    fixed_angles: list[float],
    moving_angles: list[float],
    *,
    axis_polarity: int,
    phase_degrees: float,
    tolerance_degrees: float,
) -> float:
    matched = 0
    for moving in moving_angles:
        mapped = _wrap_degrees(axis_polarity * moving + phase_degrees)
        if min(
            _angular_distance_degrees(mapped, fixed)
            for fixed in fixed_angles
        ) <= tolerance_degrees:
            matched += 1
    return matched / max(len(fixed_angles), len(moving_angles), 1)


def _pattern_phase_orbit(
    fixed: dict[str, Any],
    moving: dict[str, Any],
    *,
    axis_polarity: int,
    tolerance_degrees: float,
) -> list[float]:
    phases = {
        round(correspondence_phase_degrees(left, right, axis_polarity), 7)
        for left in fixed["angles_degrees"]
        for right in moving["angles_degrees"]
    }
    scored = [
        (
            _pattern_matching_fraction(
                fixed["angles_degrees"],
                moving["angles_degrees"],
                axis_polarity=axis_polarity,
                phase_degrees=phase,
                tolerance_degrees=tolerance_degrees,
            ),
            phase,
        )
        for phase in phases
    ]
    if not scored:
        return []
    maximum = max(row[0] for row in scored)
    if maximum < 0.75:
        return []
    return sorted(
        [_wrap_degrees(phase) for score, phase in scored if score >= maximum - 1e-9]
    )


def _directional_witnesses(
    source: dict[str, Any], axis: dict[str, Any]
) -> list[dict[str, Any]]:
    """Collect audited slots plus explicitly topology-backed generic cues."""

    rows: list[dict[str, Any]] = []
    slot_evidence = extract_key_slot_evidence(
        source, axis["origin"], axis["direction"]
    )
    if slot_evidence.get("status") == "detected":
        for candidate in slot_evidence.get("candidates") or []:
            rows.append({
                "witness_id": str(candidate["candidate_id"]),
                "kind": "topological_key_slot",
                "angle_degrees": float(candidate["center_angle_degrees"]),
                "topology_supported": True,
            })
    # This optional contract allows future audited asymmetric features to use
    # the same quotient logic.  Unqualified planar fragments are ignored.
    for index, candidate in enumerate(
        source.get("axial_orientation_witnesses") or []
    ):
        if not (
            candidate.get("asymmetric") is True
            and candidate.get("topology_supported") is True
        ):
            continue
        try:
            angle = float(candidate["angle_degrees"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "witness_id": str(candidate.get("witness_id") or f"witness:{index}"),
            "kind": str(candidate.get("kind") or "audited_asymmetric_witness"),
            "angle_degrees": _wrap_degrees(angle),
            "topology_supported": True,
        })
    return sorted(rows, key=lambda row: (row["angle_degrees"], row["witness_id"]))


def _phase_information(
    fixed_patterns: list[dict[str, Any]],
    moving_patterns: list[dict[str, Any]],
    fixed_witnesses: list[dict[str, Any]],
    moving_witnesses: list[dict[str, Any]],
    *,
    axis_polarity: int,
    hole_radius_tolerance_mm: float,
    pitch_radius_relative_tolerance: float,
    pattern_tolerance_degrees: float,
    witness_tolerance_degrees: float,
) -> dict[str, Any]:
    pair = _compatible_pattern_pair(
        fixed_patterns,
        moving_patterns,
        hole_radius_tolerance_mm=hole_radius_tolerance_mm,
        pitch_radius_relative_tolerance=pitch_radius_relative_tolerance,
    )
    interface_orbit: list[float] = []
    interface_order: int | None = None
    pattern_evidence: dict[str, Any] | None = None
    if pair is not None:
        fixed_pattern, moving_pattern = pair
        interface_orbit = _pattern_phase_orbit(
            fixed_pattern,
            moving_pattern,
            axis_polarity=axis_polarity,
            tolerance_degrees=pattern_tolerance_degrees,
        )
        if interface_orbit:
            interface_order = int(fixed_pattern["periodicity"])
            pattern_evidence = {
                "fixed_pattern_id": fixed_pattern["pattern_id"],
                "moving_pattern_id": moving_pattern["pattern_id"],
                "fixed_feature_ids": fixed_pattern["feature_ids"],
                "moving_feature_ids": moving_pattern["feature_ids"],
            }

    witness_rows: list[dict[str, Any]] = []
    for fixed in fixed_witnesses:
        for moving in moving_witnesses:
            phase = correspondence_phase_degrees(
                fixed["angle_degrees"], moving["angle_degrees"], axis_polarity
            )
            if interface_orbit:
                nearest = min(
                    interface_orbit,
                    key=lambda value: _angular_distance_degrees(value, phase),
                )
                residual = _angular_distance_degrees(nearest, phase)
            else:
                nearest = phase
                residual = 0.0
            witness_rows.append({
                "fixed_witness_id": fixed["witness_id"],
                "moving_witness_id": moving["witness_id"],
                "fixed_kind": fixed["kind"],
                "moving_kind": moving["kind"],
                "witness_phase_degrees": phase,
                "selected_interface_phase_degrees": nearest,
                "residual_degrees": residual,
            })

    if witness_rows:
        best_residual = min(row["residual_degrees"] for row in witness_rows)
        best_rows = [
            row for row in witness_rows
            if row["residual_degrees"] <= best_residual + 1e-9
        ]
        selected = sorted({
            round(float(row["selected_interface_phase_degrees"]), 7)
            for row in best_rows
        })
        if best_residual > witness_tolerance_degrees:
            status = "witness_conflict"
            active_orbit = interface_orbit
        elif len(selected) != 1:
            status = "multiple_witness_alignments"
            active_orbit = interface_orbit or [float(value) for value in selected]
        else:
            status = "resolved_by_asymmetric_witness"
            active_orbit = [_wrap_degrees(selected[0])]
        whole_order: int | None = 1
        residual: float | None = float(best_residual)
        phase_witness = best_rows
    elif fixed_witnesses or moving_witnesses:
        status = "asymmetric_witness_unpaired"
        active_orbit = interface_orbit
        whole_order = 1
        residual = None
        phase_witness = []
    elif interface_orbit:
        status = "periodic_interface_only"
        active_orbit = interface_orbit
        whole_order = interface_order
        residual = 0.0
        phase_witness = []
    else:
        status = "continuous_phase_unobservable"
        active_orbit = []
        whole_order = None
        residual = None
        phase_witness = []

    return {
        "axis_polarity": int(axis_polarity),
        "phase_convention": PHASE_CONVENTION,
        "symmetry_group": (
            f"C{interface_order}" if interface_order is not None else "SO(2)"
        ),
        "interface_symmetry_order": interface_order,
        "interface_phase_orbit_degrees": interface_orbit,
        "whole_part_symmetry_order": whole_order,
        "phase_orbit_degrees": active_orbit,
        "phase_residual_deg": residual,
        "phase_status": status,
        "phase_witness": phase_witness,
        "pattern_evidence": pattern_evidence,
    }


def construct_compound_transform(
    candidate: dict[str, Any],
    *,
    axis_polarity: int,
    phase_degrees: float,
) -> np.ndarray:
    """Construct a proper pose satisfying axis line and end-plane position.

    Radial axis coincidence is established first.  Translation is then varied
    only along the fixed axis to put the moving end face on the fixed end
    plane, so face contact cannot destroy coaxiality.
    """

    if axis_polarity not in {-1, 1}:
        raise ValueError("axis_polarity_must_be_plus_or_minus_one")
    fixed_axis = candidate["fixed_axis"]
    moving_axis = candidate["moving_axis"]
    fixed_face = candidate["fixed_end_face"]
    moving_face = candidate["moving_end_face"]

    fixed_direction = unit(np.asarray(fixed_axis["direction"], dtype=float))
    moving_direction = unit(np.asarray(moving_axis["direction"], dtype=float))
    fixed_frame = _axis_frame(fixed_direction)
    moving_frame = _axis_frame(moving_direction)
    if axis_polarity == 1:
        target_frame = fixed_frame
    else:
        # Two sign changes keep this frame right handed while mapping the
        # moving axis to -fixed_axis and moving azimuth theta to -theta.
        target_frame = fixed_frame @ np.diag([1.0, -1.0, -1.0])
    base_rotation = target_frame @ moving_frame.T
    phase_rotation = rotation_about_axis_matrix(
        [0.0, 0.0, 0.0], fixed_direction, phase_degrees
    )[:3, :3]
    rotation = phase_rotation @ base_rotation

    fixed_origin = np.asarray(fixed_axis["origin"], dtype=float)
    moving_origin = np.asarray(moving_axis["origin"], dtype=float)
    translation = fixed_origin - rotation @ moving_origin

    fixed_normal = unit(np.asarray(fixed_face["normal"], dtype=float))
    fixed_position = np.asarray(fixed_face["position"], dtype=float)
    moving_position = np.asarray(moving_face["position"], dtype=float)
    transformed_position = rotation @ moving_position + translation
    denominator = float(np.dot(fixed_normal, fixed_direction))
    if abs(denominator) <= 1e-9:
        raise ValueError("fixed_end_face_is_not_normal_to_fixed_axis")
    axial_shift = float(
        np.dot(fixed_normal, fixed_position - transformed_position) / denominator
    )
    translation = translation + axial_shift * fixed_direction

    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise RuntimeError("compound_transform_must_be_a_proper_rotation")
    return result


def _rotation_phase_degrees(
    rotation: np.ndarray,
    fixed_direction: Iterable[float],
    moving_direction: Iterable[float],
) -> float:
    fixed_first, fixed_second = _axis_basis(fixed_direction)
    moving_first, _ = _axis_basis(moving_direction)
    mapped = rotation @ moving_first
    return _wrap_degrees(math.degrees(math.atan2(
        float(np.dot(mapped, fixed_second)),
        float(np.dot(mapped, fixed_first)),
    )))


def validate_axial_compound_pose(
    candidate: dict[str, Any],
    proposal: dict[str, Any],
    *,
    collision_free: bool,
    transform: np.ndarray | Iterable[Iterable[float]] | None = None,
    axis_angle_tolerance_degrees: float = 1.0,
    radial_tolerance_mm: float = 0.1,
    plane_tolerance_mm: float = 0.1,
    face_angle_tolerance_degrees: float = 1.0,
    phase_tolerance_degrees: float = 2.0,
) -> dict[str, Any]:
    """Validate all compound constraints without equating them to semantics."""

    matrix = np.asarray(
        proposal["transform"] if transform is None else transform,
        dtype=float,
    )
    if matrix.shape != (4, 4):
        raise ValueError("transform_must_be_4x4")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    polarity = int(proposal["axis_polarity"])

    fixed_axis = candidate["fixed_axis"]
    moving_axis = candidate["moving_axis"]
    fixed_face = candidate["fixed_end_face"]
    moving_face = candidate["moving_end_face"]
    fixed_direction = unit(np.asarray(fixed_axis["direction"], dtype=float))
    moving_direction = unit(np.asarray(moving_axis["direction"], dtype=float))
    mapped_direction = unit(rotation @ moving_direction)
    target_direction = polarity * fixed_direction
    axis_angle = math.degrees(math.acos(float(np.clip(
        np.dot(mapped_direction, target_direction), -1.0, 1.0
    ))))

    fixed_origin = np.asarray(fixed_axis["origin"], dtype=float)
    mapped_origin = rotation @ np.asarray(
        moving_axis["origin"], dtype=float
    ) + translation
    origin_delta = mapped_origin - fixed_origin
    radial_residual = float(np.linalg.norm(
        origin_delta - float(np.dot(origin_delta, fixed_direction)) * fixed_direction
    ))

    fixed_normal = unit(np.asarray(fixed_face["normal"], dtype=float))
    moving_normal = unit(np.asarray(moving_face["normal"], dtype=float))
    mapped_normal = unit(rotation @ moving_normal)
    face_angle = math.degrees(math.acos(float(np.clip(
        np.dot(mapped_normal, -fixed_normal), -1.0, 1.0
    ))))
    fixed_position = np.asarray(fixed_face["position"], dtype=float)
    mapped_position = rotation @ np.asarray(
        moving_face["position"], dtype=float
    ) + translation
    plane_residual = abs(float(np.dot(
        fixed_normal, mapped_position - fixed_position
    )))

    observed_phase = _rotation_phase_degrees(
        rotation, fixed_direction, moving_direction
    )
    orbit = list(proposal.get("phase_orbit_degrees") or [])
    phase_residual = phase_residual_degrees(observed_phase, orbit)
    phase_observable = bool(orbit)
    phase_numerically_valid = (
        phase_observable and phase_residual <= phase_tolerance_degrees
    )
    determinant = float(np.linalg.det(rotation))
    proper_rotation = abs(determinant - 1.0) <= 1e-6

    geometric_checks = {
        "proper_rotation": proper_rotation,
        "coaxial_direction": axis_angle <= axis_angle_tolerance_degrees,
        "radial_axis_center": radial_residual <= radial_tolerance_mm,
        "end_face_contact": plane_residual <= plane_tolerance_mm,
        "opposed_end_face_normals": face_angle <= face_angle_tolerance_degrees,
        "phase_in_active_orbit": phase_numerically_valid,
    }
    compound_constraints_satisfied = all(geometric_checks.values())
    phase_resolved = (
        proposal.get("phase_status") == "resolved_by_asymmetric_witness"
        and len(orbit) == 1
    )
    proposal_only = bool(proposal.get("proposal_only", True))
    is_closed = bool(
        collision_free
        and compound_constraints_satisfied
        and phase_resolved
        and not proposal_only
    )
    reasons = [name for name, passed in geometric_checks.items() if not passed]
    if not collision_free:
        reasons.append("collision")
    if not phase_resolved:
        reasons.append("phase_not_uniquely_resolved")
    if proposal_only:
        reasons.append("proposal_only")
    return {
        "axis_polarity": polarity,
        "phase_convention": proposal.get("phase_convention", PHASE_CONVENTION),
        "phase_orbit_degrees": orbit,
        "whole_part_symmetry_order": proposal.get("whole_part_symmetry_order"),
        "axis_angle_residual_deg": float(axis_angle),
        "radial_center_residual_mm": radial_residual,
        "end_face_distance_residual_mm": plane_residual,
        "end_face_normal_residual_deg": float(face_angle),
        "observed_phase_degrees": observed_phase,
        "phase_residual_deg": float(phase_residual),
        "rotation_determinant": determinant,
        "collision_free": bool(collision_free),
        "checks": geometric_checks,
        "compound_constraints_satisfied": compound_constraints_satisfied,
        "proposal_only": proposal_only,
        "review_required": not is_closed,
        "is_closed": is_closed,
        "reasons": reasons,
    }


def recall_axial_compound_candidates(
    fixed_source: dict[str, Any],
    moving_source: dict[str, Any],
    *,
    axis_parallel_cosine: float = 0.985,
    minimum_end_face_relative_area: float = 0.5,
    minimum_end_face_area_mm2: float = 0.0,
    minimum_face_area_ratio: float = 0.35,
    hole_radius_tolerance_mm: float = 0.5,
    pitch_radius_relative_tolerance: float = 0.08,
    pattern_tolerance_degrees: float = 2.0,
    witness_tolerance_degrees: float = 2.0,
) -> dict[str, Any]:
    """Recall compound axis/end-face candidates from anonymous summaries."""

    fixed_axis = _main_axis(fixed_source)
    moving_axis = _main_axis(moving_source)
    if fixed_axis is None or moving_axis is None:
        return {
            "schema_version": "axial_compound_interface.v1",
            "status": "not_recalled",
            "reason": "Both parts require a finite cylindrical main axis.",
            "candidates": [],
        }
    fixed_faces = _large_end_faces(
        fixed_source,
        fixed_axis,
        axis_parallel_cosine=axis_parallel_cosine,
        minimum_relative_area=minimum_end_face_relative_area,
        minimum_area_mm2=minimum_end_face_area_mm2,
    )
    moving_faces = _large_end_faces(
        moving_source,
        moving_axis,
        axis_parallel_cosine=axis_parallel_cosine,
        minimum_relative_area=minimum_end_face_relative_area,
        minimum_area_mm2=minimum_end_face_area_mm2,
    )
    if not fixed_faces or not moving_faces:
        return {
            "schema_version": "axial_compound_interface.v1",
            "status": "not_recalled",
            "reason": (
                "Both parts require a large end face normal to their own main axis."
            ),
            "fixed_main_axis": fixed_axis,
            "moving_main_axis": moving_axis,
            "candidates": [],
        }

    fixed_patterns = _pattern_descriptors(fixed_source, fixed_axis)
    moving_patterns = _pattern_descriptors(moving_source, moving_axis)
    fixed_witnesses = _directional_witnesses(fixed_source, fixed_axis)
    moving_witnesses = _directional_witnesses(moving_source, moving_axis)
    candidates: list[dict[str, Any]] = []
    for fixed_face in fixed_faces:
        for moving_face in moving_faces:
            area_ratio = min(
                fixed_face["area_mm2"], moving_face["area_mm2"]
            ) / max(fixed_face["area_mm2"], moving_face["area_mm2"])
            if area_ratio < minimum_face_area_ratio:
                continue
            required_polarity = -(
                int(fixed_face["normal_sign_along_own_axis"])
                * int(moving_face["normal_sign_along_own_axis"])
            )
            candidate = {
                "candidate_id": (
                    f"compound:{fixed_face['feature_id']}:{moving_face['feature_id']}"
                ),
                "candidate_type": "compound_axial_end_face",
                "fixed_axis": fixed_axis,
                "moving_axis": moving_axis,
                "fixed_end_face": fixed_face,
                "moving_end_face": moving_face,
                "end_face_area_ratio": float(area_ratio),
                "required_axis_polarity_for_opposed_faces": required_polarity,
                "required_constraints": [
                    "coaxial_direction",
                    "radial_axis_center",
                    "end_face_contact",
                    "opposed_end_face_normals",
                    "phase_in_active_orbit",
                ],
                "normal_recall_method": "each_end_face_vs_its_own_main_axis",
                "proposals": [],
            }
            for polarity in (1, -1):
                phase_info = _phase_information(
                    fixed_patterns,
                    moving_patterns,
                    fixed_witnesses,
                    moving_witnesses,
                    axis_polarity=polarity,
                    hole_radius_tolerance_mm=hole_radius_tolerance_mm,
                    pitch_radius_relative_tolerance=pitch_radius_relative_tolerance,
                    pattern_tolerance_degrees=pattern_tolerance_degrees,
                    witness_tolerance_degrees=witness_tolerance_degrees,
                )
                materialized_phases = phase_info["phase_orbit_degrees"] or [0.0]
                for phase in materialized_phases:
                    transform = construct_compound_transform(
                        candidate,
                        axis_polarity=polarity,
                        phase_degrees=phase,
                    )
                    face_compatible = polarity == required_polarity
                    uniquely_resolved = (
                        phase_info["phase_status"]
                        == "resolved_by_asymmetric_witness"
                        and len(phase_info["phase_orbit_degrees"]) == 1
                    )
                    proposal_only = not (face_compatible and uniquely_resolved)
                    reasons: list[str] = []
                    if not face_compatible:
                        reasons.append("axis_polarity_does_not_oppose_end_face_normals")
                    if not uniquely_resolved:
                        reasons.append(phase_info["phase_status"])
                    candidate["proposals"].append({
                        "proposal_id": (
                            f"{candidate['candidate_id']}:p{polarity:+d}:"
                            f"phase{_wrap_degrees(phase):.7g}"
                        ),
                        **phase_info,
                        "phase_degrees": float(_wrap_degrees(phase)),
                        "transform": transform.tolist(),
                        "rotation_determinant": float(np.linalg.det(transform[:3, :3])),
                        "end_face_orientation_compatible": face_compatible,
                        "proposal_only": proposal_only,
                        "review_required": proposal_only,
                        "reasons": reasons,
                    })
            candidates.append(candidate)

    # A single locally resolved proposal may be closed by the validator.  If
    # several face/polarity proposals survive, recall has not identified a
    # unique compound interface and every survivor must remain review-only.
    resolved_survivors = [
        proposal
        for candidate in candidates
        for proposal in candidate["proposals"]
        if proposal["end_face_orientation_compatible"]
        and proposal["phase_status"] == "resolved_by_asymmetric_witness"
        and len(proposal["phase_orbit_degrees"]) == 1
    ]
    if len(resolved_survivors) > 1:
        for proposal in resolved_survivors:
            proposal["proposal_only"] = True
            proposal["review_required"] = True
            if "multiple_compound_face_or_pose_proposals" not in proposal["reasons"]:
                proposal["reasons"].append(
                    "multiple_compound_face_or_pose_proposals"
                )

    return {
        "schema_version": "axial_compound_interface.v1",
        "status": "recalled" if candidates else "not_recalled",
        "reason": (
            "Compound candidates retain axis, radial-center, end-face, and phase evidence."
            if candidates else
            "Large own-axis end faces were found, but their areas were incompatible."
        ),
        "fixed_main_axis": fixed_axis,
        "moving_main_axis": moving_axis,
        "fixed_end_faces": fixed_faces,
        "moving_end_faces": moving_faces,
        "fixed_circular_patterns": fixed_patterns,
        "moving_circular_patterns": moving_patterns,
        "fixed_directional_witnesses": fixed_witnesses,
        "moving_directional_witnesses": moving_witnesses,
        "candidates": candidates,
        "auto_accept": False,
    }


__all__ = [
    "PHASE_CONVENTION",
    "construct_compound_transform",
    "correspondence_phase_degrees",
    "phase_residual_degrees",
    "recall_axial_compound_candidates",
    "validate_axial_compound_pose",
]
