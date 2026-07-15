"""Conservative repeated edge-slot interface pose proposals.

The detector is intended for thin elongated components which seat one long
edge in one member of a repeated connector/slot family.  It consumes audited
geometry only: a stationary carrier OBB, stationary planar footprints, and a
moving OBB.  Names, file paths, case identifiers and semantic labels are never
read.

Four independent geometric evidence families are required before a proposal
is emitted:

* the moving and slot long dimensions are compatible;
* each slot floor is bounded by a geometrically mirrored pair of sloped walls;
* at least three bounded, parallel channels form an approximately equally
  spaced family; and
* the carrier thin OBB axis supplies a physically meaningful insertion axis
  for the thin elongated moving OBB.

An isolated elongated face, a repeated rail pattern without side walls, or a
face coincident with an outward carrier boundary is deliberately insufficient.
Every output remains proposal-only and must pass downstream exact collision
and engineering review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dominant_planar_envelope import derive_dominant_planar_envelope
from .transforms import matrix_to_placement


SCHEMA_VERSION = "edge_slot_interface.v2"
_EPS = 1e-9


def _unit(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    length = float(np.linalg.norm(vector))
    if length <= _EPS:
        return None
    return vector / length


def _point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    return point


def _relative_error(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), _EPS
    )


def _canonical_direction(vector: np.ndarray) -> np.ndarray:
    """Give an unoriented line a stable sign without using world semantics."""

    result = np.asarray(vector, dtype=float)
    for value in result:
        if abs(float(value)) > 1e-8:
            return result if value > 0.0 else -result
    return result


def _normalise_obb(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = summary.get("obb")
    if not isinstance(raw, Mapping):
        bbox = summary.get("bbox")
        if not isinstance(bbox, Mapping):
            return None
        minimum = _point(bbox.get("min"))
        maximum = _point(bbox.get("max"))
        if minimum is None or maximum is None or np.any(maximum <= minimum):
            return None
        raw = {
            "center": 0.5 * (minimum + maximum),
            "axes": np.eye(3),
            "dimensions": maximum - minimum,
        }

    center = _point(raw.get("center"))
    try:
        axes = np.asarray(raw.get("axes"), dtype=float)
        dimensions = np.asarray(raw.get("dimensions"), dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        center is None
        or axes.shape != (3, 3)
        or dimensions.shape != (3,)
        or not np.all(np.isfinite(axes))
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= _EPS)
    ):
        return None
    normalised_axes = [_unit(axis) for axis in axes]
    if any(axis is None for axis in normalised_axes):
        return None
    axes = np.asarray(normalised_axes, dtype=float)
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4):
        return None
    return {"center": center, "axes": axes, "dimensions": dimensions}


def _moving_functional_obb(
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, str]:
    """Prefer an audited dominant functional body and fall back conservatively."""

    derivation_status = "not_run"
    try:
        derived = derive_dominant_planar_envelope(summary)
        derivation_status = str(derived.get("status") or "unknown")
        raw = derived.get("functional_body_obb")
        if isinstance(raw, Mapping):
            normalised = _normalise_obb({"obb": raw})
            if normalised is not None:
                return normalised, "dominant_planar_envelope", derivation_status
    except (TypeError, ValueError, ArithmeticError):
        # An unavailable functional envelope is expected for sparse summaries;
        # the full OBB remains an explicitly recorded fallback, not evidence.
        derivation_status = "error"
    return _normalise_obb(summary), "full_obb_fallback", derivation_status


def _plane_dimensions(plane: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("footprint_dimensions", "extent_uv", "dimensions"):
        try:
            dimensions = np.asarray(plane.get(key), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if (
            len(dimensions) >= 2
            and np.all(np.isfinite(dimensions[:2]))
            and np.all(dimensions[:2] > _EPS)
        ):
            return dimensions[:2]
    return None


def _linear_plane_roi(
    raw_planes: Sequence[Any],
    *,
    moving_long_dimension: float,
    moving_height: float,
    minimum_aspect_ratio: float,
    maximum_short_to_height_ratio: float,
    length_tolerance: float,
) -> tuple[list[tuple[int, Mapping[str, Any]]], dict[str, int]]:
    """Cheap O(n) extent gate before normal/vector processing.

    A 120k-face carrier should not pay for normalisation, basis construction or
    family comparisons on square pads and broad board faces.  This pass reads
    only the two audited footprint extents.
    """

    selected: list[tuple[int, Mapping[str, Any]]] = []
    counters = {
        "roi_missing_extent_rejected_count": 0,
        "roi_aspect_rejected_count": 0,
        "roi_short_edge_rejected_count": 0,
        "roi_length_rejected_count": 0,
    }
    maximum_short = maximum_short_to_height_ratio * float(moving_height)
    for index, raw in enumerate(raw_planes):
        if not isinstance(raw, Mapping):
            counters["roi_missing_extent_rejected_count"] += 1
            continue
        dimensions = _plane_dimensions(raw)
        if dimensions is None:
            counters["roi_missing_extent_rejected_count"] += 1
            continue
        long_dimension = float(max(dimensions))
        short_dimension = float(min(dimensions))
        if long_dimension / max(short_dimension, _EPS) < minimum_aspect_ratio:
            counters["roi_aspect_rejected_count"] += 1
            continue
        if short_dimension > maximum_short:
            counters["roi_short_edge_rejected_count"] += 1
            continue
        if _relative_error(long_dimension, moving_long_dimension) > length_tolerance:
            counters["roi_length_rejected_count"] += 1
            continue
        selected.append((index, raw))
    return selected, counters


def _plane_axes(
    plane: Mapping[str, Any], normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    raw = plane.get("footprint_axes")
    if isinstance(raw, Sequence) and len(raw) >= 2:
        first = _unit(raw[0])
        second = _unit(raw[1])
    else:
        first = _unit(plane.get("u_axis"))
        second = _unit(plane.get("v_axis"))
    if first is None:
        return None
    first = _unit(first - float(np.dot(first, normal)) * normal)
    if first is None:
        return None
    if second is not None:
        second = second - float(np.dot(second, normal)) * normal
        second = second - float(np.dot(second, first)) * first
        second = _unit(second)
    if second is None:
        second = _unit(np.cross(normal, first))
    if second is None:
        return None
    return first, second


def _normalise_plane(
    plane: Mapping[str, Any], index: int
) -> dict[str, Any] | None:
    normal = _unit(plane.get("normal"))
    center = _point(
        plane.get("centroid", plane.get("center", plane.get("position")))
    )
    dimensions = _plane_dimensions(plane)
    if normal is None or center is None or dimensions is None:
        return None
    axes = _plane_axes(plane, normal)
    if axes is None:
        # Area plus a normal cannot safely invent a footprint orientation.
        return None
    long_index = int(np.argmax(dimensions))
    short_index = 1 - long_index
    long_axis = axes[long_index]
    short_axis = axes[short_index]
    long_axis = _canonical_direction(long_axis)
    if float(np.dot(np.cross(long_axis, short_axis), normal)) < 0.0:
        short_axis = -short_axis
    return {
        "index": int(index),
        "center": center,
        "normal": normal,
        "long_axis": long_axis,
        "short_axis": short_axis,
        "length": float(dimensions[long_index]),
        "width": float(dimensions[short_index]),
    }


def _inside_carrier_footprint(
    plane: Mapping[str, Any],
    carrier: Mapping[str, Any],
    thin_axis_index: int,
    *,
    minimum_clearance: float,
) -> tuple[bool, float]:
    """Require the complete elongated footprint to lie inside the carrier.

    The face may lie on either carrier surface along the thin direction.  The
    internal test concerns only the two in-plane axes and thus rejects a long
    exterior board-edge face even if its centre alone happens to be inside.
    """

    center_delta = np.asarray(plane["center"]) - np.asarray(carrier["center"])
    clearances = []
    for axis_index, carrier_axis in enumerate(carrier["axes"]):
        if axis_index == thin_axis_index:
            continue
        coordinate = abs(float(np.dot(center_delta, carrier_axis)))
        half_span = 0.5 * (
            abs(float(np.dot(plane["long_axis"], carrier_axis)))
            * float(plane["length"])
            + abs(float(np.dot(plane["short_axis"], carrier_axis)))
            * float(plane["width"])
        )
        clearance = (
            0.5 * float(carrier["dimensions"][axis_index])
            - coordinate
            - half_span
        )
        clearances.append(clearance)
    smallest = min(clearances) if clearances else -float("inf")
    return bool(smallest >= minimum_clearance), float(smallest)


def _bounded_channel_evidence(
    floor: dict[str, Any],
    elongated_planes: list[dict[str, Any]],
    carrier_normal: np.ndarray,
    *,
    direction_cosine: float,
    length_tolerance: float,
    mirror_tolerance: float,
) -> dict[str, Any] | None:
    """Bind one floor to the best pair of mirrored sloped channel walls.

    "Mirrored" is evaluated in the floor frame.  The two wall centres have
    opposite spacing offsets, matching opening offsets and matching long-axis
    offsets; they need not be point-reflections through the floor because real
    connector walls rise together above it.  Wall normals must have equal-sign
    carrier components and opposite-sign spacing components.
    """

    floor_center = np.asarray(floor["center"])
    floor_long = np.asarray(floor["long_axis"])
    floor_opening = np.asarray(floor["normal"])
    spacing_axis = _unit(np.cross(floor_opening, floor_long))
    if spacing_axis is None:
        return None
    spacing_axis = _canonical_direction(spacing_axis)
    wall_rows = []
    for wall in elongated_planes:
        if int(wall["index"]) == int(floor["index"]):
            continue
        if abs(float(np.dot(floor_long, wall["long_axis"]))) < direction_cosine:
            continue
        if _relative_error(floor["length"], wall["length"]) > length_tolerance:
            continue
        normal = np.asarray(wall["normal"])
        carrier_component = float(np.dot(normal, carrier_normal))
        spacing_component = float(np.dot(normal, spacing_axis))
        long_component = float(np.dot(normal, floor_long))
        # A channel wall must be genuinely sloped in the carrier/spacing plane.
        if (
            abs(carrier_component) < 0.15
            or abs(spacing_component) < 0.15
            or abs(long_component) > 0.15
        ):
            continue
        delta = np.asarray(wall["center"]) - floor_center
        long_offset = float(np.dot(delta, floor_long))
        spacing_offset = float(np.dot(delta, spacing_axis))
        opening_offset = float(np.dot(delta, floor_opening))
        if abs(long_offset) > max(0.5, 0.05 * float(floor["length"])):
            continue
        if abs(spacing_offset) <= max(0.05, 0.02 * float(floor["width"])):
            continue
        # The floor normal is the audited direction towards the channel mouth.
        if opening_offset <= max(0.05, 0.02 * float(floor["width"])):
            continue
        if opening_offset > max(
            1.0, 4.0 * float(floor["width"]) + float(wall["width"])
        ):
            continue
        wall_rows.append({
            "wall": wall,
            "carrier_normal_component": carrier_component,
            "spacing_normal_component": spacing_component,
            "long_normal_component": long_component,
            "long_offset": long_offset,
            "spacing_offset": spacing_offset,
            "opening_offset": opening_offset,
        })

    best = None
    for left_index, left in enumerate(wall_rows):
        for right in wall_rows[left_index + 1:]:
            if left["carrier_normal_component"] * right["carrier_normal_component"] <= 0.0:
                continue
            if left["spacing_normal_component"] * right["spacing_normal_component"] >= 0.0:
                continue
            if left["spacing_offset"] * right["spacing_offset"] >= 0.0:
                continue
            component_errors = [
                _relative_error(
                    abs(left["carrier_normal_component"]),
                    abs(right["carrier_normal_component"]),
                ),
                _relative_error(
                    abs(left["spacing_normal_component"]),
                    abs(right["spacing_normal_component"]),
                ),
                _relative_error(
                    abs(left["spacing_offset"]), abs(right["spacing_offset"])
                ),
                _relative_error(left["opening_offset"], right["opening_offset"]),
                _relative_error(left["wall"]["width"], right["wall"]["width"]),
            ]
            if max(component_errors) > mirror_tolerance:
                continue
            gap = abs(float(left["spacing_offset"] - right["spacing_offset"]))
            if gap < 0.10 * float(floor["width"]) or gap > 2.0 * float(floor["width"]):
                continue
            midpoint_long_error = abs(
                0.5 * float(left["long_offset"] + right["long_offset"])
            )
            midpoint_spacing_error = abs(
                0.5 * float(left["spacing_offset"] + right["spacing_offset"])
            )
            opening_mismatch = abs(
                float(left["opening_offset"] - right["opening_offset"])
            )
            if midpoint_long_error > max(0.25, 0.02 * float(floor["length"])):
                continue
            if midpoint_spacing_error > max(0.10, 0.10 * gap):
                continue
            if opening_mismatch > max(
                0.10, mirror_tolerance * max(left["opening_offset"], right["opening_offset"])
            ):
                continue
            score = (
                sum(component_errors)
                + midpoint_long_error / max(float(floor["length"]), _EPS)
                + midpoint_spacing_error / max(gap, _EPS)
                + opening_mismatch
                / max(left["opening_offset"], right["opening_offset"], _EPS)
            )
            candidate = {
                "wall_plane_indices": sorted(
                    [int(left["wall"]["index"]), int(right["wall"]["index"])]
                ),
                "channel_gap": float(gap),
                "wall_center_spacing_offsets": sorted([
                    float(left["spacing_offset"]),
                    float(right["spacing_offset"]),
                ]),
                "wall_center_opening_offsets": [
                    float(left["opening_offset"]),
                    float(right["opening_offset"]),
                ],
                "wall_carrier_normal_components": [
                    float(left["carrier_normal_component"]),
                    float(right["carrier_normal_component"]),
                ],
                "wall_spacing_normal_components": [
                    float(left["spacing_normal_component"]),
                    float(right["spacing_normal_component"]),
                ],
                "midpoint_long_error": float(midpoint_long_error),
                "midpoint_spacing_error": float(midpoint_spacing_error),
                "opening_offset_mismatch": float(opening_mismatch),
                "mirror_score": float(score),
                "floor_opening_normal": floor_opening.tolist(),
                "floor_spacing_axis": spacing_axis.tolist(),
            }
            if best is None or (candidate["mirror_score"], candidate["wall_plane_indices"]) < (
                best["mirror_score"], best["wall_plane_indices"]
            ):
                best = candidate
    return best


def _longest_equal_spacing_run(
    rows: list[dict[str, Any]],
    spacing_axis: np.ndarray,
    *,
    minimum_family_size: int,
    spacing_tolerance: float,
) -> tuple[list[dict[str, Any]], float, float] | None:
    """Return the longest contiguous approximately arithmetic progression."""

    ordered = sorted(
        rows,
        key=lambda row: (
            float(np.dot(row["center"], spacing_axis)), row["index"]
        ),
    )
    best: tuple[list[dict[str, Any]], float, float] | None = None
    for start in range(len(ordered)):
        for stop in range(start + minimum_family_size, len(ordered) + 1):
            subset = ordered[start:stop]
            coordinates = np.asarray(
                [float(np.dot(row["center"], spacing_axis)) for row in subset]
            )
            gaps = np.diff(coordinates)
            median_gap = float(np.median(gaps))
            minimum_distinct_gap = max(
                0.25, 0.5 * float(np.median([row["width"] for row in subset]))
            )
            if median_gap <= minimum_distinct_gap:
                continue
            maximum_error = float(
                np.max(np.abs(gaps - median_gap)) / max(median_gap, _EPS)
            )
            if maximum_error > spacing_tolerance:
                continue
            candidate = (subset, median_gap, maximum_error)
            if best is None or (
                len(subset), -maximum_error, -median_gap
            ) > (len(best[0]), -best[2], -best[1]):
                best = candidate
    return best


def _slot_families(
    planes: list[dict[str, Any]],
    carrier_normal: np.ndarray,
    *,
    minimum_family_size: int,
    direction_cosine: float,
    length_family_tolerance: float,
    width_family_tolerance: float,
    centerline_tolerance_ratio: float,
    layer_tolerance: float,
    spacing_tolerance: float,
) -> list[dict[str, Any]]:
    families: dict[tuple[int, ...], dict[str, Any]] = {}
    for seed in planes:
        long_axis = np.asarray(seed["long_axis"])
        spacing_axis = _unit(np.cross(carrier_normal, long_axis))
        if spacing_axis is None:
            continue
        spacing_axis = _canonical_direction(spacing_axis)
        compatible = []
        for row in planes:
            if abs(float(np.dot(long_axis, row["long_axis"]))) < direction_cosine:
                continue
            if _relative_error(seed["length"], row["length"]) > length_family_tolerance:
                continue
            if _relative_error(seed["width"], row["width"]) > width_family_tolerance:
                continue
            delta = np.asarray(row["center"]) - np.asarray(seed["center"])
            if abs(float(np.dot(delta, carrier_normal))) > layer_tolerance:
                continue
            if abs(float(np.dot(delta, long_axis))) > max(
                0.5, centerline_tolerance_ratio * float(seed["length"])
            ):
                continue
            compatible.append(row)
        run = _longest_equal_spacing_run(
            compatible,
            spacing_axis,
            minimum_family_size=minimum_family_size,
            spacing_tolerance=spacing_tolerance,
        )
        if run is None:
            continue
        members, pitch, pitch_error = run
        key = tuple(sorted(int(row["index"]) for row in members))
        if key in families:
            continue
        averaged_long = np.zeros(3)
        for row in members:
            row_axis = np.asarray(row["long_axis"])
            if float(np.dot(row_axis, long_axis)) < 0.0:
                row_axis = -row_axis
            averaged_long += row_axis
        averaged_long = _unit(averaged_long)
        if averaged_long is None:
            continue
        averaged_long = _canonical_direction(averaged_long)
        family_spacing = _unit(np.cross(carrier_normal, averaged_long))
        if family_spacing is None:
            continue
        family_spacing = _canonical_direction(family_spacing)
        families[key] = {
            "members": members,
            "long_axis": averaged_long,
            "spacing_axis": family_spacing,
            "pitch": float(pitch),
            "maximum_pitch_relative_error": float(pitch_error),
            "representative_length": float(
                np.median([row["length"] for row in members])
            ),
            "representative_width": float(
                np.median([row["width"] for row in members])
            ),
        }
    return sorted(
        families.values(),
        key=lambda family: (
            -len(family["members"]),
            family["maximum_pitch_relative_error"],
            tuple(row["index"] for row in family["members"]),
        ),
    )


def _result_base() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
        "semantic_fields_used": [],
    }


def _stopped_result(
    status: str, reason: str, audit: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **_result_base(),
        "status": status,
        "reason": reason,
        "proposals": [],
        "audit": dict(audit),
    }


def recall_edge_slot_interface_proposals(
    stationary_summary: Mapping[str, Any],
    moving_summary: Mapping[str, Any],
    *,
    maximum_proposals: int = 16,
    minimum_family_size: int = 3,
    minimum_plane_aspect_ratio: float = 5.0,
    minimum_moving_aspect_ratio: float = 2.5,
    maximum_moving_thin_ratio: float = 0.35,
    normal_alignment_cosine: float = 0.96,
    direction_cosine: float = 0.985,
    length_tolerance: float = 0.18,
    length_family_tolerance: float = 0.12,
    width_family_tolerance: float = 0.35,
    spacing_tolerance: float = 0.20,
    centerline_tolerance_ratio: float = 0.08,
    minimum_internal_clearance: float = 0.25,
    maximum_roi_short_to_moving_height_ratio: float = 0.60,
    mirror_wall_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Recall bounded review-only poses for repeated internal edge slots.

    The returned transform maps coordinates in the moving summary's frame into
    the stationary summary's frame.  The moving OBB's intermediate axis is
    treated as the insertion span: its negative edge is seated on the slot
    centreline while the moving longest axis is aligned to the slot long axis.
    """

    if not isinstance(stationary_summary, Mapping) or not isinstance(
        moving_summary, Mapping
    ):
        raise TypeError("stationary_summary and moving_summary must be mappings")
    stationary_obb = _normalise_obb(stationary_summary)
    moving_obb, moving_obb_source, functional_envelope_status = (
        _moving_functional_obb(moving_summary)
    )
    audit: dict[str, Any] = {
        "raw_stationary_plane_count": len(stationary_summary.get("planes") or []),
        "roi_stationary_plane_count": 0,
        "usable_stationary_plane_count": 0,
        "elongated_surface_plane_count": 0,
        "outward_boundary_rejected_count": 0,
        "internal_slot_plane_count": 0,
        "bounded_channel_floor_count": 0,
        "floor_without_mirror_walls_count": 0,
        "repeated_slot_family_count": 0,
        "length_compatible_family_count": 0,
        "raw_proposal_count": 0,
        "returned_proposal_count": 0,
        "proposal_limit": max(0, int(maximum_proposals)),
        "moving_obb_source": moving_obb_source,
        "functional_envelope_status": functional_envelope_status,
    }
    if stationary_obb is None or moving_obb is None:
        return _stopped_result(
            "unavailable", "Both feature summaries require a finite OBB.", audit
        )

    stationary_dimensions = np.asarray(stationary_obb["dimensions"])
    stationary_axes = np.asarray(stationary_obb["axes"])
    carrier_thin_index = int(np.argmin(stationary_dimensions))
    carrier_normal = stationary_axes[carrier_thin_index]
    in_plane_dimensions = np.delete(stationary_dimensions, carrier_thin_index)
    if float(stationary_dimensions[carrier_thin_index]) / max(
        float(np.min(in_plane_dimensions)), _EPS
    ) > maximum_moving_thin_ratio:
        return _stopped_result(
            "abstain",
            "The stationary OBB has no sufficiently thin carrier axis.",
            {**audit, "carrier_thin_axis_index": carrier_thin_index},
        )

    moving_dimensions = np.asarray(moving_obb["dimensions"])
    moving_order = np.argsort(moving_dimensions)
    moving_thin_index = int(moving_order[0])
    moving_insertion_index = int(moving_order[1])
    moving_long_index = int(moving_order[2])
    moving_long_dimension = float(moving_dimensions[moving_long_index])
    moving_insertion_dimension = float(moving_dimensions[moving_insertion_index])
    if (
        moving_long_dimension / max(moving_insertion_dimension, _EPS)
        < minimum_moving_aspect_ratio
        or float(moving_dimensions[moving_thin_index])
        / max(moving_insertion_dimension, _EPS)
        > maximum_moving_thin_ratio
    ):
        return _stopped_result(
            "abstain",
            "The moving OBB is not a thin elongated edge-insertable component.",
            {
                **audit,
                "moving_long_axis_index": moving_long_index,
                "moving_insertion_axis_index": moving_insertion_index,
                "moving_thin_axis_index": moving_thin_index,
            },
        )

    roi_planes, roi_counters = _linear_plane_roi(
        stationary_summary.get("planes") or [],
        moving_long_dimension=moving_long_dimension,
        moving_height=moving_insertion_dimension,
        minimum_aspect_ratio=minimum_plane_aspect_ratio,
        maximum_short_to_height_ratio=maximum_roi_short_to_moving_height_ratio,
        length_tolerance=length_tolerance,
    )
    audit.update(roi_counters)
    audit["roi_stationary_plane_count"] = len(roi_planes)
    usable_planes = []
    for index, raw in roi_planes:
        plane = _normalise_plane(raw, index)
        if plane is not None:
            usable_planes.append(plane)
    audit["usable_stationary_plane_count"] = len(usable_planes)

    elongated = []
    for plane in usable_planes:
        if plane["length"] / max(plane["width"], _EPS) < minimum_plane_aspect_ratio:
            continue
        # Both floors and their sloped walls have long axes in the carrier
        # plane.  Wall normals themselves are intentionally not parallel to it.
        projected_long = plane["long_axis"] - float(
            np.dot(plane["long_axis"], carrier_normal)
        ) * carrier_normal
        projected_long = _unit(projected_long)
        if projected_long is None:
            continue
        plane["long_axis"] = _canonical_direction(projected_long)
        plane["short_axis"] = _unit(np.cross(plane["normal"], plane["long_axis"]))
        if plane["short_axis"] is None:
            continue
        elongated.append(plane)
    audit["elongated_surface_plane_count"] = len(elongated)

    internal_floors = []
    for plane in elongated:
        if abs(float(np.dot(plane["normal"], carrier_normal))) < normal_alignment_cosine:
            continue
        is_inside, clearance = _inside_carrier_footprint(
            plane,
            stationary_obb,
            carrier_thin_index,
            minimum_clearance=minimum_internal_clearance,
        )
        plane["minimum_internal_clearance"] = clearance
        if not is_inside:
            audit["outward_boundary_rejected_count"] += 1
            continue
        internal_floors.append(plane)
    audit["internal_slot_plane_count"] = len(internal_floors)

    bounded_floors = []
    for floor in internal_floors:
        channel = _bounded_channel_evidence(
            floor,
            elongated,
            carrier_normal,
            direction_cosine=direction_cosine,
            length_tolerance=length_family_tolerance,
            mirror_tolerance=mirror_wall_tolerance,
        )
        if channel is None:
            audit["floor_without_mirror_walls_count"] += 1
            continue
        floor["bounded_channel"] = channel
        bounded_floors.append(floor)
    audit["bounded_channel_floor_count"] = len(bounded_floors)
    if not bounded_floors:
        return _stopped_result(
            "abstain",
            (
                "No internal slot floor is bounded by a compatible mirrored "
                "pair of sloped channel walls."
            ),
            audit,
        )

    families = _slot_families(
        bounded_floors,
        carrier_normal,
        minimum_family_size=max(3, int(minimum_family_size)),
        direction_cosine=direction_cosine,
        length_family_tolerance=length_family_tolerance,
        width_family_tolerance=width_family_tolerance,
        centerline_tolerance_ratio=centerline_tolerance_ratio,
        layer_tolerance=max(
            0.5, 0.75 * float(stationary_dimensions[carrier_thin_index])
        ),
        spacing_tolerance=spacing_tolerance,
    )
    audit["repeated_slot_family_count"] = len(families)
    if not families:
        return _stopped_result(
            "abstain",
            (
                "No family of at least three parallel, equally spaced bounded "
                "channel floors was found."
            ),
            audit,
        )

    compatible_families = []
    for family_index, family in enumerate(families):
        family["family_index"] = family_index
        family["length_relative_error"] = _relative_error(
            family["representative_length"], moving_long_dimension
        )
        if family["length_relative_error"] <= length_tolerance:
            compatible_families.append(family)
    audit["length_compatible_family_count"] = len(compatible_families)
    if not compatible_families:
        return _stopped_result(
            "abstain",
            "Repeated internal slots exist, but none is length-compatible with the moving OBB.",
            audit,
        )

    moving_axes = np.asarray(moving_obb["axes"])
    moving_basis = moving_axes.T
    moving_center = np.asarray(moving_obb["center"])
    moving_seating_edge = (
        moving_center
        - 0.5
        * moving_insertion_dimension
        * moving_axes[moving_insertion_index]
    )
    raw_proposals = []
    for family in compatible_families:
        long_axis = np.asarray(family["long_axis"])
        for slot_rank, slot in enumerate(family["members"]):
            # The audited floor normal, rather than an arbitrary OBB-axis sign,
            # points from the channel floor towards its open mouth.
            insertion_axis = _unit(slot["normal"])
            if insertion_axis is None:
                continue
            proposal_long_axis = _unit(
                long_axis - float(np.dot(long_axis, insertion_axis)) * insertion_axis
            )
            if proposal_long_axis is None:
                continue
            proposal_long_axis = _canonical_direction(proposal_long_axis)
            spacing_axis = _unit(np.cross(insertion_axis, proposal_long_axis))
            if spacing_axis is None:
                continue
            channel = slot["bounded_channel"]
            for long_sign in (1.0, -1.0):
                target_basis = np.zeros((3, 3), dtype=float)
                target_basis[:, moving_long_index] = long_sign * proposal_long_axis
                target_basis[:, moving_insertion_index] = insertion_axis
                chosen_rotation = None
                for spacing_sign in (1.0, -1.0):
                    target_basis[:, moving_thin_index] = spacing_sign * spacing_axis
                    rotation = target_basis @ moving_basis.T
                    if (
                        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                        and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-5
                    ):
                        chosen_rotation = rotation.copy()
                        break
                if chosen_rotation is None:
                    continue
                transform = np.eye(4)
                transform[:3, :3] = chosen_rotation
                transform[:3, 3] = (
                    np.asarray(slot["center"])
                    - chosen_rotation @ moving_seating_edge
                )
                evidence_families = [
                    "moving_slot_length_compatibility",
                    "bounded_mirrored_channel_walls",
                    "repeated_equally_spaced_internal_slot_family",
                    "carrier_thin_axis_insertion_role",
                ]
                raw_proposals.append({
                    "slot_family_id": int(family["family_index"]),
                    "slot_family_size": len(family["members"]),
                    "slot_rank": int(slot_rank),
                    "slot_plane_index": int(slot["index"]),
                    "floor_plane_index": int(slot["index"]),
                    "wall_plane_indices": list(channel["wall_plane_indices"]),
                    "channel_gap": float(channel["channel_gap"]),
                    "slot_center": np.asarray(slot["center"], dtype=float).tolist(),
                    "slot_long_axis": proposal_long_axis.tolist(),
                    "slot_spacing_axis": spacing_axis.tolist(),
                    "insertion_axis": np.asarray(insertion_axis).tolist(),
                    "seating_motion_axis": (-np.asarray(insertion_axis)).tolist(),
                    "slot_pitch": float(family["pitch"]),
                    "maximum_pitch_relative_error": float(
                        family["maximum_pitch_relative_error"]
                    ),
                    "slot_length": float(slot["length"]),
                    "moving_long_dimension": moving_long_dimension,
                    "length_relative_error": float(family["length_relative_error"]),
                    "minimum_internal_clearance": float(
                        slot["minimum_internal_clearance"]
                    ),
                    "floor_evidence": {
                        "floor_plane_index": int(slot["index"]),
                        "floor_center": np.asarray(slot["center"], dtype=float).tolist(),
                        "floor_opening_normal": np.asarray(slot["normal"], dtype=float).tolist(),
                        "floor_length": float(slot["length"]),
                        "floor_width": float(slot["width"]),
                        "wall_plane_indices": list(channel["wall_plane_indices"]),
                        "wall_center_spacing_offsets": list(
                            channel["wall_center_spacing_offsets"]
                        ),
                        "wall_center_opening_offsets": list(
                            channel["wall_center_opening_offsets"]
                        ),
                        "wall_carrier_normal_components": list(
                            channel["wall_carrier_normal_components"]
                        ),
                        "wall_spacing_normal_components": list(
                            channel["wall_spacing_normal_components"]
                        ),
                        "channel_gap": float(channel["channel_gap"]),
                        "mirror_score": float(channel["mirror_score"]),
                    },
                    "moving_long_axis_index": moving_long_index,
                    "moving_insertion_axis_index": moving_insertion_index,
                    "moving_thin_axis_index": moving_thin_index,
                    "long_axis_sign": int(long_sign),
                    "rotation_matrix": chosen_rotation.tolist(),
                    "transform_matrix": transform.tolist(),
                    "placement": matrix_to_placement(transform),
                    "transform_frame": "stationary_local_from_moving_local",
                    "evidence_families": evidence_families,
                    "independent_evidence_count": len(evidence_families),
                    "has_multi_evidence_support": True,
                    "evidence": [
                        {
                            "type": "moving_slot_length_compatibility",
                            "relative_error": float(family["length_relative_error"]),
                            "tolerance": float(length_tolerance),
                        },
                        {
                            "type": "bounded_mirrored_channel_walls",
                            "floor_plane_index": int(slot["index"]),
                            "wall_plane_indices": list(channel["wall_plane_indices"]),
                            "channel_gap": float(channel["channel_gap"]),
                            "mirror_score": float(channel["mirror_score"]),
                        },
                        {
                            "type": "repeated_equally_spaced_internal_slot_family",
                            "member_plane_indices": [
                                int(row["index"]) for row in family["members"]
                            ],
                            "member_wall_plane_indices": [
                                list(row["bounded_channel"]["wall_plane_indices"])
                                for row in family["members"]
                            ],
                            "family_size": len(family["members"]),
                            "pitch": float(family["pitch"]),
                            "maximum_pitch_relative_error": float(
                                family["maximum_pitch_relative_error"]
                            ),
                        },
                        {
                            "type": "carrier_thin_axis_insertion_role",
                            "carrier_thin_axis_index": carrier_thin_index,
                            "moving_insertion_axis_index": moving_insertion_index,
                        },
                    ],
                    "proposal_only": True,
                    "review_required": True,
                    "can_auto_accept": False,
                    "semantic_fields_used": [],
                })

    raw_proposals.sort(key=lambda row: (
        -int(row["slot_family_size"]),
        float(row["length_relative_error"]),
        int(row["slot_family_id"]),
        int(row["slot_rank"]),
        -int(row["long_axis_sign"]),
    ))
    audit["raw_proposal_count"] = len(raw_proposals)
    limit = max(0, int(maximum_proposals))
    proposals = raw_proposals[:limit]
    audit["returned_proposal_count"] = len(proposals)
    if not proposals:
        return _stopped_result(
            "abstain", "The proposal bound is zero or no proper rigid pose exists.", audit
        )
    return {
        **_result_base(),
        "status": "success",
        "reason": (
            "Length-compatible repeated internal channel floors with mirrored "
            "sloped walls support bounded review-only edge-insertion poses."
        ),
        "proposals": proposals,
        "audit": audit,
    }


def propose_edge_slot_interface_placements(
    stationary_summary: Mapping[str, Any],
    moving_summary: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias for :func:`recall_edge_slot_interface_proposals`."""

    return recall_edge_slot_interface_proposals(
        stationary_summary, moving_summary, **kwargs
    )


__all__ = [
    "SCHEMA_VERSION",
    "propose_edge_slot_interface_placements",
    "recall_edge_slot_interface_proposals",
]
