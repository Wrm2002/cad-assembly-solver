"""Conservative geometry-only proposals for repeated enclosure bays.

The detector is intentionally a proposal generator, not an assembly judge.  It
looks for multiple cavities supported by independent geometric evidence:

* pairs of opposing wall faces, including a divider between adjacent bays;
* one repeated rail/support footprint per bay;
* a functional-body envelope whose width, height, and depth fit each cavity;
* optionally, a common roof and an explicitly observed open-end direction.

Planar ``footprint_axes`` and ``footprint_dimensions`` are required.  Face area
alone is never converted into an invented square.  A full-part AABB may include
handles, latches, or connectors, so it is reported as protrusion risk but never
used to reject a functional-body fit.  Every result remains review-only.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .dominant_planar_envelope import derive_dominant_planar_envelope


_EPS = 1e-9


def _unit(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm <= _EPS:
        return None
    return vector / norm


def _point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    return point if np.all(np.isfinite(point)) else None


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    pivot = int(np.argmax(np.abs(axis)))
    return -axis if axis[pivot] < 0.0 else axis


def _normalise_plane(raw: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    normal = _unit(raw.get("normal"))
    center = None
    for key in ("centroid", "center", "position"):
        center = _point(raw.get(key))
        if center is not None:
            break
    axes = raw.get("footprint_axes")
    dimensions = raw.get("footprint_dimensions")
    if normal is None or center is None or not isinstance(axes, Sequence):
        return None
    if not isinstance(dimensions, Sequence) or len(axes) < 2 or len(dimensions) < 2:
        return None
    u_axis = _unit(axes[0])
    v_axis = _unit(axes[1])
    try:
        dims = np.asarray(dimensions[:2], dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        u_axis is None
        or v_axis is None
        or not np.all(np.isfinite(dims))
        or np.any(dims <= _EPS)
    ):
        return None
    # Re-project supplied UV axes into the plane.  Invalid or nearly parallel
    # axes are not silently replaced by a guessed footprint.
    u_axis = _unit(u_axis - float(np.dot(u_axis, normal)) * normal)
    if u_axis is None:
        return None
    v_axis = _unit(v_axis - float(np.dot(v_axis, normal)) * normal)
    if v_axis is None or abs(float(np.dot(u_axis, v_axis))) > 1e-3:
        return None
    return {
        "index": int(index),
        "center": center,
        "normal": normal,
        "axes": np.asarray([u_axis, v_axis], dtype=float),
        "dimensions": dims,
        "area": float(raw.get("area") or dims[0] * dims[1]),
        "source": raw,
    }


def _normalise_envelope(raw: Any, source: str) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    if raw.get("min") is not None and raw.get("max") is not None:
        minimum = _point(raw.get("min"))
        maximum = _point(raw.get("max"))
        if minimum is None or maximum is None or np.any(maximum <= minimum):
            return None
        return {
            "center": 0.5 * (minimum + maximum),
            "axes": np.eye(3),
            "dimensions": maximum - minimum,
            "source": source,
        }
    center = _point(raw.get("center"))
    try:
        axes = np.asarray(raw.get("axes"), dtype=float)
        dimensions = np.asarray(raw.get("dimensions"), dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    if (
        center is None
        or axes.shape != (3, 3)
        or not np.all(np.isfinite(axes))
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= _EPS)
    ):
        return None
    normalised = []
    for axis in axes:
        axis = _unit(axis)
        if axis is None:
            return None
        normalised.append(axis)
    axes = np.asarray(normalised, dtype=float)
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4):
        return None
    return {
        "center": center,
        "axes": axes,
        "dimensions": dimensions,
        "source": source,
    }


def _functional_body(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    # A whole-part OBB is deliberately not a functional body.  Handles,
    # latches, and connector noses commonly make it too large for the bay.
    for key in ("functional_body_obb", "functional_body", "body_obb"):
        envelope = _normalise_envelope(summary.get(key), key)
        if envelope is not None:
            return envelope
    return None


def _full_envelope(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("full_obb", "obb", "full_bbox", "bbox"):
        envelope = _normalise_envelope(summary.get(key), key)
        if envelope is not None:
            return envelope
    return None


def _envelope_audit_from_provided(body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "provided",
        "source": str(body["source"]),
        "confidence": None,
        "derivation_evidence": None,
        "excluded_planar_area": None,
        "excluded_area_fraction": None,
        "excluded_protrusion_area": None,
        "excluded_protrusion_risk": None,
        "full_envelope_protrusion_risk": None,
        "full_envelope_used_in_derivation": False,
    }


def _envelope_audit_from_derivation(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = result.get("derivation_evidence")
    return {
        "status": str(result.get("status") or "abstain"),
        "source": "derived_dominant_planar_envelope",
        "reason": result.get("reason"),
        "confidence": result.get("confidence"),
        "derivation_evidence": (
            dict(evidence) if isinstance(evidence, Mapping) else evidence
        ),
        "excluded_planar_area": result.get("excluded_planar_area"),
        "excluded_area_fraction": result.get("excluded_area_fraction"),
        "excluded_protrusion_area": result.get("excluded_protrusion_area"),
        "excluded_protrusion_risk": result.get("excluded_protrusion_risk"),
        "full_envelope_protrusion_risk": result.get(
            "full_envelope_protrusion_risk"
        ),
        "full_envelope_used_in_derivation": False,
    }


def _abstain_result(
    reason: str,
    *,
    envelope_audit: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "status": "abstain",
        "reason": reason,
        "proposals": [],
        "functional_body_envelope_audit": (
            None if envelope_audit is None else dict(envelope_audit)
        ),
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
    }
    result.update(extra)
    return result


def _open_shell(summary: Mapping[str, Any]) -> tuple[bool | None, dict[str, int | None]]:
    topology = summary.get("topology")
    if not isinstance(topology, Mapping):
        topology = summary
    solid_count = topology.get("solid_count")
    shell_count = topology.get("shell_count")
    try:
        solid_count = None if solid_count is None else int(solid_count)
        shell_count = None if shell_count is None else int(shell_count)
    except (TypeError, ValueError):
        solid_count = shell_count = None
    explicit = summary.get("open_shell")
    if explicit is not None:
        result: bool | None = bool(explicit)
    elif shell_count is not None and solid_count is not None:
        result = shell_count > 0 and solid_count == 0
    else:
        result = None
    return result, {"solid_count": solid_count, "shell_count": shell_count}


def _projected_span(plane: Mapping[str, Any], axis: np.ndarray) -> float:
    return float(sum(
        abs(float(np.dot(plane["axes"][index], axis)))
        * float(plane["dimensions"][index])
        for index in range(2)
    ))


def _interval(plane: Mapping[str, Any], axis: np.ndarray) -> tuple[float, float]:
    center = float(np.dot(plane["center"], axis))
    half = 0.5 * _projected_span(plane, axis)
    return center - half, center + half


def _intersection(
    left: tuple[float, float], right: tuple[float, float]
) -> tuple[float, float] | None:
    result = max(left[0], right[0]), min(left[1], right[1])
    return result if result[1] - result[0] > _EPS else None


def _axis_families(
    planes: Sequence[dict[str, Any]], parallel_cosine: float
) -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for plane in planes:
        canonical = _canonical_axis(plane["normal"])
        family = next((
            row for row in families
            if abs(float(np.dot(row["axis"], canonical))) >= parallel_cosine
        ), None)
        if family is None:
            family = {"axis": canonical, "planes": []}
            families.append(family)
        family["planes"].append(plane)
    return families


def _slot_pairs(
    family: Mapping[str, Any],
    height_axis: np.ndarray,
    body_dimensions: np.ndarray,
    roles: tuple[int, int, int],
) -> list[dict[str, Any]]:
    lateral = np.asarray(family["axis"], dtype=float)
    height = _unit(height_axis - float(np.dot(height_axis, lateral)) * lateral)
    if height is None:
        return []
    depth = _unit(np.cross(lateral, height))
    if depth is None:
        return []
    lateral_index, height_index, depth_index = roles
    body_lateral = float(body_dimensions[lateral_index])
    body_height = float(body_dimensions[height_index])
    body_depth = float(body_dimensions[depth_index])

    # Large CAD assemblies can contain thousands of coplanar cosmetic faces.
    # A cavity wall must itself cover most of the moving functional body's
    # height and depth.  Filter on audited footprint spans before constructing
    # the otherwise quadratic opposing-face frontier, then keep a bounded set
    # ordered by dimension compatibility.
    def wall_compatibility(row: Mapping[str, Any]) -> float | None:
        height_span = _projected_span(row, height)
        depth_span = _projected_span(row, depth)
        height_ratio = height_span / body_height
        depth_ratio = depth_span / body_depth
        if not 0.70 <= height_ratio <= 1.60:
            return None
        # Stamped dividers and guide walls are often segmented near the mouth
        # and rear connector.  Half-depth coverage is admissible only because
        # repeated rails are mandatory later and a common roof can recover the
        # full cavity extent.
        if not 0.45 <= depth_ratio <= 1.55:
            return None
        return abs(math.log(height_ratio)) + abs(math.log(depth_ratio))

    oriented: list[tuple[float, dict[str, Any]]] = []
    for row in family["planes"]:
        compatibility = wall_compatibility(row)
        if compatibility is not None:
            oriented.append((compatibility, row))
    oriented.sort(key=lambda item: (item[0], -float(item[1]["area"])))
    bounded = [row for _, row in oriented[:128]]
    positive = [
        row for row in bounded
        if float(np.dot(row["normal"], lateral)) >= 0.90
    ]
    negative = [
        row for row in bounded
        if float(np.dot(row["normal"], lateral)) <= -0.90
    ]
    slots = []
    for left in positive:
        left_coordinate = float(np.dot(left["center"], lateral))
        for right in negative:
            right_coordinate = float(np.dot(right["center"], lateral))
            width = right_coordinate - left_coordinate
            if width <= _EPS:
                continue
            width_ratio = width / body_lateral
            if not 0.94 <= width_ratio <= 1.22:
                continue
            height_interval = _intersection(
                _interval(left, height), _interval(right, height)
            )
            depth_interval = _intersection(
                _interval(left, depth), _interval(right, depth)
            )
            if height_interval is None or depth_interval is None:
                continue
            height_extent = height_interval[1] - height_interval[0]
            depth_extent = depth_interval[1] - depth_interval[0]
            height_ratio = height_extent / body_height
            depth_ratio = depth_extent / body_depth
            if not 0.82 <= height_ratio <= 1.35:
                continue
            if not 0.45 <= depth_ratio <= 1.35:
                continue
            center = (
                0.5 * (left_coordinate + right_coordinate) * lateral
                + 0.5 * sum(height_interval) * height
                + 0.5 * sum(depth_interval) * depth
            )
            fit_error = float(np.mean([
                abs(math.log(width_ratio)),
                abs(math.log(height_ratio)),
                abs(math.log(depth_ratio)),
            ]))
            slots.append({
                "center": center,
                "dimensions": np.asarray(
                    [width, height_extent, depth_extent], dtype=float
                ),
                "axes": np.asarray([lateral, height, depth], dtype=float),
                "roles": roles,
                "boundary_indices": [left["index"], right["index"]],
                "boundary_coordinates": [left_coordinate, right_coordinate],
                "fit_ratios": [width_ratio, height_ratio, depth_ratio],
                "fit_error": fit_error,
            })
    # Parallel face layers commonly duplicate one physical wall.  Keep the
    # best-fitting slot at each lateral center instead of inflating multiplicity.
    deduplicated: list[dict[str, Any]] = []
    tolerance = max(0.25, 0.03 * body_lateral)
    for slot in sorted(slots, key=lambda row: row["fit_error"]):
        coordinate = float(np.dot(slot["center"], lateral))
        if any(
            abs(coordinate - float(np.dot(other["center"], lateral))) <= tolerance
            for other in deduplicated
        ):
            continue
        deduplicated.append(slot)
    return sorted(
        deduplicated,
        key=lambda row: float(np.dot(row["center"], lateral)),
    )


def _support_and_roof(
    planes: Sequence[dict[str, Any]],
    slots: Sequence[dict[str, Any]],
    body_height: float,
    body_depth: float,
    parallel_cosine: float,
) -> dict[str, Any]:
    lateral, height, depth = slots[0]["axes"]
    mean_dimensions = np.mean([slot["dimensions"] for slot in slots], axis=0)
    slot_width, slot_height, slot_depth = map(float, mean_dimensions)
    compatible = [
        plane for plane in planes
        if abs(float(np.dot(plane["normal"], height))) >= parallel_cosine
    ]
    per_slot: list[dict[str, Any] | None] = []
    for slot in slots:
        slot_lateral = float(np.dot(slot["center"], lateral))
        slot_depth_center = float(np.dot(slot["center"], depth))
        candidates = []
        for plane in compatible:
            lateral_span = _projected_span(plane, lateral)
            depth_span = _projected_span(plane, depth)
            lateral_delta = abs(
                float(np.dot(plane["center"], lateral)) - slot_lateral
            )
            depth_delta = abs(
                float(np.dot(plane["center"], depth)) - slot_depth_center
            )
            if lateral_span > 1.20 * slot_width:
                continue
            if depth_span < 0.50 * slot_depth:
                continue
            if lateral_delta > 0.55 * slot_width:
                continue
            if depth_delta > 0.30 * slot_depth:
                continue
            candidates.append((
                abs(lateral_delta / slot_width)
                + abs(depth_delta / slot_depth)
                + abs(math.log(max(depth_span, _EPS) / slot_depth))
                # Prefer the upper, cavity-facing support surface when a rail
                # contributes parallel top and bottom faces.
                + (
                    0.20
                    if float(np.dot(plane["normal"], height)) < 0.0
                    else 0.0
                ),
                plane,
                lateral_span,
                depth_span,
            ))
        if not candidates:
            per_slot.append(None)
            continue
        _, plane, lateral_span, depth_span = min(candidates, key=lambda row: row[0])
        per_slot.append({
            "plane": plane,
            "lateral_span": lateral_span,
            "depth_span": depth_span,
            "height_coordinate": float(np.dot(plane["center"], height)),
        })

    rails_valid = all(row is not None for row in per_slot)
    if rails_valid:
        rows = [row for row in per_slot if row is not None]
        heights = [row["height_coordinate"] for row in rows]
        dimensions = np.asarray([
            [row["lateral_span"], row["depth_span"]] for row in rows
        ])
        rails_valid = (
            max(heights) - min(heights) <= max(0.5, 0.08 * body_height)
            and np.max(np.ptp(dimensions, axis=0) / np.maximum(
                np.mean(dimensions, axis=0), _EPS
            )) <= 0.15
        )

    total_left = min(slot["boundary_coordinates"][0] for slot in slots)
    total_right = max(slot["boundary_coordinates"][1] for slot in slots)
    total_width = total_right - total_left
    roof_candidates = []
    for plane in compatible:
        lateral_span = _projected_span(plane, lateral)
        depth_span = _projected_span(plane, depth)
        if lateral_span >= 0.80 * total_width and depth_span >= 0.65 * slot_depth:
            roof_candidates.append({
                "plane": plane,
                "height_coordinate": float(np.dot(plane["center"], height)),
                "lateral_span": lateral_span,
                "depth_span": depth_span,
            })

    roof = None
    if rails_valid and roof_candidates:
        rail_height = float(np.mean([
            row["height_coordinate"] for row in per_slot if row is not None
        ]))
        plausible = [
            row for row in roof_candidates
            if 0.65 * body_height
            <= abs(row["height_coordinate"] - rail_height)
            <= 1.45 * body_height
        ]
        if plausible:
            roof = min(
                plausible,
                key=lambda row: abs(
                    abs(row["height_coordinate"] - rail_height) - slot_height
                ),
            )
            # Positive height points from repeated supports toward the roof.
            if roof["height_coordinate"] < rail_height:
                height = -height
                depth = -depth
                for slot in slots:
                    slot["axes"] = np.asarray([lateral, height, depth])
                return _support_and_roof(
                    planes, slots, body_height, body_depth, parallel_cosine
                )

            # A partial divider proves lateral separation but not full depth.
            # A common roof with body-compatible depth supplies an independent
            # full-cavity extent.  Refine the depth anchor without turning the
            # roof into an acceptance decision.
            roof_depth_span = _projected_span(roof["plane"], depth)
            if 0.75 <= roof_depth_span / body_depth <= 1.35:
                roof_depth_center = float(np.dot(roof["plane"]["center"], depth))
                for slot in slots:
                    current = float(np.dot(slot["center"], depth))
                    slot["center"] += (roof_depth_center - current) * depth
                    slot["dimensions"][2] = roof_depth_span
                    slot["fit_ratios"][2] = roof_depth_span / body_depth
                    slot["fit_error"] = float(np.mean([
                        abs(math.log(value)) for value in slot["fit_ratios"]
                    ]))

            # The support-to-roof separation is more reliable than centring a
            # trimmed wall UV box on its area centroid.  Use it as the audited
            # clear height when it remains compatible with the moving body.
            for slot, rail_row in zip(slots, per_slot):
                if rail_row is None:
                    continue
                clear_height = (
                    roof["height_coordinate"] - rail_row["height_coordinate"]
                )
                if 0.75 <= clear_height / body_height <= 1.35:
                    slot["dimensions"][1] = clear_height
                    slot["fit_ratios"][1] = clear_height / body_height
                    slot["fit_error"] = float(np.mean([
                        abs(math.log(value)) for value in slot["fit_ratios"]
                    ]))

    return {
        "rails_valid": bool(rails_valid),
        "rails": per_slot,
        "roof": roof,
    }


def _direction(summary: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, Sequence) and value and isinstance(value[0], Mapping):
            value = value[0].get("outward_direction") or value[0].get("direction")
        direction = _unit(value)
        if direction is not None:
            return direction
    return None


def _rotation_for_polarity(
    moving_axes: np.ndarray,
    slot_axes: np.ndarray,
    roles: tuple[int, int, int],
    depth_sign: int,
) -> np.ndarray:
    lateral_index, height_index, depth_index = roles
    moving_basis = moving_axes.T
    target_basis = np.zeros((3, 3), dtype=float)
    target_basis[:, height_index] = slot_axes[1]
    target_basis[:, depth_index] = depth_sign * slot_axes[2]
    for lateral_sign in (1.0, -1.0):
        target_basis[:, lateral_index] = lateral_sign * slot_axes[0]
        rotation = target_basis @ moving_basis.T
        if float(np.linalg.det(rotation)) >= 1.0 - 1e-6:
            return rotation
    raise ValueError("unable to construct a proper enclosure rotation")


def _projected_envelope_dimensions(
    envelope: Mapping[str, Any], rotation: np.ndarray, frame: np.ndarray
) -> np.ndarray:
    rotated_axes = np.asarray([
        rotation @ axis for axis in envelope["axes"]
    ])
    return np.asarray([
        sum(
            abs(float(np.dot(rotated_axes[index], target_axis)))
            * float(envelope["dimensions"][index])
            for index in range(3)
        )
        for target_axis in frame
    ])


def propose_enclosure_bay_placements(
    stationary: Mapping[str, Any],
    moving: Mapping[str, Any] | None = None,
    *,
    moving_part_summary: Mapping[str, Any] | None = None,
    maximum: int = 16,
    parallel_cosine: float = 0.98,
) -> dict[str, Any]:
    """Return bounded, review-only placements for repeated enclosure bays.

    The transform contract is ``target_point = R @ source_point + t``.
    ``transform_4x4`` contains the same relative transform.  The function
    abstains unless repeated cavities, repeated supports, and a functional-body
    fit are all present; collision freedom and functional correctness remain
    unchecked.  An explicit ``functional_body_obb`` is preferred.  Otherwise
    the optional ``moving_part_summary`` (or ``moving`` itself) is passed to the
    dominant-planar envelope proposer.  A whole-part OBB is never silently used
    as the functional body.
    """

    moving_data = dict(moving) if isinstance(moving, Mapping) else {}
    part_data = (
        dict(moving_part_summary)
        if isinstance(moving_part_summary, Mapping)
        else {}
    )
    # Raw part features supply planes, topology, and the full envelope; explicit
    # moving metadata may add a reviewed body or an observed insertion end.
    combined_moving = dict(part_data)
    combined_moving.update(moving_data)

    body = _functional_body(combined_moving)
    if body is not None:
        envelope_audit = _envelope_audit_from_provided(body)
    else:
        derivation = derive_dominant_planar_envelope(
            combined_moving,
            parallel_cosine=parallel_cosine,
        )
        envelope_audit = _envelope_audit_from_derivation(derivation)
        derived_body = derivation.get("functional_body_obb")
        body = _normalise_envelope(
            derived_body,
            "derived_dominant_planar_envelope",
        )
        if derivation.get("status") != "proposed" or body is None:
            return _abstain_result(
                "Functional-body envelope derivation abstained: "
                + str(derivation.get("reason") or "insufficient evidence"),
                envelope_audit=envelope_audit,
            )
    if body is None:
        return _abstain_result(
            "A functional-body envelope is required.",
            envelope_audit=envelope_audit,
        )
    planes = [
        plane for index, raw in enumerate(stationary.get("planes") or [])
        if isinstance(raw, Mapping)
        and (plane := _normalise_plane(raw, index)) is not None
    ]
    if len(planes) < 4:
        return _abstain_result(
            "Too few audited planar footprints form opposing bays.",
            envelope_audit=envelope_audit,
        )

    candidates = []
    for family in _axis_families(planes, parallel_cosine):
        if len(family["planes"]) < 4:
            continue
        representative = max(family["planes"], key=lambda row: row["area"])
        for height_seed in representative["axes"]:
            for roles in itertools.permutations(range(3)):
                slots = _slot_pairs(
                    family,
                    height_seed,
                    body["dimensions"],
                    roles,
                )
                if len(slots) < 2:
                    continue
                # Keep the most coherent repeated family.  A wider enclosure
                # is never split merely because its width/body ratio is near 2.
                mean_dimensions = np.mean(
                    [slot["dimensions"] for slot in slots], axis=0
                )
                coherent = [
                    slot for slot in slots
                    if np.max(abs(slot["dimensions"] / mean_dimensions - 1.0))
                    <= 0.12
                ]
                if len(coherent) < 2:
                    continue
                support = _support_and_roof(
                    planes,
                    coherent,
                    float(body["dimensions"][roles[1]]),
                    float(body["dimensions"][roles[2]]),
                    parallel_cosine,
                )
                # Repeated support/rail evidence is deliberately mandatory.
                # Opposing walls plus an AABB-like fit alone are too ambiguous.
                if not support["rails_valid"]:
                    continue
                candidates.append({
                    "slots": coherent,
                    "roles": roles,
                    "support": support,
                    "fit_error": float(np.mean([
                        slot["fit_error"] for slot in coherent
                    ])),
                })

    if not candidates:
        return _abstain_result(
            (
                "No repeated bay has opposing walls, repeated supports, and "
                "a compatible functional-body envelope."
            ),
            envelope_audit=envelope_audit,
            audited_plane_count=len(planes),
        )

    family = min(
        candidates,
        key=lambda row: (
            row["fit_error"],
            -len(row["slots"]),
            row["roles"],
        ),
    )
    slots = family["slots"]
    roles = family["roles"]
    support = family["support"]
    stationary_opening = _direction(
        stationary,
        ("opening_direction", "opening_directions", "openings"),
    )
    moving_external_end = _direction(
        combined_moving,
        (
            "external_end_direction",
            "open_end_direction",
            "insertion_end_direction",
        ),
    )
    full_envelope = _full_envelope(combined_moving)
    envelope_audit = dict(envelope_audit)
    envelope_audit["full_envelope_source"] = (
        None if full_envelope is None else full_envelope["source"]
    )
    stationary_open_shell, stationary_topology = _open_shell(stationary)
    moving_open_shell, moving_topology = _open_shell(combined_moving)
    any_open_shell = bool(stationary_open_shell or moving_open_shell)

    evidence = [
        "opposing_wall_pairs",
        "repeated_equivalent_slots",
        "paired_repeated_supports",
        "functional_body_envelope_fit",
    ]
    if support["roof"] is not None:
        evidence.append("common_roof")
    if stationary_opening is not None and moving_external_end is not None:
        evidence.append("observed_open_end_polarity")

    proposals = []
    for slot_index, slot in enumerate(slots):
        rail = support["rails"][slot_index]
        target_center = np.asarray(slot["center"], dtype=float).copy()
        if rail is not None:
            height = slot["axes"][1]
            support_height = float(np.dot(rail["plane"]["center"], height))
            current_height = float(np.dot(target_center, height))
            desired_height = support_height + 0.5 * float(
                body["dimensions"][roles[1]]
            )
            target_center += (desired_height - current_height) * height
        for depth_sign in (1, -1):
            rotation = _rotation_for_polarity(
                body["axes"], slot["axes"], roles, depth_sign
            )
            translation = target_center - rotation @ body["center"]
            determinant = float(np.linalg.det(rotation))
            opening_alignment = None
            polarity = "ambiguous"
            if stationary_opening is not None and moving_external_end is not None:
                opening_alignment = float(np.dot(
                    rotation @ moving_external_end, stationary_opening
                ))
                polarity = "preferred" if opening_alignment >= 0.5 else "opposite"
            full_dimensions = None
            protrusion = False
            if full_envelope is not None:
                full_dimensions = _projected_envelope_dimensions(
                    full_envelope, rotation, slot["axes"]
                )
                protrusion = bool(np.any(
                    full_dimensions > 1.02 * slot["dimensions"]
                ))
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = translation
            fit_score = math.exp(-float(slot["fit_error"]))
            polarity_bonus = (
                0.0 if opening_alignment is None
                else 0.10 * max(-1.0, min(1.0, opening_alignment))
            )
            proposal_score = max(0.0, min(
                1.0,
                0.75 * fit_score
                + 0.10
                + (0.05 if support["roof"] is not None else 0.0)
                + polarity_bonus,
            ))
            proposals.append({
                "candidate_id": f"enclosure_slot_{slot_index}_polarity_{depth_sign:+d}",
                "slot_index": int(slot_index),
                "slot_center": target_center.tolist(),
                "slot_dimensions_lateral_height_depth": slot["dimensions"].tolist(),
                "moving_axis_roles": {
                    "lateral": int(roles[0]),
                    "height": int(roles[1]),
                    "depth": int(roles[2]),
                },
                "depth_polarity": int(depth_sign),
                "opening_polarity": polarity,
                "opening_alignment": opening_alignment,
                "rotation_matrix": rotation.tolist(),
                "translation": translation.tolist(),
                "transform_4x4": transform.tolist(),
                "placement": {
                    "rotation_matrix": rotation.tolist(),
                    "translate": translation.tolist(),
                },
                "determinant": determinant,
                "boundary_plane_indices": list(slot["boundary_indices"]),
                "support_plane_index": (
                    None if rail is None else int(rail["plane"]["index"])
                ),
                "roof_plane_index": (
                    None if support["roof"] is None
                    else int(support["roof"]["plane"]["index"])
                ),
                "body_fit_ratios_lateral_height_depth": list(slot["fit_ratios"]),
                "functional_body_source": body["source"],
                "functional_body_envelope_source": envelope_audit["source"],
                "functional_body_derivation_status": envelope_audit["status"],
                "functional_body_derivation_confidence": envelope_audit[
                    "confidence"
                ],
                "functional_body_excluded_protrusion_risk": envelope_audit[
                    "excluded_protrusion_risk"
                ],
                "full_envelope_source": (
                    None if full_envelope is None else full_envelope["source"]
                ),
                "full_envelope_dimensions_in_slot_frame": (
                    None if full_dimensions is None else full_dimensions.tolist()
                ),
                "full_bbox_protrusion_risk": protrusion,
                "full_envelope_protrusion_risk": protrusion,
                "full_bbox_used_as_rejection_gate": False,
                "full_envelope_used_as_rejection_gate": False,
                "evidence": list(evidence),
                "independent_evidence_count": len(evidence),
                "proposal_score": round(float(proposal_score), 9),
                "open_shell": any_open_shell,
                "stationary_open_shell": stationary_open_shell,
                "moving_open_shell": moving_open_shell,
                "stationary_topology": stationary_topology,
                "moving_topology": moving_topology,
                "collision_status": "unchecked",
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            })

    proposals.sort(key=lambda row: (
        -float(row["proposal_score"]),
        row["slot_index"],
        -row["depth_polarity"],
    ))
    proposals = proposals[: max(1, int(maximum))]
    placement_protrusion_risk = any(
        bool(row["full_envelope_protrusion_risk"]) for row in proposals
    )
    derived_protrusion_risk = bool(
        envelope_audit.get("excluded_protrusion_risk")
    )
    envelope_audit["placement_frame_full_envelope_protrusion_risk"] = bool(
        placement_protrusion_risk
    )
    envelope_audit["combined_protrusion_risk"] = bool(
        placement_protrusion_risk or derived_protrusion_risk
    )
    return {
        "status": "proposed",
        "reason": (
            "Repeated enclosure bays have multi-evidence geometric support; "
            "all placements still require review and collision validation."
        ),
        "proposals": proposals,
        "slot_count": len(slots),
        "independent_evidence": evidence,
        "functional_body_envelope_audit": envelope_audit,
        "functional_body_protrusion_risk": envelope_audit[
            "combined_protrusion_risk"
        ],
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
    }


def propose_enclosure_bays(
    stationary: Mapping[str, Any],
    moving: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Short alias for :func:`propose_enclosure_bay_placements`."""

    return propose_enclosure_bay_placements(stationary, moving, **kwargs)
