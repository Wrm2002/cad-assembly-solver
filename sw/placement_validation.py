"""Validate manifest placements using residual, graph, and bbox checks."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from constraints import CLEARANCE, COAXIAL, PLANAR_ALIGN, PLANAR_MATE, POCKET_MATE, match_features
from diagnostics import _choose_reference, _graph_analysis, _identity_placement
from features import extract_features
from match_scoring import score_matches


_DOMINANT_AXIS_GUARD_MAX_SELECTED_RADIUS_RATIO = 0.5
_DOMINANT_AXIS_GUARD_MAX_ANGLE_DEG = 2.0
_DOMINANT_AXIS_GUARD_MAX_RADIAL_DISTANCE_MM = 1.0


def _dot(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(v):
    return math.sqrt(_dot(v, v))


def _unit(v):
    length = _norm(v)
    return [float(x) / length for x in v] if length > 1e-12 else [0.0, 0.0, 1.0]


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _rotate(v, axis, degrees):
    axis = _unit(axis)
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    cross = _cross(axis, v)
    projection = _dot(axis, v)
    return [
        v[i] * c + cross[i] * s + axis[i] * projection * (1.0 - c)
        for i in range(3)
    ]


def _axis_to_rotation(source, target):
    source, target = _unit(source), _unit(target)
    dot = max(-1.0, min(1.0, _dot(source, target)))
    if dot > 1.0 - 1e-9:
        return [0, 0, 1], 0.0
    if dot < -1.0 + 1e-9:
        axis = _cross(source, [1, 0, 0] if abs(source[0]) < 0.9 else [0, 1, 0])
        return _unit(axis), 180.0
    return _unit(_cross(source, target)), math.degrees(math.acos(dot))


def transform_vector(vector, placement):
    result = [float(value) for value in vector]
    rotations = placement.get("rotate_sequence", [])
    if not rotations and "rotate_axis_angle" in placement:
        rotations = [{"axis_angle": placement["rotate_axis_angle"]}]
    for spec in rotations:
        if "axis_angle" in spec:
            axis_angle = spec["axis_angle"]
            result = _rotate(result, axis_angle[:3], axis_angle[3])
        elif "axis_to" in spec:
            axis, angle = _axis_to_rotation(
                spec["axis_to"]["from"],
                spec["axis_to"].get("to", [0, 0, 1]),
            )
            result = _rotate(result, axis, angle)
    return result


def transform_point(point, placement):
    rotated = transform_vector(point, placement)
    translation = placement.get("translate", [0, 0, 0])
    return [rotated[i] + float(translation[i]) for i in range(3)]


def _line_distance(p1, d1, p2, d2):
    d1, d2 = _unit(d1), _unit(d2)
    cross = _cross(d1, d2)
    if _norm(cross) < 1e-8:
        return _norm(_cross([p2[i] - p1[i] for i in range(3)], d1))
    return abs(_dot([p2[i] - p1[i] for i in range(3)], _unit(cross)))


def _feature(match, features, side, collection):
    index = 0 if side == "a" else 1
    part = match["parts"][index]
    feature_index = int(match.get(f"feat_{side}_idx", -1))
    values = features.get(part, {}).get(collection, [])
    return values[feature_index] if 0 <= feature_index < len(values) else {}


def _dominant_cylinder(part_features):
    """Return the largest usable cylinder and its source index."""

    usable = []
    for index, cylinder in enumerate(part_features.get("cylinders", [])):
        try:
            radius = abs(float(cylinder["radius"]))
            origin = cylinder["origin"]
            axis = cylinder["axis"]
            if (
                not math.isfinite(radius)
                or radius <= 1e-12
                or len(origin) != 3
                or len(axis) != 3
                or _norm([float(value) for value in axis]) <= 1e-12
                or any(
                    not math.isfinite(float(value))
                    for value in [*origin, *axis]
                )
            ):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        usable.append((radius, index, cylinder))
    if not usable:
        return None
    radius, index, cylinder = max(usable, key=lambda row: row[0])
    return {"radius": radius, "index": index, "feature": cylinder}


def _dominant_axis_guard_record(
    selected_a,
    selected_b,
    features_a,
    features_b,
    placement_a,
    placement_b,
):
    """Diagnose accidental matching through two subordinate cylinders.

    A bolt-hole-sized pair can have a perfect selected-axis residual while the
    two parts' dominant cylindrical axes remain far apart.  The supplementary
    guard is intentionally narrow: it activates only when *both* selected
    radii are no more than half of their respective dominant cylinder radius.
    Candidate generation remains unchanged; consumers decide whether a
    required, failed guard is allowed to close the connection.
    """

    thresholds = {
        "max_selected_radius_ratio": (
            _DOMINANT_AXIS_GUARD_MAX_SELECTED_RADIUS_RATIO
        ),
        "max_axis_angle_deg": _DOMINANT_AXIS_GUARD_MAX_ANGLE_DEG,
        "max_radial_distance_mm": (
            _DOMINANT_AXIS_GUARD_MAX_RADIAL_DISTANCE_MM
        ),
    }
    result = {
        "dominant_axis_guard_required": False,
        "dominant_axis_guard_evaluable": False,
        "dominant_axis_guard_passed": None,
        "dominant_axis_guard_reason": "not_evaluable",
        "dominant_axis_guard_thresholds": thresholds,
    }
    dominant_a = _dominant_cylinder(features_a)
    dominant_b = _dominant_cylinder(features_b)
    try:
        selected_radius_a = abs(float(selected_a["radius"]))
        selected_radius_b = abs(float(selected_b["radius"]))
    except (KeyError, TypeError, ValueError):
        result["dominant_axis_guard_reason"] = "selected_radius_unavailable"
        return result
    if dominant_a is None or dominant_b is None:
        result["dominant_axis_guard_reason"] = "dominant_cylinder_unavailable"
        return result

    ratio_a = selected_radius_a / max(float(dominant_a["radius"]), 1e-12)
    ratio_b = selected_radius_b / max(float(dominant_b["radius"]), 1e-12)
    result.update({
        "dominant_axis_guard_evaluable": True,
        "dominant_axis_selected_radius_mm_a": selected_radius_a,
        "dominant_axis_selected_radius_mm_b": selected_radius_b,
        "dominant_axis_max_radius_mm_a": float(dominant_a["radius"]),
        "dominant_axis_max_radius_mm_b": float(dominant_b["radius"]),
        "dominant_axis_selected_radius_ratio_a": ratio_a,
        "dominant_axis_selected_radius_ratio_b": ratio_b,
        "dominant_axis_feature_index_a": int(dominant_a["index"]),
        "dominant_axis_feature_index_b": int(dominant_b["index"]),
    })
    required = (
        ratio_a <= _DOMINANT_AXIS_GUARD_MAX_SELECTED_RADIUS_RATIO + 1e-12
        and ratio_b <= _DOMINANT_AXIS_GUARD_MAX_SELECTED_RADIUS_RATIO + 1e-12
    )
    result["dominant_axis_guard_required"] = required

    cylinder_a = dominant_a["feature"]
    cylinder_b = dominant_b["feature"]
    point_a = transform_point(cylinder_a["origin"], placement_a)
    point_b = transform_point(cylinder_b["origin"], placement_b)
    direction_a = _unit(transform_vector(cylinder_a["axis"], placement_a))
    direction_b = _unit(transform_vector(cylinder_b["axis"], placement_b))
    axis_angle = math.degrees(math.acos(max(
        -1.0,
        min(1.0, abs(_dot(direction_a, direction_b))),
    )))
    radial_distance = _line_distance(
        point_a, direction_a, point_b, direction_b
    )
    result.update({
        "dominant_axis_angle_deg": axis_angle,
        "dominant_axis_radial_distance_mm": radial_distance,
    })
    if not required:
        result["dominant_axis_guard_reason"] = (
            "not_required_selected_cylinders_are_not_both_small"
        )
        return result

    passed = (
        axis_angle <= _DOMINANT_AXIS_GUARD_MAX_ANGLE_DEG
        and radial_distance <= _DOMINANT_AXIS_GUARD_MAX_RADIAL_DISTANCE_MM
    )
    result["dominant_axis_guard_passed"] = passed
    if passed:
        result["dominant_axis_guard_reason"] = "required_and_satisfied"
    else:
        failures = []
        if axis_angle > _DOMINANT_AXIS_GUARD_MAX_ANGLE_DEG:
            failures.append("dominant_axis_angle_exceeds_threshold")
        if radial_distance > _DOMINANT_AXIS_GUARD_MAX_RADIAL_DISTANCE_MM:
            failures.append("dominant_axis_radial_distance_exceeds_threshold")
        result["dominant_axis_guard_reason"] = ";".join(failures)
    return result


def _bbox_axis_interval(part_features, placement, axis):
    bbox = part_features.get("bbox")
    if not bbox:
        return None
    low, high = bbox["min"], bbox["max"]
    values = []
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            for z in (low[2], high[2]):
                point = transform_point([x, y, z], placement)
                values.append(_dot(point, axis))
    return [min(values), max(values)]


def _plane_footprint(feature, placement):
    """Return a transformed rectangular footprint when it is explicit.

    Plane position and normal describe an infinite support plane.  They cannot
    establish that two trimmed faces meet.  The feature extractor also records
    the two trimmed UV spans and their world axes; use those fields only when
    the complete bounded description is present, leaving legacy plane records
    on their existing residual path.
    """

    axes = feature.get("footprint_axes")
    dimensions = (
        feature.get("footprint_dimensions")
        or feature.get("extent_uv")
        or feature.get("dimensions")
    )
    center = feature.get("centroid") or feature.get("position")
    if (
        not isinstance(axes, (list, tuple))
        or len(axes) != 2
        or not isinstance(dimensions, (list, tuple))
        or len(dimensions) != 2
        or not isinstance(center, (list, tuple))
        or len(center) != 3
    ):
        return None
    try:
        dimensions = [abs(float(value)) for value in dimensions]
        world_axes = [
            _unit(transform_vector(axis, placement)) for axis in axes
        ]
        world_center = transform_point(center, placement)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if (
        min(dimensions) <= 1e-9
        or any(not math.isfinite(value) for value in dimensions)
        or any(_norm(axis) <= 1e-9 for axis in world_axes)
        or abs(_dot(world_axes[0], world_axes[1])) >= 0.999
    ):
        return None
    return {
        "center": world_center,
        "axes": world_axes,
        "dimensions": dimensions,
    }


def _polygon_area_2d(points):
    if len(points) < 3:
        return 0.0
    return 0.5 * abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[index][1] * points[(index + 1) % len(points)][0]
        for index in range(len(points))
    ))


def _cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _clip_convex_polygon_2d(subject, clip):
    """Clip ``subject`` by a counter-clockwise convex polygon."""

    output = list(subject)
    tolerance = 1e-10
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        edge = [
            clip_end[0] - clip_start[0],
            clip_end[1] - clip_start[1],
        ]

        def inside(point):
            relative = [
                point[0] - clip_start[0],
                point[1] - clip_start[1],
            ]
            return _cross_2d(edge, relative) >= -tolerance

        def intersection(first, second):
            segment = [second[0] - first[0], second[1] - first[1]]
            denominator = _cross_2d(segment, edge)
            if abs(denominator) <= tolerance:
                return list(second)
            offset = [
                clip_start[0] - first[0],
                clip_start[1] - first[1],
            ]
            scale = _cross_2d(offset, edge) / denominator
            return [
                first[0] + scale * segment[0],
                first[1] + scale * segment[1],
            ]

        source = output
        output = []
        if not source:
            break
        previous = source[-1]
        previous_inside = inside(previous)
        for current in source:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
    return output


def _bounded_plane_overlap(first, second):
    """Measure projected overlap of two transformed trimmed-plane rectangles."""

    first_axis = _unit(first["axes"][0])
    raw_second_axis = first["axes"][1]
    second_axis = [
        raw_second_axis[index]
        - _dot(raw_second_axis, first_axis) * first_axis[index]
        for index in range(3)
    ]
    if _norm(second_axis) <= 1e-9:
        return None
    second_axis = _unit(second_axis)
    first_center = first["center"]

    first_half = [value * 0.5 for value in first["dimensions"]]
    first_polygon = [
        [-first_half[0], -first_half[1]],
        [first_half[0], -first_half[1]],
        [first_half[0], first_half[1]],
        [-first_half[0], first_half[1]],
    ]

    other_center = second["center"]
    other_half = [value * 0.5 for value in second["dimensions"]]
    other_polygon = []
    for sign_u, sign_v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corner = [
            other_center[index]
            + sign_u * other_half[0] * second["axes"][0][index]
            + sign_v * other_half[1] * second["axes"][1][index]
            for index in range(3)
        ]
        relative = [
            corner[index] - first_center[index] for index in range(3)
        ]
        other_polygon.append([
            _dot(relative, first_axis),
            _dot(relative, second_axis),
        ])

    clipped = _clip_convex_polygon_2d(other_polygon, first_polygon)
    overlap_area = _polygon_area_2d(clipped)
    first_area = first["dimensions"][0] * first["dimensions"][1]
    second_area = _polygon_area_2d(other_polygon)
    smaller_area = min(first_area, second_area)
    overlap_ratio = (
        overlap_area / smaller_area if smaller_area > 1e-12 else 0.0
    )
    center_delta = [
        other_center[index] - first_center[index] for index in range(3)
    ]
    tangential_distance = math.hypot(
        _dot(center_delta, first_axis),
        _dot(center_delta, second_axis),
    )
    return {
        "bounded_overlap_area_mm2": max(0.0, float(overlap_area)),
        "bounded_overlap_ratio": max(0.0, min(1.0, float(overlap_ratio))),
        "tangential_distance": float(tangential_distance),
        "tangential_distance_mm": float(tangential_distance),
    }


def constraint_residual(match, features, placements):
    kind = match["type"]
    a_name, b_name = match["parts"]
    a_place, b_place = placements.get(a_name, {}), placements.get(b_name, {})
    if kind in {COAXIAL, CLEARANCE}:
        a = _feature(match, features, "a", "cylinders")
        b = _feature(match, features, "b", "cylinders")
        if not a or not b:
            return {"type": kind, "parts": [a_name, b_name], "valid": False}
        pa = transform_point(a["origin"], a_place)
        pb = transform_point(b["origin"], b_place)
        da = _unit(transform_vector(a["axis"], a_place))
        db = _unit(transform_vector(b["axis"], b_place))
        axis_angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(_dot(da, db))))))
        radial = _line_distance(pa, da, pb, db)
        interval_a = _bbox_axis_interval(features.get(a_name, {}), a_place, da)
        interval_b = _bbox_axis_interval(features.get(b_name, {}), b_place, da)
        axial_overlap = None
        axial_overlap_ratio = None
        if interval_a and interval_b:
            overlap = min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0])
            axial_overlap = max(0.0, overlap)
            len_a = max(interval_a[1] - interval_a[0], 1e-9)
            len_b = max(interval_b[1] - interval_b[0], 1e-9)
            axial_overlap_ratio = axial_overlap / min(len_a, len_b)
        result = {
            "type": kind,
            "parts": [a_name, b_name],
            "valid": True,
            "axis_angle_deg": axis_angle,
            "radial_distance": radial,
            "axial_interval_a": interval_a,
            "axial_interval_b": interval_b,
            "axial_overlap": axial_overlap,
            "axial_overlap_ratio": axial_overlap_ratio,
            "residual": radial + axis_angle,
        }
        result.update(_dominant_axis_guard_record(
            a,
            b,
            features.get(a_name, {}),
            features.get(b_name, {}),
            a_place,
            b_place,
        ))
        return result
    if kind in {PLANAR_MATE, PLANAR_ALIGN}:
        a = _feature(match, features, "a", "planes")
        b = _feature(match, features, "b", "planes")
        if not a or not b:
            return {"type": kind, "parts": [a_name, b_name], "valid": False}
        pa = transform_point(a["position"], a_place)
        pb = transform_point(b["position"], b_place)
        na = _unit(transform_vector(a["normal"], a_place))
        nb = _unit(transform_vector(b["normal"], b_place))
        expected = -1.0 if kind == PLANAR_MATE else 1.0
        angle = math.degrees(math.acos(max(-1.0, min(1.0, expected * _dot(na, nb)))))
        distance = abs(_dot([pb[i] - pa[i] for i in range(3)], na))
        result = {
            "type": kind,
            "parts": [a_name, b_name],
            "valid": True,
            "normal_angle_deg": angle,
            "plane_distance": distance,
            "residual": distance + angle,
        }
        footprint_a = _plane_footprint(a, a_place)
        footprint_b = _plane_footprint(b, b_place)
        bounded = (
            _bounded_plane_overlap(footprint_a, footprint_b)
            if footprint_a is not None and footprint_b is not None
            else None
        )
        result["bounded_footprint_available"] = bounded is not None
        if bounded is not None:
            result.update(bounded)
        return result
    if kind == POCKET_MATE:
        pocket_a = match.get("pocket_a") or {}
        pocket_b = match.get("pocket_b") or {}
        center_a = match.get("_center_a") or pocket_a.get("center")
        center_b = match.get("_center_b") or pocket_b.get("center")
        if center_a and center_b:
            a = transform_point(center_a, a_place)
            b = transform_point(center_b, b_place)
            distance = _norm([a[i] - b[i] for i in range(3)])
            return {
                "type": kind,
                "parts": [a_name, b_name],
                "valid": True,
                "pocket_center_distance": distance,
                "residual": distance,
            }
    return {"type": kind, "parts": [a_name, b_name], "valid": False}


def transformed_bbox(features, placement):
    bbox = features.get("bbox")
    if not bbox:
        return None
    low, high = bbox["min"], bbox["max"]
    corners = [
        transform_point([x, y, z], placement)
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]
    return {
        "min": [min(point[i] for point in corners) for i in range(3)],
        "max": [max(point[i] for point in corners) for i in range(3)],
    }


def bbox_collisions(features, placements):
    boxes = {
        part: transformed_bbox(part_features, placements.get(part, {}))
        for part, part_features in features.items()
    }
    collisions = []
    for a, b in combinations(features, 2):
        if not boxes[a] or not boxes[b]:
            continue
        overlap = [
            min(boxes[a]["max"][i], boxes[b]["max"][i])
            - max(boxes[a]["min"][i], boxes[b]["min"][i])
            for i in range(3)
        ]
        if min(overlap) <= 1e-6:
            continue
        overlap_volume = math.prod(overlap)
        volumes = []
        for part in (a, b):
            size = [boxes[part]["max"][i] - boxes[part]["min"][i] for i in range(3)]
            volumes.append(max(math.prod(size), 1e-9))
        ratio = overlap_volume / min(volumes)
        volume_ratio = min(volumes) / max(volumes)
        contained_index = 0 if volumes[0] <= volumes[1] else 1
        smaller_part = (a, b)[contained_index]
        strict_containment = ratio >= 0.95 and volume_ratio <= 0.80
        collisions.append(
            {
                "parts": [a, b],
                "overlap": overlap,
                "overlap_volume": overlap_volume,
                "minimum_part_volume_ratio": ratio,
                "bbox_volume_ratio": volume_ratio,
                "smaller_part": smaller_part,
                "is_strict_containment": strict_containment,
                "severe": ratio >= 0.25,
                "method": "transformed_bbox",
            }
        )
    return collisions


def exact_shape_collisions(
    folder,
    components,
    *,
    minimum_volume=0.001,
    minimum_part_ratio=0.0001,
):
    """Confirm solid penetration using OCCT Boolean Common.

    Contact through a face or edge has zero common volume and is not reported.
    The function returns an explicit check status so callers never mistake an
    unavailable Boolean check for a collision-free result.
    """
    folder = Path(folder).resolve()
    try:
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from build_assembly import build_transform, load_step
    except Exception as exc:
        return {
            "status": "unavailable",
            "method": "occt_boolean_common_volume",
            "collisions": [],
            "errors": [f"OCCT import failed: {exc}"],
        }

    def volume(shape):
        properties = GProp_GProps()
        brepgprop.VolumeProperties(shape, properties)
        return max(0.0, float(properties.Mass()))

    shapes = {}
    volumes = {}
    errors = []
    for component in components:
        source = component["source"]
        try:
            shape = load_step(str(folder / source))
            transform = build_transform(component.get("placement", {}))
            if transform.Form() != 0:
                shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
            shapes[source] = shape
            volumes[source] = volume(shape)
        except Exception as exc:
            errors.append(f"{source}: load/transform failed: {exc}")

    collisions = []
    for a, b in combinations(sorted(shapes), 2):
        try:
            operation = BRepAlgoAPI_Common(shapes[a], shapes[b])
            operation.Build()
            if not operation.IsDone():
                errors.append(f"{a} <-> {b}: Boolean Common did not complete")
                continue
            common_volume = volume(operation.Shape())
            smaller = max(min(volumes[a], volumes[b]), 1e-12)
            ratio = common_volume / smaller
            if (
                common_volume >= float(minimum_volume)
                and ratio >= float(minimum_part_ratio)
            ):
                collisions.append(
                    {
                        "parts": [a, b],
                        "intersection_volume_mm3": common_volume,
                        "minimum_part_volume_ratio": ratio,
                        "severe": True,
                        "method": "occt_boolean_common_volume",
                    }
                )
        except Exception as exc:
            errors.append(f"{a} <-> {b}: Boolean Common failed: {exc}")
    return {
        "status": "success" if not errors else "partial",
        "method": "occt_boolean_common_volume",
        "collisions": collisions,
        "errors": errors,
    }


def _bbox_overlap_volume(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
    *,
    tolerance: float = 1e-6,
) -> tuple[float, tuple[float, float, float]]:
    overlap = tuple(
        max(
            0.0,
            min(first[index + 3], second[index + 3])
            - max(first[index], second[index]),
        )
        for index in range(3)
    )
    if any(value <= tolerance for value in overlap):
        return 0.0, overlap
    return overlap[0] * overlap[1] * overlap[2], overlap


def _bbox_clearance_translation_for_second(
    first: tuple[float, float, float, float, float, float],
    second: tuple[float, float, float, float, float, float],
    *,
    clearance: float = 0.05,
) -> tuple[float, float, float]:
    """Return the smallest axis translation that separates ``second`` AABB."""

    volume, overlap = _bbox_overlap_volume(first, second, tolerance=0.0)
    if volume <= 0.0:
        return (0.0, 0.0, 0.0)
    options = []
    for axis in range(3):
        positive = first[axis + 3] - second[axis] + float(clearance)
        negative = first[axis] - second[axis + 3] - float(clearance)
        options.append((abs(positive), axis, positive))
        options.append((abs(negative), axis, negative))
    _, axis, value = min(options)
    result = [0.0, 0.0, 0.0]
    result[axis] = float(value)
    return tuple(result)


def exact_shape_collisions_solid_broadphase(
    folder,
    components,
    *,
    minimum_volume=0.001,
    minimum_part_ratio=0.0001,
    maximum_solid_pair_checks=512,
):
    """Validate large multi-solid STEP through per-solid AABB broad phase.

    Whole-compound Boolean Common can stall or exhaust memory on vendor chassis
    models. This variant transforms each component once, enumerates its solids,
    and runs Boolean Common only for solid pairs whose AABBs overlap in all
    three axes. A capped or open-shell check returns ``partial`` rather than a
    false collision-free success.
    """

    folder = Path(folder).resolve()
    components = list(components)
    try:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
        from build_assembly import build_transform, load_step
    except Exception as exc:
        return {
            "status": "unavailable",
            "method": "occt_per_solid_aabb_then_boolean_common",
            "collisions": [],
            "errors": [f"OCCT import failed: {exc}"],
            "checked_solid_pair_count": 0,
            "collision_result": "uncertain",
            "collision_free": None,
            "component_audit": {},
            "coverage_audit": {
                "status": "unknown",
                "complete": False,
                "collision_scope": "solid_topology_only",
                "reason": "OCCT imports are unavailable; no topology was checked.",
            },
        }

    def volume(shape):
        properties = GProp_GProps()
        brepgprop.VolumeProperties(shape, properties)
        return max(0.0, float(properties.Mass()))

    def bounds(shape):
        box = Bnd_Box()
        box.SetGap(0.0)
        brepbndlib.Add(shape, box)
        return tuple(float(value) for value in box.Get())

    def topology_coverage(shape, solid_rows):
        """Audit which component faces are represented by checked solids.

        ``TopExp_Explorer(shape, TopAbs_SOLID)`` is intentionally retained for
        Boolean collision checks, but a STEP compound may contain both valid
        solids and additional orphan faces/open shells.  Mapping topology is a
        linear operation and lets us expose that unvalidated geometry without
        attempting an impractical all-face Boolean pass.
        """

        all_faces = TopTools_IndexedMapOfShape()
        solid_faces = TopTools_IndexedMapOfShape()
        all_shells = TopTools_IndexedMapOfShape()
        solid_shells = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_FACE, all_faces)
        topexp.MapShapes(shape, TopAbs_SHELL, all_shells)
        for row in solid_rows:
            topexp.MapShapes(row["shape"], TopAbs_FACE, solid_faces)
            topexp.MapShapes(row["shape"], TopAbs_SHELL, solid_shells)

        uncovered_face_count = sum(
            not solid_faces.Contains(all_faces.FindKey(index))
            for index in range(1, all_faces.Size() + 1)
        )
        uncovered_shell_count = sum(
            not solid_shells.Contains(all_shells.FindKey(index))
            for index in range(1, all_shells.Size() + 1)
        )
        total_face_count = int(all_faces.Size())
        covered_face_count = total_face_count - int(uncovered_face_count)
        face_coverage_ratio = (
            covered_face_count / total_face_count
            if total_face_count
            else (1.0 if solid_rows else 0.0)
        )
        complete = (
            bool(solid_rows)
            and uncovered_face_count == 0
            and uncovered_shell_count == 0
        )
        return {
            "status": "complete" if complete else "partial",
            "complete": complete,
            "collision_scope": "solid_topology_only",
            "topology_face_count": total_face_count,
            "solid_covered_face_count": covered_face_count,
            "uncovered_face_count": int(uncovered_face_count),
            "solid_face_coverage_ratio": float(face_coverage_ratio),
            "topology_shell_count": int(all_shells.Size()),
            "solid_covered_shell_count": int(all_shells.Size())
            - int(uncovered_shell_count),
            "uncovered_shell_count": int(uncovered_shell_count),
            "has_orphan_or_open_shell_geometry": bool(
                uncovered_face_count or uncovered_shell_count or not solid_rows
            ),
        }

    solids_by_component: dict[str, list[dict[str, Any]]] = {}
    component_volumes: dict[str, float] = {}
    component_audit: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    open_shell_components: list[str] = []
    open_shell_only_components: list[str] = []
    mixed_solid_open_shell_components: list[str] = []
    for component in components:
        source = str(component["source"])
        try:
            source_path = Path(source)
            shape = load_step(
                str(source_path if source_path.is_absolute() else folder / source_path)
            )
            transform = build_transform(component.get("placement", {}))
            if transform.Form() != 0:
                shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            rows = []
            while explorer.More():
                solid = explorer.Current()
                try:
                    row_volume = volume(solid)
                    row_bounds = bounds(solid)
                except Exception as exc:
                    errors.append(
                        f"{source}: solid audit failed: {type(exc).__name__}: {exc}"
                    )
                    explorer.Next()
                    continue
                rows.append({
                    "solid_index": len(rows),
                    "shape": solid,
                    "bounds": row_bounds,
                    "volume": row_volume,
                })
                explorer.Next()
            solids_by_component[source] = rows
            component_volumes[source] = sum(row["volume"] for row in rows)
            try:
                coverage = topology_coverage(shape, rows)
            except Exception as exc:
                coverage = {
                    "status": "unknown",
                    "complete": False,
                    "collision_scope": "solid_topology_only",
                    "topology_face_count": None,
                    "solid_covered_face_count": None,
                    "uncovered_face_count": None,
                    "solid_face_coverage_ratio": None,
                    "topology_shell_count": None,
                    "solid_covered_shell_count": None,
                    "uncovered_shell_count": None,
                    "has_orphan_or_open_shell_geometry": None,
                }
                errors.append(
                    f"{source}: topology coverage audit failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            component_audit[source] = {
                "solid_count": len(rows),
                "solid_volume_mm3": component_volumes[source],
                **coverage,
            }
            if not rows:
                open_shell_components.append(source)
                open_shell_only_components.append(source)
                errors.append(
                    f"{source}: no solid topology; open-shell collision result is incomplete"
                )
            elif not coverage.get("complete"):
                open_shell_components.append(source)
                mixed_solid_open_shell_components.append(source)
                uncovered = coverage.get("uncovered_face_count")
                errors.append(
                    f"{source}: solid collision scope leaves "
                    f"{uncovered if uncovered is not None else 'unknown'} "
                    "orphan/open-shell face(s) unchecked"
                )
        except Exception as exc:
            errors.append(
                f"{source}: load/transform failed: {type(exc).__name__}: {exc}"
            )

    collisions = []
    checked = 0
    broadphase_candidates = 0
    truncated_pairs = []
    for source_a, source_b in combinations(sorted(solids_by_component), 2):
        left = solids_by_component[source_a]
        right = solids_by_component[source_b]
        candidates = []
        for solid_a in left:
            for solid_b in right:
                proxy, overlap = _bbox_overlap_volume(
                    solid_a["bounds"], solid_b["bounds"]
                )
                if proxy > 0.0:
                    candidates.append((proxy, overlap, solid_a, solid_b))
        candidates.sort(key=lambda row: row[0], reverse=True)
        broadphase_candidates += len(candidates)
        if len(candidates) > int(maximum_solid_pair_checks):
            truncated_pairs.append({
                "parts": [source_a, source_b],
                "candidate_count": len(candidates),
                "checked_count": int(maximum_solid_pair_checks),
            })
            errors.append(
                f"{source_a} <-> {source_b}: solid broad phase truncated "
                f"{len(candidates)} candidates to {int(maximum_solid_pair_checks)}"
            )
        common_total = 0.0
        common_rows = []
        for _, overlap, solid_a, solid_b in candidates[: int(maximum_solid_pair_checks)]:
            checked += 1
            try:
                operation = BRepAlgoAPI_Common(solid_a["shape"], solid_b["shape"])
                operation.Build()
                if not operation.IsDone():
                    errors.append(
                        f"{source_a}[{solid_a['solid_index']}] <-> "
                        f"{source_b}[{solid_b['solid_index']}]: Boolean Common did not complete"
                    )
                    continue
                common_volume = volume(operation.Shape())
                if common_volume <= 0.0:
                    continue
                common_total += common_volume
                clearance_translation = _bbox_clearance_translation_for_second(
                    solid_a["bounds"], solid_b["bounds"]
                )
                common_rows.append({
                    "solid_indices": [
                        solid_a["solid_index"], solid_b["solid_index"]
                    ],
                    "intersection_volume_mm3": common_volume,
                    "bbox_overlap_mm": list(overlap),
                    "clearance_translation_for_second_part_mm": list(
                        clearance_translation
                    ),
                })
            except Exception as exc:
                errors.append(
                    f"{source_a}[{solid_a['solid_index']}] <-> "
                    f"{source_b}[{solid_b['solid_index']}]: Boolean Common failed: {exc}"
                )
        smaller = max(
            min(
                component_volumes.get(source_a, 0.0),
                component_volumes.get(source_b, 0.0),
            ),
            1e-12,
        )
        ratio = common_total / smaller
        if common_total >= float(minimum_volume) and ratio >= float(minimum_part_ratio):
            grouped_clearance: dict[tuple[int, int], list[float]] = {}
            for row in common_rows:
                vector = row["clearance_translation_for_second_part_mm"]
                axis = max(range(3), key=lambda index: abs(float(vector[index])))
                value = float(vector[axis])
                if abs(value) <= 1e-12:
                    continue
                grouped_clearance.setdefault(
                    (axis, 1 if value > 0.0 else -1), []
                ).append(value)
            clearance_proposals = []
            for (axis, _sign), values in grouped_clearance.items():
                value = max(values, key=abs)
                vector = [0.0, 0.0, 0.0]
                vector[axis] = value
                clearance_proposals.append(vector)
            clearance_proposals.sort(
                key=lambda vector: sum(abs(value) for value in vector)
            )
            collisions.append({
                "parts": [source_a, source_b],
                "intersection_volume_mm3": common_total,
                "minimum_part_volume_ratio": ratio,
                "part_volume_ratios": {
                    source_a: common_total / max(
                        component_volumes.get(source_a, 0.0), 1e-12
                    ),
                    source_b: common_total / max(
                        component_volumes.get(source_b, 0.0), 1e-12
                    ),
                },
                "severe": True,
                "method": "occt_per_solid_aabb_then_boolean_common",
                "solid_intersections": common_rows[:32],
                "solid_intersection_count": len(common_rows),
                "clearance_translation_proposals_for_second_part_mm": (
                    clearance_proposals[:6]
                ),
            })
    total_common_volume = sum(
        float(row.get("intersection_volume_mm3", 0.0)) for row in collisions
    )
    positive_solid_pairs = sum(
        int(row.get("solid_intersection_count", 0)) for row in collisions
    )
    requested_component_count = len(components)
    loaded_component_count = len(component_audit)
    partial_topology_components = sorted(
        source
        for source, audit in component_audit.items()
        if not audit.get("complete")
    )
    topology_coverage_complete = (
        loaded_component_count == requested_component_count
        and not partial_topology_components
    )
    exact_check_complete = (
        requested_component_count > 0
        and topology_coverage_complete
        and not errors
    )
    coverage_audit = {
        "status": "complete" if exact_check_complete else "partial",
        "complete": exact_check_complete,
        "topology_coverage_complete": topology_coverage_complete,
        "collision_scope": "solid_topology_only",
        "component_count_requested": requested_component_count,
        "component_count_loaded": loaded_component_count,
        "fully_covered_component_count": sum(
            bool(audit.get("complete")) for audit in component_audit.values()
        ),
        "partially_covered_components": partial_topology_components,
        "open_shell_only_components": sorted(open_shell_only_components),
        "mixed_solid_open_shell_components": sorted(
            mixed_solid_open_shell_components
        ),
        "topology_face_count": sum(
            int(audit.get("topology_face_count") or 0)
            for audit in component_audit.values()
        ),
        "solid_covered_face_count": sum(
            int(audit.get("solid_covered_face_count") or 0)
            for audit in component_audit.values()
        ),
        "uncovered_face_count": sum(
            int(audit.get("uncovered_face_count") or 0)
            for audit in component_audit.values()
        ),
        "unknown_topology_audit_components": sorted(
            source
            for source, audit in component_audit.items()
            if audit.get("status") == "unknown"
        ),
        "truncated_component_pair_count": len(truncated_pairs),
        "unchecked_open_shell_geometry_requires_review": bool(
            partial_topology_components
        ),
        "reason": (
            "All loaded component faces are represented by checked solids and "
            "all scheduled solid Boolean checks completed."
            if exact_check_complete
            else "At least one component or solid-pair check was not fully "
            "covered; a zero solid-collision count is not proof of collision-free geometry."
        ),
    }
    collision_result = (
        "collision_detected"
        if collisions
        else ("no_collision_detected" if exact_check_complete else "uncertain")
    )
    collision_free = (
        False
        if collisions
        else (True if exact_check_complete else None)
    )
    return {
        "status": "success" if exact_check_complete else "partial",
        "method": "occt_per_solid_aabb_then_boolean_common",
        "collisions": collisions,
        "errors": errors,
        "collision_result": collision_result,
        "collision_free": collision_free,
        "component_audit": component_audit,
        "coverage_audit": coverage_audit,
        "open_shell_components": sorted(open_shell_components),
        "open_shell_only_components": sorted(open_shell_only_components),
        "mixed_solid_open_shell_components": sorted(
            mixed_solid_open_shell_components
        ),
        "broadphase_candidate_count": broadphase_candidates,
        "checked_solid_pair_count": checked,
        "maximum_solid_pair_checks_per_component_pair": int(
            maximum_solid_pair_checks
        ),
        "truncated_component_pairs": truncated_pairs,
        "collision_summary": {
            "colliding_component_pair_count": len(collisions),
            "positive_solid_pair_count": positive_solid_pairs,
            "total_intersection_volume_mm3": total_common_volume,
            "component_pair_volume_ratio_sum": sum(
                float(row.get("minimum_part_volume_ratio", 0.0))
                for row in collisions
            ),
            "clearance_vectors_are_proposals_only": True,
        },
    }


def validate_assembly(folder, matches_path=None, residual_threshold=5.0):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "assembly_manifest.json").read_text(encoding="utf-8"))
    step_files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
        and not path.name.lower().startswith("assembly")
    )
    features = {path.name: extract_features(str(path)) for path in step_files}
    if matches_path:
        matches = json.loads(Path(matches_path).read_text(encoding="utf-8"))
    else:
        matches = score_matches(match_features(features), features)
    placements = {
        component["source"]: component.get("placement", {})
        for component in manifest.get("components", [])
    }
    reference = _choose_reference(features, matches)
    graph, reachable = _graph_analysis(list(features), matches, reference)
    unsolved = sorted(set(features) - reachable | (set(features) - set(placements)))
    identity = sorted(part for part, placement in placements.items() if _identity_placement(placement))
    residuals = [constraint_residual(match, features, placements) for match in matches]
    valid_values = [item["residual"] for item in residuals if item.get("valid")]
    collisions = bbox_collisions(features, placements)
    severe_bbox_candidates = [collision for collision in collisions if collision["severe"]]
    warnings = []
    if unsolved:
        warnings.append(f"unsolved parts: {', '.join(unsolved)}")
    ambiguous = [part for part in identity if part != reference and part not in unsolved]
    if ambiguous:
        warnings.append(f"identity placement is ambiguous: {', '.join(ambiguous)}")
    if graph["connected_component_count"] > 1:
        warnings.append("assembly graph is disconnected")
    if severe_bbox_candidates:
        warnings.append(
            f"{len(severe_bbox_candidates)} severe bbox overlap candidate(s); "
            "OCCT boolean confirmation required"
        )
    maximum = max(valid_values, default=None)
    if maximum is not None and maximum > residual_threshold:
        warnings.append(f"constraint residual exceeds {residual_threshold}")
    if unsolved:
        status = "partial_success" if len(unsolved) < len(features) else "failed"
    elif maximum is not None and maximum > residual_threshold:
        status = "partial_success"
    else:
        status = "success"
    return {
        "schema_version": 1,
        "status": status,
        "num_parts": len(features),
        "num_solved_parts": len(features) - len(unsolved),
        "num_unsolved_parts": len(unsolved),
        "unsolved_parts": unsolved,
        "reference_part": reference,
        "identity_placements": identity,
        "graph": graph,
        "constraint_residuals": residuals,
        "max_constraint_residual": maximum,
        "collision_count": len(collisions),
        "bbox_collision_count": len(collisions),
        "possible_severe_bbox_count": len(severe_bbox_candidates),
        "severe_penetration_count": 0,
        "collision_method": "bbox_precheck_only",
        "collisions": collisions,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--matches")
    parser.add_argument("--residual-threshold", type=float, default=5.0)
    args = parser.parse_args()
    result = validate_assembly(args.folder, args.matches, args.residual_threshold)
    output = Path(args.folder).resolve() / "assembly_validation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"status={result['status']} warnings={len(result['warnings'])}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
