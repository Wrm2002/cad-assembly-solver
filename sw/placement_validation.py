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
        return {
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
        return {
            "type": kind,
            "parts": [a_name, b_name],
            "valid": True,
            "normal_angle_deg": angle,
            "plane_distance": distance,
            "residual": distance + angle,
        }
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
        collisions.append(
            {
                "parts": [a, b],
                "overlap": overlap,
                "overlap_volume": overlap_volume,
                "minimum_part_volume_ratio": ratio,
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
