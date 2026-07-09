"""Build a reversible coarse interface proxy from a full part-feature model.

The proxy never changes CAD geometry.  It clusters only geometrically
equivalent plane and cylinder records, keeps aggregate evidence, and stores
index ranges that point back to every full-resolution feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(vector: list[float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm <= 1e-15:
        raise ValueError(f"zero-length direction: {vector}")
    return tuple(float(value) / norm for value in vector)


def _canonical_direction(
    vector: list[float],
) -> tuple[tuple[float, float, float], int]:
    unit = _unit(vector)
    for value in unit:
        if abs(value) > 1e-12:
            if value < 0:
                return tuple(-component for component in unit), -1
            break
    return unit, 1


def _quantize(value: float, tolerance: float) -> int:
    if tolerance <= 0:
        raise ValueError("quantization tolerance must be positive")
    return int(round(float(value) / tolerance))


def _index(feature: dict[str, Any]) -> int:
    return int(str(feature["feature_id"]).rsplit(":", 1)[1])


def compact_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(set(indices))
    ranges = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def range_cardinality(ranges: list[list[int]]) -> int:
    return sum(end - start + 1 for start, end in ranges)


def _log_bucket(value: float, ratio: float) -> int:
    if value <= 0:
        return -10**9
    return int(math.floor(math.log(value) / math.log(ratio)))


def _radius_bucket(
    radius: float,
    *,
    relative_tolerance: float,
    absolute_tolerance_mm: float,
) -> tuple[str, int]:
    transition = absolute_tolerance_mm / relative_tolerance
    if radius <= transition:
        return (
            "absolute",
            _quantize(radius, absolute_tolerance_mm),
        )
    return (
        "relative",
        _log_bucket(radius / transition, 1.0 + relative_tolerance),
    )


def _families(
    clusters: list[dict[str, Any]],
    *,
    kind: str,
    key_function: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters, start=1):
        grouped[key_function(cluster)].append(index)
    result = []
    for family_number, key in enumerate(sorted(grouped), start=1):
        child_indices = grouped[key]
        children = [clusters[index - 1] for index in child_indices]
        result.append({
            "family_id": f"{kind}_family_{family_number:05d}",
            "kind": kind,
            "signature": list(key) if isinstance(key, tuple) else key,
            "child_cluster_count": len(children),
            "child_cluster_index_ranges": compact_ranges(child_indices),
            "full_feature_count": sum(
                child["member_count"] for child in children
            ),
            "maximum_member_area": max(
                child["maximum_member_area"] for child in children
            ),
        })
    return result


def _plane_key(
    feature: dict[str, Any],
    *,
    direction_tolerance: float,
    linear_tolerance_mm: float,
):
    parameters = feature["parameters"]
    normal, sign = _canonical_direction(parameters["normal"])
    position = [float(value) for value in parameters["position"]]
    offset = sum(
        component * coordinate
        for component, coordinate in zip(normal, position)
    )
    return (
        tuple(
            _quantize(component, direction_tolerance)
            for component in normal
        ),
        _quantize(offset, linear_tolerance_mm),
    ), normal, sign


def _cylinder_key(
    feature: dict[str, Any],
    *,
    direction_tolerance: float,
    linear_tolerance_mm: float,
    radius_tolerance_mm: float,
):
    parameters = feature["parameters"]
    axis, sign = _canonical_direction(parameters["axis"])
    origin = [float(value) for value in parameters["origin"]]
    axial = sum(
        component * coordinate
        for component, coordinate in zip(axis, origin)
    )
    perpendicular = tuple(
        coordinate - axial * component
        for coordinate, component in zip(origin, axis)
    )
    return (
        _quantize(parameters["radius"], radius_tolerance_mm),
        tuple(
            _quantize(component, direction_tolerance)
            for component in axis
        ),
        tuple(
            _quantize(component, linear_tolerance_mm)
            for component in perpendicular
        ),
    ), axis, sign


def _bounds(points: list[list[float]]) -> dict[str, list[float]]:
    return {
        "minimum": [
            min(float(point[axis]) for point in points)
            for axis in range(3)
        ],
        "maximum": [
            max(float(point[axis]) for point in points)
            for axis in range(3)
        ],
    }


def _cluster(
    features: list[dict[str, Any]],
    key_function: Callable,
    *,
    kind: str,
    position_field: str,
    key_options: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[tuple[dict[str, Any], Any, int]]] = (
        defaultdict(list)
    )
    for feature in features:
        key, direction, sign = key_function(feature, **key_options)
        grouped[key].append((feature, direction, sign))
    clusters = []
    for cluster_number, key in enumerate(sorted(grouped), start=1):
        members = grouped[key]
        representative, direction, _ = max(
            members,
            key=lambda item: float(
                item[0]["parameters"].get("area") or 0.0
            ),
        )
        indices = [_index(item[0]) for item in members]
        ranges = compact_ranges(indices)
        areas = [
            float(item[0]["parameters"].get("area") or 0.0)
            for item in members
        ]
        positions = [
            [
                float(value)
                for value in item[0]["parameters"][position_field]
            ]
            for item in members
        ]
        positive = sum(item[2] > 0 for item in members)
        clusters.append({
            "proxy_id": f"{kind}_cluster_{cluster_number:06d}",
            "kind": kind,
            "canonical_direction": list(direction),
            "representative_feature_index": _index(representative),
            "representative_parameters": representative["parameters"],
            "member_count": len(members),
            "member_index_ranges": ranges,
            "member_mapping_complete": (
                range_cardinality(ranges) == len(members)
            ),
            "orientation_counts": {
                "canonical": positive,
                "opposite": len(members) - positive,
            },
            "total_area": sum(areas),
            "maximum_member_area": max(areas, default=0.0),
            "minimum_member_area": min(areas, default=0.0),
            "position_bounds_mm": _bounds(positions),
        })
    return clusters


def build_proxy(
    full: dict[str, Any],
    *,
    source_path: Path,
    direction_tolerance: float = 1e-6,
    linear_tolerance_mm: float = 1e-5,
    radius_tolerance_mm: float = 1e-5,
    planar_area_family_ratio: float = 1.25,
    cylinder_radius_relative_tolerance: float = 0.08,
    cylinder_radius_absolute_tolerance_mm: float = 1.0,
) -> dict[str, Any]:
    planes = full.get("planar_faces", [])
    cylinders = full.get("cylindrical_faces", [])
    holes = full.get("holes", [])
    plane_clusters = _cluster(
        planes,
        _plane_key,
        kind="plane_interface",
        position_field="position",
        key_options={
            "direction_tolerance": direction_tolerance,
            "linear_tolerance_mm": linear_tolerance_mm,
        },
    )
    cylinder_options = {
        "direction_tolerance": direction_tolerance,
        "linear_tolerance_mm": linear_tolerance_mm,
        "radius_tolerance_mm": radius_tolerance_mm,
    }
    cylinder_clusters = _cluster(
        cylinders,
        _cylinder_key,
        kind="cylindrical_interface",
        position_field="origin",
        key_options=cylinder_options,
    )
    hole_clusters = _cluster(
        holes,
        _cylinder_key,
        kind="hole_interface",
        position_field="origin",
        key_options=cylinder_options,
    )
    plane_families = _families(
        plane_clusters,
        kind="plane",
        key_function=lambda cluster: _log_bucket(
            float(
                cluster["representative_parameters"].get("area")
                or 0.0
            ),
            planar_area_family_ratio,
        ),
    )
    cylinder_families = _families(
        cylinder_clusters,
        kind="cylinder",
        key_function=lambda cluster: _radius_bucket(
            float(
                cluster["representative_parameters"].get("radius")
                or 0.0
            ),
            relative_tolerance=cylinder_radius_relative_tolerance,
            absolute_tolerance_mm=(
                cylinder_radius_absolute_tolerance_mm
            ),
        ),
    )
    full_count = len(planes) + len(cylinders)
    proxy_count = len(plane_clusters) + len(cylinder_clusters)
    mapping_complete = (
        sum(cluster["member_count"] for cluster in plane_clusters)
        == len(planes)
        and sum(
            cluster["member_count"] for cluster in cylinder_clusters
        )
        == len(cylinders)
        and sum(cluster["member_count"] for cluster in hole_clusters)
        == len(holes)
        and all(
            cluster["member_mapping_complete"]
            for cluster in (
                plane_clusters + cylinder_clusters + hole_clusters
            )
        )
    )
    return {
        "schema_version": "1.0.0",
        "part_id": full["part_id"],
        "representation": "reversible coarse interface proxy",
        "source_feature_file": str(source_path.resolve()),
        "source_feature_sha256": sha256_file(source_path),
        "units": full.get("units"),
        "coordinate_frame": full.get("coordinate_frame"),
        "bbox": full.get("bbox"),
        "volume": full.get("volume"),
        "center_of_mass": full.get("center_of_mass"),
        "principal_axes": full.get("principal_axes"),
        "geometric_class": full.get("geometric_class"),
        "plane_interfaces": plane_clusters,
        "cylindrical_interfaces": cylinder_clusters,
        "hole_interfaces": hole_clusters,
        "plane_families": plane_families,
        "cylinder_families": cylinder_families,
        "compression": {
            "full_plane_count": len(planes),
            "proxy_plane_count": len(plane_clusters),
            "full_cylinder_count": len(cylinders),
            "proxy_cylinder_count": len(cylinder_clusters),
            "full_hole_count": len(holes),
            "proxy_hole_count": len(hole_clusters),
            "coarse_plane_family_count": len(plane_families),
            "coarse_cylinder_family_count": len(cylinder_families),
            "matching_feature_reduction_fraction": (
                1.0 - proxy_count / max(full_count, 1)
            ),
            "all_members_accounted_for": mapping_complete,
            "loss_policy": (
                "No source feature is deleted. Cluster ranges point back "
                "to the full-resolution feature model for refinement."
            ),
        },
        "settings": {
            "direction_component_tolerance": direction_tolerance,
            "plane_offset_tolerance_mm": linear_tolerance_mm,
            "axis_offset_tolerance_mm": linear_tolerance_mm,
            "radius_tolerance_mm": radius_tolerance_mm,
            "planar_area_family_ratio": planar_area_family_ratio,
            "cylinder_radius_family_relative_tolerance": (
                cylinder_radius_relative_tolerance
            ),
            "cylinder_radius_family_absolute_tolerance_mm": (
                cylinder_radius_absolute_tolerance_mm
            ),
        },
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
