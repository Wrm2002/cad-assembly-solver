"""Conservative planar-footprint pose proposals for thin CAD components.

This module is deliberately a *recall* aid, not an assembly classifier.  It
looks for a thin moving OBB whose two in-plane dimensions fit a bounded planar
face on the stationary part.  By default, a proposal is emitted only when the
stationary face belongs to a co-centred, parallel multi-plane layer.  This is a
second geometric evidence family; a single coincidental planar contact is not
enough.

An optional local cylindrical-layout check can add a third evidence family.
It is intentionally stricter than a pairwise-distance signature: radius,
surface polarity, installation layer and transformed point correspondence all
have to agree.  Anonymous part names, file names, case ids and source labels
are never inspected.

Every returned transform is proposal-only.  It must still pass exact pose,
collision, group-consistency and (when available) functional review gates.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import matrix_to_placement


SCHEMA_VERSION = "planar_footprint.v1"


def _unit(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    length = float(np.linalg.norm(vector))
    if length <= 1e-10:
        return None
    return vector / length


def _point(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        return None
    return result


def _proper_basis(u: Any, v: Any, normal: Any) -> np.ndarray | None:
    """Return a right-handed orthonormal basis with vectors as columns."""

    n = _unit(normal)
    u_vector = _unit(u)
    if n is None or u_vector is None:
        return None
    u_vector = u_vector - float(np.dot(u_vector, n)) * n
    u_vector = _unit(u_vector)
    if u_vector is None:
        return None
    v_vector = _unit(v)
    if v_vector is None or abs(float(np.dot(v_vector, n))) > 1e-4:
        v_vector = np.cross(n, u_vector)
    else:
        v_vector = v_vector - float(np.dot(v_vector, n)) * n
        v_vector = _unit(v_vector)
        if v_vector is None:
            return None
        if float(np.dot(np.cross(u_vector, v_vector), n)) < 0.0:
            v_vector = -v_vector
    # Recompute v so numerical drift cannot create a reflected pose.
    v_vector = _unit(np.cross(n, u_vector))
    if v_vector is None:
        return None
    return np.column_stack((u_vector, v_vector, n))


def _fallback_plane_basis(normal: np.ndarray) -> np.ndarray:
    seed = np.asarray([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, normal))) > 0.85:
        seed = np.asarray([0.0, 1.0, 0.0])
    u = _unit(seed - float(np.dot(seed, normal)) * normal)
    assert u is not None
    v = _unit(np.cross(normal, u))
    assert v is not None
    return np.column_stack((u, v, normal))


def _bbox_obb(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    bbox = summary.get("bbox")
    if not isinstance(bbox, Mapping):
        return None
    minimum = _point(bbox.get("min"))
    maximum = _point(bbox.get("max"))
    if minimum is None or maximum is None:
        return None
    dimensions = maximum - minimum
    if np.any(dimensions <= 1e-9):
        return None
    return {
        "center": ((minimum + maximum) * 0.5).tolist(),
        "axes": np.eye(3).tolist(),
        "dimensions": dimensions.tolist(),
        "method": "axis_aligned_bbox_fallback",
    }


def _normalise_obb(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = summary.get("obb") or _bbox_obb(summary)
    if not isinstance(raw, Mapping):
        return None
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
        or np.any(dimensions <= 1e-9)
    ):
        return None
    normalised_axes = [_unit(axis) for axis in axes]
    if any(axis is None for axis in normalised_axes):
        return None
    axes = np.asarray(normalised_axes, dtype=float)
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4):
        return None
    return {
        "center": center,
        "axes": axes,
        "dimensions": dimensions,
        "method": str(raw.get("method") or "provided_obb"),
    }


def _plane_dimensions(plane: Mapping[str, Any]) -> np.ndarray | None:
    for key in ("footprint_dimensions", "extent_uv", "dimensions"):
        raw = plane.get(key)
        if raw is None:
            continue
        try:
            values = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if (
            len(values) >= 2
            and np.all(np.isfinite(values[:2]))
            and np.all(values[:2] > 1e-9)
        ):
            return values[:2]
    # Area alone cannot recover aspect ratio and is therefore not a safe size
    # gate.  Deliberately abstain instead of assuming a square face.
    return None


def _normalise_plane(plane: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    normal = _unit(plane.get("normal"))
    center_value = plane.get("centroid")
    if center_value is None:
        center_value = plane.get("center")
    if center_value is None:
        center_value = plane.get("position")
    center = _point(center_value)
    dimensions = _plane_dimensions(plane)
    if normal is None or center is None or dimensions is None:
        return None

    axes = plane.get("footprint_axes")
    basis = None
    if isinstance(axes, (list, tuple)) and len(axes) >= 2:
        basis = _proper_basis(axes[0], axes[1], normal)
    if basis is None and plane.get("u_axis") is not None:
        v_axis = plane.get("v_axis")
        if v_axis is None:
            u_axis = _unit(plane.get("u_axis"))
            if u_axis is not None and normal is not None:
                v_axis = np.cross(normal, u_axis)
        basis = _proper_basis(
            plane.get("u_axis"),
            v_axis,
            normal,
        )
    if basis is None:
        basis = _fallback_plane_basis(normal)

    return {
        "index": int(index),
        "center": center,
        "normal": basis[:, 2],
        "u_axis": basis[:, 0],
        "v_axis": basis[:, 1],
        "dimensions": dimensions,
        "area": float(plane.get("area") or dimensions[0] * dimensions[1]),
        "source": plane,
    }


def _occt_shape_summary(shape: Any) -> dict[str, Any]:
    """Extract the minimum audited geometry needed from a TopoDS shape.

    Imports stay local so synthetic summaries and installations without OCCT
    can still use the module.  A planar footprint uses actual face vertices;
    face area alone is never converted into an invented aspect ratio.
    """

    from OCC.Core.Bnd import Bnd_OBB
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
    from OCC.Core.TopExp import TopExp_Explorer
    try:
        from OCC.Core.TopoDS import Face as topods_Face, Vertex as topods_Vertex
    except ImportError:  # pragma: no cover - older pythonocc binding
        from OCC.Core.TopoDS import topods_Face, topods_Vertex

    obb = Bnd_OBB()
    brepbndlib.AddOBB(shape, obb, False, False, False)
    if obb.IsVoid():
        raise ValueError("OCCT shape has no finite OBB")
    obb_center = obb.Center()
    obb_axes = (obb.XDirection(), obb.YDirection(), obb.ZDirection())
    summary: dict[str, Any] = {
        "obb": {
            "center": [obb_center.X(), obb_center.Y(), obb_center.Z()],
            "axes": [[axis.X(), axis.Y(), axis.Z()] for axis in obb_axes],
            "dimensions": [
                2.0 * float(obb.XHSize()),
                2.0 * float(obb.YHSize()),
                2.0 * float(obb.ZHSize()),
            ],
            "method": "occt_brepbndlib_addobb",
        },
        "planes": [],
        "cylinders": [],
    }

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods_Face(explorer.Current())
        adaptor = BRepAdaptor_Surface(face)
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties)
        centroid = properties.CentreOfMass()
        center = np.asarray([centroid.X(), centroid.Y(), centroid.Z()])

        if adaptor.GetType() == GeomAbs_Plane:
            plane = adaptor.Plane()
            normal_direction = plane.Axis().Direction()
            u_direction = plane.XAxis().Direction()
            normal = np.asarray([
                normal_direction.X(), normal_direction.Y(), normal_direction.Z()
            ])
            u_axis = np.asarray([
                u_direction.X(), u_direction.Y(), u_direction.Z()
            ])
            v_axis = np.cross(normal, u_axis)
            points = []
            vertex_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
            while vertex_explorer.More():
                point = BRep_Tool.Pnt(topods_Vertex(vertex_explorer.Current()))
                points.append(np.asarray([point.X(), point.Y(), point.Z()]))
                vertex_explorer.Next()
            if len(points) >= 2:
                projected_u = [float(np.dot(point - center, u_axis)) for point in points]
                projected_v = [float(np.dot(point - center, v_axis)) for point in points]
                dimensions = [
                    max(projected_u) - min(projected_u),
                    max(projected_v) - min(projected_v),
                ]
                if min(dimensions) > 1e-9:
                    summary["planes"].append({
                        "normal": normal.tolist(),
                        "centroid": center.tolist(),
                        "footprint_axes": [u_axis.tolist(), v_axis.tolist()],
                        "footprint_dimensions": dimensions,
                        "area": float(properties.Mass()),
                    })

        elif adaptor.GetType() == GeomAbs_Cylinder:
            cylinder = adaptor.Cylinder()
            direction = cylinder.Axis().Direction()
            origin = cylinder.Axis().Location()
            summary["cylinders"].append({
                "radius": float(cylinder.Radius()),
                "origin": [origin.X(), origin.Y(), origin.Z()],
                "centroid": center.tolist(),
                "axis": [direction.X(), direction.Y(), direction.Z()],
                # OCCT normal polarity is optional and deliberately not
                # fabricated here.  Thus raw cylinder geometry cannot become
                # layout evidence until a richer extractor supplies polarity.
                "surface_polarity": "unknown",
                "area": float(properties.Mass()),
            })

        explorer.Next()
    return summary


def _coerce_summary(value: Any) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        if value.get("shape") is not None:
            extracted = _occt_shape_summary(value["shape"])
            # Explicit audited fields (for example cylinder polarity) are
            # richer than the minimal shape extractor and therefore win.
            merged = dict(extracted)
            merged.update(value)
            return merged, "mapping_with_occt_shape"
        return dict(value), "feature_summary"

    if isinstance(value, (str, os.PathLike, Path)):
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        path = str(value)
        if reader.ReadFile(path) != IFSelect_RetDone:
            raise ValueError(f"Could not read STEP file: {path}")
        reader.TransferRoots()
        return _occt_shape_summary(reader.OneShape()), "step_path"

    if hasattr(value, "ShapeType"):
        return _occt_shape_summary(value), "occt_shape"
    raise TypeError("expected a feature summary, STEP path, or OCCT TopoDS shape")


def _relative_errors(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.maximum(np.abs(first), np.abs(second)), 1e-9)
    return np.abs(first - second) / denominator


def _phase_size_errors(
    moving_dimensions: np.ndarray,
    stationary_dimensions: np.ndarray,
    phase_degrees: int,
) -> np.ndarray:
    ordered = moving_dimensions
    if phase_degrees % 180 == 90:
        ordered = ordered[::-1]
    return _relative_errors(ordered, stationary_dimensions)


def _parallel_layer_companions(
    target: dict[str, Any],
    planes: list[dict[str, Any]],
    *,
    normal_alignment_threshold: float,
    size_tolerance: float,
    center_tolerance_ratio: float,
    maximum_layer_spacing_ratio: float,
) -> list[dict[str, Any]]:
    result = []
    normal = target["normal"]
    minimum_span = float(min(target["dimensions"]))
    center_tolerance = max(0.25, center_tolerance_ratio * minimum_span)
    maximum_spacing = max(0.5, maximum_layer_spacing_ratio * minimum_span)
    for other in planes:
        if other["index"] == target["index"]:
            continue
        normal_dot = abs(float(np.dot(normal, other["normal"])))
        if normal_dot < normal_alignment_threshold:
            continue
        dimension_errors = _relative_errors(
            np.sort(target["dimensions"]), np.sort(other["dimensions"])
        )
        if float(np.max(dimension_errors)) > size_tolerance:
            continue
        delta = other["center"] - target["center"]
        layer_spacing = abs(float(np.dot(delta, normal)))
        in_plane = delta - float(np.dot(delta, normal)) * normal
        if (
            layer_spacing <= 1e-5
            or layer_spacing > maximum_spacing
            or float(np.linalg.norm(in_plane)) > center_tolerance
        ):
            continue
        result.append({
            "plane_index": int(other["index"]),
            "layer_spacing": layer_spacing,
            "in_plane_center_error": float(np.linalg.norm(in_plane)),
            "normal_alignment": normal_dot,
            "size_error": float(np.max(dimension_errors)),
        })
    return sorted(result, key=lambda row: (
        row["layer_spacing"], row["in_plane_center_error"], row["plane_index"]
    ))


def _cylinder_point(cylinder: Mapping[str, Any]) -> np.ndarray | None:
    # Centroid (or an explicit layer point) is required.  An arbitrary point
    # on an infinite cylinder axis does not establish an installation layer.
    if cylinder.get("layer_point") is not None:
        return _point(cylinder.get("layer_point"))
    return _point(cylinder.get("centroid"))


def _strict_cylinder_layout_evidence(
    stationary: Mapping[str, Any],
    moving: Mapping[str, Any],
    transform: np.ndarray,
    plane_normal: np.ndarray,
    footprint_dimensions: np.ndarray,
    *,
    radius_tolerance: float,
    axis_alignment_threshold: float,
    position_tolerance_ratio: float,
    layer_tolerance_ratio: float,
) -> dict[str, Any] | None:
    stationary_rows = list(stationary.get("cylinders") or [])
    moving_rows = list(moving.get("cylinders") or [])
    if len(stationary_rows) < 3 or len(moving_rows) < 3:
        return None

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    position_tolerance = max(
        0.25, position_tolerance_ratio * float(np.linalg.norm(footprint_dimensions))
    )
    layer_tolerance = max(
        0.15, layer_tolerance_ratio * float(min(footprint_dimensions))
    )

    pairs = []
    for moving_index, moving_cylinder in enumerate(moving_rows):
        moving_axis = _unit(moving_cylinder.get("axis"))
        moving_point = _cylinder_point(moving_cylinder)
        moving_polarity = str(moving_cylinder.get("surface_polarity") or "unknown")
        try:
            moving_radius = float(moving_cylinder.get("radius"))
        except (TypeError, ValueError):
            continue
        if (
            moving_axis is None
            or moving_point is None
            or moving_polarity not in {"concave", "convex"}
            or abs(float(np.dot(rotation @ moving_axis, plane_normal)))
            < axis_alignment_threshold
        ):
            continue
        transformed_point = rotation @ moving_point + translation
        transformed_axis = rotation @ moving_axis

        for stationary_index, stationary_cylinder in enumerate(stationary_rows):
            stationary_axis = _unit(stationary_cylinder.get("axis"))
            stationary_point = _cylinder_point(stationary_cylinder)
            stationary_polarity = str(
                stationary_cylinder.get("surface_polarity") or "unknown"
            )
            try:
                stationary_radius = float(stationary_cylinder.get("radius"))
            except (TypeError, ValueError):
                continue
            if (
                stationary_axis is None
                or stationary_point is None
                or stationary_polarity != moving_polarity
                or abs(float(np.dot(stationary_axis, plane_normal)))
                < axis_alignment_threshold
                or abs(float(np.dot(stationary_axis, transformed_axis)))
                < axis_alignment_threshold
            ):
                continue
            relative_radius_error = abs(stationary_radius - moving_radius) / max(
                stationary_radius, moving_radius, 1e-9
            )
            if relative_radius_error > radius_tolerance:
                continue
            delta = stationary_point - transformed_point
            layer_error = abs(float(np.dot(delta, plane_normal)))
            in_plane = delta - float(np.dot(delta, plane_normal)) * plane_normal
            position_error = float(np.linalg.norm(in_plane))
            if layer_error <= layer_tolerance and position_error <= position_tolerance:
                pairs.append({
                    "moving_index": moving_index,
                    "stationary_index": stationary_index,
                    "position_error": position_error,
                    "layer_error": layer_error,
                    "radius_error": relative_radius_error,
                    "transformed_point": transformed_point,
                    "stationary_point": stationary_point,
                })

    # Greedy one-to-one correspondence is deterministic and sufficient for a
    # bounded evidence gate.  It is not an optimizer or a source-label match.
    pairs.sort(key=lambda row: (
        row["position_error"] + row["layer_error"],
        row["radius_error"],
        row["moving_index"],
        row["stationary_index"],
    ))
    selected = []
    used_moving: set[int] = set()
    used_stationary: set[int] = set()
    for row in pairs:
        if (
            row["moving_index"] in used_moving
            or row["stationary_index"] in used_stationary
        ):
            continue
        selected.append(row)
        used_moving.add(row["moving_index"])
        used_stationary.add(row["stationary_index"])

    if len(selected) < 3:
        return None

    # A repeated layout must contain at least two non-zero intervals.  This
    # avoids treating duplicate cylindrical faces on one axis as an array.
    interval_values = []
    for left in range(len(selected)):
        for right in range(left + 1, len(selected)):
            delta = (
                selected[left]["stationary_point"]
                - selected[right]["stationary_point"]
            )
            in_plane = delta - float(np.dot(delta, plane_normal)) * plane_normal
            interval = float(np.linalg.norm(in_plane))
            if interval > position_tolerance:
                interval_values.append(interval)
    distinct_intervals = []
    for value in sorted(interval_values):
        if not distinct_intervals or abs(value - distinct_intervals[-1]) > position_tolerance:
            distinct_intervals.append(value)
    if len(distinct_intervals) < 2:
        return None

    return {
        "type": "cylinder_layout_correspondence",
        "independent_family": "local_cylinder_layout",
        "correspondence_count": len(selected),
        "distinct_interval_count": len(distinct_intervals),
        "maximum_position_error": max(row["position_error"] for row in selected),
        "maximum_layer_error": max(row["layer_error"] for row in selected),
        "maximum_radius_error": max(row["radius_error"] for row in selected),
        "requires_radius": True,
        "requires_known_polarity": True,
        "requires_installation_layer": True,
        "requires_transformed_position_correspondence": True,
        "distance_only_signature_used": False,
    }


def recall_planar_footprint_proposals(
    stationary: Any,
    moving: Any,
    *,
    maximum: int = 32,
    size_tolerance: float = 0.18,
    max_thinness_ratio: float = 0.35,
    normal_alignment_threshold: float = 0.985,
    center_tolerance_ratio: float = 0.06,
    maximum_layer_spacing_ratio: float = 0.25,
    require_multi_plane: bool = True,
    enable_cylinder_layout_evidence: bool = True,
    cylinder_radius_tolerance: float = 0.05,
    cylinder_position_tolerance_ratio: float = 0.02,
    cylinder_layer_tolerance_ratio: float = 0.02,
) -> dict[str, Any]:
    """Recall bounded planar footprint/socket transforms.

    ``stationary`` and ``moving`` may be feature-summary dictionaries, STEP
    paths, or OCCT ``TopoDS_Shape`` objects.  Feature summaries should expose
    ``obb`` plus stationary planes with ``normal``, ``centroid``,
    ``footprint_axes`` and ``footprint_dimensions``.  ``bbox`` is an audited
    axis-aligned OBB fallback for the moving part.

    ``transform_matrix`` and ``placement`` map moving-local coordinates into
    stationary-local coordinates.  If the stationary part already has a
    world placement, the caller must left-compose that placement; this module
    intentionally does not inspect or mutate a group pose.

    The default gate requires two independent evidence families:

    1. thin-part footprint dimensions with a proper normal alignment; and
    2. a co-centred, parallel, similarly sized multi-plane stationary layer.

    Setting ``require_multi_plane=False`` is intended for recall audits only;
    emitted rows remain proposal-only and explicitly report one evidence
    family.  It never turns a single plane into an accepted assembly.
    """

    if maximum < 1:
        raise ValueError("maximum must be positive")
    if not 0.0 <= size_tolerance < 1.0:
        raise ValueError("size_tolerance must be in [0, 1)")
    if not 0.0 < max_thinness_ratio < 1.0:
        raise ValueError("max_thinness_ratio must be in (0, 1)")
    if not 0.0 < normal_alignment_threshold <= 1.0:
        raise ValueError("normal_alignment_threshold must be in (0, 1]")

    stationary_summary, stationary_source = _coerce_summary(stationary)
    moving_summary, moving_source = _coerce_summary(moving)
    stationary_obb = _normalise_obb(stationary_summary)
    moving_obb = _normalise_obb(moving_summary)
    raw_planes = list(stationary_summary.get("planes") or [])
    planes = [
        row
        for index, plane in enumerate(raw_planes)
        if isinstance(plane, Mapping)
        for row in [_normalise_plane(plane, index)]
        if row is not None
    ]

    audit: dict[str, Any] = {
        "stationary_source": stationary_source,
        "moving_source": moving_source,
        "raw_stationary_plane_count": len(raw_planes),
        "usable_stationary_plane_count": len(planes),
        "size_compatible_plane_count": 0,
        "multi_plane_supported_count": 0,
        "cylinder_layout_evidence_count": 0,
        "require_multi_plane": bool(require_multi_plane),
        "size_tolerance": float(size_tolerance),
        "normal_alignment_threshold": float(normal_alignment_threshold),
        "anonymous_semantic_fields_used": False,
        "distance_only_cylinder_signature_used": False,
        "stationary_carrier_obb_available": stationary_obb is not None,
        "carrier_perpendicular_wall_count": 0,
        "outward_edge_alignment_proposal_count": 0,
    }
    base = {
        "schema_version": SCHEMA_VERSION,
        "proposals": [],
        "audit": audit,
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
    }
    if moving_obb is None:
        return {
            **base,
            "status": "unavailable",
            "reason": "Moving part has no finite OBB or bbox.",
        }
    if not planes:
        return {
            **base,
            "status": "unavailable",
            "reason": "No stationary plane has an explicit 2D footprint.",
        }

    dimensions = moving_obb["dimensions"]
    order = np.argsort(dimensions)
    thin_axis = int(order[0])
    in_plane_axes = [int(order[1]), int(order[2])]
    thinness_ratio = float(dimensions[thin_axis] / dimensions[order[1]])
    audit.update({
        "moving_obb_method": moving_obb["method"],
        "moving_dimensions": dimensions.tolist(),
        "moving_thin_axis": thin_axis,
        "moving_thinness_ratio": thinness_ratio,
    })
    if thinness_ratio > max_thinness_ratio:
        return {
            **base,
            "status": "abstain",
            "reason": "Moving OBB is not sufficiently thin for planar footprint recall.",
        }

    moving_u = moving_obb["axes"][in_plane_axes[0]]
    moving_v = moving_obb["axes"][in_plane_axes[1]]
    moving_normal = moving_obb["axes"][thin_axis]
    moving_basis = _proper_basis(moving_u, moving_v, moving_normal)
    if moving_basis is None:
        return {
            **base,
            "status": "unavailable",
            "reason": "Moving OBB axes do not form a proper footprint basis.",
        }
    moving_footprint = dimensions[in_plane_axes]

    generated = []
    transform_keys: set[tuple[float, ...]] = set()
    for plane in planes:
        carrier_axis = None
        carrier_outward = None
        carrier_plane_alignment = None
        interface_orientation_class = "unknown"
        if stationary_obb is not None:
            carrier_thin_axis = int(np.argmin(stationary_obb["dimensions"]))
            carrier_axis = stationary_obb["axes"][carrier_thin_axis]
            carrier_plane_alignment = abs(float(np.dot(
                plane["normal"], carrier_axis
            )))
            if carrier_plane_alignment >= 0.90:
                interface_orientation_class = "carrier_parallel_surface"
            elif carrier_plane_alignment <= 0.20:
                interface_orientation_class = "carrier_perpendicular_wall"
                audit["carrier_perpendicular_wall_count"] += 1
                signed_distance = float(np.dot(
                    plane["center"] - stationary_obb["center"], carrier_axis
                ))
                carrier_outward = (
                    carrier_axis if signed_distance >= 0.0 else -carrier_axis
                )
        compatible_phases = []
        phase_errors: dict[int, np.ndarray] = {}
        for phase in (0, 90, 180, 270):
            errors = _phase_size_errors(
                moving_footprint, plane["dimensions"], phase
            )
            phase_errors[phase] = errors
            if float(np.max(errors)) <= size_tolerance:
                compatible_phases.append(phase)
        if not compatible_phases:
            continue
        audit["size_compatible_plane_count"] += 1

        companions = _parallel_layer_companions(
            plane,
            planes,
            normal_alignment_threshold=normal_alignment_threshold,
            size_tolerance=size_tolerance,
            center_tolerance_ratio=center_tolerance_ratio,
            maximum_layer_spacing_ratio=maximum_layer_spacing_ratio,
        )
        if companions:
            audit["multi_plane_supported_count"] += 1
        if require_multi_plane and not companions:
            continue

        anchor_indices = [plane["index"]]
        anchor_indices.extend(row["plane_index"] for row in companions)
        anchor_indices = sorted(set(anchor_indices))
        anchor_planes = [
            candidate for candidate in planes if candidate["index"] in anchor_indices
        ]
        equivalence_class = "layer:" + ",".join(str(index) for index in anchor_indices)

        for anchor_plane in anchor_planes:
            stationary_basis = np.column_stack((
                anchor_plane["u_axis"],
                anchor_plane["v_axis"],
                anchor_plane["normal"],
            ))
            for normal_sign in (-1, 1):
                signed_normal = normal_sign * stationary_basis[:, 2]
                signed_v = normal_sign * stationary_basis[:, 1]
                signed_basis = np.column_stack((
                    stationary_basis[:, 0], signed_v, signed_normal
                ))
                for phase in compatible_phases:
                    anchor_size_errors = _phase_size_errors(
                        moving_footprint, anchor_plane["dimensions"], phase
                    )
                    if float(np.max(anchor_size_errors)) > size_tolerance:
                        continue
                    radians = math.radians(float(phase))
                    phase_matrix = np.asarray([
                        [math.cos(radians), -math.sin(radians), 0.0],
                        [math.sin(radians), math.cos(radians), 0.0],
                        [0.0, 0.0, 1.0],
                    ])
                    target_basis = signed_basis @ phase_matrix
                    rotation = target_basis @ moving_basis.T
                    if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-5:
                        continue
                    # A thin plate against a wall perpendicular to the
                    # carrier's broad face is an insertion interface, not a
                    # centred planar seat.  When the moving plate is taller
                    # than the wall, centre alignment puts half of the excess
                    # through the carrier.  Recall a bounded outward/inward
                    # edge alignment along the carrier normal.  Both remain
                    # review-only; the outward branch is merely ordered first
                    # and exact collision decides physical feasibility.
                    alignment_variants = [("center", np.zeros(3), 0.0)]
                    if (
                        interface_orientation_class
                        == "carrier_perpendicular_wall"
                        and carrier_axis is not None
                        and carrier_outward is not None
                    ):
                        plane_axes = (
                            anchor_plane["u_axis"], anchor_plane["v_axis"]
                        )
                        carrier_in_plane_axis = int(np.argmax([
                            abs(float(np.dot(axis, carrier_axis)))
                            for axis in plane_axes
                        ]))
                        carrier_axis_alignment = abs(float(np.dot(
                            plane_axes[carrier_in_plane_axis], carrier_axis
                        )))
                        ordered_moving_dimensions = (
                            moving_footprint[::-1]
                            if phase % 180 == 90
                            else moving_footprint
                        )
                        excess = max(
                            0.0,
                            0.5 * float(
                                ordered_moving_dimensions[carrier_in_plane_axis]
                                - anchor_plane["dimensions"][carrier_in_plane_axis]
                            ),
                        )
                        if carrier_axis_alignment >= 0.90 and excess >= 0.10:
                            alignment_variants = [
                                (
                                    "outward_edge",
                                    excess * carrier_outward,
                                    excess,
                                ),
                                ("center", np.zeros(3), 0.0),
                                (
                                    "inward_edge",
                                    -excess * carrier_outward,
                                    -excess,
                                ),
                            ]
                    # Seating aligns one of the moving OBB's two thin support
                    # faces with the stationary plane.  Aligning OBB centres
                    # would straddle the board/socket plane and create an
                    # artificial half-thickness penetration.  Both sides stay
                    # in the recall frontier because surface orientation alone
                    # does not prove which side is accessible.
                    support_offset = 0.5 * float(dimensions[thin_axis])
                    for support_polarity in (-1, 1):
                        for (
                            edge_alignment_strategy,
                            in_plane_shift,
                            edge_alignment_offset,
                        ) in alignment_variants:
                            transform = np.eye(4)
                            transform[:3, :3] = rotation
                            target_center = (
                                anchor_plane["center"]
                                + support_polarity
                                * support_offset
                                * anchor_plane["normal"]
                                + in_plane_shift
                            )
                            transform[:3, 3] = (
                                target_center - rotation @ moving_obb["center"]
                            )
                            transform_key = tuple(np.round(transform, 8).reshape(-1))
                            if transform_key in transform_keys:
                                continue
                            transform_keys.add(transform_key)

                            evidence = [{
                            "type": "planar_footprint_size_and_normal",
                            "independent_family": "planar_footprint",
                            "maximum_relative_size_error": float(
                                np.max(anchor_size_errors)
                            ),
                            "relative_size_errors": anchor_size_errors.tolist(),
                            "normal_alignment": 1.0,
                            "proper_rotation": True,
                            "support_surface_aligned": True,
                                "support_offset_mm": support_offset,
                                "interface_orientation_class": (
                                    interface_orientation_class
                                ),
                                "in_plane_alignment_strategy": (
                                    edge_alignment_strategy
                                ),
                            }]
                            if companions:
                                evidence.append({
                                "type": "co_centered_parallel_plane_layers",
                                "independent_family": "multi_plane_layer",
                                "companion_plane_indices": [
                                    row["plane_index"] for row in companions
                                ],
                                "minimum_layer_spacing": min(
                                    row["layer_spacing"] for row in companions
                                ),
                                "maximum_in_plane_center_error": max(
                                    row["in_plane_center_error"] for row in companions
                                ),
                                })

                            if enable_cylinder_layout_evidence:
                                cylinder_evidence = _strict_cylinder_layout_evidence(
                                stationary_summary,
                                moving_summary,
                                transform,
                                anchor_plane["normal"],
                                moving_footprint,
                                radius_tolerance=cylinder_radius_tolerance,
                                axis_alignment_threshold=normal_alignment_threshold,
                                position_tolerance_ratio=cylinder_position_tolerance_ratio,
                                layer_tolerance_ratio=cylinder_layer_tolerance_ratio,
                                )
                                if cylinder_evidence is not None:
                                    evidence.append(cylinder_evidence)
                                    audit["cylinder_layout_evidence_count"] += 1

                            families = sorted({
                                row["independent_family"] for row in evidence
                            })
                            if edge_alignment_strategy == "outward_edge":
                                audit["outward_edge_alignment_proposal_count"] += 1
                            generated.append({
                            "stationary_plane_index": int(plane["index"]),
                            "anchor_plane_index": int(anchor_plane["index"]),
                            "equivalence_class_id": equivalence_class,
                            "equivalent_anchor_plane_indices": anchor_indices,
                            "equivalent_anchor_count": len(anchor_indices),
                            "phase_degrees": int(phase),
                            "normal_sign": int(normal_sign),
                            "support_polarity": int(support_polarity),
                            "support_offset_mm": support_offset,
                                "support_surface_aligned": True,
                                "interface_orientation_class": (
                                    interface_orientation_class
                                ),
                                "carrier_plane_normal_alignment": (
                                    carrier_plane_alignment
                                ),
                                "in_plane_alignment_strategy": (
                                    edge_alignment_strategy
                                ),
                                "in_plane_edge_alignment_offset_mm": float(
                                    edge_alignment_offset
                                ),
                            "moving_thin_axis": thin_axis,
                            "moving_footprint_axes": in_plane_axes,
                            "moving_footprint_dimensions": moving_footprint.tolist(),
                            "stationary_footprint_dimensions": plane["dimensions"].tolist(),
                            "anchor": anchor_plane["center"].tolist(),
                            "rotation_matrix": rotation.tolist(),
                            "transform_matrix": transform.tolist(),
                            "transform_frame": "stationary_local_from_moving_local",
                            "placement": matrix_to_placement(transform),
                            "evidence": evidence,
                            "evidence_families": families,
                            "evidence_count": len(evidence),
                            "independent_evidence_count": len(families),
                            "has_multi_evidence_support": len(families) >= 2,
                            "proposal_only": True,
                            "review_required": True,
                            "can_auto_accept": False,
                            "semantic_fields_used": [],
                                "distance_only_cylinder_signature_used": False,
                            })

    generated.sort(key=lambda row: (
        -int(row["independent_evidence_count"]),
        max(row["evidence"][0]["relative_size_errors"]),
        row["equivalence_class_id"],
        row["anchor_plane_index"],
        {
            "outward_edge": 0,
            "center": 1,
            "inward_edge": 2,
        }.get(str(row.get("in_plane_alignment_strategy")), 3),
        row["normal_sign"],
        row["support_polarity"],
        row["phase_degrees"],
    ))
    # Preserve repeated socket classes, both support sides *and* both physical
    # faces of a multi-plane layer under a small caller quota.  A plain prefix
    # (or buckets that omit ``anchor_plane_index``) silently spends the whole
    # budget on one layer face; case-independent insertion depth is then lost
    # before exact validation.  Greedy least-used strata keep the frontier
    # bounded while retaining these discrete geometric alternatives.
    remaining_rows = list(enumerate(generated))
    selected = []
    equivalence_use: dict[str, int] = {}
    support_use: dict[tuple[str, int], int] = {}
    anchor_use: dict[tuple[str, int], int] = {}
    normal_use: dict[tuple[str, int], int] = {}
    phase_use: dict[tuple[str, int], int] = {}
    while remaining_rows and len(selected) < maximum:
        def diversity_key(item: tuple[int, dict[str, Any]]) -> tuple[int, ...]:
            original_index, row = item
            equivalence = str(row["equivalence_class_id"])
            support = int(row["support_polarity"])
            anchor = int(row["anchor_plane_index"])
            normal = int(row["normal_sign"])
            phase = int(row["phase_degrees"])
            return (
                equivalence_use.get(equivalence, 0),
                support_use.get((equivalence, support), 0),
                anchor_use.get((equivalence, anchor), 0),
                normal_use.get((equivalence, normal), 0),
                phase_use.get((equivalence, phase), 0),
                original_index,
            )

        chosen = min(remaining_rows, key=diversity_key)
        remaining_rows.remove(chosen)
        row = chosen[1]
        selected.append(row)
        equivalence = str(row["equivalence_class_id"])
        support = int(row["support_polarity"])
        anchor = int(row["anchor_plane_index"])
        normal = int(row["normal_sign"])
        phase = int(row["phase_degrees"])
        equivalence_use[equivalence] = equivalence_use.get(equivalence, 0) + 1
        support_use[(equivalence, support)] = (
            support_use.get((equivalence, support), 0) + 1
        )
        anchor_use[(equivalence, anchor)] = (
            anchor_use.get((equivalence, anchor), 0) + 1
        )
        normal_use[(equivalence, normal)] = (
            normal_use.get((equivalence, normal), 0) + 1
        )
        phase_use[(equivalence, phase)] = (
            phase_use.get((equivalence, phase), 0) + 1
        )
    for index, row in enumerate(selected, start=1):
        row["proposal_id"] = f"planar_footprint_{index:03d}"
    proposals = selected
    audit["generated_before_bound"] = len(generated)
    audit["equivalence_class_count"] = len({
        str(row["equivalence_class_id"]) for row in generated
    })
    audit["equivalence_support_normal_stratum_count"] = len({
        (
            str(row["equivalence_class_id"]),
            int(row["support_polarity"]),
            int(row["normal_sign"]),
        )
        for row in generated
    })
    audit["equivalence_anchor_stratum_count"] = len({
        (
            str(row["equivalence_class_id"]),
            int(row["anchor_plane_index"]),
        )
        for row in generated
    })
    audit["selected_count"] = len(proposals)
    if not proposals:
        if audit["size_compatible_plane_count"] == 0:
            reason = "No stationary planar footprint passed the size gate."
        elif require_multi_plane:
            reason = "Size-compatible planes lacked co-centred parallel layer support."
        else:
            reason = "No proper bounded footprint transform was generated."
        return {**base, "status": "abstain", "reason": reason}
    return {
        **base,
        "status": "success",
        "reason": "Bounded proposal-only planar footprint frontier generated.",
        "proposals": proposals,
    }


__all__ = ["SCHEMA_VERSION", "recall_planar_footprint_proposals"]
