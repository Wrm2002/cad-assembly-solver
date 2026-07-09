"""Deterministic STEP part indexing for pool-level matching."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from contracts import (
    BoundingBox,
    DetectionStatus,
    FeatureSummary,
    PartFeature,
)
from features import extract_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _occt_mass_properties(path: Path) -> dict[str, Any]:
    """Read volume/COM/inertia axes; return explicit errors, never guessed values."""
    try:
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            return {"available": False, "error": "STEPControl_Reader.ReadFile failed"}
        reader.TransferRoots()
        shape = reader.OneShape()
        bounds = Bnd_Box()
        bounds.SetGap(0.0)
        brepbndlib.Add(shape, bounds)
        x1, y1, z1, x2, y2, z2 = bounds.Get()
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        center = props.CentreOfMass()
        result = {
            "available": True,
            "volume": float(props.Mass()),
            "center_of_mass": [center.X(), center.Y(), center.Z()],
            "bbox": {
                "min": [x1, y1, z1],
                "max": [x2, y2, z2],
            },
            "principal_axes": [],
            "principal_axes_method": "unavailable",
        }
        try:
            principal = props.PrincipalProperties()
            axes = [
                principal.FirstAxisOfInertia(),
                principal.SecondAxisOfInertia(),
                principal.ThirdAxisOfInertia(),
            ]
            result["principal_axes"] = [
                [axis.X(), axis.Y(), axis.Z()] for axis in axes
            ]
            result["principal_axes_method"] = "occt_volume_inertia"
        except Exception as exc:
            result["principal_axes_error"] = str(exc)
        return result
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _feature_id(part_id: str, kind: str, index: int) -> str:
    return f"{part_id}:{kind}:{index}"


def _hole_candidates(part_id, cylinders, characteristic_length):
    """Return conservative, explicitly heuristic cylindrical-interface candidates."""
    candidates = []
    maximum_radius = max(characteristic_length * 0.25, 1e-9)
    for index, cylinder in enumerate(cylinders):
        radius = float(cylinder.get("radius", 0.0))
        polarity = cylinder.get("surface_polarity", "unknown")
        if polarity not in {"concave", "unknown", None}:
            # An outward-facing cylindrical wall is a pin, boss, shaft, or
            # outer race candidate—not a hole. Keep the fallback for text-only
            # STEP parsing where topology polarity is unavailable.
            continue
        if 0 < radius <= maximum_radius:
            candidates.append(
                FeatureSummary(
                    feature_id=_feature_id(part_id, "hole", index),
                    kind="cylindrical_interface_candidate",
                    parameters={
                        "radius": radius,
                        "origin": cylinder.get("origin"),
                        "axis": cylinder.get("axis"),
                        "area": cylinder.get("area"),
                        "surface_polarity": polarity,
                    },
                    detection_status=DetectionStatus.heuristic,
                    reason=(
                        "Concave or topology-unknown cylindrical face is small "
                        "relative to the part bbox; through-hole semantics are "
                        "not asserted."
                    ),
                )
            )
    return candidates


def _hole_patterns(part_id, holes, relative_tolerance=0.03):
    groups: list[list[FeatureSummary]] = []
    for hole in holes:
        radius = float(hole.parameters["radius"])
        for group in groups:
            reference = float(group[0].parameters["radius"])
            if abs(radius - reference) <= relative_tolerance * max(radius, reference, 1e-9):
                group.append(hole)
                break
        else:
            groups.append([hole])
    patterns = []
    for index, group in enumerate(groups):
        if len(group) < 2:
            continue
        patterns.append(
            FeatureSummary(
                feature_id=_feature_id(part_id, "hole_pattern", index),
                kind="equal_radius_cylindrical_pattern_candidate",
                parameters={
                    "count": len(group),
                    "mean_radius": sum(
                        float(item.parameters["radius"]) for item in group
                    ) / len(group),
                    "members": [item.feature_id for item in group],
                },
                detection_status=DetectionStatus.heuristic,
                reason="Grouped only by radius; bolt-circle semantics are not asserted.",
            )
        )
    return patterns


def index_part(
    path: str | Path,
    part_id: str | None = None,
    functional_semantics: dict[str, Any] | None = None,
) -> PartFeature:
    path = Path(path).resolve()
    part_id = part_id or path.stem
    raw = extract_features(str(path))
    mass = _occt_mass_properties(path)
    bbox = mass.get("bbox") or raw.get("bbox")
    if not bbox:
        raise ValueError(f"no bounding box extracted from {path}")
    minimum = [float(value) for value in bbox["min"]]
    maximum = [float(value) for value in bbox["max"]]
    size = [maximum[index] - minimum[index] for index in range(3)]
    diagonal = math.sqrt(sum(value * value for value in size))
    cylinders = raw.get("cylinders", [])
    planes = raw.get("planes", [])
    cylinder_models = [
        FeatureSummary(
            feature_id=_feature_id(part_id, "cylinder", index),
            kind="cylinder",
            parameters={
                key: cylinder.get(key)
                for key in (
                    "radius",
                    "origin",
                    "axis",
                    "area",
                    "surface_polarity",
                    "normal_radial_dot",
                )
                if cylinder.get(key) is not None
            },
        )
        for index, cylinder in enumerate(cylinders)
    ]
    plane_models = [
        FeatureSummary(
            feature_id=_feature_id(part_id, "plane", index),
            kind="plane",
            parameters={
                key: plane.get(key)
                for key in (
                    "position",
                    "normal",
                    "area",
                    "surface_orientation",
                )
                if plane.get(key) is not None
            },
        )
        for index, plane in enumerate(planes)
    ]
    holes = _hole_candidates(part_id, cylinders, diagonal)
    patterns = _hole_patterns(part_id, holes)
    if cylinders and planes:
        geometric_class = "mixed"
    elif cylinders:
        geometric_class = "cylindrical"
    elif planes:
        geometric_class = "planar"
    else:
        geometric_class = "unclassified"
    axes = mass.get("principal_axes") or [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    axes_method = mass.get("principal_axes_method", "bbox_coordinate_axes_fallback")
    if not mass.get("principal_axes"):
        axes_method = "bbox_coordinate_axes_fallback"
    return PartFeature(
        part_id=part_id,
        source_file=str(path),
        source_sha256=_sha256(path),
        bbox=BoundingBox(minimum=minimum, maximum=maximum, size=size),
        volume=mass.get("volume"),
        center_of_mass=mass.get("center_of_mass"),
        principal_axes=axes,
        principal_axes_method=axes_method,
        planar_faces=plane_models,
        cylindrical_faces=cylinder_models,
        holes=holes,
        hole_patterns=patterns,
        geometric_class=geometric_class,
        functional_semantics=functional_semantics or {},
        extraction={
            "extractor": "features.extract_features + OCCT mass properties",
            "occt_stats": raw.get("occt_stats", {}),
            "mass_properties": mass,
        },
    )
