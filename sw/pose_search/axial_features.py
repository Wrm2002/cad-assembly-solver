"""Generic B-Rep axial orientation evidence and rotation hypotheses.

This module deliberately knows nothing about case IDs, filenames, part roles,
or a fixed set of angles. It converts circular/cylindrical B-Rep features
around a supplied joint axis into a small, auditable set of rotations. Equal
spaced patterns describe *geometric* symmetry only; they are never treated as
functional evidence or an automatic pose decision.

Key-slot topology needs face-edge adjacency and concavity information that the
current lightweight feature schema does not retain. This module consequently
emits an explicit ``unknown`` result rather than pretending planar fragments
are a keyway.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from .transforms import unit
from .key_slot_features import extract_key_slot_evidence


def _wrap_degrees(value: float) -> float:
    """Map an axial angle to [-180, 180), preserving a canonical 180."""

    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if abs(wrapped + 180.0) < 1e-9 else wrapped


def _angular_distance_degrees(left: float, right: float) -> float:
    return abs(_wrap_degrees(float(left) - float(right)))


def _axis_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = unit(np.cross(axis, seed))
    return first, unit(np.cross(axis, first))


def _angle_about_axis(
    point: np.ndarray,
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
) -> tuple[float, float]:
    """Return polar angle and radius of *point* around the joint axis."""

    offset = point - axis_origin
    axial = float(np.dot(offset, axis_direction))
    radial = offset - axial * axis_direction
    radius = float(np.linalg.norm(radial))
    first, second = _axis_basis(axis_direction)
    angle = math.degrees(
        math.atan2(float(np.dot(radial, second)), float(np.dot(radial, first)))
    )
    return _wrap_degrees(angle), radius


@dataclass(frozen=True)
class AxialCircularFeature:
    """One circular/cylindrical feature measured in a joint-axis frame."""

    feature_id: str
    radius_mm: float
    angle_degrees: float
    radial_distance_mm: float
    source_kind: str


@dataclass(frozen=True)
class CircularPattern:
    """A same-radius angular feature set around one axis."""

    pattern_id: str
    radius_mm: float
    angles_degrees: tuple[float, ...]
    feature_ids: tuple[str, ...]
    periodicity: int | None
    geometry_symmetric: bool


@dataclass(frozen=True)
class RotationHypothesis:
    """A bounded axial rotation proposal with transparent provenance."""

    rotation_degrees: float
    score: float
    geometry_symmetry_only: bool
    evidence_kind: str
    fixed_pattern_id: str | None
    moving_pattern_id: str | None
    matching_fraction: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AxialPlanarWitness:
    """An off-axis planar directional cue, weaker than a confirmed key-slot."""

    feature_id: str
    angle_degrees: float
    radial_distance_mm: float
    area_mm2: float
    source_kind: str = "plane"


def _iter_circular_entities(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield normalized circular entities from either features or B-Rep graph."""

    if "nodes" in source:
        for node in source.get("nodes") or []:
            geometry = str(
                node.get("surface_type") or node.get("curve_type") or ""
            ).lower()
            if geometry not in {"cylinder", "circle"}:
                continue
            origin = node.get("axis_origin") or node.get("centroid")
            direction = node.get("axis_direction")
            radius = node.get("radius")
            if origin is None or direction is None or radius is None:
                continue
            yield {
                "feature_id": str(node.get("node_id")),
                "origin": origin,
                "axis": direction,
                "radius": radius,
                "source_kind": geometry,
            }
        return

    for index, feature in enumerate(source.get("cylinders") or []):
        if feature.get("origin") is None or feature.get("axis") is None:
            continue
        yield {
            "feature_id": f"cylinder:{index}",
            "origin": feature["origin"],
            "axis": feature["axis"],
            "radius": feature.get("radius"),
            "source_kind": "cylinder",
        }


def _iter_planar_entities(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if "nodes" in source:
        for node in source.get("nodes") or []:
            if str(node.get("surface_type") or "").lower() != "plane":
                continue
            centroid = node.get("centroid") or node.get("position")
            normal = node.get("normal")
            if centroid is None or normal is None:
                continue
            yield {
                "feature_id": str(node.get("node_id")),
                "position": centroid,
                "normal": normal,
                "area": node.get("area", 0.0),
            }
        return
    for index, feature in enumerate(source.get("planes") or []):
        if feature.get("position") is None or feature.get("normal") is None:
            continue
        yield {
            "feature_id": f"plane:{index}",
            "position": feature["position"],
            "normal": feature["normal"],
            "area": feature.get("area", 0.0),
        }


def extract_axial_circular_features(
    source: dict[str, Any],
    axis_origin: Iterable[float],
    axis_direction: Iterable[float],
    *,
    axis_parallel_cosine: float = 0.985,
    minimum_radial_distance_mm: float = 1.0,
) -> list[AxialCircularFeature]:
    """Extract off-axis circles/cylinders parallel to a supplied joint axis.

    A centered bore is intentionally excluded: it identifies the joint axis but
    provides no rotational orientation. The caller may pass the project's
    feature dictionary or the audited B-Rep graph emitted for JoinABLe.
    """

    origin = np.asarray(axis_origin, dtype=float)
    direction = unit(np.asarray(axis_direction, dtype=float))
    output: list[AxialCircularFeature] = []
    for entity in _iter_circular_entities(source):
        try:
            feature_origin = np.asarray(entity["origin"], dtype=float)
            feature_axis = unit(np.asarray(entity["axis"], dtype=float))
            feature_radius = float(entity["radius"])
        except (TypeError, ValueError):
            continue
        if (
            feature_origin.shape != (3,)
            or feature_radius <= 1e-9
            or abs(float(np.dot(feature_axis, direction))) < axis_parallel_cosine
        ):
            continue
        angle, radial_distance = _angle_about_axis(
            feature_origin, origin, direction
        )
        # A main coaxial bore/outer cylinder does not define an azimuth.
        if radial_distance < max(
            float(minimum_radial_distance_mm), feature_radius * 1.25
        ):
            continue
        output.append(AxialCircularFeature(
            feature_id=str(entity["feature_id"]),
            radius_mm=feature_radius,
            angle_degrees=angle,
            radial_distance_mm=radial_distance,
            source_kind=str(entity["source_kind"]),
        ))
    return sorted(
        output, key=lambda row: (row.radius_mm, row.angle_degrees, row.feature_id)
    )


def extract_axial_planar_witnesses(
    source: dict[str, Any],
    axis_origin: Iterable[float],
    axis_direction: Iterable[float],
    *,
    maximum_axis_normal_cosine: float = 0.25,
    minimum_radial_distance_mm: float = 1.0,
) -> list[AxialPlanarWitness]:
    """Return off-axis side planes as weak directional witnesses.

    This is deliberately not called a key-slot detector: the present feature
    contract lacks face-edge adjacency, concavity, and bounded face extents.
    It can nevertheless provide diverse rotation *proposals* for a later
    multi-part scorer, with provenance showing that the evidence is weak.
    """

    origin = np.asarray(axis_origin, dtype=float)
    direction = unit(np.asarray(axis_direction, dtype=float))
    output: list[AxialPlanarWitness] = []
    for entity in _iter_planar_entities(source):
        try:
            position = np.asarray(entity["position"], dtype=float)
            normal = unit(np.asarray(entity["normal"], dtype=float))
            area = float(entity.get("area") or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            position.shape != (3,)
            or abs(float(np.dot(normal, direction))) > maximum_axis_normal_cosine
        ):
            continue
        angle, radial_distance = _angle_about_axis(position, origin, direction)
        if radial_distance < minimum_radial_distance_mm:
            continue
        output.append(AxialPlanarWitness(
            feature_id=str(entity["feature_id"]),
            angle_degrees=angle,
            radial_distance_mm=radial_distance,
            area_mm2=max(0.0, area),
        ))
    return sorted(output, key=lambda row: (row.angle_degrees, row.feature_id))


def _deduplicate_angles(
    features: list[AxialCircularFeature], tolerance_degrees: float
) -> list[AxialCircularFeature]:
    """Keep one topology representative per coincident circular edge/face."""

    output: list[AxialCircularFeature] = []
    for feature in sorted(features, key=lambda row: row.angle_degrees):
        if output and _angular_distance_degrees(
            feature.angle_degrees, output[-1].angle_degrees
        ) <= tolerance_degrees:
            continue
        output.append(feature)
    if len(output) > 1 and _angular_distance_degrees(
        output[0].angle_degrees, output[-1].angle_degrees
    ) <= tolerance_degrees:
        output.pop()
    return output


def _regular_periodicity(
    angles: list[float], tolerance_degrees: float
) -> int | None:
    if len(angles) < 2:
        return None
    ordered = sorted((value + 360.0) % 360.0 for value in angles)
    gaps = [
        (ordered[(index + 1) % len(ordered)] - ordered[index]) % 360.0
        for index in range(len(ordered))
    ]
    expected = 360.0 / len(ordered)
    if all(abs(gap - expected) <= tolerance_degrees for gap in gaps):
        return len(ordered)
    return None


def build_circular_patterns(
    features: list[AxialCircularFeature],
    *,
    radius_tolerance_mm: float = 0.25,
    angle_tolerance_degrees: float = 2.0,
) -> list[CircularPattern]:
    """Group off-axis circles into same-radius angular patterns."""

    groups: list[list[AxialCircularFeature]] = []
    for feature in sorted(features, key=lambda row: row.radius_mm):
        if (
            not groups
            or abs(feature.radius_mm - groups[-1][0].radius_mm)
            > radius_tolerance_mm
        ):
            groups.append([feature])
        else:
            groups[-1].append(feature)
    patterns = []
    for index, group in enumerate(groups):
        unique = _deduplicate_angles(group, angle_tolerance_degrees)
        if len(unique) < 2:
            continue
        angles = tuple(float(row.angle_degrees) for row in unique)
        period = _regular_periodicity(list(angles), angle_tolerance_degrees)
        patterns.append(CircularPattern(
            pattern_id=f"circle_pattern:{index}",
            radius_mm=float(np.mean([row.radius_mm for row in unique])),
            angles_degrees=angles,
            feature_ids=tuple(row.feature_id for row in unique),
            periodicity=period,
            geometry_symmetric=period is not None,
        ))
    return patterns


def _pattern_fraction(
    fixed_angles: tuple[float, ...],
    moving_angles: tuple[float, ...],
    rotation_degrees: float,
    tolerance_degrees: float,
) -> float:
    if not fixed_angles or not moving_angles:
        return 0.0
    matched = 0
    for moving in moving_angles:
        rotated = _wrap_degrees(moving + rotation_degrees)
        if min(
            _angular_distance_degrees(rotated, fixed)
            for fixed in fixed_angles
        ) <= tolerance_degrees:
            matched += 1
    return matched / max(len(fixed_angles), len(moving_angles))


def generate_axial_rotation_hypotheses(
    fixed_source: dict[str, Any],
    moving_source: dict[str, Any],
    *,
    fixed_axis_origin: Iterable[float],
    fixed_axis_direction: Iterable[float],
    moving_axis_origin: Iterable[float],
    moving_axis_direction: Iterable[float],
    radius_tolerance_mm: float = 0.25,
    angle_tolerance_degrees: float = 2.0,
) -> dict[str, Any]:
    """Generate generic axial rotations from circular B-Rep feature patterns.

    The returned score only measures circular-feature correspondence. It is
    candidate evidence, *not* a semantic verdict. Uniform patterns generate
    explicit geometry-symmetry variants with no functional preference.
    """

    fixed_features = extract_axial_circular_features(
        fixed_source, fixed_axis_origin, fixed_axis_direction
    )
    moving_features = extract_axial_circular_features(
        moving_source, moving_axis_origin, moving_axis_direction
    )
    fixed_patterns = build_circular_patterns(
        fixed_features,
        radius_tolerance_mm=radius_tolerance_mm,
        angle_tolerance_degrees=angle_tolerance_degrees,
    )
    moving_patterns = build_circular_patterns(
        moving_features,
        radius_tolerance_mm=radius_tolerance_mm,
        angle_tolerance_degrees=angle_tolerance_degrees,
    )
    fixed_plane_witnesses = extract_axial_planar_witnesses(
        fixed_source, fixed_axis_origin, fixed_axis_direction
    )
    moving_plane_witnesses = extract_axial_planar_witnesses(
        moving_source, moving_axis_origin, moving_axis_direction
    )
    fixed_slot_evidence = extract_key_slot_evidence(
        fixed_source, fixed_axis_origin, fixed_axis_direction
    )
    moving_slot_evidence = extract_key_slot_evidence(
        moving_source, moving_axis_origin, moving_axis_direction
    )

    candidates: list[RotationHypothesis] = [RotationHypothesis(
        rotation_degrees=0.0,
        score=0.0,
        geometry_symmetry_only=False,
        evidence_kind="axis_alignment_anchor",
        fixed_pattern_id=None,
        moving_pattern_id=None,
        matching_fraction=0.0,
        reason="Always preserve the unrotated axis-alignment hypothesis.",
    )]
    # A periodic interface can admit several rotational alignments even if the
    # counterpart has no observable matching circles (for example, a flange
    # against a shaft). These are recall-oriented interface hypotheses only:
    # the pattern does not prove the *whole part* is functionally symmetric.
    for side, patterns in (("fixed", fixed_patterns), ("moving", moving_patterns)):
        for pattern in patterns:
            if not pattern.periodicity:
                continue
            for index in range(pattern.periodicity):
                candidates.append(RotationHypothesis(
                    rotation_degrees=360.0 * index / pattern.periodicity,
                    score=0.0,
                    geometry_symmetry_only=True,
                    evidence_kind="single_interface_periodicity",
                    fixed_pattern_id=(pattern.pattern_id if side == "fixed" else None),
                    moving_pattern_id=(pattern.pattern_id if side == "moving" else None),
                    matching_fraction=0.0,
                    reason=(
                        "A periodic circular interface permits this rotation "
                        "as a proposal; whole-part functional equivalence is unknown."
                    ),
                ))
    for fixed in fixed_patterns:
        for moving in moving_patterns:
            if abs(fixed.radius_mm - moving.radius_mm) > radius_tolerance_mm:
                continue
            offsets = {
                _wrap_degrees(left - right)
                for left in fixed.angles_degrees
                for right in moving.angles_degrees
            }
            uniform_pair = (
                fixed.geometry_symmetric
                and moving.geometry_symmetric
                and fixed.periodicity == moving.periodicity
            )
            for offset in offsets:
                fraction = _pattern_fraction(
                    fixed.angles_degrees,
                    moving.angles_degrees,
                    offset,
                    angle_tolerance_degrees,
                )
                if fraction <= 0.0:
                    continue
                candidates.append(RotationHypothesis(
                    rotation_degrees=offset,
                    score=float(fraction),
                    geometry_symmetry_only=uniform_pair,
                    evidence_kind=(
                        "uniform_circular_pattern_symmetry"
                        if uniform_pair else "circular_pattern_correspondence"
                    ),
                    fixed_pattern_id=fixed.pattern_id,
                    moving_pattern_id=moving.pattern_id,
                    matching_fraction=float(fraction),
                    reason=(
                        "Equally spaced circular patterns preserve this rotation; "
                        "functional equivalence remains unknown."
                        if uniform_pair else
                        "Off-axis circular features align at this axial rotation."
                    ),
                ))

    # When patterns cannot be paired (for example a shaft keyway versus an
    # external flange feature), side planes may still supply *diverse* axis
    # rotations. They are intentionally weak: a global feature scorer must
    # later decide whether any particular pairing has functional meaning.
    for fixed in fixed_plane_witnesses:
        for moving in moving_plane_witnesses:
            if fixed.area_mm2 > 0.0 and moving.area_mm2 > 0.0:
                area_ratio = min(fixed.area_mm2, moving.area_mm2) / max(
                    fixed.area_mm2, moving.area_mm2
                )
                if area_ratio < 0.05:
                    continue
            else:
                area_ratio = 0.0
            candidates.append(RotationHypothesis(
                rotation_degrees=_wrap_degrees(
                    fixed.angle_degrees - moving.angle_degrees
                ),
                score=0.05 * float(area_ratio),
                geometry_symmetry_only=False,
                evidence_kind="weak_planar_directional_witness",
                fixed_pattern_id=fixed.feature_id,
                moving_pattern_id=moving.feature_id,
                matching_fraction=0.0,
                reason=(
                    "Off-axis planar directions define a rotation proposal; "
                    "they are not a confirmed key-slot or functional mate."
                ),
            ))

    # A detected slot is substantially stronger than an arbitrary side plane,
    # but still proves only local geometry/topology.  It may seed the pose
    # frontier; it is never interpreted as a functional relation or final
    # semantic orientation.
    if (
        fixed_slot_evidence.get("status") == "detected"
        and moving_slot_evidence.get("status") == "detected"
    ):
        for fixed_slot in fixed_slot_evidence.get("candidates", []):
            for moving_slot in moving_slot_evidence.get("candidates", []):
                fixed_width = float(fixed_slot["wall_separation_mm"])
                moving_width = float(moving_slot["wall_separation_mm"])
                ratio = min(fixed_width, moving_width) / max(
                    fixed_width, moving_width, 1e-12
                )
                if ratio < 0.5:
                    continue
                candidates.append(RotationHypothesis(
                    rotation_degrees=_wrap_degrees(
                        float(fixed_slot["center_angle_degrees"])
                        - float(moving_slot["center_angle_degrees"])
                    ),
                    score=0.20 * ratio,
                    geometry_symmetry_only=False,
                    evidence_kind="topological_slot_directional_correspondence",
                    fixed_pattern_id=str(fixed_slot["candidate_id"]),
                    moving_pattern_id=str(moving_slot["candidate_id"]),
                    matching_fraction=ratio,
                    reason=(
                        "Closed concave slot topology aligns at this rotation; "
                        "functional assembly meaning remains unverified."
                    ),
                ))

    # Deduplicate by angle while retaining the strongest non-symmetry witness.
    best: dict[float, RotationHypothesis] = {}
    for row in candidates:
        key = round(_wrap_degrees(row.rotation_degrees), 6)
        previous = best.get(key)
        quality = (
            not row.geometry_symmetry_only,
            row.matching_fraction,
            row.evidence_kind != "axis_alignment_anchor",
        )
        previous_quality = None if previous is None else (
            not previous.geometry_symmetry_only,
            previous.matching_fraction,
            previous.evidence_kind != "axis_alignment_anchor",
        )
        if previous is None or quality > previous_quality:
            best[key] = row
    hypotheses = sorted(
        best.values(),
        key=lambda row: (
            -row.score,
            row.geometry_symmetry_only,
            abs(row.rotation_degrees),
            row.rotation_degrees,
        ),
    )
    serialized = [row.to_dict() for row in hypotheses]
    return {
        "schema_version": "axial_orientation_evidence.v1",
        "fixed_circular_features": [asdict(row) for row in fixed_features],
        "moving_circular_features": [asdict(row) for row in moving_features],
        "fixed_patterns": [asdict(row) for row in fixed_patterns],
        "moving_patterns": [asdict(row) for row in moving_patterns],
        "fixed_planar_witnesses": [asdict(row) for row in fixed_plane_witnesses],
        "moving_planar_witnesses": [asdict(row) for row in moving_plane_witnesses],
        "fixed_key_slot_evidence": fixed_slot_evidence,
        "moving_key_slot_evidence": moving_slot_evidence,
        "rotation_hypotheses": serialized,
        "geometric_directional_orientation_available": any(
            row["evidence_kind"] in {
                "circular_pattern_correspondence",
                "topological_slot_directional_correspondence",
            }
            and not row["geometry_symmetry_only"]
            for row in serialized
        ),
        # Geometry is intentionally not promoted to functional semantics.  A
        # future multi-part/global scorer may consume it, but must add CAD
        # metadata or independent assembly evidence before semantic acceptance.
        "functional_orientation_available": False,
        "key_slot_status": {
            "fixed": fixed_slot_evidence.get("status"),
            "moving": moving_slot_evidence.get("status"),
        },
    }
