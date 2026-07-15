"""Leakage-free, conservative precision validation for exact CAD poses.

The validator consumes only geometric solver audits.  It deliberately does
not inspect candidate ids, file/part names, roles, assembly families, or case
ids.  A collision-free OCCT result is necessary but is not sufficient: a
valid result also needs contact support and at least two independent geometric
evidence types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class PrecisionTolerances:
    """General-purpose conservative defaults, expressed in CAD millimetres."""

    maximum_axis_distance_mm: float = 0.20
    maximum_axis_angle_degrees: float = 1.0
    maximum_plane_gap_mm: float = 0.20
    maximum_hole_pattern_rms_mm: float = 0.20
    maximum_occt_common_volume_mm3: float = 0.01
    minimum_verified_insertion_depth_mm: float = 0.10
    maximum_contact_gap_normalized: float = 0.05
    minimum_independent_evidence_count: int = 2


_ALLOWED_EVIDENCE_TYPES = {
    "axis_alignment",
    "planar_contact",
    "surface_contact",
    "repeated_axis_pattern",
    "axial_support_contact",
    "prismatic_profile_fit",
    "verified_insertion_depth",
    "rigid_interface_fit",
}

_NUMERIC_PRECISION_FIELDS = {
    "axis_distance_mm",
    "axis_angle_degrees",
    "plane_gap_mm",
    "hole_pattern_rms_mm",
    "insertion_depth_mm",
    "axial_support_offset_mm",
    "clearance_mm",
}


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(mapping: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _finite_number(mapping.get(key))
        if value is not None:
            return value
    return None


def _normalise_evidence_types(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []
    return sorted({str(item) for item in values if str(item) in _ALLOWED_EVIDENCE_TYPES})


def filter_precision_evidence(
    manifold_type: str, provenance: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return a strict geometric whitelist from an untrusted provenance row.

    This helper is shared by the manifold solver when serialising factor
    audits.  Arbitrary provenance strings are never copied into its result.
    """

    source = provenance if isinstance(provenance, Mapping) else {}
    explicit = source.get("precision_evidence")
    explicit = explicit if isinstance(explicit, Mapping) else {}
    result: dict[str, Any] = {}
    evidence_types = set(_normalise_evidence_types(explicit.get("evidence_types")))

    for key in _NUMERIC_PRECISION_FIELDS:
        value = _finite_number(explicit.get(key))
        if value is not None:
            result[key] = value

    kind = str(manifold_type or "").strip().lower()
    multi_axis = bool(source.get("multi_interface_ransac"))
    multi_prismatic = bool(source.get("multi_interface_prismatic"))

    if multi_axis:
        pattern_rms = _first_number(
            source, ("hole_pattern_rms_mm", "pattern_rms_mm", "residual_mm")
        )
        if pattern_rms is not None:
            result["hole_pattern_rms_mm"] = abs(pattern_rms)
            evidence_types.add("repeated_axis_pattern")

        clearance = _first_number(
            source,
            (
                "plane_gap_mm",
                "contact_gap_mm",
                "axial_clearance_mm",
                "clearance_mm",
            ),
        )
        offset = _first_number(source, ("axial_support_offset_mm", "offset_mm"))
        # New producers should emit clearance_mm.  offset_mm remains a
        # conservative legacy fallback for existing RANSAC artifacts.
        gap = clearance if clearance is not None else offset
        if gap is not None:
            result["plane_gap_mm"] = abs(gap)
            result["axial_support_offset_mm"] = float(offset if offset is not None else gap)
            evidence_types.add("axial_support_contact")

    if multi_prismatic:
        evidence_types.add("prismatic_profile_fit")

    insertion_verified = bool(
        explicit.get("insertion_depth_verified")
        or source.get("insertion_depth_verified")
        or source.get("actual_insertion_depth_validated")
    )
    insertion_depth = _first_number(
        explicit,
        ("insertion_depth_mm", "verified_insertion_depth_mm"),
    )
    if insertion_depth is None:
        insertion_depth = _first_number(
            source, ("verified_insertion_depth_mm", "insertion_depth_mm")
        )
    if insertion_verified and insertion_depth is not None:
        result["insertion_depth_mm"] = insertion_depth
        result["insertion_depth_verified"] = True
        evidence_types.add("verified_insertion_depth")
    elif insertion_verified:
        # A boolean assertion without a measurement is not precision evidence.
        result["insertion_depth_verified"] = False

    if evidence_types:
        result["evidence_types"] = sorted(evidence_types)
    return result


def _coerce_tolerances(
    value: PrecisionTolerances | Mapping[str, Any] | None,
) -> PrecisionTolerances:
    if value is None:
        result = PrecisionTolerances()
    elif isinstance(value, PrecisionTolerances):
        result = value
    elif isinstance(value, Mapping):
        known = asdict(PrecisionTolerances())
        unknown = sorted(set(value) - set(known))
        if unknown:
            raise ValueError(f"unknown_precision_tolerances:{','.join(unknown)}")
        updates = {
            key: int(item) if key == "minimum_independent_evidence_count" else float(item)
            for key, item in value.items()
        }
        result = replace(PrecisionTolerances(), **updates)
    else:
        raise TypeError("tolerances_must_be_mapping_or_PrecisionTolerances")

    numeric = asdict(result)
    if any(
        not math.isfinite(float(item)) or float(item) < 0.0
        for key, item in numeric.items()
        if key != "minimum_independent_evidence_count"
    ):
        raise ValueError("precision_tolerances_must_be_finite_and_nonnegative")
    if int(result.minimum_independent_evidence_count) < 2:
        raise ValueError("minimum_independent_evidence_count_must_be_at_least_two")
    return result


def _maximum(values: Iterable[float | None]) -> float | None:
    finite = [abs(value) for value in values if value is not None and math.isfinite(value)]
    return max(finite) if finite else None


def _occt_common_volume(exact: Mapping[str, Any], exact_status: str) -> float | None:
    volumes: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {
                    "occt_common_volume_mm3",
                    "occt_common_volume",
                    "common_volume_mm3",
                    "intersection_volume_mm3",
                }:
                    number = _finite_number(item)
                    if number is not None:
                        volumes.append(abs(number))
                elif isinstance(item, (Mapping, list, tuple)):
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(exact)
    if volumes:
        return max(volumes)
    occt = exact.get("occt")
    if (
        exact_status == "valid"
        and isinstance(occt, Mapping)
        and str(occt.get("status", "")).lower() == "success"
        and not (occt.get("collisions") or [])
    ):
        return 0.0
    return None


def _contact_from_geometry_audit(
    hypothesis: Mapping[str, Any], maximum_gap: float
) -> tuple[bool | None, str]:
    audit = hypothesis.get("geometry_residual_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    selected = [
        row
        for row in (audit.get("pair_scores") or [])
        if isinstance(row, Mapping) and bool(row.get("selected_constraint_edge"))
    ]
    if not selected:
        return None, "not_available"
    gaps = [_finite_number(row.get("contact_gap_normalized")) for row in selected]
    supported = all(value is not None and value <= maximum_gap for value in gaps)
    return supported, "geometry_residual_audit"


def _resolve_contact_gate(
    hypothesis: Mapping[str, Any],
    contact_gate: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None,
    tolerances: PrecisionTolerances,
) -> tuple[bool | None, str, str | None]:
    if contact_gate is None:
        return (*_contact_from_geometry_audit(
            hypothesis, tolerances.maximum_contact_gap_normalized
        ), None)
    try:
        gate = contact_gate(hypothesis) if callable(contact_gate) else contact_gate
    except Exception as exc:  # A gate failure cannot become acceptance.
        return None, "contact_gate", f"contact_gate_error:{type(exc).__name__}"
    if not isinstance(gate, Mapping):
        return None, "contact_gate", "contact_gate_invalid_result"
    if "supported" in gate:
        return bool(gate.get("supported")), "contact_gate", None
    status = str(gate.get("precision_status", gate.get("status", "unknown"))).lower()
    if status == "valid":
        return True, "contact_gate", None
    if status in {"failed", "review", "unsupported"}:
        return False, "contact_gate", None
    return None, "contact_gate", None


def validate_precision_pose(
    hypothesis: Mapping[str, Any],
    *,
    contact_gate: Mapping[str, Any] | Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    tolerances: PrecisionTolerances | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tier one exact pose as ``valid``, ``review``, or ``failed``.

    ``failed`` is reserved for exact collision failure or a measured residual
    outside an injected general tolerance.  Missing or insufficient evidence
    goes to ``review``; it is never silently converted into acceptance.
    """

    if not isinstance(hypothesis, Mapping):
        raise TypeError("hypothesis_must_be_a_mapping")
    limits = _coerce_tolerances(tolerances)
    exact = hypothesis.get("exact_validation")
    exact = exact if isinstance(exact, Mapping) else {}
    exact_status = str(exact.get("status", "not_checked")).lower()
    factor_rows = [
        row for row in (hypothesis.get("factor_residuals") or [])
        if isinstance(row, Mapping)
    ]

    axis_distances: list[float | None] = []
    axis_angles: list[float | None] = []
    plane_gaps: list[float | None] = []
    pattern_rms_values: list[float | None] = []
    verified_depths: list[float] = []
    evidence_types: set[str] = set()
    has_prismatic = False
    insertion_depth_verified = False

    for row in factor_rows:
        kind = str(row.get("manifold_type", "")).strip().lower()
        raw_provenance = row.get("provenance")
        raw_provenance = raw_provenance if isinstance(raw_provenance, Mapping) else {}
        precision = row.get("precision_evidence")
        precision = precision if isinstance(precision, Mapping) else {}
        merged_precision = dict(filter_precision_evidence(kind, raw_provenance))
        merged_precision.update(filter_precision_evidence(
            kind, {"precision_evidence": precision}
        ))
        row_evidence_types = set(
            _normalise_evidence_types(merged_precision.get("evidence_types"))
        )
        evidence_types.update(row_evidence_types)

        multi_axis = bool(raw_provenance.get("multi_interface_ransac")) or (
            "repeated_axis_pattern" in row_evidence_types
        )
        is_axis = (
            "axis_coincidence" in kind
            or "coaxial" in kind
            or kind in {"axis", "cylindrical", "revolute"}
            or (multi_axis and "axis" in kind)
        )
        is_plane = "plane" in kind or "planar" in kind
        is_prismatic = "prismatic" in kind or bool(
            raw_provenance.get("multi_interface_prismatic")
        )
        has_prismatic = has_prismatic or is_prismatic

        translation = _finite_number(row.get("projected_translation_residual_mm"))
        rotation = _finite_number(row.get("projected_rotation_residual_degrees"))
        if is_axis:
            axis_distances.append(translation)
            axis_angles.append(rotation)
            if not multi_axis:
                evidence_types.add("axis_alignment")
        if is_plane:
            plane_gaps.append(translation)
            evidence_types.add("planar_contact")
        if kind in {"frame", "rigid", "rigid_frame"}:
            evidence_types.add("rigid_interface_fit")

        axis_distances.append(_finite_number(merged_precision.get("axis_distance_mm")))
        axis_angles.append(_finite_number(merged_precision.get("axis_angle_degrees")))
        plane_gaps.append(_finite_number(merged_precision.get("plane_gap_mm")))
        pattern_rms_values.append(
            _finite_number(merged_precision.get("hole_pattern_rms_mm"))
        )
        depth = _finite_number(merged_precision.get("insertion_depth_mm"))
        if bool(merged_precision.get("insertion_depth_verified")) and depth is not None:
            insertion_depth_verified = True
            verified_depths.append(depth)
            evidence_types.add("verified_insertion_depth")

    top_precision = hypothesis.get("precision_evidence")
    if isinstance(top_precision, Mapping):
        evidence_types.update(_normalise_evidence_types(top_precision.get("evidence_types")))
        axis_distances.append(_finite_number(top_precision.get("axis_distance_mm")))
        axis_angles.append(_finite_number(top_precision.get("axis_angle_degrees")))
        plane_gaps.append(_finite_number(top_precision.get("plane_gap_mm")))
        pattern_rms_values.append(_finite_number(top_precision.get("hole_pattern_rms_mm")))
        depth = _finite_number(top_precision.get("insertion_depth_mm"))
        if bool(top_precision.get("insertion_depth_verified")) and depth is not None:
            insertion_depth_verified = True
            verified_depths.append(depth)
            evidence_types.add("verified_insertion_depth")

    contact_supported, contact_source, contact_error = _resolve_contact_gate(
        hypothesis, contact_gate, limits
    )
    if contact_gate is None and contact_supported is None:
        # A RANSAC axial support event is itself geometric contact support.
        contact_supported = "axial_support_contact" in evidence_types
        contact_source = "precision_evidence" if contact_supported else "not_available"
    if contact_supported and contact_source != "precision_evidence":
        # Do not double-count a sampled planar gate as evidence independent of
        # the selected plane itself.  For non-planar constraints, measured
        # surface contact is an independent observation.
        only_planar = bool(factor_rows) and all(
            "plane" in str(row.get("manifold_type", "")).lower()
            or "planar" in str(row.get("manifold_type", "")).lower()
            for row in factor_rows
        )
        evidence_types.add("planar_contact" if only_planar else "surface_contact")

    axis_distance = _maximum(axis_distances)
    axis_angle = _maximum(axis_angles)
    plane_gap = _maximum(plane_gaps)
    pattern_rms = _maximum(pattern_rms_values)
    insertion_depth = min(verified_depths) if verified_depths else None
    occt_volume = _occt_common_volume(exact, exact_status)
    independent_types = sorted(evidence_types & _ALLOWED_EVIDENCE_TYPES)

    failed_reasons: list[str] = []
    review_reasons: list[str] = []
    if exact_status == "failed":
        failed_reasons.append("exact_validation_failed")
    elif exact_status != "valid":
        review_reasons.append(f"exact_validation_{exact_status}")

    measured_checks = (
        (axis_distance, limits.maximum_axis_distance_mm, "axis_distance_exceeds_tolerance"),
        (axis_angle, limits.maximum_axis_angle_degrees, "axis_angle_exceeds_tolerance"),
        (plane_gap, limits.maximum_plane_gap_mm, "plane_gap_exceeds_tolerance"),
        (pattern_rms, limits.maximum_hole_pattern_rms_mm, "hole_pattern_rms_exceeds_tolerance"),
        (occt_volume, limits.maximum_occt_common_volume_mm3, "occt_common_volume_exceeds_tolerance"),
    )
    for measured, maximum, reason in measured_checks:
        if measured is not None and measured > maximum:
            failed_reasons.append(reason)
    if (
        insertion_depth_verified
        and insertion_depth is not None
        and insertion_depth < limits.minimum_verified_insertion_depth_mm
    ):
        failed_reasons.append("verified_insertion_depth_below_tolerance")

    if exact_status == "valid" and contact_supported is not True:
        review_reasons.append(
            contact_error or "occt_valid_but_contact_support_missing"
        )
    if len(independent_types) < int(limits.minimum_independent_evidence_count):
        review_reasons.append("insufficient_independent_precision_evidence")
    if len(independent_types) == 1 and independent_types[0] in {
        "axis_alignment", "planar_contact"
    }:
        review_reasons.append("single_interface_constraint_requires_review")
    simple_single_interface = len(factor_rows) == 1 and (
        "axis_coincidence" in str(factor_rows[0].get("manifold_type", "")).lower()
        or "plane_coincidence" in str(factor_rows[0].get("manifold_type", "")).lower()
    )
    if simple_single_interface and evidence_types <= {
        "axis_alignment", "planar_contact", "surface_contact"
    }:
        review_reasons.append("single_interface_constraint_requires_review")
    if has_prismatic and not insertion_depth_verified:
        review_reasons.append("prismatic_insertion_depth_not_verified")
    if hypothesis.get("unresolved_manifold_dofs"):
        review_reasons.append("unresolved_manifold_dofs")
    optimizer = hypothesis.get("optimizer")
    if isinstance(optimizer, Mapping) and str(optimizer.get("status", "converged")) != "converged":
        review_reasons.append("optimizer_not_converged")

    failed_reasons = list(dict.fromkeys(failed_reasons))
    review_reasons = list(dict.fromkeys(review_reasons))
    if failed_reasons:
        status, reasons = "failed", failed_reasons
    elif review_reasons:
        status, reasons = "review", review_reasons
    else:
        status, reasons = "valid", ["exact_pose_has_multi_evidence_precision_support"]

    return {
        "schema_version": "precision_pose_validation.v1",
        "axis_distance_mm": axis_distance,
        "axis_angle_degrees": axis_angle,
        "plane_gap_mm": plane_gap,
        "hole_pattern_rms_mm": pattern_rms,
        "insertion_depth_mm": insertion_depth,
        "occt_common_volume_mm3": occt_volume,
        "independent_evidence_count": len(independent_types),
        "independent_evidence_types": independent_types,
        "contact_supported": contact_supported,
        "contact_support_source": contact_source,
        "precision_status": status,
        "status": status,
        "review_required": status == "review",
        "reason": reasons[0],
        "reasons": reasons,
        "tolerances": asdict(limits),
    }


# Descriptive alias for callers that prefer a noun phrase.
precision_pose_validation = validate_precision_pose


__all__ = [
    "PrecisionTolerances",
    "filter_precision_evidence",
    "precision_pose_validation",
    "validate_precision_pose",
]
