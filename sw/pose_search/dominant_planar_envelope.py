"""Infer a review-only functional-body OBB from dominant planar consensus.

The full bounding box of an engineering part is often a poor insertion
envelope: handles, latches, cable glands, and connector noses can dominate its
extrema.  This module deliberately ignores that envelope while fitting.  It
looks instead for the rectangular-prism body supported by the largest amount
of mutually consistent planar-face area.

The inference is conservative.  A proposal requires at least two nearly
orthogonal normal families and either two opposing face pairs, or one opposing
pair plus an orthogonal end face (the ``end + two sides`` pattern).  Every
supporting face must agree with the inferred in-plane dimensions within the
configured tolerance.  A full AABB/OBB is consulted only *after* fitting to
report excluded protrusion risk; it can never enlarge the functional body.

This is a proposal primitive, not an assembly or acceptance decision.  All
successful results therefore carry ``proposal_only=True``,
``review_required=True``, and ``can_auto_accept=False``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_EPS = 1e-9


def _unit(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return None
    return vector / norm


def _point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    return point if np.all(np.isfinite(point)) else None


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Give an unoriented normal family a stable sign."""

    result = np.asarray(axis, dtype=float)
    pivot = int(np.argmax(np.abs(result)))
    return -result if result[pivot] < 0.0 else result


def _normalise_plane(
    raw: Mapping[str, Any], index: int
) -> dict[str, Any] | None:
    normal = _unit(raw.get("normal"))
    center = None
    for key in ("centroid", "center", "position"):
        center = _point(raw.get(key))
        if center is not None:
            break
    axes = raw.get("footprint_axes")
    dimensions = raw.get("footprint_dimensions")
    if (
        normal is None
        or center is None
        or not isinstance(axes, Sequence)
        or not isinstance(dimensions, Sequence)
        or len(axes) < 2
        or len(dimensions) < 2
    ):
        return None
    first = _unit(axes[0])
    second = _unit(axes[1])
    try:
        dims = np.asarray(dimensions[:2], dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        first is None
        or second is None
        or not np.all(np.isfinite(dims))
        or np.any(dims <= _EPS)
    ):
        return None

    # Preserve only audited planar footprints.  Area alone never determines a
    # rectangle, and a bad UV frame is not silently repaired into one.
    first = _unit(first - float(np.dot(first, normal)) * normal)
    second = _unit(second - float(np.dot(second, normal)) * normal)
    if first is None or second is None or abs(float(np.dot(first, second))) > 1e-3:
        return None
    try:
        area = float(raw.get("area", dims[0] * dims[1]))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(area) or area <= _EPS:
        return None
    raw_index = raw.get("index")
    try:
        feature_index = int(index if raw_index is None else raw_index)
    except (TypeError, ValueError):
        feature_index = int(index)
    return {
        "index": feature_index,
        "normal": normal,
        "center": center,
        "axes": np.asarray([first, second], dtype=float),
        "dimensions": dims,
        "area": area,
    }


def _projected_span(plane: Mapping[str, Any], axis: np.ndarray) -> float:
    return float(sum(
        abs(float(np.dot(plane["axes"][offset], axis)))
        * float(plane["dimensions"][offset])
        for offset in range(2)
    ))


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda row: row[0])
    threshold = 0.5 * sum(max(0.0, float(weight)) for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, float(weight))
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _normal_families(
    planes: Sequence[dict[str, Any]], parallel_cosine: float
) -> list[dict[str, Any]]:
    """Area-weighted greedy clustering of unoriented plane normals."""

    families: list[dict[str, Any]] = []
    for plane in sorted(planes, key=lambda row: -float(row["area"])):
        normal = _canonical_axis(plane["normal"])
        family = next((
            row for row in families
            if abs(float(np.dot(row["axis"], normal))) >= parallel_cosine
        ), None)
        if family is None:
            family = {"axis": normal, "planes": [], "area": 0.0}
            families.append(family)
        family["planes"].append(plane)
        family["area"] += float(plane["area"])

        # The principal eigenvector of the signed-invariant normal scatter is
        # stable even when opposite exterior faces have opposite normals.
        scatter = sum(
            float(item["area"]) * np.outer(item["normal"], item["normal"])
            for item in family["planes"]
        )
        values, vectors = np.linalg.eigh(scatter)
        family["axis"] = _canonical_axis(vectors[:, int(np.argmax(values))])
    return sorted(families, key=lambda row: -float(row["area"]))


def _candidate_bases(
    families: Sequence[Mapping[str, Any]],
    orthogonal_cosine: float,
    maximum_families: int = 10,
) -> list[np.ndarray]:
    bases: list[np.ndarray] = []
    for left, right in itertools.combinations(families[:maximum_families], 2):
        first = _canonical_axis(np.asarray(left["axis"], dtype=float))
        seed = np.asarray(right["axis"], dtype=float)
        if abs(float(np.dot(first, seed))) > orthogonal_cosine:
            continue
        second = _unit(seed - float(np.dot(seed, first)) * first)
        if second is None:
            continue
        second = _canonical_axis(second)
        third = _unit(np.cross(first, second))
        if third is None:
            continue
        basis = np.asarray([first, second, third], dtype=float)
        if float(np.linalg.det(basis)) < 0.0:
            basis[2] *= -1.0
        # Avoid repeated bases created by the three pair choices of a box.
        if any(
            all(
                max(abs(float(np.dot(axis, other_axis))) for other_axis in other)
                >= 1.0 - 1e-5
                for axis in basis
            )
            for other in bases
        ):
            continue
        bases.append(basis)
    return bases


def _assigned_planes(
    planes: Sequence[dict[str, Any]],
    basis: np.ndarray,
    parallel_cosine: float,
) -> list[list[dict[str, Any]]]:
    assigned: list[list[dict[str, Any]]] = [[], [], []]
    for plane in planes:
        similarities = np.abs(basis @ plane["normal"])
        axis_index = int(np.argmax(similarities))
        if float(similarities[axis_index]) >= parallel_cosine:
            assigned[axis_index].append(plane)
    return assigned


def _coordinate_layers(
    planes: Sequence[dict[str, Any]], axis: np.ndarray
) -> list[dict[str, Any]]:
    if not planes:
        return []
    coordinates = [float(np.dot(plane["center"], axis)) for plane in planes]
    coordinate_span = max(coordinates) - min(coordinates)
    footprint_scale = float(np.median([
        max(float(value) for value in plane["dimensions"])
        for plane in planes
    ]))
    tolerance = max(1e-6, 0.002 * max(coordinate_span, footprint_scale, 1.0))
    layers: list[dict[str, Any]] = []
    for coordinate, plane in sorted(zip(coordinates, planes), key=lambda row: row[0]):
        if not layers or coordinate - layers[-1]["last_coordinate"] > tolerance:
            layers.append({"coordinates": [], "planes": [], "last_coordinate": coordinate})
        layer = layers[-1]
        layer["coordinates"].append(coordinate)
        layer["planes"].append(plane)
        layer["last_coordinate"] = coordinate
    for layer in layers:
        weights = [float(plane["area"]) for plane in layer["planes"]]
        layer["coordinate"] = _weighted_median(layer["coordinates"], weights)
        layer["area"] = float(sum(weights))
    return layers


def _span_error(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    basis: np.ndarray,
    normal_axis: int,
) -> float:
    errors = []
    for axis_index in range(3):
        if axis_index == normal_axis:
            continue
        first = _projected_span(left, basis[axis_index])
        second = _projected_span(right, basis[axis_index])
        mean = 0.5 * (first + second)
        if mean <= _EPS:
            return math.inf
        errors.append(abs(first - second) / mean)
    return max(errors, default=math.inf)


def _opposing_pair_intervals(
    planes: Sequence[dict[str, Any]],
    basis: np.ndarray,
    axis_index: int,
    parallel_cosine: float,
    dimension_tolerance: float,
) -> list[dict[str, Any]]:
    axis = basis[axis_index]
    layers = _coordinate_layers(planes, axis)
    # Thousands of cosmetic coplanar fragments must not create an unbounded
    # quadratic search.  Main-body exterior layers rank highly by face area.
    layers = sorted(layers, key=lambda row: -float(row["area"]))[:32]
    layers.sort(key=lambda row: float(row["coordinate"]))
    intervals = []
    for lower_offset, lower in enumerate(layers):
        lower_planes = sorted((
            plane for plane in lower["planes"]
            if float(np.dot(plane["normal"], axis)) <= -parallel_cosine
        ), key=lambda row: -float(row["area"]))[:8]
        if not lower_planes:
            continue
        for upper in layers[lower_offset + 1:]:
            extent = float(upper["coordinate"] - lower["coordinate"])
            if extent <= _EPS:
                continue
            upper_planes = sorted((
                plane for plane in upper["planes"]
                if float(np.dot(plane["normal"], axis)) >= parallel_cosine
            ), key=lambda row: -float(row["area"]))[:8]
            if not upper_planes:
                continue
            matches = []
            for first in lower_planes:
                for second in upper_planes:
                    error = _span_error(first, second, basis, axis_index)
                    if error <= dimension_tolerance:
                        matches.append((
                            float(first["area"] + second["area"]),
                            -error,
                            first,
                            second,
                            error,
                        ))
            if not matches:
                continue
            _, _, first, second, error = max(matches, key=lambda row: row[:2])
            intervals.append({
                "low": float(lower["coordinate"]),
                "high": float(upper["coordinate"]),
                "kind": "opposing_pair",
                "support_plane_indices": [first["index"], second["index"]],
                "support_area": float(first["area"] + second["area"]),
                "dimension_error": float(error),
            })
    return _deduplicate_intervals(intervals, dimension_tolerance)[:8]


def _footprint_consensus_intervals(
    assigned: Sequence[Sequence[dict[str, Any]]],
    basis: np.ndarray,
    target_axis: int,
    dimension_tolerance: float,
) -> list[dict[str, Any]]:
    records = []
    for normal_axis, family_planes in enumerate(assigned):
        if normal_axis == target_axis:
            continue
        for plane in sorted(
            family_planes, key=lambda row: -float(row["area"])
        )[:128]:
            length = _projected_span(plane, basis[target_axis])
            if length <= _EPS:
                continue
            center = float(np.dot(plane["center"], basis[target_axis]))
            records.append({
                "low": center - 0.5 * length,
                "high": center + 0.5 * length,
                "center": center,
                "length": length,
                "area": float(plane["area"]),
                "plane_index": plane["index"],
                "normal_axis": normal_axis,
            })
    clusters: list[list[dict[str, Any]]] = []
    for record in sorted(records, key=lambda row: -float(row["area"])):
        cluster = next((
            rows for rows in clusters
            if _intervals_compatible(rows[0], record, dimension_tolerance)
        ), None)
        if cluster is None:
            clusters.append([record])
        else:
            cluster.append(record)
    intervals = []
    for cluster in clusters:
        indices = sorted({int(row["plane_index"]) for row in cluster})
        if len(indices) < 2:
            continue
        weights = [float(row["area"]) for row in cluster]
        low = _weighted_median([float(row["low"]) for row in cluster], weights)
        high = _weighted_median([float(row["high"]) for row in cluster], weights)
        if high - low <= _EPS:
            continue
        lengths = [float(row["length"]) for row in cluster]
        median_length = _weighted_median(lengths, weights)
        max_error = max(abs(length / median_length - 1.0) for length in lengths)
        intervals.append({
            "low": low,
            "high": high,
            "kind": "footprint_consensus",
            "support_plane_indices": indices,
            "support_normal_axes": sorted({
                int(row["normal_axis"]) for row in cluster
            }),
            "support_area": float(sum(weights)),
            "dimension_error": float(max_error),
        })
    return _deduplicate_intervals(intervals, dimension_tolerance)[:8]


def _intervals_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    tolerance: float,
) -> bool:
    left_length = float(left["high"] - left["low"])
    right_length = float(right["high"] - right["low"])
    mean_length = 0.5 * (left_length + right_length)
    if mean_length <= _EPS:
        return False
    center_delta = abs(
        0.5 * float(left["low"] + left["high"])
        - 0.5 * float(right["low"] + right["high"])
    )
    return (
        abs(left_length - right_length) / mean_length <= tolerance
        and center_delta / mean_length <= tolerance
    )


def _deduplicate_intervals(
    intervals: Sequence[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    ordered = sorted(
        intervals,
        key=lambda row: (
            -float(row["support_area"]),
            0 if row["kind"] == "opposing_pair" else 1,
            float(row["dimension_error"]),
        ),
    )
    for interval in ordered:
        duplicate = next((
            row for row in result
            if _intervals_compatible(row, interval, min(0.03, tolerance))
        ), None)
        if duplicate is None:
            result.append(interval)
        elif (
            interval["kind"] == "opposing_pair"
            and duplicate["kind"] != "opposing_pair"
        ):
            result[result.index(duplicate)] = interval
    return result


def _evaluate_candidate(
    planes: Sequence[dict[str, Any]],
    basis: np.ndarray,
    intervals: Sequence[Mapping[str, Any]],
    total_area: float,
    parallel_cosine: float,
    dimension_tolerance: float,
) -> dict[str, Any] | None:
    lows = np.asarray([float(row["low"]) for row in intervals])
    highs = np.asarray([float(row["high"]) for row in intervals])
    dimensions = highs - lows
    if np.any(dimensions <= _EPS):
        return None
    sides: list[dict[str, list[int]]] = [
        {"low": [], "high": []} for _ in range(3)
    ]
    explained = []
    errors = []
    for plane in planes:
        similarities = basis @ plane["normal"]
        normal_axis = int(np.argmax(np.abs(similarities)))
        signed_similarity = float(similarities[normal_axis])
        if abs(signed_similarity) < parallel_cosine:
            continue
        coordinate = float(np.dot(plane["center"], basis[normal_axis]))
        coordinate_tolerance = max(1e-6, 0.025 * float(dimensions[normal_axis]))
        side = None
        if (
            abs(coordinate - lows[normal_axis]) <= coordinate_tolerance
            and signed_similarity <= -parallel_cosine
        ):
            side = "low"
        elif (
            abs(coordinate - highs[normal_axis]) <= coordinate_tolerance
            and signed_similarity >= parallel_cosine
        ):
            side = "high"
        if side is None:
            continue
        plane_errors = []
        for axis_index in range(3):
            if axis_index == normal_axis:
                continue
            span = _projected_span(plane, basis[axis_index])
            plane_errors.append(abs(span / dimensions[axis_index] - 1.0))
        error = max(plane_errors, default=math.inf)
        if error > dimension_tolerance:
            continue
        explained.append(plane)
        errors.append(error)
        sides[normal_axis][side].append(int(plane["index"]))

    explained_indices = {int(plane["index"]) for plane in explained}
    normal_axes = {
        axis_index
        for axis_index, side_rows in enumerate(sides)
        if side_rows["low"] or side_rows["high"]
    }
    pair_axes = [
        axis_index for axis_index, side_rows in enumerate(sides)
        if side_rows["low"] and side_rows["high"]
    ]
    if len(explained_indices) < 3 or len(normal_axes) < 2 or not pair_axes:
        return None
    if len(pair_axes) >= 2:
        evidence_pattern = "multiple_opposing_pairs"
    else:
        pair_axis = pair_axes[0]
        orthogonal_end_axes = [
            axis_index for axis_index in normal_axes if axis_index != pair_axis
        ]
        if not orthogonal_end_axes:
            return None
        evidence_pattern = "end_face_plus_two_sides"

    explained_area = float(sum(plane["area"] for plane in explained))
    coverage = explained_area / max(total_area, _EPS)
    return {
        "basis": basis,
        "lows": lows,
        "highs": highs,
        "dimensions": dimensions,
        "center": sum(
            0.5 * float(lows[index] + highs[index]) * basis[index]
            for index in range(3)
        ),
        "intervals": intervals,
        "sides": sides,
        "pair_axes": pair_axes,
        "normal_axes": sorted(normal_axes),
        "evidence_pattern": evidence_pattern,
        "explained_planes": explained,
        "explained_indices": explained_indices,
        "explained_area": explained_area,
        "coverage": coverage,
        "mean_dimension_error": float(np.mean(errors)),
        "max_dimension_error": float(max(errors)),
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
        axes = np.asarray(raw.get("axes"), dtype=float).reshape(3, 3)
        dimensions = np.asarray(raw.get("dimensions"), dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    normalised_axes = [_unit(axis) for axis in axes]
    if (
        center is None
        or any(axis is None for axis in normalised_axes)
        or not np.all(np.isfinite(dimensions))
        or np.any(dimensions <= _EPS)
    ):
        return None
    axes = np.asarray(normalised_axes, dtype=float)
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4):
        return None
    return {
        "center": center,
        "axes": axes,
        "dimensions": dimensions,
        "source": source,
    }


def _full_envelope(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in ("full_obb", "obb", "full_bbox", "bbox"):
        envelope = _normalise_envelope(summary.get(key), key)
        if envelope is not None:
            return envelope
    return None


def _project_envelope(
    envelope: Mapping[str, Any], basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centers = basis @ np.asarray(envelope["center"], dtype=float)
    source_axes = np.asarray(envelope["axes"], dtype=float)
    source_dimensions = np.asarray(envelope["dimensions"], dtype=float)
    half_extents = 0.5 * np.asarray([
        sum(
            abs(float(np.dot(target_axis, source_axes[index])))
            * float(source_dimensions[index])
            for index in range(3)
        )
        for target_axis in basis
    ])
    return centers - half_extents, centers + half_extents


def _excluded_protrusion_area(
    planes: Sequence[dict[str, Any]],
    excluded_indices: set[int],
    basis: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> tuple[float, list[int]]:
    dimensions = highs - lows
    protruding = []
    area = 0.0
    for plane in planes:
        if int(plane["index"]) not in excluded_indices:
            continue
        is_outside = False
        for axis_index, axis in enumerate(basis):
            center = float(np.dot(plane["center"], axis))
            # A planar footprint has no thickness along its normal.
            half_span = 0.5 * _projected_span(plane, axis)
            tolerance = max(1e-6, 0.025 * float(dimensions[axis_index]))
            if (
                center - half_span < lows[axis_index] - tolerance
                or center + half_span > highs[axis_index] + tolerance
            ):
                is_outside = True
                break
        if is_outside:
            protruding.append(int(plane["index"]))
            area += float(plane["area"])
    return area, sorted(protruding)


def _abstain(
    reason: str,
    *,
    valid_plane_count: int,
    normal_family_count: int,
) -> dict[str, Any]:
    return {
        "status": "abstain",
        "reason": reason,
        "functional_body_obb": None,
        "confidence": 0.0,
        "derivation_evidence": {
            "audited_plane_count": int(valid_plane_count),
            "normal_family_count": int(normal_family_count),
        },
        "excluded_planar_area": None,
        "excluded_area_fraction": None,
        "excluded_protrusion_area": None,
        "excluded_protrusion_risk": None,
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
    }


def derive_dominant_planar_envelope(
    summary: Mapping[str, Any],
    *,
    parallel_cosine: float = 0.98,
    orthogonal_cosine: float = 0.12,
    dimension_tolerance: float = 0.15,
    minimum_explained_area_fraction: float = 0.30,
) -> dict[str, Any]:
    """Derive the dominant rectangular functional-body envelope.

    Parameters are dimensionless and no case name, file name, or absolute
    coordinate is inspected.  The returned OBB follows the repository
    convention: ``axes`` is a list of three world-space unit axes and
    ``dimensions[i]`` is the full extent along ``axes[i]``.
    """

    planes = [
        plane for index, raw in enumerate(summary.get("planes") or [])
        if isinstance(raw, Mapping)
        and (plane := _normalise_plane(raw, index)) is not None
    ]
    if len(planes) < 3:
        return _abstain(
            "At least three audited planar footprints are required.",
            valid_plane_count=len(planes),
            normal_family_count=0,
        )
    total_area = float(sum(plane["area"] for plane in planes))
    families = _normal_families(planes, parallel_cosine)
    bases = _candidate_bases(families, orthogonal_cosine)
    if not bases:
        return _abstain(
            "No two dominant plane-normal families are nearly orthogonal.",
            valid_plane_count=len(planes),
            normal_family_count=len(families),
        )

    candidates = []
    for basis in bases:
        assigned = _assigned_planes(planes, basis, parallel_cosine)
        per_axis: list[list[dict[str, Any]]] = []
        for axis_index in range(3):
            pair_intervals = _opposing_pair_intervals(
                assigned[axis_index],
                basis,
                axis_index,
                parallel_cosine,
                dimension_tolerance,
            )
            footprint_intervals = _footprint_consensus_intervals(
                assigned,
                basis,
                axis_index,
                dimension_tolerance,
            )
            intervals = _deduplicate_intervals(
                pair_intervals + footprint_intervals,
                dimension_tolerance,
            )[:8]
            per_axis.append(intervals)
        if any(not rows for rows in per_axis):
            continue
        for interval_choice in itertools.product(*per_axis):
            candidate = _evaluate_candidate(
                planes,
                basis,
                interval_choice,
                total_area,
                parallel_cosine,
                dimension_tolerance,
            )
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        return _abstain(
            (
                "No maximum-consensus prism has an opposing exterior pair, "
                "orthogonal support, and dimension agreement within tolerance."
            ),
            valid_plane_count=len(planes),
            normal_family_count=len(families),
        )

    # Explained planar area is the primary objective.  Pair count and fit
    # consistency only break ties; neither the full bbox nor its volume enters.
    best = max(candidates, key=lambda row: (
        float(row["explained_area"]),
        len(row["pair_axes"]),
        -float(row["mean_dimension_error"]),
        -float(np.prod(row["dimensions"])),
    ))
    if float(best["coverage"]) < minimum_explained_area_fraction:
        return _abstain(
            (
                "The best rectangular consensus explains too little audited "
                "planar area to identify a dominant functional body."
            ),
            valid_plane_count=len(planes),
            normal_family_count=len(families),
        )

    basis = np.asarray(best["basis"], dtype=float)
    if float(np.linalg.det(basis)) < 0.0:
        basis[2] *= -1.0
        # Bounds on a flipped axis must be reflected as well.
        low = -float(best["highs"][2])
        high = -float(best["lows"][2])
        best["lows"][2], best["highs"][2] = low, high
    center = sum(
        0.5 * float(best["lows"][index] + best["highs"][index]) * basis[index]
        for index in range(3)
    )
    dimensions = np.asarray(best["highs"] - best["lows"], dtype=float)
    explained_indices = set(best["explained_indices"])
    all_indices = {int(plane["index"]) for plane in planes}
    excluded_indices = all_indices - explained_indices
    excluded_area = float(total_area - best["explained_area"])
    protruding_area, protruding_indices = _excluded_protrusion_area(
        planes,
        excluded_indices,
        basis,
        np.asarray(best["lows"], dtype=float),
        np.asarray(best["highs"], dtype=float),
    )

    full = _full_envelope(summary)
    full_risk = False
    full_projection = None
    if full is not None:
        full_low, full_high = _project_envelope(full, basis)
        tolerance = 0.025 * dimensions
        excess_low = np.maximum(0.0, best["lows"] - full_low)
        excess_high = np.maximum(0.0, full_high - best["highs"])
        full_risk = bool(np.any(excess_low > tolerance) or np.any(excess_high > tolerance))
        full_projection = {
            "source": full["source"],
            "low": full_low.tolist(),
            "high": full_high.tolist(),
            "excess_low": excess_low.tolist(),
            "excess_high": excess_high.tolist(),
        }

    coverage = float(best["coverage"])
    consistency = max(
        0.0,
        1.0 - float(best["mean_dimension_error"]) / max(dimension_tolerance, _EPS),
    )
    pair_strength = 1.0 if len(best["pair_axes"]) >= 2 else 0.72
    family_strength = min(1.0, len(best["normal_axes"]) / 3.0)
    confidence = max(0.0, min(
        1.0,
        0.45 * min(1.0, coverage / 0.75)
        + 0.25 * consistency
        + 0.20 * pair_strength
        + 0.10 * family_strength,
    ))
    interval_evidence = [
        {
            "body_axis": int(axis_index),
            "source": row["kind"],
            "support_plane_indices": list(row["support_plane_indices"]),
            "support_area": float(row["support_area"]),
            "dimension_error": float(row["dimension_error"]),
        }
        for axis_index, row in enumerate(best["intervals"])
    ]
    body_obb = {
        "center": np.asarray(center, dtype=float).tolist(),
        "axes": basis.tolist(),
        "dimensions": dimensions.tolist(),
        "method": "dominant_planar_consensus",
    }
    return {
        "status": "proposed",
        "reason": (
            "A dominant rectangular body is supported by independent, "
            "dimension-consistent planar evidence; review is still required."
        ),
        "functional_body_obb": body_obb,
        "body_obb": body_obb,
        "confidence": round(float(confidence), 9),
        "derivation_evidence": {
            "evidence_pattern": best["evidence_pattern"],
            "audited_plane_count": len(planes),
            "normal_family_count": len(families),
            "supporting_normal_axes": list(best["normal_axes"]),
            "opposing_pair_axes": list(best["pair_axes"]),
            "supporting_plane_indices": sorted(explained_indices),
            "supporting_plane_count": len(explained_indices),
            "explained_planar_area": float(best["explained_area"]),
            "explained_area_fraction": coverage,
            "mean_dimension_error": float(best["mean_dimension_error"]),
            "max_dimension_error": float(best["max_dimension_error"]),
            "dimension_tolerance": float(dimension_tolerance),
            "interval_evidence": interval_evidence,
            "proper_basis_determinant": float(np.linalg.det(basis)),
            "full_envelope_used_in_derivation": False,
        },
        "excluded_plane_indices": sorted(excluded_indices),
        "excluded_planar_area": excluded_area,
        "excluded_area_fraction": excluded_area / max(total_area, _EPS),
        "excluded_protrusion_plane_indices": protruding_indices,
        "excluded_protrusion_area": float(protruding_area),
        "excluded_protrusion_risk": bool(protruding_area > _EPS or full_risk),
        "full_envelope_protrusion_risk": bool(full_risk),
        "full_envelope_projection": full_projection,
        "full_envelope_used_as_body_fit": False,
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
    }


def infer_functional_body_obb(
    summary: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Alias for :func:`derive_dominant_planar_envelope`."""

    return derive_dominant_planar_envelope(summary, **kwargs)
