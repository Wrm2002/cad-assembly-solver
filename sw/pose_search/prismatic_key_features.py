"""Conservative detection of a separate rectangular key and slot compatibility.

The detector describes local B-Rep geometry only.  It neither assumes a part
name nor asserts that a prism is functionally a key.  A match is therefore a
review-level insertion *proposal*, not a relation label or pose acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from typing import Any, Iterable

import numpy as np

from .transforms import unit


@dataclass(frozen=True)
class PrismaticKeyFeature:
    feature_id: str
    center_point_mm: tuple[float, float, float]
    principal_directions: tuple[tuple[float, float, float], ...]
    dimensions_mm: tuple[float, float, float]
    long_axis: tuple[float, float, float]
    convex_edge_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vector(row: dict[str, Any], field: str) -> np.ndarray | None:
    try:
        value = np.asarray(row.get(field), dtype=float)
    except (TypeError, ValueError):
        return None
    if value.shape != (3,) or not bool(np.all(np.isfinite(value))):
        return None
    return value


def extract_prismatic_key_feature(source: dict[str, Any]) -> dict[str, Any]:
    """Find a finite, convex, three-axis planar prism in an audited graph."""

    topology = (source.get("metadata") or {}).get("edge_topology_features") or {}
    if not topology.get("available"):
        return {
            "schema_version": "prismatic_key_evidence.v1",
            "status": "unknown",
            "feature": None,
            "reason": "Audited local edge topology is unavailable.",
        }
    planes = []
    for row in source.get("nodes") or []:
        if row.get("entity_type") != "face" or str(row.get("surface_type")).lower() != "plane":
            continue
        center, normal = _vector(row, "centroid"), _vector(row, "normal")
        try:
            normal = None if normal is None else unit(normal)
            area = float(row.get("area") or 0.0)
        except (TypeError, ValueError):
            continue
        if center is not None and normal is not None and area > 1e-9:
            planes.append((str(row["node_id"]), center, normal))
    opposing = []
    for left, right in itertools.combinations(planes, 2):
        if float(np.dot(left[2], right[2])) > -0.96:
            continue
        separation = abs(float(np.dot(right[1] - left[1], left[2])))
        if separation > 1e-6:
            opposing.append((left, right, separation))
    selected = None
    for trio in itertools.combinations(opposing, 3):
        directions = [row[0][2] for row in trio]
        if all(abs(float(np.dot(a, b))) <= 0.2 for a, b in itertools.combinations(directions, 2)):
            selected = trio
            break
    if selected is None:
        return {
            "schema_version": "prismatic_key_evidence.v1",
            "status": "not_detected",
            "feature": None,
            "reason": "No three mutually perpendicular pairs of finite opposing planar faces.",
        }
    face_ids = {face[0] for pair in selected for face in pair[:2]}
    convex_edges = [
        row for row in source.get("nodes") or []
        if row.get("entity_type") == "edge"
        and row.get("topology_feature_status") == "success"
        and row.get("convexity") == "convex"
        and len(set(map(str, row.get("adjacent_face_ids") or [])).intersection(face_ids)) == 2
    ]
    if len(convex_edges) < 4:
        return {
            "schema_version": "prismatic_key_evidence.v1",
            "status": "not_detected",
            "feature": None,
            "reason": "Planar prism candidate lacks sufficient convex local edge support.",
        }
    directions = [unit(pair[0][2]) for pair in selected]
    dimensions = [float(pair[2]) for pair in selected]
    long_index = int(np.argmax(dimensions))
    center = np.mean([face[1] for pair in selected for face in pair[:2]], axis=0)
    feature = PrismaticKeyFeature(
        feature_id="convex_planar_prism:0",
        center_point_mm=tuple(float(value) for value in center),
        principal_directions=tuple(tuple(float(value) for value in direction) for direction in directions),
        dimensions_mm=tuple(dimensions),
        long_axis=tuple(float(value) for value in directions[long_index]),
        convex_edge_count=len(convex_edges),
    )
    return {
        "schema_version": "prismatic_key_evidence.v1",
        "status": "detected",
        "feature": feature.to_dict(),
        "reason": "Finite convex three-axis planar prism; functional role remains unknown.",
    }


def match_prismatic_key_to_slots(
    key_evidence: dict[str, Any],
    slot_evidence: dict[str, Any],
    slot_axis_direction: Iterable[float],
) -> dict[str, Any]:
    """Return topology-backed key/slot insertion proposals without acceptance."""

    if key_evidence.get("status") != "detected" or slot_evidence.get("status") != "detected":
        return {"status": "unknown", "candidates": [], "reason": "Key or slot topology is unavailable."}
    key = key_evidence["feature"]
    axis = unit(np.asarray(slot_axis_direction, dtype=float))
    long_axis = unit(np.asarray(key["long_axis"], dtype=float))
    if abs(float(np.dot(axis, long_axis))) < 0.9:
        return {"status": "not_detected", "candidates": [], "reason": "Prism long axis is incompatible with slot axis."}
    dimensions = sorted(float(value) for value in key["dimensions_mm"])
    rows = []
    for slot in slot_evidence.get("candidates") or []:
        width = float(slot["wall_separation_mm"])
        compatible = [value for value in dimensions[:-1] if value <= width * 1.10]
        if not compatible:
            continue
        fit_ratio = max(0.0, min(1.0, max(compatible) / max(width, 1e-9)))
        rows.append({
            "key_feature_id": key["feature_id"],
            "slot_candidate_id": slot["candidate_id"],
            "key_cross_section_mm": max(compatible),
            "slot_width_mm": width,
            "fit_ratio": fit_ratio,
            "evidence_count": 3,
            "reason": (
                "Convex prism long axis and cross-section are geometrically "
                "compatible with a closed concave slot; functional meaning and "
                "insertion depth remain unverified."
            ),
        })
    return {
        "status": "detected" if rows else "not_detected",
        "candidates": rows,
        "review_required": True,
    }
