"""Recognize labelled assembly relations for STEP parts known to form one assembly.

This entry point deliberately bypasses mixed-pool membership, provenance,
semantic-review, and forced partitioning logic.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_assembly import build_assembly
from constraints import (
    CLEARANCE,
    COAXIAL,
    PLANAR_ALIGN,
    PLANAR_MATE,
    POCKET_MATE,
    match_features,
)
from contracts import KnownGroupAssemblyResult
from direct_assembly_graph import (
    build_pair_candidates,
    canonical_pair,
    select_direct_connections,
    stable_id,
)
from coordinate_solver import placements_to_manifest
from features import extract_features
from match_scoring import score_matches
from placement_validation import (
    bbox_collisions,
    constraint_residual,
    exact_shape_collisions,
    transform_point,
    transform_vector,
)
from small_assembly_solver import solve_small_assembly


RELATION_COLLECTION = {
    COAXIAL: "cylinders",
    CLEARANCE: "cylinders",
    PLANAR_MATE: "planes",
    PLANAR_ALIGN: "planes",
    POCKET_MATE: "pockets",
}
FEATURE_KIND = {
    COAXIAL: "cylinder",
    CLEARANCE: "cylinder",
    PLANAR_MATE: "plane",
    PLANAR_ALIGN: "plane",
    POCKET_MATE: "pocket",
}

def _bbox_diagonal_from_features(part_features: dict[str, Any]) -> float:
    bbox = part_features.get("bbox") or {}
    low = bbox.get("min")
    high = bbox.get("max")
    if not low or not high:
        return 0.0
    return math.sqrt(
        sum((float(high[index]) - float(low[index])) ** 2 for index in range(3))
    )


def _bbox_dimensions(part_features: dict[str, Any]) -> list[float]:
    bbox = part_features.get("bbox") or {}
    low = bbox.get("min")
    high = bbox.get("max")
    if not low or not high:
        return []
    return [
        abs(float(high[index]) - float(low[index])) for index in range(3)
    ]


def _nonzero_dims(values: list[float] | tuple[float, ...]) -> list[float]:
    return sorted(abs(float(value)) for value in values if abs(float(value)) > 1e-6)


def _size_similarity(values_a, values_b) -> float:
    dims_a = _nonzero_dims(values_a or [])
    dims_b = _nonzero_dims(values_b or [])
    comparable = min(len(dims_a), len(dims_b))
    if comparable == 0:
        return 0.0
    dims_a = dims_a[:comparable]
    dims_b = dims_b[:comparable]
    return sum(
        min(a, b) / max(a, b, 1e-9)
        for a, b in zip(dims_a, dims_b)
    ) / comparable


def _best_pocket_fit(
    connection: dict[str, Any],
    *,
    require_planar_support: bool,
) -> dict[str, Any]:
    """Pure-geometry gate for pocket/slot assembly evidence."""
    if require_planar_support and not (
        PLANAR_MATE in set(connection.get("relation_types") or [])
        or PLANAR_ALIGN in set(connection.get("relation_types") or [])
    ):
        return {
            "supported": False,
            "reason": "pocket rejected: no planar seating/alignment evidence",
        }
    best: dict[str, Any] | None = None
    for match in connection.get("matches") or []:
        if match.get("type") != POCKET_MATE:
            continue
        size_a = match.get("_size_a") or []
        size_b = match.get("_size_b") or []
        dims_a = _nonzero_dims(size_a)
        dims_b = _nonzero_dims(size_b)
        if len(dims_a) < 2 or len(dims_b) < 2:
            candidate = {
                "supported": False,
                "reason": "pocket rejected: degenerate pocket dimensions",
            }
        else:
            reason = match.get("reason") or {}
            similarity = float(
                reason.get("pocket_size_similarity", _size_similarity(size_a, size_b))
            )
            direction = abs(float(reason.get("direction_compatibility", 0.0)))
            wall = abs(float(reason.get("wall_normal_compatibility", 0.0)))
            orientation = max(direction, wall)
            insertion_depth = min(dims_a + dims_b)
            max_depth = max(dims_a + dims_b)
            depth_ratio = insertion_depth / max(max_depth, 1e-9)
            supported = (
                similarity >= 0.40
                and orientation >= 0.70
                and 0.02 <= depth_ratio <= 1.0
            )
            candidate = {
                "supported": supported,
                "size_similarity": similarity,
                "direction_compatibility": direction,
                "wall_normal_compatibility": wall,
                "orientation_support": orientation,
                "insertion_depth_proxy": insertion_depth,
                "depth_ratio": depth_ratio,
                "reason": (
                    "pocket accepted: fit/orientation/depth/planar gate"
                    if supported
                    else "pocket rejected: fit/orientation/depth gate failed"
                ),
            }
        if best is None or (
            candidate.get("supported", False),
            float(candidate.get("size_similarity", 0.0)),
            float(candidate.get("orientation_support", 0.0)),
        ) > (
            best.get("supported", False),
            float(best.get("size_similarity", 0.0)),
            float(best.get("orientation_support", 0.0)),
        ):
            best = candidate
    return best or {"supported": False, "reason": "no pocket candidate present"}


def _planar_insert_geometry_support(
    connection: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    """Weak pure-geometry pocket support when pocket detection misses a slot.

    Requirements:
    - direct pair is non-axial;
    - planar seating evidence exists;
    - one part is materially smaller by bbox diagonal;
    - the interface is localized: selected planar evidence is not a broad,
      equal-area flange-like face.
    """
    present = set(connection.get("relation_types") or [])
    if present & {COAXIAL, CLEARANCE}:
        return {
            "supported": False,
            "reason": "planar insert rejected: axial evidence dominates",
        }
    if not (present & {PLANAR_MATE, PLANAR_ALIGN}):
        return {
            "supported": False,
            "reason": "planar insert rejected: no planar evidence",
        }
    parts = list(connection.get("parts") or [])
    if len(parts) != 2:
        return {"supported": False, "reason": "planar insert rejected: invalid pair"}
    diagonals = [
        _bbox_diagonal_from_features(features.get(part, {})) for part in parts
    ]
    if min(diagonals) <= 0.0:
        return {
            "supported": False,
            "reason": "planar insert rejected: missing bbox dimensions",
        }
    size_ratio = min(diagonals) / max(diagonals)
    planar_matches = [
        match for match in connection.get("matches") or []
        if match.get("type") in {PLANAR_MATE, PLANAR_ALIGN}
    ]
    localized = False
    best_area_ratio = 1.0
    best_distance = None
    for match in planar_matches:
        reason = match.get("reason") or {}
        area_ratio = float(reason.get("area_ratio", 1.0))
        distance = abs(float(reason.get("distance", 999.0)))
        best_area_ratio = min(best_area_ratio, area_ratio)
        best_distance = distance if best_distance is None else min(best_distance, distance)
        localized = localized or (
            area_ratio <= 0.60 and distance <= 2.0
        )
    supported = size_ratio <= 0.70 and localized
    return {
        "supported": supported,
        "size_ratio": size_ratio,
        "best_area_ratio": best_area_ratio,
        "best_plane_distance": best_distance,
        "reason": (
            "weak pocket accepted: small-to-large localized planar insert geometry"
            if supported
            else "weak pocket rejected: size/localized planar gate failed"
        ),
    }


def _assembly_method_relation_types(
    connection: dict[str, Any],
    features: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Infer a stable "how assembled" label from selected pair evidence.

    Pose-closed constraints can under-report relations because one local
    constraint may be enough to place a part.  For the public assembly method we
    use all selected-pair evidence, with conservative ordering and suppression
    of common false pocket detections on coaxial fastening pairs.
    """
    available = list(connection.get("relation_types") or [])
    present = set(available)
    reasons = [f"selected_pair_evidence={sorted(present)}"]

    labels: list[str] = []
    if CLEARANCE in present:
        labels.append(CLEARANCE)
        reasons.append("clearance evidence indicates shaft/bore or insert fit")
        return labels, reasons

    axial = COAXIAL in present
    planar = PLANAR_MATE in present or PLANAR_ALIGN in present
    pocket = POCKET_MATE in present

    if axial:
        labels.append(COAXIAL)
        if PLANAR_MATE in present:
            labels.append(PLANAR_MATE)
        elif PLANAR_ALIGN in present:
            labels.append(PLANAR_ALIGN)
        if pocket:
            reasons.append(
                "pocket evidence suppressed because coaxial fastening evidence is stronger"
            )
        return labels, reasons

    if pocket:
        pocket_fit = _best_pocket_fit(
            connection, require_planar_support=True
        )
        reasons.append(f"pocket_geometry_gate={pocket_fit}")
        if pocket_fit.get("supported"):
            labels.append(POCKET_MATE)
            if PLANAR_MATE in present:
                labels.append(PLANAR_MATE)
            elif PLANAR_ALIGN in present:
                labels.append(PLANAR_ALIGN)
            return labels, reasons
        # If a noisy pocket candidate failed but the same selected pair has a
        # localized small-insert planar signature, retain pocket as weak
        # geometry.  No file-name or case-id information is used.
        planar_insert = _planar_insert_geometry_support(connection, features)
        reasons.append(f"planar_insert_geometry_gate={planar_insert}")
        if planar_insert.get("supported"):
            labels.append(POCKET_MATE)
            labels.append(PLANAR_MATE if PLANAR_MATE in present else PLANAR_ALIGN)
            return labels, reasons
        labels.append(PLANAR_MATE if PLANAR_MATE in present else PLANAR_ALIGN)
        return labels, reasons

    if planar:
        planar_insert = _planar_insert_geometry_support(connection, features)
        reasons.append(f"planar_insert_geometry_gate={planar_insert}")
        if planar_insert.get("supported"):
            labels.append(POCKET_MATE)
        labels.append(PLANAR_MATE if PLANAR_MATE in present else PLANAR_ALIGN)
        return labels, reasons

    if available:
        labels.append(available[0])
    return labels, reasons


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _joinable_index(path: Path | None, parts: list[str], pool_id: str) -> tuple[dict, dict]:
    if path is None or not path.is_file():
        return {}, {
            "status": "not_provided",
            "reason": "No cached JoinABLe report was supplied; analytic geometry remains active.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = set(parts)
    indexed = {}
    ignored = 0
    for row in payload.get("pairs", []):
        if row.get("pool_id") not in {None, pool_id}:
            continue
        pair = canonical_pair((row.get("part_a", ""), row.get("part_b", "")))
        if set(pair) <= valid and row.get("status") == "success" and row.get("candidates"):
            indexed[pair] = {
                "pair_id": row.get("pair_id"),
                "pair_features": row.get("pair_features") or {},
                "top_interface_candidates": (row.get("candidates") or [])[:10],
            }
        else:
            ignored += 1
    return indexed, {
        "status": "success",
        "report": str(path.resolve()),
        "supported_pair_count": len(indexed),
        "ignored_pair_count": ignored,
        "role": "learned_interface_localization_evidence",
    }


def _joinable_relation_family(candidate: dict[str, Any]) -> str:
    explicit = str(candidate.get("family_hint", "")).lower()
    if "planar" in explicit:
        return "planar"
    if "coaxial" in explicit or "cylind" in explicit:
        return "axial"
    entity_types = []
    for key in ("part_a_entity", "part_b_entity", "entity_a", "entity_b"):
        entity = candidate.get(key) or {}
        entity_types.append(
            str(entity.get("geometry_type") or entity.get("joinable_entity_type") or "").lower()
        )
    if entity_types and all("plane" in value for value in entity_types):
        return "planar"
    if any("cylind" in value or "circle" in value for value in entity_types):
        return "axial"
    return "unknown"


def _apply_joinable_support(
    matches: list[dict[str, Any]],
    joinable_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use learned interface family/rank as bounded support, never fabrication."""
    supported = []
    for match in matches:
        row = dict(match)
        learned = joinable_by_pair.get(canonical_pair(row["parts"]))
        if not learned:
            supported.append(row)
            continue
        candidates = learned.get("top_interface_candidates") or []
        desired = (
            "axial" if row["type"] in {COAXIAL, CLEARANCE}
            else "planar" if row["type"] in {PLANAR_MATE, PLANAR_ALIGN}
            else "unknown"
        )
        matching = [
            candidate for candidate in candidates
            if _joinable_relation_family(candidate) in {desired, "unknown"}
        ]
        if matching:
            best = min(matching, key=lambda candidate: int(candidate.get("rank", 999999)))
            rank = max(1, int(best.get("rank", 1)))
            probability = float(
                best.get("softmax_probability", best.get("probability", 0.0)) or 0.0
            )
            boost = min(0.08, 0.04 / rank + 0.04 * probability)
            row["score"] = round(min(1.0, float(row.get("score", 0.0)) + boost), 6)
            row["confidence"] = (
                "high" if row["score"] >= 0.75
                else "medium" if row["score"] >= 0.5 else "low"
            )
            row["joinable_support"] = {
                "pair_id": learned.get("pair_id"),
                "rank": rank,
                "probability": probability,
                "relation_family": _joinable_relation_family(best),
                "score_boost": round(boost, 6),
                "candidate": best,
            }
        supported.append(row)
    return supported


def _matrix_from_placement(placement: dict[str, Any]) -> list[list[float]]:
    origin = transform_point([0.0, 0.0, 0.0], placement)
    axes = []
    for point in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
        transformed = transform_point(point, placement)
        axes.append([transformed[i] - origin[i] for i in range(3)])
    return [
        [axes[column][row] for column in range(3)] + [float(origin[row])]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _rigid_inverse(matrix: list[list[float]]) -> list[list[float]]:
    rotation_t = [[matrix[column][row] for column in range(3)] for row in range(3)]
    translation = [matrix[row][3] for row in range(3)]
    inverse = [
        rotation_t[row]
        + [-sum(rotation_t[row][k] * translation[k] for k in range(3))]
        for row in range(3)
    ]
    inverse.append([0.0, 0.0, 0.0, 1.0])
    return inverse


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[
        sum(float(a[row][k]) * float(b[k][column]) for k in range(4))
        for column in range(4)
    ] for row in range(4)]


def _relative_transform(part_a: str, part_b: str, placements: dict) -> list[list[float]]:
    global_a = _matrix_from_placement(placements[part_a])
    global_b = _matrix_from_placement(placements[part_b])
    return _multiply(_rigid_inverse(global_b), global_a)


def _interface(match: dict[str, Any], side: str, features: dict) -> dict[str, Any]:
    relation = str(match["type"])
    part_position = 0 if side == "a" else 1
    index_key = "feat_a_idx" if side == "a" else "feat_b_idx"
    part = match["parts"][part_position]
    index = match.get(index_key)
    collection = RELATION_COLLECTION[relation]
    geometry = {}
    if index is not None:
        rows = features.get(part, {}).get(collection, [])
        if 0 <= int(index) < len(rows):
            raw = rows[int(index)]
            for key in (
                "radius", "axis", "origin", "normal", "position", "area",
                "surface_polarity", "size", "direction", "wall_normal",
            ):
                if key in raw:
                    geometry[key] = raw[key]
    return {
        "feature_kind": FEATURE_KIND[relation],
        "feature_index": int(index) if index is not None else None,
        "feature_id": f"{part}:{FEATURE_KIND[relation]}:{index}" if index is not None else None,
        "geometry": geometry,
    }


def _fingerprint(match: dict[str, Any]) -> tuple[Any, ...]:
    a, b = match["parts"]
    ia, ib = match.get("feat_a_idx"), match.get("feat_b_idx")
    if str(a) <= str(b):
        return match["type"], str(a), str(b), ia, ib
    return match["type"], str(b), str(a), ib, ia


def _selected_evidence_fingerprints(solution: dict[str, Any]) -> set[tuple[Any, ...]]:
    return {
        _fingerprint(match)
        for mate in solution.get("selected_mates", [])
        for match in mate.get("evidence", [])
    }


def _best_constraints_per_type(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)
    return [
        max(group, key=lambda item: float(item.get("score", 0.0)))
        for _, group in sorted(grouped.items())
    ]


def _residual_record(match: dict[str, Any], features: dict, placements: dict) -> dict:
    return constraint_residual(match, features, placements)


def _constraint_satisfied(match: dict[str, Any], record: dict[str, Any]) -> bool:
    if not record.get("valid"):
        return False
    relation = match["type"]
    if relation == COAXIAL:
        return (
            float(record.get("axis_angle_deg", 999.0)) <= 2.0
            and float(record.get("radial_distance", 999.0)) <= 1.0
        )
    if relation == CLEARANCE:
        axial_overlap_ratio = record.get("axial_overlap_ratio")
        return (
            float(record.get("axis_angle_deg", 999.0)) <= 2.0
            and float(record.get("radial_distance", 999.0)) <= 1.0
            and axial_overlap_ratio is not None
            and float(axial_overlap_ratio) >= 0.15
        )
    if relation in {PLANAR_MATE, PLANAR_ALIGN}:
        return (
            float(record.get("normal_angle_deg", 999.0)) <= 5.0
            and float(record.get("plane_distance", 999.0)) <= 5.0
        )
    if relation == POCKET_MATE:
        return float(record.get("pocket_center_distance", 999.0)) <= 2.0
    return False


def _pose_closure(candidate: dict, graph: dict, features: dict) -> dict[str, Any]:
    placements = candidate["placements"]
    connection_rows = []
    fully_closed = True
    for connection in graph["selected"]:
        by_type: dict[str, bool] = defaultdict(bool)
        for match in connection["matches"]:
            record = _residual_record(match, features, placements)
            by_type[match["type"]] = by_type[match["type"]] or _constraint_satisfied(match, record)
        available = set(connection["relation_types"])
        axial_available = bool(available & {COAXIAL, CLEARANCE})
        planar_available = bool(available & {PLANAR_MATE, PLANAR_ALIGN})
        pocket_available = POCKET_MATE in available
        axial_satisfied = any(by_type[kind] for kind in {COAXIAL, CLEARANCE})
        planar_satisfied = any(by_type[kind] for kind in {PLANAR_MATE, PLANAR_ALIGN})
        pocket_satisfied = bool(by_type[POCKET_MATE])
        any_satisfied = any(by_type.values())
        # When pocket evidence is part of the selected relation, planar contact
        # alone is not enough; otherwise a fan/block can sit near a face and be
        # incorrectly accepted as inserted.
        if pocket_available:
            closed = pocket_satisfied and (not planar_available or planar_satisfied)
        # When both axial and seating evidence exist, both must close.  This
        # prevents a coaxial-but-separated flange pose from being called valid.
        elif axial_available and planar_available:
            closed = axial_satisfied and planar_satisfied
        else:
            closed = any_satisfied
        fully_closed = fully_closed and closed
        connection_rows.append({
            "connection_id": connection["connection_id"],
            "parts": connection["parts"],
            "closed": closed,
            "satisfied_relation_types": sorted(
                kind for kind, satisfied in by_type.items() if satisfied
            ),
            "available_relation_types": sorted(available),
        })
    closed_count = sum(row["closed"] for row in connection_rows)
    return {
        "fully_closed": fully_closed,
        "closed_connection_count": closed_count,
        "connection_count": len(connection_rows),
        "closure_ratio": closed_count / max(1, len(connection_rows)),
        "connections": connection_rows,
    }


def _pose_contact_support(candidate: dict, graph: dict, features: dict) -> dict[str, Any]:
    """Cheap contact/fit objective inspired by JoinABLe overlap/contact search.

    This is not a neural contact-area estimator.  It turns already available
    residuals into a bounded pose-quality signal: closed constraints add contact
    support, near-miss constraints add smaller support, and weak single-contact
    poses remain distinguishable from multi-evidence fits.
    """
    placements = candidate["placements"]
    support = 0.0
    max_support = 0.0
    per_connection = []
    for connection in graph.get("selected") or []:
        best_by_type: dict[str, tuple[float, dict[str, Any]]] = {}
        for match in connection.get("matches") or []:
            record = _residual_record(match, features, placements)
            if not record.get("valid"):
                continue
            relation = match.get("type")
            score = 0.0
            if relation in {COAXIAL, CLEARANCE}:
                angle = abs(float(record.get("axis_angle_deg", 999.0)))
                radial = abs(float(record.get("radial_distance", 999.0)))
                axial_overlap_ratio = float(record.get("axial_overlap_ratio") or 0.0)
                score = (
                    0.35 * max(0.0, 1.0 - angle / 5.0)
                    + 0.35 * max(0.0, 1.0 - radial / 3.0)
                    + 0.30 * min(1.0, axial_overlap_ratio / 0.50)
                )
            elif relation in {PLANAR_MATE, PLANAR_ALIGN}:
                angle = abs(float(record.get("normal_angle_deg", 999.0)))
                distance = abs(float(record.get("plane_distance", 999.0)))
                score = (
                    0.55 * max(0.0, 1.0 - angle / 10.0)
                    + 0.45 * max(0.0, 1.0 - distance / 10.0)
                )
            elif relation == POCKET_MATE:
                distance = abs(float(record.get("pocket_center_distance", 999.0)))
                score = max(0.0, 1.0 - distance / 10.0)
            current = best_by_type.get(str(relation))
            if current is None or score > current[0]:
                best_by_type[str(relation)] = (score, record)
        available = set(connection.get("relation_types") or [])
        expected_types = []
        if POCKET_MATE in available:
            expected_types.append(POCKET_MATE)
            if available & {PLANAR_MATE, PLANAR_ALIGN}:
                expected_types.append(PLANAR_MATE if PLANAR_MATE in available else PLANAR_ALIGN)
        elif available & {COAXIAL, CLEARANCE}:
            expected_types.append(CLEARANCE if CLEARANCE in available else COAXIAL)
            if available & {PLANAR_MATE, PLANAR_ALIGN}:
                expected_types.append(PLANAR_MATE if PLANAR_MATE in available else PLANAR_ALIGN)
        elif available & {PLANAR_MATE, PLANAR_ALIGN}:
            expected_types.append(PLANAR_MATE if PLANAR_MATE in available else PLANAR_ALIGN)
        else:
            expected_types.extend(sorted(available))
        if not expected_types:
            expected_types = sorted(best_by_type)
        row_support = 0.0
        row_max = max(1, len(expected_types))
        for relation in expected_types:
            row_support += best_by_type.get(relation, (0.0, {}))[0]
        support += row_support
        max_support += row_max
        per_connection.append({
            "connection_id": connection.get("connection_id"),
            "parts": connection.get("parts"),
            "expected_relation_types": expected_types,
            "support_score": row_support / row_max,
            "best_relation_scores": {
                relation: value[0] for relation, value in best_by_type.items()
            },
        })
    normalized = support / max(max_support, 1.0)
    return {
        "contact_support_score": normalized,
        "raw_contact_support": support,
        "max_contact_support": max_support,
        "connections": per_connection,
    }


def _pose_overlap_cost(candidate: dict, graph: dict, features: dict) -> dict[str, Any]:
    """Broad-phase overlap cost used before expensive OCCT exact checks."""
    placements = candidate["placements"]
    selected_pairs = {
        canonical_pair(row.get("parts", [])) for row in graph.get("selected") or []
    }
    closed_pairs = {
        canonical_pair(row.get("parts", []))
        for row in _pose_closure(candidate, graph, features).get("connections", [])
        if row.get("closed")
    }
    collisions = bbox_collisions(features, placements)
    selected_overlap = 0.0
    non_edge_overlap = 0.0
    severe_non_edge_count = 0
    rows = []
    for item in collisions:
        pair = canonical_pair(item.get("parts", []))
        ratio = float(item.get("minimum_part_volume_ratio", 0.0))
        if pair in selected_pairs:
            # Closed selected pairs can have AABB overlap for valid insertion.
            weight = 0.15 if pair in closed_pairs else 0.60
            selected_overlap += weight * min(1.0, ratio)
            kind = "selected_pair_overlap"
        else:
            weight = 2.0
            non_edge_overlap += weight * min(1.0, ratio)
            severe_non_edge_count += int(ratio >= 0.05)
            kind = "non_edge_overlap"
        rows.append({**item, "pair_kind": kind, "weighted_cost": weight * min(1.0, ratio)})
    total = selected_overlap + non_edge_overlap
    return {
        "bbox_overlap_cost": total,
        "selected_pair_overlap_cost": selected_overlap,
        "non_edge_overlap_cost": non_edge_overlap,
        "severe_non_edge_overlap_count": severe_non_edge_count,
        "bbox_overlap_items": rows[:20],
        "bbox_overlap_item_count": len(rows),
    }


def _pose_precheck(
    candidate: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    closure = _pose_closure(candidate, graph, features)
    contact = _pose_contact_support(candidate, graph, features)
    overlap = _pose_overlap_cost(candidate, graph, features)
    closure_ratio = float(closure.get("closure_ratio", 0.0))
    score = (
        4.0 * closure_ratio
        + 1.5 * float(contact.get("contact_support_score", 0.0))
        - 1.2 * float(overlap.get("bbox_overlap_cost", 0.0))
        + 0.05 * float(candidate.get("total_score", 0.0))
    )
    return {
        "constraint_closure": closure,
        "contact_objective": contact,
        "overlap_objective": overlap,
        "group_pose_precheck_score": score,
    }


def _exact_status_rank(exact: dict[str, Any]) -> int:
    if exact.get("status") == "success" and not exact.get("collisions"):
        return 3
    if exact.get("status") == "success":
        return 1
    return 0


def _group_pose_final_score(
    candidate: dict[str, Any],
    exact: dict[str, Any],
    precheck: dict[str, Any],
) -> float:
    closure = precheck["constraint_closure"]
    contact = precheck["contact_objective"]
    overlap = precheck["overlap_objective"]
    collision_penalty = sum(
        min(1.0, float(row.get("minimum_part_volume_ratio", 0.0)))
        for row in exact.get("collisions", [])
    )
    exact_bonus = 2.0 if exact.get("status") == "success" and not exact.get("collisions") else 0.0
    exact_penalty = 3.0 * collision_penalty if exact.get("status") == "success" else 0.5
    return (
        5.0 * float(closure.get("closure_ratio", 0.0))
        + 2.0 * float(contact.get("contact_support_score", 0.0))
        - 1.5 * float(overlap.get("bbox_overlap_cost", 0.0))
        + exact_bonus
        - exact_penalty
        + 0.05 * float(candidate.get("total_score", 0.0))
    )


def _top_k_joint_proposals(
    pair_candidates: list[dict[str, Any]],
    *,
    k: int = 8,
) -> list[dict[str, Any]]:
    proposals = []
    for candidate in pair_candidates:
        rows = []
        for rank, match in enumerate(candidate.get("matches", [])[:k], 1):
            rows.append({
                "rank": rank,
                "relation_type": match.get("type"),
                "score": float(match.get("score", 0.0)),
                "confidence": match.get("confidence"),
                "feat_a_idx": match.get("feat_a_idx"),
                "feat_b_idx": match.get("feat_b_idx"),
                "candidate_origin": (match.get("reason") or {}).get("candidate_origin"),
                "reason": match.get("reason") or {},
                "joinable_support": match.get("joinable_support"),
            })
        proposals.append({
            "connection_id": candidate.get("connection_id"),
            "parts": candidate.get("parts"),
            "pair_score": candidate.get("score"),
            "relation_types": candidate.get("relation_types"),
            "providers": candidate.get("providers"),
            "top_k": rows,
        })
    return proposals


def _conservative_pose_output(
    result: dict[str, Any],
    pose_audit: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_rank = result.get("collision_validation", {}).get("selected_pose_rank")
    selected = next(
        (row for row in pose_audit if row.get("rank") == selected_rank),
        {},
    )
    group_record = {
        "assembly_id": result.get("assembly_id"),
        "parts": result.get("parts", []),
        "pose_status": result.get("pose_status"),
        "assembly_connected": result.get("assembly_connected"),
        "selected_pose_rank": selected_rank,
        "selected_candidate_origin": selected.get("candidate_origin"),
        "group_pose_final_score": selected.get("group_pose_final_score"),
        "closure": selected.get("constraint_closure"),
        "contact_objective": selected.get("contact_objective"),
        "overlap_objective": selected.get("overlap_objective"),
        "collision_validation": result.get("collision_validation", {}),
        "direct_connections": [
            {
                "connection_id": row.get("connection_id"),
                "parts": row.get("parts"),
                "primary_relation_type": row.get("primary_relation_type"),
                "assembly_method_relation_types": row.get("assembly_method_relation_types"),
                "constraint_closed_in_selected_pose": row.get("constraint_closed_in_selected_pose"),
                "review_required": row.get("review_required"),
            }
            for row in result.get("direct_connections", [])
        ],
    }
    accepted, review, rejected = [], [], []
    reasons = []
    collision = result.get("collision_validation", {})
    if not result.get("assembly_connected"):
        review.append(group_record)
        reasons.append("assembly_graph_not_connected")
    elif result.get("pose_status") == "valid":
        accepted.append(group_record)
        reasons.append("full_closure_and_exact_collision_free")
    elif result.get("pose_status") == "failed":
        rejected.append(group_record)
        reasons.append("exact_collision_or_pose_failure")
    else:
        review.append(group_record)
        if collision.get("status") != "success":
            reasons.append("exact_collision_not_available_or_budgeted")
        if selected.get("constraint_closure", {}).get("fully_closed") is not True:
            reasons.append("not_all_selected_constraints_closed")
        if selected.get("overlap_objective", {}).get("severe_non_edge_overlap_count", 0):
            reasons.append("severe_non_edge_overlap_in_precheck")
        if not reasons:
            reasons.append("conservative_uncertain_pose")
    group_record["decision_reasons"] = reasons
    return {
        "schema_version": "known_group_conservative_pose.v1",
        "policy": "accepted_requires_connected_full_closure_and_exact_collision_free",
        "accepted_groups": accepted,
        "review_groups": review,
        "rejected_groups": rejected,
        "unresolved_parts": result.get("unresolved_parts", []),
        "metrics": {
            "accepted_group_count": len(accepted),
            "review_group_count": len(review),
            "rejected_group_count": len(rejected),
            "unresolved_parts_count": len(result.get("unresolved_parts", [])),
            "checked_pose_count": result.get("collision_validation", {}).get("checked_pose_count"),
            "pose_status": result.get("pose_status"),
        },
    }


def _portable_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for component in components:
        item = dict(component)
        item["source"] = f"../{Path(item['source']).name}"
        result.append(item)
    return result


def _identity_pose_candidate(features: dict[str, Any]) -> dict[str, Any]:
    placements = {
        part: {"translate": [0.0, 0.0, 0.0]} for part in features
    }
    return {
        "placements": placements,
        "components": placements_to_manifest(features, placements),
        "selected_mates": [],
        "score": 0.0,
        "penalty": 0.0,
        "total_score": -0.05,
        "penalty_details": {"identity_prior": 0.05},
        "candidate_origin": "identity_input_pose",
    }


def _main_cylinder_axis(part_features: dict[str, Any]) -> list[float] | None:
    cylinders = part_features.get("cylinders") or []
    if not cylinders:
        return None
    cylinder = max(cylinders, key=lambda row: float(row.get("radius", 0.0)))
    axis = cylinder.get("axis")
    if not axis:
        return None
    return [float(value) for value in axis]


def _shift_placement_along_axis(
    placement: dict[str, Any],
    axis: list[float],
    offset: float,
) -> dict[str, Any]:
    shifted = json.loads(json.dumps(placement))
    translation = list(shifted.get("translate", [0.0, 0.0, 0.0]))
    shifted["translate"] = [
        float(translation[index]) + float(axis[index]) * float(offset)
        for index in range(3)
    ]
    return shifted


def _dot(a: list[float], b: list[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in v))


def _unit(v: list[float]) -> list[float]:
    length = _norm(v)
    if length <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [float(value) / length for value in v]


def _orthonormal_basis(normal: list[float]) -> tuple[list[float], list[float]]:
    n = _unit(normal)
    seed = [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = _unit(_cross(n, seed))
    v = _unit(_cross(n, u))
    return u, v


def _translate_placement(
    placement: dict[str, Any],
    vector: list[float],
) -> dict[str, Any]:
    shifted = json.loads(json.dumps(placement))
    translation = list(shifted.get("translate", [0.0, 0.0, 0.0]))
    shifted["translate"] = [
        float(translation[index]) + float(vector[index])
        for index in range(3)
    ]
    return shifted


def _flip_axis_direction(
    placement: dict[str, Any],
    center_axis: list[float],
    pivot_local: list[float],
) -> dict[str, Any]:
    """Flip axial direction while preserving a local axis point in world space."""
    flipped = json.loads(json.dumps(placement))
    pivot_before = transform_point(pivot_local, placement)
    perpendicular, _ = _orthonormal_basis(center_axis)
    rotations = list(flipped.get("rotate_sequence", []))
    rotations.append({
        "axis_angle": [
            float(perpendicular[0]),
            float(perpendicular[1]),
            float(perpendicular[2]),
            180.0,
        ]
    })
    flipped["rotate_sequence"] = rotations
    pivot_after = transform_point(pivot_local, flipped)
    translation = list(flipped.get("translate", [0.0, 0.0, 0.0]))
    flipped["translate"] = [
        float(translation[index]) + pivot_before[index] - pivot_after[index]
        for index in range(3)
    ]
    return flipped


def _clearance_axis_feature(
    part_features: dict[str, Any],
    center_radius: float,
) -> dict[str, Any] | None:
    cylinders = list(part_features.get("cylinders") or [])
    larger = [
        row for row in cylinders
        if float(row.get("radius", 0.0)) > center_radius + 0.05
    ]
    if larger:
        return min(larger, key=lambda row: float(row.get("radius", 0.0)))
    if cylinders:
        return max(cylinders, key=lambda row: float(row.get("radius", 0.0)))
    return None


def _feature_for_match(
    match: dict[str, Any],
    part: str,
    features: dict[str, Any],
    collection: str,
) -> dict[str, Any] | None:
    if part not in match.get("parts", []):
        return None
    side = "a" if match["parts"][0] == part else "b"
    index = match.get(f"feat_{side}_idx")
    if index is None:
        return None
    rows = features.get(part, {}).get(collection, [])
    if 0 <= int(index) < len(rows):
        return rows[int(index)]
    return None


def _candidate_from_placements(
    features: dict[str, Any],
    placements: dict[str, Any],
    source_candidate: dict[str, Any],
    *,
    origin: str,
    score_penalty: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "placements": placements,
        "components": placements_to_manifest(features, placements),
        "selected_mates": source_candidate.get("selected_mates", []),
        "score": source_candidate.get("score", 0.0),
        "penalty": source_candidate.get("penalty", 0.0),
        "total_score": float(source_candidate.get("total_score", 0.0)) - score_penalty,
        "penalty_details": {
            **(source_candidate.get("penalty_details") or {}),
            f"{origin}_penalty": score_penalty,
        },
        "candidate_origin": origin,
        **extra,
    }


def _axial_slide_pose_candidates(
    search: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    """Generate collision-avoidance DOF candidates for coaxial/clearance groups.

    This is generic: if two or more selected satellite parts connect to the same
    center through axial evidence, keep their solver rotations but sample
    translations along the center part's main cylinder axis.
    """
    base_placements = search.get("placements") or {}
    if not base_placements:
        return []
    axial_neighbors: dict[str, list[str]] = defaultdict(list)
    for connection in graph.get("selected") or []:
        if not (set(connection.get("relation_types") or []) & {COAXIAL, CLEARANCE}):
            continue
        parts = list(connection.get("parts") or [])
        if len(parts) != 2:
            continue
        for center, satellite in ((parts[0], parts[1]), (parts[1], parts[0])):
            if _main_cylinder_axis(features.get(center, {})):
                axial_neighbors[center].append(satellite)
    candidates = []
    for center, satellites in axial_neighbors.items():
        satellites = sorted(set(satellites))
        if len(satellites) < 2:
            continue
        axis = _main_cylinder_axis(features.get(center, {}))
        if not axis:
            continue
        center_cylinders = features.get(center, {}).get("cylinders") or []
        center_radius = max(
            (float(row.get("radius", 0.0)) for row in center_cylinders),
            default=0.0,
        )
        center_diag = max(_bbox_diagonal_from_features(features.get(center, {})), 1.0)
        offsets = [
            factor * center_diag
            for factor in (
                -2.5, -2.0, -1.5, -1.25, -1.0, -0.875, -0.75,
                -0.625, -0.5, -0.375, -0.25, 0.25, 0.375, 0.5,
                0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0, 2.5
            )
        ]
        # The first two satellites are enough for the current 2-flange shaft
        # topology; for larger groups this remains bounded.
        movable = satellites[:3]
        offset_grids = []
        if len(movable) == 2:
            symmetric_grids = []
            for factor in (0.5, 0.625, 0.75, 0.875, 1.0):
                value = factor * center_diag
                symmetric_grids.extend([(-value, value), (value, -value)])
            offset_grids = symmetric_grids + [
                (first, second)
                for first in offsets
                for second in offsets
                if abs(first - second) >= 0.75 * center_diag
            ]
            offset_grids.sort(
                key=lambda grid: (
                    sum(abs(value) for value in grid),
                    max(abs(value) for value in grid),
                    grid,
                )
            )
        else:
            offset_grids = [
                (-center_diag, 0.0, center_diag),
                (-1.5 * center_diag, 0.0, 1.5 * center_diag),
                (-2.0 * center_diag, 0.0, 2.0 * center_diag),
            ]
        for grid in offset_grids[:max_candidates]:
            placements = json.loads(json.dumps(base_placements))
            for part, offset in zip(movable, grid):
                if part not in placements:
                    continue
                placements[part] = _shift_placement_along_axis(
                    placements[part], axis, float(offset)
                )
            normalized_slide = (
                sum(abs(float(offset)) for offset in grid)
                / max(center_diag, 1.0)
            )
            candidates.append({
                "placements": placements,
                "components": placements_to_manifest(features, placements),
                "selected_mates": search.get("selected_mates", []),
                "score": search.get("score", 0.0),
                "penalty": search.get("penalty", 0.0),
                "total_score": (
                    float(search.get("total_score", 0.0))
                    - 0.10
                    - 0.05 * normalized_slide
                ),
                "penalty_details": {
                    **(search.get("penalty_details") or {}),
                    "axial_slide_prior": 0.10,
                    "normalized_axial_slide": normalized_slide,
                },
                "candidate_origin": "axial_slide_dof_search",
                "axial_slide": {
                    "center_part": center,
                    "axis": axis,
                    "offsets": {
                        part: float(offset) for part, offset in zip(movable, grid)
                    },
                },
            })
            # Orientation-consistent variant: satellites placed on opposite
            # sides of the center shaft should not retain the same axial
            # direction.  Point the dominant extrusion axis away from the
            # center; collision/closure validation remains the final gate.
            oriented_placements = json.loads(json.dumps(placements))
            flipped_parts = []
            for part, offset in zip(movable, grid):
                clearance_feature = _clearance_axis_feature(
                    features.get(part, {}),
                    center_radius,
                )
                if not clearance_feature or part not in oriented_placements:
                    continue
                part_axis_local = clearance_feature.get("axis")
                pivot_local = clearance_feature.get("origin")
                if not part_axis_local or not pivot_local:
                    continue
                part_axis_world = _unit(transform_vector(
                    part_axis_local,
                    oriented_placements[part],
                ))
                side_sign = -1.0 if float(offset) < 0.0 else 1.0
                if side_sign * _dot(part_axis_world, _unit(axis)) < 0.0:
                    oriented_placements[part] = _flip_axis_direction(
                        oriented_placements[part],
                        axis,
                        [float(value) for value in pivot_local],
                    )
                    flipped_parts.append(part)
            if flipped_parts:
                candidates.append({
                    "placements": oriented_placements,
                    "components": placements_to_manifest(features, oriented_placements),
                    "selected_mates": search.get("selected_mates", []),
                    "score": search.get("score", 0.0),
                    "penalty": search.get("penalty", 0.0),
                    "total_score": (
                        float(search.get("total_score", 0.0))
                        - 0.12
                        - 0.05 * normalized_slide
                    ),
                    "penalty_details": {
                        **(search.get("penalty_details") or {}),
                        "axial_orientation_prior": 0.12,
                        "normalized_axial_slide": normalized_slide,
                    },
                    "candidate_origin": "axial_slide_orientation_search",
                    "axial_slide": {
                        "center_part": center,
                        "axis": axis,
                        "offsets": {
                            part: float(offset) for part, offset in zip(movable, grid)
                        },
                        "flipped_parts": flipped_parts,
                    },
                })
    return candidates


def _planar_slide_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    """Search bounded in-plane translations for planar mate ambiguity."""
    generated = []
    for source_candidate in source_candidates[:12]:
        placements = source_candidate.get("placements") or {}
        for connection in graph.get("selected") or []:
            if not (set(connection.get("relation_types") or []) & {PLANAR_MATE, PLANAR_ALIGN}):
                continue
            parts = list(connection.get("parts") or [])
            if len(parts) != 2 or any(part not in placements for part in parts):
                continue
            diagonals = {
                part: _bbox_diagonal_from_features(features.get(part, {}))
                for part in parts
            }
            if min(diagonals.values()) <= 0.0:
                continue
            movable = min(parts, key=lambda part: diagonals[part])
            stationary = max(parts, key=lambda part: diagonals[part])
            if diagonals[movable] / max(diagonals[stationary], 1e-9) > 0.85:
                # Similar-sized planar pairs are usually flange/cover contacts;
                # large sliding search is more likely to create false poses.
                continue
            planar_matches = [
                match for match in connection.get("matches") or []
                if match.get("type") in {PLANAR_MATE, PLANAR_ALIGN}
            ][:3]
            for match in planar_matches:
                plane = _feature_for_match(match, stationary, features, "planes")
                if plane is None:
                    plane = _feature_for_match(match, movable, features, "planes")
                    normal_part = movable
                else:
                    normal_part = stationary
                if plane is None:
                    continue
                normal = transform_vector(
                    plane.get("normal", [0, 0, 1]),
                    placements.get(normal_part, {}),
                )
                u, v = _orthonormal_basis(normal)
                scale = max(2.0, min(50.0, 0.25 * diagonals[movable]))
                offsets = [-scale, -0.5 * scale, 0.5 * scale, scale]
                for du in offsets:
                    for dv in offsets:
                        move = [
                            u[index] * du + v[index] * dv
                            for index in range(3)
                        ]
                        new_placements = json.loads(json.dumps(placements))
                        new_placements[movable] = _translate_placement(
                            new_placements[movable], move
                        )
                        normalized = _norm(move) / max(diagonals[movable], 1.0)
                        generated.append(
                            _candidate_from_placements(
                                features,
                                new_placements,
                                source_candidate,
                                origin="planar_slide_dof_search",
                                score_penalty=0.08 + 0.04 * normalized,
                                extra={
                                    "planar_slide": {
                                        "connection_id": connection["connection_id"],
                                        "movable_part": movable,
                                        "stationary_part": stationary,
                                        "offset": move,
                                        "basis_u": u,
                                        "basis_v": v,
                                        "normalized_slide": normalized,
                                    }
                                },
                            )
                        )
                        if len(generated) >= max_candidates:
                            return generated
    return generated


def _pocket_depth_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    """Search bounded insertion-depth translations for pocket mates."""
    generated = []
    for source_candidate in source_candidates[:12]:
        placements = source_candidate.get("placements") or {}
        for connection in graph.get("selected") or []:
            if POCKET_MATE not in set(connection.get("relation_types") or []):
                continue
            parts = list(connection.get("parts") or [])
            if len(parts) != 2 or any(part not in placements for part in parts):
                continue
            diagonals = {
                part: _bbox_diagonal_from_features(features.get(part, {}))
                for part in parts
            }
            if min(diagonals.values()) <= 0.0:
                continue
            movable = min(parts, key=lambda part: diagonals[part])
            stationary = max(parts, key=lambda part: diagonals[part])
            pocket_matches = [
                match for match in connection.get("matches") or []
                if match.get("type") == POCKET_MATE
            ][:3]
            for match in pocket_matches:
                if match["parts"][0] == stationary:
                    direction = match.get("_dir_a")
                    size_stationary = match.get("_size_a") or []
                    size_movable = match.get("_size_b") or []
                elif match["parts"][1] == stationary:
                    direction = match.get("_dir_b")
                    size_stationary = match.get("_size_b") or []
                    size_movable = match.get("_size_a") or []
                else:
                    direction = match.get("_dir_a") or match.get("_dir_b")
                    size_stationary = match.get("_size_a") or []
                    size_movable = match.get("_size_b") or []
                if not direction:
                    continue
                axis = _unit(transform_vector(direction, placements.get(stationary, {})))
                depth_dims = _nonzero_dims(size_stationary) + _nonzero_dims(size_movable)
                depth = min(depth_dims) if depth_dims else min(diagonals[movable], 20.0)
                depth = max(1.0, min(depth, 40.0))
                offsets = [0.5 * depth, 1.0 * depth, 0.25 * depth]
                for offset in offsets:
                    for sign_axis in (axis, [-value for value in axis]):
                        move = [sign_axis[index] * offset for index in range(3)]
                        new_placements = json.loads(json.dumps(placements))
                        new_placements[movable] = _translate_placement(
                            new_placements[movable], move
                        )
                        normalized = abs(offset) / max(diagonals[movable], 1.0)
                        generated.append(
                            _candidate_from_placements(
                                features,
                                new_placements,
                                source_candidate,
                                origin="pocket_depth_dof_search",
                                score_penalty=0.06 + 0.04 * normalized,
                                extra={
                                    "pocket_depth": {
                                        "connection_id": connection["connection_id"],
                                        "movable_part": movable,
                                        "stationary_part": stationary,
                                        "axis": sign_axis,
                                        "offset": offset,
                                        "depth_proxy": depth,
                                        "normalized_depth_move": normalized,
                                    }
                                },
                            )
                        )
                        if len(generated) >= max_candidates:
                            return generated
    return generated


def _axis_alignment_rotation(
    source_axis: list[float],
    target_axis: list[float],
) -> dict[str, Any] | None:
    source = _unit(source_axis)
    target = _unit(target_axis)
    dot = max(-1.0, min(1.0, _dot(source, target)))
    if dot > 1.0 - 1e-9:
        return None
    if dot < -1.0 + 1e-9:
        axis, _ = _orthonormal_basis(source)
        return {"axis_angle": [axis[0], axis[1], axis[2], 180.0]}
    axis = _unit(_cross(source, target))
    angle = math.degrees(math.acos(dot))
    return {"axis_angle": [axis[0], axis[1], axis[2], angle]}


def _placement_from_joinable_axis_parameters(
    movable_axis_local: list[float],
    movable_origin_local: list[float],
    target_axis_world: list[float],
    target_origin_world: list[float],
    *,
    offset: float,
    rotation_degrees: float,
) -> dict[str, Any]:
    """Build a placement from JoinABLe-style joint parameters.

    JoinABLe aligns the two predicted joint axes, then searches three residual
    parameters: offset along the joint axis, rotation about the joint axis, and
    a flip that reverses the joint-axis direction.  The caller supplies the
    already-flipped target axis; this helper materializes offset/rotation.
    """
    target_axis = _unit(target_axis_world)
    rotate_sequence: list[dict[str, Any]] = []
    alignment = _axis_alignment_rotation(movable_axis_local, target_axis)
    if alignment is not None:
        rotate_sequence.append(alignment)
    if abs(rotation_degrees) > 1e-9:
        rotate_sequence.append({
            "axis_angle": [
                target_axis[0],
                target_axis[1],
                target_axis[2],
                float(rotation_degrees),
            ]
        })
    rotated_origin = transform_vector(
        movable_origin_local,
        {"rotate_sequence": rotate_sequence},
    )
    target_point = [
        float(target_origin_world[index]) + target_axis[index] * float(offset)
        for index in range(3)
    ]
    return {
        "translate": [
            target_point[index] - rotated_origin[index]
            for index in range(3)
        ],
        "rotate_sequence": rotate_sequence,
    }


def _joinable_axis_data_for_match(
    match: dict[str, Any],
    stationary: str,
    movable: str,
    features: dict[str, Any],
    placements: dict[str, Any],
) -> tuple[list[float], list[float], list[float], list[float], str] | None:
    relation_type = match.get("type")
    if relation_type in {COAXIAL, CLEARANCE}:
        stationary_feature = _feature_for_match(match, stationary, features, "cylinders")
        movable_feature = _feature_for_match(match, movable, features, "cylinders")
        if not stationary_feature or not movable_feature:
            return None
        target_axis = _unit(transform_vector(
            stationary_feature.get("axis", [0, 0, 1]),
            placements.get(stationary, {}),
        ))
        target_origin = transform_point(
            stationary_feature.get("origin", [0, 0, 0]),
            placements.get(stationary, {}),
        )
        return (
            [float(value) for value in movable_feature.get("axis", [0, 0, 1])],
            [float(value) for value in movable_feature.get("origin", [0, 0, 0])],
            target_axis,
            target_origin,
            "axial",
        )
    if relation_type in {PLANAR_MATE, PLANAR_ALIGN}:
        stationary_feature = _feature_for_match(match, stationary, features, "planes")
        movable_feature = _feature_for_match(match, movable, features, "planes")
        if not stationary_feature or not movable_feature:
            return None
        stationary_normal = _unit(transform_vector(
            stationary_feature.get("normal", [0, 0, 1]),
            placements.get(stationary, {}),
        ))
        target_axis = (
            [-value for value in stationary_normal]
            if relation_type == PLANAR_MATE else stationary_normal
        )
        target_origin = transform_point(
            stationary_feature.get("point", [0, 0, 0]),
            placements.get(stationary, {}),
        )
        return (
            [float(value) for value in movable_feature.get("normal", [0, 0, 1])],
            [float(value) for value in movable_feature.get("point", [0, 0, 0])],
            target_axis,
            target_origin,
            "planar",
        )
    if relation_type == POCKET_MATE:
        parts = list(match.get("parts") or [])
        if len(parts) != 2:
            return None
        stationary_side = "a" if parts[0] == stationary else "b"
        movable_side = "a" if parts[0] == movable else "b"
        target_pocket = match.get(f"pocket_{stationary_side}") or {}
        movable_pocket = match.get(f"pocket_{movable_side}") or {}
        target_direction = (
            match.get(f"_dir_{stationary_side}")
            or target_pocket.get("direction")
        )
        movable_direction = (
            match.get(f"_dir_{movable_side}")
            or movable_pocket.get("direction")
        )
        target_center = (
            match.get(f"_center_{stationary_side}")
            or target_pocket.get("center")
        )
        movable_center = (
            match.get(f"_center_{movable_side}")
            or movable_pocket.get("center")
        )
        if (
            not target_direction
            or not movable_direction
            or not target_center
            or not movable_center
        ):
            return None
        target_axis = _unit(transform_vector(
            target_direction,
            placements.get(stationary, {}),
        ))
        # Pocket insertions usually close opposing normals/directions.
        target_axis = [-value for value in target_axis]
        target_origin = transform_point(
            target_center,
            placements.get(stationary, {}),
        )
        return (
            [float(value) for value in movable_direction],
            [float(value) for value in movable_center],
            target_axis,
            [float(value) for value in target_origin],
            "pocket",
        )
    return None


def _joinable_pose_parameter_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    """Discrete reproduction of JoinABLe's joint pose search.

    We do not use JoinABLe's neural axis predictor here.  Existing analytic
    pair evidence supplies the top-k joint axes; this routine searches the same
    residual pose parameters described by the paper/code: axial offset,
    rotation around the joint axis, and flip of the joint-axis direction.
    """
    generated: list[dict[str, Any]] = []
    axial_degree: dict[str, int] = defaultdict(int)
    for connection in graph.get("selected") or []:
        if set(connection.get("relation_types") or []) & {COAXIAL, CLEARANCE}:
            for part in connection.get("parts") or []:
                axial_degree[part] += 1
    central_axial_parts = {
        part for part, degree in axial_degree.items()
        if degree >= 2 and _main_cylinder_axis(features.get(part, {}))
    }
    for source_candidate in source_candidates[:12]:
        placements = source_candidate.get("placements") or {}
        for connection in graph.get("selected") or []:
            parts = list(connection.get("parts") or [])
            if len(parts) != 2 or any(part not in placements for part in parts):
                continue
            diagonals = {
                part: _bbox_diagonal_from_features(features.get(part, {}))
                for part in parts
            }
            relation_types = set(connection.get("relation_types") or [])
            if relation_types & {COAXIAL, CLEARANCE}:
                center_candidates = [part for part in parts if part in central_axial_parts]
                stationary = center_candidates[0] if center_candidates else min(
                    parts,
                    key=lambda part: (
                        max((float(row.get("radius", 0.0)) for row in features.get(part, {}).get("cylinders", [])), default=0.0),
                        diagonals[part],
                    ),
                )
                movable = parts[1] if parts[0] == stationary else parts[0]
                relation_priority = [CLEARANCE, COAXIAL, PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE]
                offset_scale = max(1.0, min(120.0, 0.6 * (diagonals[stationary] + diagonals[movable])))
                offset_factors = [0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75, -1.0, 1.0]
                rotation_angles = [0.0, 90.0, 180.0, 270.0]
            elif POCKET_MATE in relation_types:
                stationary = max(parts, key=lambda part: diagonals[part])
                movable = parts[1] if parts[0] == stationary else parts[0]
                relation_priority = [POCKET_MATE, PLANAR_MATE, PLANAR_ALIGN]
                offset_scale = max(1.0, min(80.0, 0.5 * diagonals[movable]))
                offset_factors = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0]
                rotation_angles = [0.0, 180.0, 90.0, 270.0]
            elif relation_types & {PLANAR_MATE, PLANAR_ALIGN}:
                stationary = max(parts, key=lambda part: diagonals[part])
                movable = parts[1] if parts[0] == stationary else parts[0]
                relation_priority = [PLANAR_MATE, PLANAR_ALIGN]
                offset_scale = max(1.0, min(50.0, 0.25 * diagonals[movable]))
                offset_factors = [0.0, -0.1, 0.1, -0.25, 0.25]
                rotation_angles = [0.0, 180.0, 90.0, 270.0]
            else:
                continue
            matches = []
            for relation_type in relation_priority:
                matches.extend([
                    match for match in connection.get("matches") or []
                    if match.get("type") == relation_type
                ])
            for match in matches[:4]:
                axis_data = _joinable_axis_data_for_match(
                    match, stationary, movable, features, placements
                )
                if axis_data is None:
                    continue
                (
                    movable_axis_local,
                    movable_origin_local,
                    target_axis_world,
                    target_origin_world,
                    joint_kind,
                ) = axis_data
                for flip in (False, True):
                    flipped_axis = (
                        [-value for value in target_axis_world]
                        if flip else target_axis_world
                    )
                    for offset_factor in offset_factors:
                        offset = float(offset_factor) * offset_scale
                        for rotation_degrees in rotation_angles:
                            new_placements = json.loads(json.dumps(placements))
                            new_placements[movable] = _placement_from_joinable_axis_parameters(
                                movable_axis_local,
                                movable_origin_local,
                                flipped_axis,
                                target_origin_world,
                                offset=offset,
                                rotation_degrees=rotation_degrees,
                            )
                            normalized_offset = abs(offset) / max(offset_scale, 1.0)
                            generated.append(
                                _candidate_from_placements(
                                    features,
                                    new_placements,
                                    source_candidate,
                                    origin="joinable_pose_parameter_search",
                                    score_penalty=(
                                        0.14
                                        + 0.03 * normalized_offset
                                        + (0.02 if flip else 0.0)
                                        + (0.01 if rotation_degrees else 0.0)
                                    ),
                                    extra={
                                        "joinable_pose_search": {
                                            "connection_id": connection["connection_id"],
                                            "joint_kind": joint_kind,
                                            "stationary_part": stationary,
                                            "movable_part": movable,
                                            "match_type": match.get("type"),
                                            "offset": offset,
                                            "offset_scale": offset_scale,
                                            "rotation_degrees": rotation_degrees,
                                            "flip": flip,
                                            "target_axis_world": flipped_axis,
                                            "target_origin_world": target_origin_world,
                                        }
                                    },
                                )
                            )
                            if len(generated) >= max_candidates:
                                return generated
    return generated


def _joinable_multi_axial_pose_candidates(
    search: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    """Jointly search JoinABLe parameters for shaft-with-multiple-satellites.

    Pair-wise pose search can fix one flange/hub while making the other one
    invalid.  For a central shaft connected to two or more axial satellites,
    search their offsets/flips as one bounded group move.
    """
    base_placements = search.get("placements") or {}
    if not base_placements:
        return []
    axial_connections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for connection in graph.get("selected") or []:
        if not (set(connection.get("relation_types") or []) & {COAXIAL, CLEARANCE}):
            continue
        parts = list(connection.get("parts") or [])
        if len(parts) != 2:
            continue
        for center, satellite in ((parts[0], parts[1]), (parts[1], parts[0])):
            if (
                center in base_placements
                and satellite in base_placements
                and _main_cylinder_axis(features.get(center, {}))
            ):
                axial_connections[center].append({
                    "connection": connection,
                    "satellite": satellite,
                })
    generated: list[dict[str, Any]] = []
    for center, rows in axial_connections.items():
        satellites = []
        seen_satellites = set()
        for row in rows:
            satellite = row["satellite"]
            if satellite in seen_satellites:
                continue
            seen_satellites.add(satellite)
            satellites.append(row)
        if len(satellites) < 2:
            continue
        center_diag = max(_bbox_diagonal_from_features(features.get(center, {})), 1.0)
        center_axis = _unit(transform_vector(
            _main_cylinder_axis(features.get(center, {})) or [0, 0, 1],
            base_placements.get(center, {}),
        ))
        movable_rows = satellites[:3]
        if len(movable_rows) == 2:
            offset_grids = []
            for factor in (0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25):
                value = factor * center_diag
                offset_grids.extend([(-value, value), (value, -value)])
            flip_patterns = [
                (False, True),
                (True, False),
                (False, False),
                (True, True),
            ]
        else:
            offset_grids = [
                (-center_diag, 0.0, center_diag),
                (-1.25 * center_diag, 0.0, 1.25 * center_diag),
            ]
            flip_patterns = [
                tuple(False for _ in movable_rows),
                tuple(index % 2 == 0 for index, _ in enumerate(movable_rows)),
            ]
        rotation_angles = [0.0, 180.0]
        for offsets in offset_grids:
            for flips in flip_patterns:
                for rotation_degrees in rotation_angles:
                    placements = json.loads(json.dumps(base_placements))
                    move_details = {}
                    failed = False
                    for row, offset, flip in zip(movable_rows, offsets, flips):
                        satellite = row["satellite"]
                        connection = row["connection"]
                        matches = [
                            match for match in connection.get("matches") or []
                            if match.get("type") in {CLEARANCE, COAXIAL}
                        ]
                        axis_data = None
                        for match in matches[:3]:
                            axis_data = _joinable_axis_data_for_match(
                                match, center, satellite, features, placements
                            )
                            if axis_data is not None:
                                break
                        if axis_data is None:
                            failed = True
                            break
                        (
                            movable_axis_local,
                            movable_origin_local,
                            target_axis_world,
                            target_origin_world,
                            joint_kind,
                        ) = axis_data
                        target_axis = [-value for value in target_axis_world] if flip else target_axis_world
                        placements[satellite] = _placement_from_joinable_axis_parameters(
                            movable_axis_local,
                            movable_origin_local,
                            target_axis,
                            target_origin_world,
                            offset=float(offset),
                            rotation_degrees=rotation_degrees,
                        )
                        move_details[satellite] = {
                            "connection_id": connection["connection_id"],
                            "joint_kind": joint_kind,
                            "offset": float(offset),
                            "flip": bool(flip),
                            "rotation_degrees": float(rotation_degrees),
                            "target_axis_world": target_axis,
                            "target_origin_world": target_origin_world,
                        }
                    if failed:
                        continue
                    normalized = sum(abs(float(offset)) for offset in offsets) / max(
                        center_diag * len(offsets),
                        1.0,
                    )
                    generated.append(
                        _candidate_from_placements(
                            features,
                            placements,
                            search,
                            origin="joinable_multi_axial_pose_search",
                            score_penalty=0.12 + 0.04 * normalized + (0.01 if rotation_degrees else 0.0),
                            extra={
                                "joinable_multi_axial_search": {
                                    "center_part": center,
                                    "center_axis": center_axis,
                                    "satellites": move_details,
                                    "normalized_offset": normalized,
                                }
                            },
                        )
                    )
                    if len(generated) >= max_candidates:
                        return generated
    return generated


def _augment_pose_candidates(
    search: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
) -> dict[str, Any]:
    augmented = dict(search)
    candidates = []
    candidates.append(_identity_pose_candidate(features))
    candidates.extend(list(search.get("pose_candidates") or []))
    if not search.get("pose_candidates"):
        candidates.append({
            "placements": search["placements"],
            "components": search["components"],
            "selected_mates": search.get("selected_mates", []),
            "score": search.get("score", 0.0),
            "penalty": search.get("penalty", 0.0),
            "total_score": search.get("total_score", 0.0),
            "penalty_details": search.get("penalty_details", {}),
            "candidate_origin": "solver_primary",
        })
    total_surface_count = sum(
        len(row.get("planes", [])) + len(row.get("cylinders", []))
        for row in features.values()
    )
    large_step_case = total_surface_count > 3000
    if not large_step_case:
        candidates.extend(
            _axial_slide_pose_candidates(
                search, graph, features, max_candidates=160
            )
        )
        # Run planar/pocket DOF search over the already bounded candidate
        # frontier, including identity and axial candidates.
        candidates.extend(
            _planar_slide_pose_candidates(
                candidates, graph, features, max_candidates=80
            )
        )
        candidates.extend(
            _pocket_depth_pose_candidates(
                candidates, graph, features, max_candidates=80
            )
        )
        candidates.extend(
            _joinable_multi_axial_pose_candidates(
                search, graph, features, max_candidates=80
            )
        )
        candidates.extend(
            _joinable_pose_parameter_candidates(
                candidates, graph, features, max_candidates=200
            )
        )
    else:
        # Large STEP cases can have tens of thousands of surfaces, so broad
        # planar expansion is still disabled.  Pocket depth search, however,
        # operates only on the already selected bounded frontier and is needed
        # for slide-in assemblies such as fan-module into fan-cage.
        candidates.extend(
            _pocket_depth_pose_candidates(
                candidates, graph, features, max_candidates=2
            )
        )
        candidates.extend(
            _joinable_pose_parameter_candidates(
                candidates, graph, features, max_candidates=48
            )
        )
    # Deduplicate by coarse translations and rotations.
    seen = set()
    unique = []
    for candidate in candidates:
        key = json.dumps(
            {
                part: candidate["placements"].get(part, {})
                for part in sorted(candidate.get("placements", {}))
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    augmented["pose_candidates"] = unique
    augmented["complete_pose_candidate_count"] = len(unique)
    augmented["candidate_augmentation"] = {
        "identity_candidate_added": True,
        "planar_slide_search_enabled": True,
        "pocket_depth_search_enabled": True,
        "joinable_multi_axial_pose_search_enabled": True,
        "joinable_pose_parameter_search_enabled": True,
        "large_step_case": large_step_case,
        "total_surface_count": total_surface_count,
        "total_pose_candidates_after_augmentation": len(unique),
    }
    return augmented


def _evaluate_pose_candidates(
    case_dir: Path,
    output_dir: Path,
    search: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bool, int]:
    candidates = list(search.get("pose_candidates") or [])
    if not candidates:
        candidates = [{
            "placements": search["placements"],
            "components": search["components"],
            "selected_mates": search.get("selected_mates", []),
            "score": search.get("score", 0.0),
            "penalty": search.get("penalty", 0.0),
            "total_score": search.get("total_score", 0.0),
        }]
    audit_by_rank: dict[int, dict[str, Any]] = {}
    evaluated = []
    large_step_case = bool(
        (search.get("candidate_augmentation") or {}).get("large_step_case")
    )
    exact_check_budget = max(1, len(candidates))  # always check all known-group candidates
    prechecked = []
    for rank, candidate in enumerate(candidates, 1):
        precheck = _pose_precheck(candidate, graph, features)
        prechecked.append((rank, candidate, precheck))

    exact_rank_budget = {
        rank
        for rank, _, _ in sorted(
            prechecked,
            key=lambda item: (
                bool(item[2]["constraint_closure"].get("fully_closed")),
                float(item[2]["constraint_closure"].get("closure_ratio", 0.0)),
                float(item[2].get("group_pose_precheck_score", 0.0)),
                float(item[1].get("total_score", 0.0)),
            ),
            reverse=True,
        )[:exact_check_budget]
    }

    for rank, candidate, precheck in prechecked:
        components = _portable_components(candidate["components"])
        closure = precheck["constraint_closure"]
        if large_step_case and not closure["fully_closed"]:
            exact = {
                "status": "skipped_constraint_not_closed",
                "method": "closure_short_circuit",
                "collisions": [],
                "errors": [],
            }
        elif rank not in exact_rank_budget:
            exact = {
                "status": (
                    "skipped_large_step_exact_budget"
                    if large_step_case else "skipped_exact_budget"
                ),
                "method": (
                    "large_step_exact_budget"
                    if large_step_case else "bounded_exact_budget"
                ),
                "collisions": [],
                "errors": [],
            }
        else:
            exact = exact_shape_collisions(output_dir, components)
        final_score = _group_pose_final_score(candidate, exact, precheck)
        row = {
            "rank": rank,
            "candidate_origin": candidate.get("candidate_origin", "solver_beam"),
            "axial_slide": candidate.get("axial_slide"),
            "planar_slide": candidate.get("planar_slide"),
            "pocket_depth": candidate.get("pocket_depth"),
            "joinable_pose_search": candidate.get("joinable_pose_search"),
            "joinable_multi_axial_search": candidate.get("joinable_multi_axial_search"),
            "total_score": candidate.get("total_score"),
            "collision_status": exact["status"],
            "collision_count": len(exact["collisions"]),
            "collisions": exact["collisions"],
            "errors": exact.get("errors", []),
            "constraint_closure": closure,
            "contact_objective": precheck["contact_objective"],
            "overlap_objective": {
                key: value for key, value in precheck["overlap_objective"].items()
                if key != "bbox_overlap_items"
            },
            "bbox_overlap_items": precheck["overlap_objective"].get("bbox_overlap_items", []),
            "group_pose_precheck_score": precheck["group_pose_precheck_score"],
            "group_pose_final_score": final_score,
        }
        audit_by_rank[rank] = row
        evaluated.append((candidate, exact, closure, rank, final_score))
    collision_free = [
        item for item in evaluated
        if item[1]["status"] == "success" and not item[1]["collisions"]
    ]
    fully_valid = [item for item in collision_free if item[2]["fully_closed"]]
    if fully_valid:
        selected, selected_exact, selected_closure, selected_rank, _selected_score = max(
            fully_valid,
            key=lambda item: (
                float(item[4]),
                -int(item[3]),
            ),
        )
    elif collision_free:
        selected, selected_exact, selected_closure, selected_rank, _selected_score = max(
            collision_free,
            key=lambda item: (
                item[2]["closure_ratio"],
                float(item[4]),
            ),
        )
    else:
        selected, selected_exact, selected_closure, selected_rank, _selected_score = max(
            evaluated,
            key=lambda item: (
                _exact_status_rank(item[1]),
                item[2]["closure_ratio"],
                float(item[4]),
            ),
        )
    audit = [audit_by_rank[index] for index in sorted(audit_by_rank)]
    if selected_rank in audit_by_rank:
        audit_by_rank[selected_rank]["selected_by_group_pose_optimizer"] = True
    return (
        selected,
        selected_exact,
        audit,
        bool(selected_closure["fully_closed"]),
        int(selected_rank),
    )


def _apply_axial_contact_to_clearance_pairs(
    placements: dict[str, Any],
    features: dict[str, Any],
    selected_pairs: set[tuple[str, str]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Post-process: slide clearance-connected parts to shaft ends.

    When two parts share the same reference (e.g. two flanges on one shaft),
    one goes to each end. A 2mm safety gap prevents bbox inaccuracy from
    causing geometric penetration.
    """
    from coordinate_solver import (
        _global_vector, _part_bbox_interval_along_axis, _vec_norm,
    )

    GAP = 2.0
    result = dict(placements)

    # Collect clearance targets grouped by reference
    ref_targets: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for match in matches:
        if match.get("type") != "clearance":
            continue
        pair = canonical_pair(match["parts"])
        if pair not in selected_pairs:
            continue
        a, b = match["parts"]
        pa = result.get(a, {}).get("translate", [0.0, 0.0, 0.0])
        pb = result.get(b, {}).get("translate", [0.0, 0.0, 0.0])
        ref = a if sum(v*v for v in pa) <= sum(v*v for v in pb) else b
        tgt = b if ref == a else a
        ref_targets[ref].append((ref, tgt, match))

    for ref_part, targets in ref_targets.items():
        ref_feats = features.get(ref_part)
        if not ref_feats or not ref_feats.get("cylinders"):
            continue
        ref_placement = result.get(ref_part, {})
        axis = _global_vector(ref_feats["cylinders"][0]["axis"], ref_placement)
        axis_u = _vec_norm(axis)
        ref_iv = _part_bbox_interval_along_axis(ref_feats, ref_placement, axis)
        if not ref_iv:
            continue
        ref_min, ref_max = ref_iv

        for i, (ref, tgt, match) in enumerate(targets):
            tgt_feats = features.get(tgt)
            tgt_placement = result.get(tgt, {})
            if not tgt_feats:
                continue
            tgt_iv = _part_bbox_interval_along_axis(tgt_feats, tgt_placement, axis)
            if not tgt_iv:
                continue
            tgt_min, tgt_max = tgt_iv
            current = list(tgt_placement.get("translate", [0.0, 0.0, 0.0]))

            # Determine which end this target should go to
            if len(targets) == 1:
                # Single target → nearest end
                to_min = abs(tgt_max - ref_min)
                to_max = abs(tgt_min - ref_max)
                if to_min < to_max:
                    slide = ref_min - tgt_max - GAP
                else:
                    slide = ref_max - tgt_min + GAP
            else:
                # Multiple targets → alternate ends
                if i == 0:
                    slide = ref_min - tgt_max - GAP
                else:
                    slide = ref_max - tgt_min + GAP

            new_tgt = dict(tgt_placement)
            new_tgt["translate"] = [current[k] + slide * axis_u[k] for k in range(3)]
            result[tgt] = new_tgt

    return result


def run_known_group_assembly(
    case_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    joinable_report: str | Path | None = None,
    beam_width: int = 20,
) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir else case_dir / "known_group_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    step_files = sorted(
        path for path in case_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".step", ".stp"}
        and not path.name.lower().startswith("assembly")
    )
    if not 1 <= len(step_files) <= 5:
        raise ValueError("known-group entry point supports 1..5 STEP parts")
    parts = [path.name for path in step_files]
    features = {path.name: extract_features(str(path)) for path in step_files}
    raw_matches = match_features(features, {
        "preserve_cylindrical_face_hypotheses": True,
        "maximum_cylindrical_hypotheses_per_radius_bucket": 8,
        "maximum_planar_hypotheses_per_normal_bucket": 8,
        "minimum_local_plane_area_mm2": 10.0,
        "local_component_diagonal_mm": 80.0,
    })
    # ── Inject JoinABLe-learned joint axis constraints ──
    try:
        from joinable_joint_axis import inject_joinable_constraints as _inject_joinable
        raw_matches = _inject_joinable(case_dir, features, raw_matches, top_k=5)
    except Exception:
        pass
    # ──────────────────────────────────────────────────────
    joinable_path = Path(joinable_report).resolve() if joinable_report else None
    joinable_by_pair, joinable_audit = _joinable_index(joinable_path, parts, case_dir.name)
    scored = _apply_joinable_support(
        score_matches(raw_matches, features), joinable_by_pair
    )
    pair_candidates = build_pair_candidates(scored, joinable_by_pair)
    # Compute part weights from file sizes — larger files are more likely
    # to be the structural center (chassis, baseplate, main housing).
    part_weights = {path.name: float(path.stat().st_size) for path in step_files}
    graph = select_direct_connections(parts, pair_candidates, conservative=True, part_weights=part_weights)
    selected_pairs = {
        canonical_pair(row["parts"]) for row in graph["selected"]
    }
    solver_matches = [
        row for row in scored if canonical_pair(row["parts"]) in selected_pairs
    ]
    search = solve_small_assembly(
        features,
        solver_matches,
        beam_width=beam_width,
        target_branching=min(3, max(1, len(parts) - 1)),
    )
    search = _augment_pose_candidates(search, graph, features)
    selected_pose, exact, pose_audit, pose_fully_closed, selected_pose_rank = _evaluate_pose_candidates(
        case_dir, output_dir, search, graph, features
    )
    components = _portable_components(selected_pose["components"])
    placements = selected_pose["placements"]
    # ── Axial contact correction ──
    # For every clearance pair, slide the target part along the shared axis
    # until bounding boxes touch.  This fixes the missing planar-mate when
    # shaft shoulder and flange face are not parallel in local coords.
    placements = _apply_axial_contact_to_clearance_pairs(
        placements, features, selected_pairs, solver_matches
    )
    selected_pose["placements"] = placements
    # Update component placements from the corrected placements dict
    for comp in selected_pose.get("components", []):
        label = comp.get("label", "")
        for part_name, plac in placements.items():
            if Path(part_name).stem == label or part_name == label:
                comp["placement"] = plac
                break
    components = _portable_components(selected_pose["components"])
    # ─────────────────────────────
    manifest = {
        "schema_version": "2.0.0",
        "assembly_name": case_dir.name,
        "global_units": "mm",
        "components": components,
    }
    _write(output_dir / "assembly_manifest.json", manifest)
    build_assembly(
        str(output_dir / "assembly_manifest.json"),
        str(output_dir / "assembly.step"),
    )

    used = _selected_evidence_fingerprints(selected_pose)
    constraints = []
    connections = []
    for selected in graph["selected"]:
        a, b = selected["parts"]
        constraint_ids = []
        selected_rows = []
        by_type = defaultdict(list)
        for row in selected["matches"]:
            by_type[row["type"]].append(row)
        for relation_type, rows in sorted(by_type.items()):
            ranked_rows = []
            for row in rows:
                record = _residual_record(row, features, placements)
                ranked_rows.append((
                    _constraint_satisfied(row, record),
                    -float(record.get("residual", 1e12) or 1e12),
                    float(row.get("score", 0.0)),
                    row,
                ))
            best = max(ranked_rows, key=lambda item: item[:3])
            if best[0]:
                selected_rows.append(best[3])
        for row in selected_rows:
            relation = row["type"]
            constraint_id = stable_id(
                "R", selected["connection_id"], relation,
                str(row.get("feat_a_idx")), str(row.get("feat_b_idx")),
            )
            constraint_ids.append(constraint_id)
            residual_record = constraint_residual(row, features, placements)
            residual = residual_record.get("residual")
            constraints.append({
                "constraint_id": constraint_id,
                "connection_id": selected["connection_id"],
                "parts": list(row["parts"]),
                "relation_type": relation,
                "interface_a": _interface(row, "a", features),
                "interface_b": _interface(row, "b", features),
                "score": float(row.get("score", 0.0)),
                "confidence": row.get("confidence", "low"),
                "providers": selected["providers"],
                "used_for_pose": _fingerprint(row) in used,
                "constraint_residual": (
                    float(residual)
                    if residual is not None and math.isfinite(float(residual))
                    else None
                ),
                "evidence": {
                    **(row.get("reason") or {}),
                    "pose_residual": residual_record,
                    **(
                        {"joinable_support": row["joinable_support"]}
                        if row.get("joinable_support") else {}
                    ),
                },
            })
        satisfied_types = [row["type"] for row in selected_rows]
        method_types, method_reasons = _assembly_method_relation_types(
            selected, features
        )
        primary_row = max(
            selected_rows or selected["matches"],
            key=lambda row: float(row.get("score", 0.0)),
        )
        connections.append({
            "connection_id": selected["connection_id"],
            "parts": [a, b],
            "primary_relation_type": (
                method_types[0] if method_types else primary_row["type"]
            ),
            "supporting_relation_types": satisfied_types,
            "assembly_method_relation_types": method_types,
            "assembly_method_reason": method_reasons,
            "constraint_ids": constraint_ids,
            "score": selected["score"],
            "confidence": selected["confidence"],
            "selection_role": selected["selection_role"],
            "constraint_closed_in_selected_pose": bool(selected_rows),
            "review_required": not bool(selected_rows),
            "providers": selected["providers"],
            "relative_transform_a_to_b": _relative_transform(a, b, placements),
            "joinable_interface_candidates": (
                (selected.get("joinable") or {}).get("top_interface_candidates", [])
            ),
        })

    if exact["status"] == "success":
        if exact["collisions"]:
            pose_status = "failed"
        else:
            pose_status = (
                "valid"
                if pose_fully_closed and graph["connected"]
                else "uncertain"
            )
    else:
        pose_status = "uncertain"
    limitations = []
    if joinable_audit["status"] != "success":
        limitations.append("JoinABLe缓存未提供；本次仅使用解析几何接口候选。")
    if not graph["connected"]:
        limitations.append("候选关系图未连接全部输入零件。")
    if pose_status != "valid":
        limitations.append("未找到经OCCT确认无实体穿透的完整位姿。")
    result = {
        "schema_version": "2.0.0",
        "task": "known_group_assembly_relation_recognition",
        "assembly_id": case_dir.name,
        "input_assumption": "all_parts_belong_to_one_assembly",
        "parts": parts,
        "reference_part": search["reference_part"],
        "assembly_connected": bool(graph["connected"]),
        "pose_status": pose_status,
        "direct_connections": connections,
        "assembly_relations": constraints,
        "components": components,
        "unresolved_parts": graph["unresolved_parts"],
        "collision_validation": {
            **exact,
            "checked_pose_count": len(pose_audit),
            "selected_pose_rank": selected_pose_rank,
        },
        "candidate_summary": {
            "raw_constraint_count": len(raw_matches),
            "scored_constraint_count": len(scored),
            "candidate_pair_count": len(pair_candidates),
            "selected_connection_count": len(connections),
            "selection_method": graph["selection_method"],
            "joinable": joinable_audit,
        },
        "limitations": limitations,
    }
    validated = KnownGroupAssemblyResult.model_validate(result).model_dump(mode="json")
    top_k_joint_proposals = _top_k_joint_proposals(pair_candidates, k=8)
    _write(output_dir / "candidate_relations.json", {
        "parts": parts,
        "pair_candidates": [
            {key: value for key, value in row.items() if key != "matches"}
            for row in pair_candidates
        ],
        "top_k_joint_proposals": top_k_joint_proposals,
        "scored_constraints": scored,
        "joinable": joinable_audit,
    })
    _write(output_dir / "top_k_joint_proposals.json", {
        "schema_version": "top_k_joint_proposals.v1",
        "assembly_id": case_dir.name,
        "top_k_per_pair": 8,
        "proposal_count": sum(len(row["top_k"]) for row in top_k_joint_proposals),
        "pair_proposals": top_k_joint_proposals,
    })
    _write(output_dir / "pose_validation.json", {
        "search_status": search["status"],
        "reference_part": search["reference_part"],
        "expanded_states": search["expanded_states"],
        "complete_pose_candidate_count": search["complete_pose_candidate_count"],
        "candidate_augmentation": search.get("candidate_augmentation", {}),
        "group_pose_optimizer": {
            "enabled": True,
            "exact_check_policy": (
                "bounded_top_precheck_candidates; large STEP exact disabled by default"
            ),
            "selected_pose_rank": selected_pose_rank,
        },
        "pose_audit": pose_audit,
        "selected_exact_collision": exact,
    })
    _write(output_dir / "assembly_relations.json", validated)
    _write(
        output_dir / "conservative_pose_output.json",
        _conservative_pose_output(validated, pose_audit),
    )
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--joinable-report")
    parser.add_argument("--beam-width", type=int, default=20)
    args = parser.parse_args()
    result = run_known_group_assembly(
        args.case_dir,
        output_dir=args.output_dir,
        joinable_report=args.joinable_report,
        beam_width=args.beam_width,
    )
    print(json.dumps({
        "assembly_id": result["assembly_id"],
        "assembly_connected": result["assembly_connected"],
        "pose_status": result["pose_status"],
        "direct_connection_count": len(result["direct_connections"]),
    }, ensure_ascii=False))
    return 0 if result["assembly_connected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
