"""Recognize labelled assembly relations for STEP parts known to form one assembly.

This entry point deliberately bypasses mixed-pool membership, provenance,
semantic-review, and forced partitioning logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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
    exact_shape_collisions_solid_broadphase,
    transform_point,
    transform_vector,
)
from pose_search import (
    compose_group_pose_hypotheses,
    enumerate_axis_role_frames,
    load_joinable_pair_pose_candidate_directory,
    matrix_to_placement,
    placement_to_matrix,
    recall_edge_slot_interface_proposals,
    propose_enclosure_bay_placements,
    recall_axial_compound_candidates,
    recall_planar_footprint_proposals,
    validate_axial_compound_pose,
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


# Pair-level planar summaries are invariant across the retained topology
# frontier.  Cache them for one known-group run so a 100k-face carrier is not
# re-normalised once per topology.  ``run_known_group_assembly`` clears this
# private cache before extracting a new input group.
_PLANAR_FOOTPRINT_RECALL_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_AXIAL_COMPOUND_RECALL_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_ENCLOSURE_BAY_RECALL_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_EDGE_SLOT_RECALL_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attach_brep_graph_sidecars(
    features: dict[str, dict[str, Any]],
    step_files: list[Path],
    graph_dir: str | Path | None,
) -> dict[str, Any]:
    """Attach hash-verified topology evidence without using names as labels.

    Filenames only locate a sidecar for the corresponding input STEP.  The
    STEP SHA-256 must be declared and match.  An unverified same-stem sidecar
    is never allowed to contribute topology evidence; all downstream phase
    evidence comes from anonymous B-Rep nodes and local topology.
    """

    audit = {
        "status": "not_provided" if graph_dir is None else "loaded",
        "graph_dir": None if graph_dir is None else str(Path(graph_dir).resolve()),
        "loaded_parts": [],
        "missing_parts": [],
        "rejected_parts": [],
    }
    if graph_dir is None:
        return audit
    root = Path(graph_dir).resolve()
    if not root.is_dir():
        audit["status"] = "unavailable"
        audit["reason"] = "B-Rep graph sidecar directory does not exist."
        return audit
    for step_path in step_files:
        candidates = [
            root / f"{step_path.stem}.brep_graph.json",
            root / f"{step_path.stem}_graph.json",
            root / f"{step_path.stem}.json",
        ]
        graph_path = next((path for path in candidates if path.is_file()), None)
        if graph_path is None:
            audit["missing_parts"].append(step_path.name)
            continue
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            declared_hash = payload.get("source_geometry_sha256")
            if not declared_hash:
                raise ValueError("source_geometry_sha256_missing")
            if str(declared_hash).lower() != _file_sha256(step_path):
                raise ValueError("source_geometry_sha256_mismatch")
            topology = (payload.get("metadata") or {}).get(
                "edge_topology_features"
            ) or {}
            if topology.get("available") is not True:
                raise ValueError("edge_topology_features_unavailable")
            nodes = list(payload.get("nodes") or [])
            if not nodes:
                raise ValueError("brep_graph_nodes_missing")
            features[step_path.name]["nodes"] = nodes
            features[step_path.name]["metadata"] = payload.get("metadata") or {}
            features[step_path.name]["brep_graph_sidecar"] = {
                "path": str(graph_path),
                "hash_verified": True,
                "schema_version": payload.get("schema_version"),
            }
            audit["loaded_parts"].append(step_path.name)
        except Exception as exc:
            audit["rejected_parts"].append({
                "part": step_path.name,
                "graph": str(graph_path),
                "reason": str(exc),
                "exception_type": type(exc).__name__,
            })
    if audit["rejected_parts"]:
        audit["status"] = "partial"
    elif audit["missing_parts"]:
        audit["status"] = "partial"
    elif len(audit["loaded_parts"]) != len(step_files):
        audit["status"] = "partial"
    return audit

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
    dominant_axis_guard_satisfied = (
        not bool(record.get("dominant_axis_guard_required"))
        or record.get("dominant_axis_guard_passed") is True
    )
    if relation == COAXIAL:
        return (
            float(record.get("axis_angle_deg", 999.0)) <= 2.0
            and float(record.get("radial_distance", 999.0)) <= 1.0
            and dominant_axis_guard_satisfied
        )
    if relation == CLEARANCE:
        axial_overlap_ratio = record.get("axial_overlap_ratio")
        return (
            float(record.get("axis_angle_deg", 999.0)) <= 2.0
            and float(record.get("radial_distance", 999.0)) <= 1.0
            and axial_overlap_ratio is not None
            and float(axial_overlap_ratio) >= 0.15
            and dominant_axis_guard_satisfied
        )
    if relation in {PLANAR_MATE, PLANAR_ALIGN}:
        bounded_overlap_ratio = record.get("bounded_overlap_ratio")
        bounded_overlap_satisfied = (
            bounded_overlap_ratio is None
            or float(bounded_overlap_ratio) >= 0.01
        )
        return (
            float(record.get("normal_angle_deg", 999.0)) <= 5.0
            and float(record.get("plane_distance", 999.0)) <= 5.0
            and bounded_overlap_satisfied
        )
    if relation == POCKET_MATE:
        return float(record.get("pocket_center_distance", 999.0)) <= 2.0
    return False


def _pose_closure(candidate: dict, graph: dict, features: dict) -> dict[str, Any]:
    placements = candidate["placements"]
    footprint_history = list(candidate.get("planar_footprint_history") or [])
    if not footprint_history and candidate.get("planar_footprint"):
        footprint_history = [candidate["planar_footprint"]]
    footprint_by_connection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in footprint_history:
        connection_id = row.get("connection_id")
        if connection_id is not None:
            footprint_by_connection[str(connection_id)].append(row)
    compound_history = list(candidate.get("axial_compound_history") or [])
    if not compound_history and candidate.get("axial_compound_interface"):
        compound_history = [candidate["axial_compound_interface"]]
    compound_by_connection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in compound_history:
        connection_id = row.get("connection_id")
        if connection_id is not None:
            compound_by_connection[str(connection_id)].append(row)
    centering_history = list(
        candidate.get("axial_group_centering_history") or []
    )
    if not centering_history and candidate.get("axial_group_centering"):
        centering_history = [candidate["axial_group_centering"]]
    centering_by_connection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in centering_history:
        connection_id = row.get("connection_id")
        if connection_id is not None:
            centering_by_connection[str(connection_id)].append(row)
    centering_required_connection_ids = {
        str(value) for value in (
            candidate.get(
                "axial_group_centering_required_connection_ids"
            ) or []
        )
    }
    enclosure_history = list(candidate.get("enclosure_bay_history") or [])
    if not enclosure_history and candidate.get("enclosure_bay"):
        enclosure_history = [candidate["enclosure_bay"]]
    enclosure_by_connection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enclosure_history:
        connection_id = row.get("connection_id")
        if connection_id is not None:
            enclosure_by_connection[str(connection_id)].append(row)
    edge_slot_history = list(candidate.get("edge_slot_history") or [])
    if not edge_slot_history and candidate.get("edge_slot_interface"):
        edge_slot_history = [candidate["edge_slot_interface"]]
    edge_slot_by_connection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edge_slot_history:
        connection_id = row.get("connection_id")
        if connection_id is not None:
            edge_slot_by_connection[str(connection_id)].append(row)
    containment_pairs = {
        canonical_pair(row.get("parts") or [])
        for row in bbox_collisions(features, placements)
        if row.get("is_strict_containment")
    }
    connection_rows = []
    fully_closed = True
    review_required = False
    for connection in graph["selected"]:
        footprint_rows = footprint_by_connection.get(
            str(connection.get("connection_id")), []
        )
        footprint_supported = any(
            row.get("has_multi_evidence_support") is True
            and int(row.get("independent_evidence_count", 0)) >= 2
            and row.get("proposal_only") is True
            and row.get("can_auto_accept") is False
            for row in footprint_rows
        )
        enclosure_rows = enclosure_by_connection.get(
            str(connection.get("connection_id")), []
        )
        current_enclosure_rows = []
        for row in enclosure_rows:
            stationary = row.get("stationary_part")
            movable = row.get("movable_part")
            expected_transform = row.get("transform_4x4")
            pose_preserved = False
            translation_residual = None
            rotation_residual_degrees = None
            if (
                stationary in placements
                and movable in placements
                and expected_transform is not None
            ):
                try:
                    stationary_world = placement_to_matrix(
                        placements[stationary]
                    )
                    movable_world = placement_to_matrix(placements[movable])
                    current_relative = np.linalg.inv(
                        stationary_world
                    ) @ movable_world
                    expected = np.asarray(expected_transform, dtype=float)
                    translation_residual = float(np.linalg.norm(
                        current_relative[:3, 3] - expected[:3, 3]
                    ))
                    rotation_delta = (
                        expected[:3, :3].T @ current_relative[:3, :3]
                    )
                    cosine = max(-1.0, min(
                        1.0,
                        (float(np.trace(rotation_delta)) - 1.0) * 0.5,
                    ))
                    rotation_residual_degrees = math.degrees(
                        math.acos(cosine)
                    )
                    pose_preserved = (
                        translation_residual <= 1e-3
                        and rotation_residual_degrees <= 0.05
                    )
                except Exception:
                    pose_preserved = False
            supported = (
                pose_preserved
                and int(row.get("independent_evidence_count", 0)) >= 4
                and row.get("proposal_only") is True
                and row.get("can_auto_accept") is False
            )
            current_enclosure_rows.append({
                "candidate_id": row.get("candidate_id"),
                "slot_index": row.get("slot_index"),
                "depth_polarity": row.get("depth_polarity"),
                "independent_evidence_count": row.get(
                    "independent_evidence_count"
                ),
                "evidence": row.get("evidence") or [],
                "functional_body_derivation_status": row.get(
                    "functional_body_derivation_status"
                ),
                "functional_body_excluded_protrusion_risk": row.get(
                    "functional_body_excluded_protrusion_risk"
                ),
                "proposal_only": bool(row.get("proposal_only")),
                "review_required": bool(row.get("review_required")),
                "pose_preserved": pose_preserved,
                "translation_residual_mm": translation_residual,
                "rotation_residual_degrees": rotation_residual_degrees,
                "supported": supported,
            })
        enclosure_supported = any(
            row["supported"] for row in current_enclosure_rows
        )
        edge_slot_rows = edge_slot_by_connection.get(
            str(connection.get("connection_id")), []
        )
        current_edge_slot_rows = []
        for row in edge_slot_rows:
            stationary = row.get("stationary_part")
            movable = row.get("movable_part")
            expected_transform = row.get("transform_matrix")
            pose_preserved = False
            translation_residual = None
            rotation_residual_degrees = None
            if (
                stationary in placements
                and movable in placements
                and expected_transform is not None
            ):
                try:
                    stationary_world = placement_to_matrix(placements[stationary])
                    movable_world = placement_to_matrix(placements[movable])
                    current_relative = np.linalg.inv(stationary_world) @ movable_world
                    expected = np.asarray(expected_transform, dtype=float)
                    translation_residual = float(np.linalg.norm(
                        current_relative[:3, 3] - expected[:3, 3]
                    ))
                    rotation_delta = expected[:3, :3].T @ current_relative[:3, :3]
                    cosine = max(-1.0, min(
                        1.0,
                        (float(np.trace(rotation_delta)) - 1.0) * 0.5,
                    ))
                    rotation_residual_degrees = math.degrees(math.acos(cosine))
                    pose_preserved = (
                        translation_residual <= 1e-3
                        and rotation_residual_degrees <= 0.05
                    )
                except Exception:
                    pose_preserved = False
            supported = (
                pose_preserved
                and int(row.get("independent_evidence_count", 0)) >= 4
                and row.get("has_multi_evidence_support") is True
                and row.get("proposal_only") is True
                and row.get("review_required") is True
                and row.get("can_auto_accept") is False
            )
            current_edge_slot_rows.append({
                "slot_family_id": row.get("slot_family_id"),
                "slot_family_size": row.get("slot_family_size"),
                "slot_rank": row.get("slot_rank"),
                "floor_plane_index": row.get("floor_plane_index"),
                "wall_plane_indices": row.get("wall_plane_indices") or [],
                "channel_gap": row.get("channel_gap"),
                "slot_pitch": row.get("slot_pitch"),
                "independent_evidence_count": row.get(
                    "independent_evidence_count"
                ),
                "evidence_families": row.get("evidence_families") or [],
                "proposal_only": bool(row.get("proposal_only")),
                "review_required": bool(row.get("review_required")),
                "pose_preserved": pose_preserved,
                "translation_residual_mm": translation_residual,
                "rotation_residual_degrees": rotation_residual_degrees,
                "supported": supported,
            })
        edge_slot_supported = any(
            row["supported"] for row in current_edge_slot_rows
        )
        compound_rows = compound_by_connection.get(
            str(connection.get("connection_id")), []
        )
        current_compound_rows: list[dict[str, Any]] = []
        for row in compound_rows:
            fixed = row.get("fixed_part")
            moving = row.get("moving_part")
            interface = row.get("compound_candidate")
            proposal = row.get("compound_proposal")
            if (
                fixed not in placements
                or moving not in placements
                or not isinstance(interface, dict)
                or not isinstance(proposal, dict)
            ):
                continue
            try:
                fixed_world = placement_to_matrix(placements[fixed])
                moving_world = placement_to_matrix(placements[moving])
                relative = np.linalg.inv(fixed_world) @ moving_world
                validation = validate_axial_compound_pose(
                    interface,
                    proposal,
                    collision_free=True,
                    transform=relative,
                )
                validation["collision_free"] = None
                validation["collision_scope"] = "deferred_to_exact_occt"
                validation["is_closed"] = False
                physical_closed = _compound_physical_constraints_satisfied(
                    validation, proposal
                )
            except Exception as exc:
                validation = {
                    "compound_constraints_satisfied": False,
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                }
                physical_closed = False
            current_compound_rows.append({
                "candidate_id": row.get("candidate_id"),
                "proposal_id": row.get("proposal_id"),
                "axis_polarity": row.get("axis_polarity"),
                "phase_convention": row.get("phase_convention"),
                "phase_orbit_degrees": row.get("phase_orbit_degrees"),
                "whole_part_symmetry_order": row.get(
                    "whole_part_symmetry_order"
                ),
                "phase_status": row.get("phase_status"),
                "phase_witness": row.get("phase_witness") or [],
                "proposal_only": bool(row.get("proposal_only")),
                "review_required": bool(row.get("review_required")),
                "compound_constraints_satisfied": physical_closed,
                "current_pose_validation": validation,
            })
        centering_rows = centering_by_connection.get(
            str(connection.get("connection_id")), []
        )
        current_centering_rows: list[dict[str, Any]] = []
        for row in centering_rows:
            support_rows = compound_by_connection.get(
                str(row.get("support_connection_id")), []
            )
            shaft_rows = compound_by_connection.get(
                str(row.get("connection_id")), []
            )
            support_rows = [
                candidate_row for candidate_row in support_rows
                if candidate_row.get("candidate_id")
                == row.get("support_candidate_id")
                and candidate_row.get("proposal_id")
                == row.get("support_proposal_id")
                and candidate_row.get("fixed_part")
                in set(row.get("support_parts") or [])
                and candidate_row.get("moving_part")
                in set(row.get("support_parts") or [])
            ]
            shaft_rows = [
                candidate_row for candidate_row in shaft_rows
                if candidate_row.get("candidate_id")
                == row.get("shaft_candidate_id")
                and candidate_row.get("proposal_id")
                == row.get("shaft_proposal_id")
                and row.get("shaft_part")
                in {
                    candidate_row.get("fixed_part"),
                    candidate_row.get("moving_part"),
                }
            ]
            diagnostics = None
            for support_row in support_rows:
                for shaft_row in shaft_rows:
                    current = _axial_group_centering_diagnostics(
                        support_row,
                        shaft_row,
                        placements,
                        features,
                        graph,
                    )
                    if diagnostics is None or current.get("supported") is True:
                        diagnostics = current
                    if current.get("supported") is True:
                        break
                if diagnostics and diagnostics.get("supported") is True:
                    break
            if diagnostics is None:
                diagnostics = {
                    "schema_version": "axial_group_centering.v1",
                    "pattern_detected": False,
                    "supported": False,
                    "reason": "referenced_compound_history_missing",
                    "proposal_only": True,
                    "review_required": True,
                    "can_auto_accept": False,
                }
            current_centering_rows.append({
                **diagnostics,
                "propagated_rigid_dependents": row.get(
                    "propagated_rigid_dependents"
                ) or [],
            })
        group_centering_supported = any(
            row.get("supported") is True for row in current_centering_rows
        )
        supported_compound_rows = [
            row for row in current_compound_rows
            if row["compound_constraints_satisfied"]
        ]
        compound_supported = bool(supported_compound_rows)
        compound_review_required = bool(
            compound_rows
            and (
                not compound_supported
                or all(
                    row["proposal_only"] or row["review_required"]
                    for row in supported_compound_rows
                )
            )
        )
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
        compound_clearance_overlap_supported = bool(
            CLEARANCE not in available or by_type[CLEARANCE]
        )
        pair = canonical_pair(connection.get("parts") or [])
        planar_evidence_count = sum(
            bool(by_type[kind]) for kind in {PLANAR_MATE, PLANAR_ALIGN}
        )
        containment_fallback = (
            pair in containment_pairs
            and planar_evidence_count >= 2
        )
        # When pocket evidence is part of the selected relation, planar contact
        # alone is not enough; otherwise a fan/block can sit near a face and be
        # incorrectly accepted as inserted.
        if pocket_available:
            closed = (
                pocket_satisfied and (not planar_available or planar_satisfied)
            ) or containment_fallback
            connection_review_required = containment_fallback and not pocket_satisfied
        # When both axial and seating evidence exist, both must close.  This
        # prevents a coaxial-but-separated flange pose from being called valid.
        elif axial_available and planar_available:
            closed = axial_satisfied and planar_satisfied
            connection_review_required = False
        else:
            closed = any_satisfied
            connection_review_required = False
        if footprint_supported:
            # A dimension-matched support face plus a co-centred parallel
            # layer is sufficient to send this pose to exact physical
            # validation.  It is never an automatic semantic/functional
            # acceptance: the source proposal guard forces human review.
            closed = True
            connection_review_required = True
        centering_required = (
            str(connection.get("connection_id"))
            in centering_required_connection_ids
            or bool(centering_rows)
        )
        if compound_rows and centering_required:
            # A small axial body spanning two mirrored/opposed supports must
            # retain group-level centre and two-sided overlap evidence.  Its
            # original one-end-face compound stop is intentionally stale after
            # centring and must not be used as a fallback closure.
            closed = group_centering_supported
            connection_review_required = True
        elif compound_rows:
            # The candidate was generated from a compound interface, so later
            # residual refinements must preserve all of its axis/end-face/phase
            # constraints.  Never fall back to an axial-only closure if the
            # compound residual has become stale.
            # A compound end-face stop does not by itself prove that a
            # clearance/shaft-hole relation is inserted.  The selected
            # clearance edge must also retain finite axial overlap.
            closed = (
                compound_supported
                and compound_clearance_overlap_supported
            )
            connection_review_required = bool(
                compound_review_required
                or not compound_clearance_overlap_supported
            )
        elif centering_required:
            closed = False
            connection_review_required = True
        if enclosure_rows:
            # Repeated opposing walls, repeated rails/supports and a derived
            # functional-body fit can close this physical insertion edge for
            # exact validation.  The transform must still be the recalled
            # transform; later generic refinements cannot inherit stale bay
            # evidence.  Functional correctness remains review-only.
            closed = enclosure_supported
            connection_review_required = True
        if edge_slot_rows:
            # Repeated bounded channel floors and mirrored walls support a
            # physical edge insertion proposal.  Functional correctness is
            # deliberately not inferred, so this path is always review-only.
            closed = edge_slot_supported
            connection_review_required = True
        fully_closed = fully_closed and closed
        review_required = review_required or connection_review_required
        satisfied_relation_types = sorted(
            kind for kind, satisfied in by_type.items() if satisfied
        )
        if footprint_supported:
            satisfied_relation_types.append(
                "planar_footprint_multi_evidence"
            )
        if compound_supported:
            satisfied_relation_types.append(
                "compound_coaxial_radial_end_face_phase"
            )
        if group_centering_supported:
            satisfied_relation_types.append(
                "group_centered_two_sided_axial_insertion"
            )
        if enclosure_supported:
            satisfied_relation_types.append(
                "repeated_enclosure_bay_multi_evidence"
            )
        if edge_slot_supported:
            satisfied_relation_types.append(
                "repeated_bounded_edge_slot_multi_evidence"
            )
        connection_rows.append({
            "connection_id": connection["connection_id"],
            "parts": connection["parts"],
            "closed": closed,
            "satisfied_relation_types": satisfied_relation_types,
            "available_relation_types": sorted(available),
            "closure_evidence": (
                "axial_group_centered_two_sided_insertion"
                if group_centering_supported
                else "axial_group_centering_residual_failed"
                if centering_required
                else "axial_compound_interface"
                if compound_supported and compound_clearance_overlap_supported
                else "axial_compound_clearance_overlap_failed"
                if compound_supported
                else "axial_compound_residual_failed"
                if compound_rows
                else "repeated_enclosure_bay_multi_evidence"
                if enclosure_supported
                else "enclosure_bay_pose_residual_failed"
                if enclosure_rows
                else "repeated_bounded_edge_slot_multi_evidence"
                if edge_slot_supported
                else "edge_slot_pose_residual_failed"
                if edge_slot_rows
                else "planar_footprint_multi_evidence_proposal"
                if footprint_supported
                else "multi_planar_plus_strict_containment"
                if connection_review_required
                else "selected_interface_residuals"
            ),
            "review_required": connection_review_required,
            "planar_footprint_evidence": [
                {
                    "proposal_id": row.get("proposal_id"),
                    "equivalence_class_id": row.get(
                        "equivalence_class_id"
                    ),
                    "independent_evidence_count": row.get(
                        "independent_evidence_count"
                    ),
                    "phase_degrees": row.get("phase_degrees"),
                    "support_polarity": row.get("support_polarity"),
                }
                for row in footprint_rows
            ],
            "axial_compound_evidence": current_compound_rows,
            "axial_group_centering_evidence": current_centering_rows,
            "enclosure_bay_evidence": current_enclosure_rows,
            "edge_slot_evidence": current_edge_slot_rows,
        })
    closed_count = sum(row["closed"] for row in connection_rows)
    return {
        "fully_closed": fully_closed,
        "closed_connection_count": closed_count,
        "connection_count": len(connection_rows),
        "closure_ratio": closed_count / max(1, len(connection_rows)),
        "connections": connection_rows,
        "review_required": review_required,
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
            # A strict small-in-large containment is an insertion proposal, not
            # a material collision.  It receives only a tiny ranking cost and
            # still requires exact OCCT plus the downstream precision gate.
            if item.get("is_strict_containment"):
                weight = 0.02
                kind = "selected_pair_containment"
            else:
                # Closed selected pairs can have AABB overlap for valid insertion.
                weight = 0.15 if pair in closed_pairs else 0.60
                kind = "selected_pair_overlap"
            selected_overlap += weight * min(1.0, ratio)
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
    side_consistency = candidate.get("carrier_open_side_consistency") or {}
    # This is only a bounded ranking preference between otherwise feasible
    # review candidates.  It is intentionally not an acceptance condition and
    # cannot compensate for missing closure, contact, or exact validation.
    side_consistency_bonus = (
        0.12 if side_consistency.get("supported") is True else 0.0
    )
    score = (
        4.0 * closure_ratio
        + 1.5 * float(contact.get("contact_support_score", 0.0))
        - 1.2 * float(overlap.get("bbox_overlap_cost", 0.0))
        + 0.05 * float(candidate.get("total_score", 0.0))
        + side_consistency_bonus
    )
    return {
        "constraint_closure": closure,
        "contact_objective": contact,
        "overlap_objective": overlap,
        "group_pose_precheck_score": score,
        "carrier_open_side_consistency_bonus": side_consistency_bonus,
    }


def _exact_status_rank(exact: dict[str, Any]) -> int:
    if exact.get("status") == "success" and not exact.get("collisions"):
        return 3
    if exact.get("status") == "success":
        return 1
    return 0


def _exact_collision_risk(exact: dict[str, Any]) -> tuple[float, int, float]:
    collisions = exact.get("collisions") or []
    return (
        sum(
            max(0.0, float(row.get("minimum_part_volume_ratio", 0.0)))
            for row in collisions
        ),
        sum(
            max(1, int(row.get("solid_intersection_count", 1)))
            for row in collisions
        ),
        sum(
            max(0.0, float(row.get("intersection_volume_mm3", 0.0)))
            for row in collisions
        ),
    )


def _localized_interference_review(
    exact: dict[str, Any],
    closure: dict[str, Any],
) -> dict[str, Any]:
    ratio, pair_count, volume = _exact_collision_risk(exact)
    eligible = (
        exact.get("status") == "success"
        and bool(exact.get("collisions"))
        and bool(closure.get("fully_closed"))
        and bool(closure.get("review_required"))
        and ratio <= 0.002
        and pair_count <= 2
    )
    return {
        "eligible_for_review": eligible,
        "component_volume_ratio_sum": ratio,
        "positive_solid_pair_count": pair_count,
        "intersection_volume_mm3": volume,
        "reason": (
            "Localized low-ratio interference on a containment-derived pose; "
            "possible compliant clip or CAD interference remains human-review only."
            if eligible
            else "Collision exceeds the localized-interference review gate."
        ),
        "can_auto_accept": False,
    }


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
    precision = result.get("precision_pose_validation")
    if not isinstance(precision, dict):
        precision = selected.get("precision_pose_validation")
    if not isinstance(precision, dict):
        precision = {
            "precision_status": "not_checked",
            "review_required": True,
            "reason": "precision_pose_validation_missing",
        }
    group_record = {
        "assembly_id": result.get("assembly_id"),
        "parts": result.get("parts", []),
        "pose_status": result.get("pose_status"),
        "assembly_connected": result.get("assembly_connected"),
        "selected_pose_rank": selected_rank,
        "selected_candidate_origin": selected.get("candidate_origin"),
        "selected_candidate_proposal_only": bool(
            selected.get("proposal_only")
        ),
        "selected_candidate_review_required": bool(
            selected.get("review_required")
        ),
        "selected_candidate_can_auto_accept": selected.get(
            "can_auto_accept"
        ),
        "obb_insertion": selected.get("obb_insertion"),
        "obb_insertion_history": selected.get("obb_insertion_history"),
        "planar_footprint": selected.get("planar_footprint"),
        "planar_footprint_history": selected.get(
            "planar_footprint_history"
        ),
        "axial_compound_interface": selected.get(
            "axial_compound_interface"
        ),
        "axial_compound_history": selected.get(
            "axial_compound_history"
        ),
        "axial_group_centering": selected.get(
            "axial_group_centering"
        ),
        "axial_group_centering_history": selected.get(
            "axial_group_centering_history"
        ),
        "enclosure_bay": selected.get("enclosure_bay"),
        "enclosure_bay_history": selected.get(
            "enclosure_bay_history"
        ),
        "edge_slot_interface": selected.get("edge_slot_interface"),
        "edge_slot_history": selected.get("edge_slot_history"),
        "carrier_open_side_consistency": selected.get(
            "carrier_open_side_consistency"
        ),
        "carrier_open_side_consistency_history": selected.get(
            "carrier_open_side_consistency_history"
        ),
        "group_pose_final_score": selected.get("group_pose_final_score"),
        "closure": selected.get("constraint_closure"),
        "contact_objective": selected.get("contact_objective"),
        "overlap_objective": selected.get("overlap_objective"),
        "collision_validation": result.get("collision_validation", {}),
        "precision_pose_validation": precision,
        "direct_connections": [
            {
                "connection_id": row.get("connection_id"),
                "parts": row.get("parts"),
                "primary_relation_type": row.get("primary_relation_type"),
                "assembly_method_relation_types": row.get("assembly_method_relation_types"),
                "constraint_closed_in_selected_pose": row.get("constraint_closed_in_selected_pose"),
                "closure_evidence": row.get("closure_evidence"),
                "axial_compound_evidence": row.get(
                    "axial_compound_evidence"
                ) or [],
                "axial_group_centering_evidence": row.get(
                    "axial_group_centering_evidence"
                ) or [],
                "enclosure_bay_evidence": row.get(
                    "enclosure_bay_evidence"
                ) or [],
                "review_required": row.get("review_required"),
            }
            for row in result.get("direct_connections", [])
        ],
    }
    accepted, review, rejected = [], [], []
    reasons = []
    collision = result.get("collision_validation", {})
    proposal_guard = bool(
        selected.get("proposal_only")
        or selected.get("review_required")
        or selected.get("can_auto_accept") is False
        or (selected.get("obb_insertion") or {}).get("review_required")
        or (selected.get("planar_footprint") or {}).get(
            "review_required"
        )
        or (selected.get("axial_compound_interface") or {}).get(
            "review_required"
        )
        or (selected.get("enclosure_bay") or {}).get(
            "review_required"
        )
        or (selected.get("constraint_closure") or {}).get(
            "review_required"
        )
    )
    if not result.get("assembly_connected"):
        review.append(group_record)
        reasons.append("assembly_graph_not_connected")
    elif proposal_guard:
        review.append(group_record)
        reasons.append("proposal_only_pose_requires_human_review")
    elif (
        result.get("pose_status") == "valid"
        and precision.get("precision_status", precision.get("status"))
        == "valid"
    ):
        accepted.append(group_record)
        reasons.append("full_closure_exact_collision_and_precision_gate_valid")
    elif result.get("pose_status") == "valid":
        review.append(group_record)
        reasons.append(
            str(precision.get("reason") or "precision_pose_validation_not_valid")
        )
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
        "policy": (
            "accepted_requires_connected_full_closure_exact_collision_and_"
            "multi_evidence_precision_gate"
        ),
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


def _portable_components(
    components: list[dict[str, Any]],
    source_dir: str | Path,
    manifest_dir: str | Path,
) -> list[dict[str, Any]]:
    """Make component sources portable from the actual manifest directory.

    The previous implementation assumed every output directory was a direct
    child of the input case and blindly emitted ``../<part>``.  Independent
    frozen-exam directories therefore pointed at nonexistent STEP files.
    """

    source_root = Path(source_dir).resolve()
    manifest_root = Path(manifest_dir).resolve()
    result = []
    for component in components:
        item = dict(component)
        source = Path(str(item["source"]))
        if not source.is_absolute():
            source = source_root / source.name
        item["source"] = Path(
            os.path.relpath(source.resolve(), manifest_root)
        ).as_posix()
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


def _rigid_attachment_cluster(
    root: str,
    graph: dict[str, Any],
    *,
    blocked: set[str] | None = None,
) -> list[str]:
    """Return the bounded non-axial attachment subtree rooted at ``root``.

    Pose recall often moves an axial carrier after a small prismatic child has
    already been seated on it.  Moving only the carrier destroys that relative
    placement.  We therefore propagate the same world-space rigid delta across
    edges supported exclusively by planar/pocket evidence.  Axial edges are
    never crossed, and callers can block other independently sampled roots.

    A single planar edge is still weak evidence: propagation only preserves a
    proposal and does not make the edge accepted.  The group closure gate keeps
    such a connection review-only.
    """

    blocked = set(blocked or set())
    blocked.discard(root)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for connection in graph.get("selected") or []:
        parts = list(connection.get("parts") or [])
        relation_types = set(connection.get("relation_types") or [])
        if len(parts) != 2:
            continue
        if relation_types & {COAXIAL, CLEARANCE}:
            continue
        if not relation_types & {PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE}:
            continue
        left, right = parts
        adjacency[left].add(right)
        adjacency[right].add(left)

    cluster: list[str] = []
    pending = [root]
    visited = set(blocked)
    while pending:
        part = pending.pop()
        if part in visited:
            continue
        visited.add(part)
        cluster.append(part)
        pending.extend(sorted(adjacency.get(part, set()) - visited, reverse=True))
    return cluster


def _set_placement_with_rigid_dependents(
    placements: dict[str, Any],
    root: str,
    new_root_placement: dict[str, Any],
    graph: dict[str, Any],
    *,
    blocked: set[str] | None = None,
) -> list[str]:
    """Set one root placement and apply its world delta to attached children."""

    if root not in placements:
        return []
    old_root = placement_to_matrix(placements[root])
    new_root = placement_to_matrix(new_root_placement)
    delta = new_root @ np.linalg.inv(old_root)
    cluster = _rigid_attachment_cluster(root, graph, blocked=blocked)
    for part in cluster:
        if part not in placements:
            continue
        if part == root:
            placements[part] = json.loads(json.dumps(new_root_placement))
            continue
        placements[part] = matrix_to_placement(
            delta @ placement_to_matrix(placements[part])
        )
    return cluster


def _main_cylinder_radius(part_features: dict[str, Any]) -> float | None:
    cylinders = list(part_features.get("cylinders") or [])
    if not cylinders:
        return None
    radius = max(float(row.get("radius", 0.0)) for row in cylinders)
    return radius if radius > 0.0 else None


def _world_bbox_axis_interval(
    part_features: dict[str, Any],
    placement: dict[str, Any],
    axis_world: list[float],
) -> tuple[float, float] | None:
    """Project all transformed AABB corners onto one world-space axis."""

    bbox = part_features.get("bbox") or {}
    low = bbox.get("min")
    high = bbox.get("max")
    if not low or not high or len(low) != 3 or len(high) != 3:
        return None
    axis = np.asarray(_unit(axis_world), dtype=float)
    values = []
    for x in (float(low[0]), float(high[0])):
        for y in (float(low[1]), float(high[1])):
            for z in (float(low[2]), float(high[2])):
                point = np.asarray(
                    transform_point([x, y, z], placement), dtype=float
                )
                values.append(float(np.dot(point, axis)))
    return min(values), max(values)


def _axial_terminal_face_interval(
    part_features: dict[str, Any],
    placement: dict[str, Any],
    axis_world: list[float],
) -> tuple[float, float] | None:
    """Find a conservative pair of opposed terminal faces for a shaft-like part.

    Only planes normal to the dominant cylinder axis are considered.  The pair
    must have opposed normals, comparable areas, and substantial separation.
    This deliberately abstains on ambiguous stepped bodies instead of treating
    a shoulder or keyway end wall as a terminal face.
    """

    local_axis = _main_cylinder_axis(part_features)
    if local_axis is None:
        return None
    local_axis_np = np.asarray(_unit(local_axis), dtype=float)
    aligned: list[dict[str, Any]] = []
    for plane in part_features.get("planes") or []:
        normal = plane.get("normal")
        position = plane.get("position") or plane.get("centroid")
        area = float(plane.get("area", 0.0))
        if not normal or not position or area <= 0.0:
            continue
        normal_np = np.asarray(_unit(normal), dtype=float)
        if abs(float(np.dot(normal_np, local_axis_np))) < math.cos(
            math.radians(2.0)
        ):
            continue
        aligned.append({
            "normal": normal_np,
            "position": np.asarray(position, dtype=float),
            "area": area,
        })
    if len(aligned) < 2:
        return None
    max_area = max(row["area"] for row in aligned)
    eligible = [row for row in aligned if row["area"] >= 0.25 * max_area]
    if len(eligible) < 2:
        return None
    coordinates = [
        float(np.dot(row["position"], local_axis_np)) for row in eligible
    ]
    minimum = min(coordinates)
    maximum = max(coordinates)
    if maximum - minimum <= 1e-3:
        return None
    tolerance = max(1e-6, 1e-5 * (maximum - minimum))
    low_rows = [
        row for row, coordinate in zip(eligible, coordinates)
        if abs(coordinate - minimum) <= tolerance
    ]
    high_rows = [
        row for row, coordinate in zip(eligible, coordinates)
        if abs(coordinate - maximum) <= tolerance
    ]
    left = max(low_rows, key=lambda row: row["area"])
    right = max(high_rows, key=lambda row: row["area"])
    if float(np.dot(left["normal"], right["normal"])) > -math.cos(
        math.radians(2.0)
    ):
        return None
    area_ratio = min(left["area"], right["area"]) / max(
        left["area"], right["area"]
    )
    if area_ratio < 0.80:
        return None
    axis = np.asarray(_unit(axis_world), dtype=float)
    projected = [
        float(np.dot(
            np.asarray(transform_point(row["position"].tolist(), placement)),
            axis,
        ))
        for row in (left, right)
    ]
    return min(projected), max(projected)


def _axial_support_face_interval(
    part_features: dict[str, Any],
    placement: dict[str, Any],
    axis_world: list[float],
) -> tuple[float, float] | None:
    """Project significant dominant-axis end/shoulder planes for a support."""

    local_axis = _main_cylinder_axis(part_features)
    if local_axis is None:
        return None
    local_axis_np = np.asarray(_unit(local_axis), dtype=float)
    aligned = []
    for plane in part_features.get("planes") or []:
        normal = plane.get("normal")
        position = plane.get("position") or plane.get("centroid")
        area = float(plane.get("area", 0.0))
        if not normal or not position or area <= 0.0:
            continue
        if abs(float(np.dot(
            np.asarray(_unit(normal), dtype=float), local_axis_np
        ))) < math.cos(math.radians(2.0)):
            continue
        aligned.append((area, position))
    if len(aligned) < 2:
        return None
    max_area = max(area for area, _ in aligned)
    significant = [
        position for area, position in aligned
        if area >= max(1.0, 0.08 * max_area)
    ]
    if len(significant) < 2:
        return None
    axis = np.asarray(_unit(axis_world), dtype=float)
    values = [
        float(np.dot(
            np.asarray(transform_point(position, placement), dtype=float),
            axis,
        ))
        for position in significant
    ]
    if max(values) - min(values) <= 1e-3:
        return None
    return min(values), max(values)


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
    inherited_proposal_guard = {
        key: source_candidate[key]
        for key in (
            "proposal_only",
            "review_required",
            "can_auto_accept",
            "obb_insertion",
            "obb_insertion_history",
            "obb_refined_connection_ids",
            "planar_footprint",
            "planar_footprint_history",
            "planar_footprint_refined_connection_ids",
            "axial_compound_interface",
            "axial_compound_history",
            "axial_compound_refined_connection_ids",
            "axial_group_centering",
            "axial_group_centering_history",
            "axial_group_centering_required_connection_ids",
            "enclosure_bay",
            "enclosure_bay_history",
            "enclosure_bay_refined_connection_ids",
            "edge_slot_interface",
            "edge_slot_history",
            "edge_slot_refined_connection_ids",
            "carrier_open_side_consistency",
            "carrier_open_side_consistency_history",
        )
        if key in source_candidate
    }
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
        **inherited_proposal_guard,
        **extra,
    }


def _inject_joinable_group_pose_seed(
    search: dict[str, Any],
    features: dict[str, Any],
    pose_report_root: str | Path | None,
) -> dict[str, Any]:
    """Compose cached pair poses into one generic group-pose proposal.

    The proposal is admitted only when all parts are connected and all pair
    cycles are consistent.  It still goes through the same closure, overlap and
    exact OCCT checks as every analytic pose candidate.
    """

    if pose_report_root is None:
        return search
    seeds = load_joinable_pair_pose_candidate_directory(
        pose_report_root, limit_per_pair=8
    )
    hypotheses = compose_group_pose_hypotheses(
        features.keys(),
        seeds,
        maximum_candidates_per_pair=4,
        maximum_combinations=64,
    )
    updated = dict(search)
    updated["joinable_group_pose_composition"] = {
        "pair_pose_candidate_count": len(seeds),
        "group_hypothesis_count": len(hypotheses),
        "complete_hypothesis_count": sum(
            row["status"] == "complete" for row in hypotheses
        ),
        "review_hypothesis_count": sum(
            row["review_required"] for row in hypotheses
        ),
        "hypotheses": hypotheses,
    }
    complete = [row for row in hypotheses if row["status"] == "complete"]
    if not complete:
        return updated
    candidates = []
    for index, composition in enumerate(complete):
        candidates.append(_candidate_from_placements(
            features,
            composition["placements"],
            search,
            origin="joinable_pair_pose_composition",
            score_penalty=0.02 + 0.001 * index,
            extra={
                "joinable_group_pose": {
                    "combination_index": composition["combination_index"],
                    "reference_part": composition["reference_part"],
                    "usable_pair_seed_count": composition[
                        "usable_pair_seed_count"
                    ],
                    "pair_seed_sources": composition["pair_seed_sources"],
                    "pair_seed_score_sum": composition[
                        "pair_seed_score_sum"
                    ],
                    "cycle_checks": composition["cycle_checks"],
                }
            },
        ))
    updated["pose_candidates"] = (
        list(search.get("pose_candidates") or []) + candidates
    )
    return updated


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
            propagated_dependents: dict[str, list[str]] = {}
            blocked_roots = set(movable) | {center}
            for part, offset in zip(movable, grid):
                if part not in placements:
                    continue
                shifted = _shift_placement_along_axis(
                    placements[part], axis, float(offset)
                )
                cluster = _set_placement_with_rigid_dependents(
                    placements,
                    part,
                    shifted,
                    graph,
                    blocked=blocked_roots - {part},
                )
                propagated_dependents[part] = [
                    member for member in cluster if member != part
                ]
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
                    "propagated_rigid_dependents": propagated_dependents,
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
                    flipped = _flip_axis_direction(
                        oriented_placements[part],
                        axis,
                        [float(value) for value in pivot_local],
                    )
                    cluster = _set_placement_with_rigid_dependents(
                        oriented_placements,
                        part,
                        flipped,
                        graph,
                        blocked=blocked_roots - {part},
                    )
                    propagated_dependents[part] = [
                        member for member in cluster if member != part
                    ]
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
                        "propagated_rigid_dependents": propagated_dependents,
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
            stationary_feature.get("centroid")
            or stationary_feature.get("point")
            or stationary_feature.get("position", [0, 0, 0]),
            placements.get(stationary, {}),
        )
        return (
            [float(value) for value in movable_feature.get("normal", [0, 0, 1])],
            [
                float(value) for value in (
                    movable_feature.get("centroid")
                    or movable_feature.get("point")
                    or movable_feature.get("position", [0, 0, 0])
                )
            ],
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


def _joinable_parameter_jobs(
    matches: list[dict[str, Any]],
    offset_factors: list[float],
    rotation_angles: list[float],
) -> list[tuple[dict[str, Any], bool, float, float]]:
    """Interleave feature, offset, flip, and rotation probes fairly.

    A small per-edge quota must still see more than the first flip/rotation
    stratum.  Ordering the finite lattice by its maximum complexity band gives
    the best match at the identity parameters first, then quickly introduces a
    second match, signed offsets, a flip, and a 180-degree rotation.
    """

    magnitudes = sorted({abs(float(value)) for value in offset_factors})
    magnitude_band = {value: index for index, value in enumerate(magnitudes)}

    def rotation_band(angle: float) -> int:
        normalized = abs(float(angle)) % 360.0
        normalized = min(normalized, 360.0 - normalized)
        if normalized <= 1e-9:
            return 0
        if abs(normalized - 180.0) <= 1e-9:
            return 1
        return 2

    jobs = []
    for match_rank, match in enumerate(matches[:4]):
        for offset_factor in offset_factors:
            offset_rank = magnitude_band[abs(float(offset_factor))]
            for flip in (False, True):
                flip_rank = int(flip)
                for rotation_degrees in rotation_angles:
                    rotate_rank = rotation_band(rotation_degrees)
                    complexity = max(
                        match_rank, offset_rank, flip_rank, rotate_rank
                    )
                    jobs.append((
                        (
                            complexity,
                            match_rank + offset_rank + flip_rank + rotate_rank,
                            match_rank,
                            offset_rank,
                            flip_rank,
                            rotate_rank,
                            abs(float(offset_factor)),
                            float(offset_factor),
                            float(rotation_degrees),
                        ),
                        match,
                        flip,
                        float(offset_factor),
                        float(rotation_degrees),
                    ))
    jobs.sort(key=lambda row: row[0])
    return [row[1:] for row in jobs]


def _joinable_pose_parameter_candidates_for_connection(
    source_candidates: list[dict[str, Any]],
    connection: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int,
    refinement_phase: str,
) -> list[dict[str, Any]]:
    """Generate a bounded residual-pose frontier for exactly one graph edge."""

    if max_candidates <= 0:
        return []
    generated: list[dict[str, Any]] = []
    axial_degree: dict[str, int] = defaultdict(int)
    for selected_connection in graph.get("selected") or []:
        if set(selected_connection.get("relation_types") or []) & {COAXIAL, CLEARANCE}:
            for part in selected_connection.get("parts") or []:
                axial_degree[part] += 1
    central_axial_parts = {
        part for part, degree in axial_degree.items()
        if degree >= 2 and _main_cylinder_axis(features.get(part, {}))
    }
    active_sources = source_candidates[: min(12, max_candidates)]
    source_quotas = _fair_quotas(max_candidates, len(active_sources))
    for source_candidate, source_quota in zip(active_sources, source_quotas):
        if source_quota <= 0:
            continue
        generated_for_source = 0
        placements = source_candidate.get("placements") or {}
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
        for match, flip, offset_factor, rotation_degrees in _joinable_parameter_jobs(
            matches, offset_factors, rotation_angles
        ):
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
            flipped_axis = (
                [-value for value in target_axis_world]
                if flip else target_axis_world
            )
            offset = float(offset_factor) * offset_scale
            new_placements = json.loads(json.dumps(placements))
            new_movable_placement = _placement_from_joinable_axis_parameters(
                movable_axis_local,
                movable_origin_local,
                flipped_axis,
                target_origin_world,
                offset=offset,
                rotation_degrees=rotation_degrees,
            )
            propagated_cluster = _set_placement_with_rigid_dependents(
                new_placements,
                movable,
                new_movable_placement,
                graph,
                blocked={stationary},
            )
            normalized_offset = abs(offset) / max(offset_scale, 1.0)
            detail = {
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
                "refinement_phase": refinement_phase,
                "propagated_rigid_dependents": [
                    part for part in propagated_cluster if part != movable
                ],
            }
            history = list(
                source_candidate.get("joinable_pose_refinement_history") or []
            ) + [detail]
            refined_connection_ids = list(dict.fromkeys(
                str(row.get("connection_id")) for row in history
                if row.get("connection_id") is not None
            ))
            generated.append(
                _candidate_from_placements(
                    features,
                    new_placements,
                    source_candidate,
                    origin=(
                        "joinable_two_edge_pose_composition"
                        if len(refined_connection_ids) >= 2
                        else "joinable_pose_parameter_search"
                    ),
                    score_penalty=(
                        0.14
                        + 0.03 * normalized_offset
                        + (0.02 if flip else 0.0)
                        + (0.01 if rotation_degrees else 0.0)
                    ),
                    extra={
                        "joinable_pose_search": detail,
                        "joinable_pose_refinement_history": history,
                        "joinable_pose_refined_connection_ids": refined_connection_ids,
                    },
                )
            )
            generated_for_source += 1
            if (
                len(generated) >= max_candidates
                or generated_for_source >= source_quota
            ):
                break
        if len(generated) >= max_candidates:
            return generated
    return generated


def _fair_quotas(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return [0] * max(0, count)
    base, remainder = divmod(int(total), int(count))
    return [base + int(index < remainder) for index in range(count)]


def _cached_axial_compound_recall(
    fixed_features: dict[str, Any],
    moving_features: dict[str, Any],
) -> dict[str, Any]:
    """Recall one anonymous axial/end-face interface once per known group."""

    key = (id(fixed_features), id(moving_features))
    cached = _AXIAL_COMPOUND_RECALL_CACHE.get(key)
    if cached is not None:
        return cached
    fixed_radius = max(
        (float(row.get("radius", 0.0)) for row in fixed_features.get("cylinders", [])),
        default=0.0,
    )
    moving_radius = max(
        (float(row.get("radius", 0.0)) for row in moving_features.get("cylinders", [])),
        default=0.0,
    )
    radius_ratio = (
        min(fixed_radius, moving_radius) / max(fixed_radius, moving_radius)
        if min(fixed_radius, moving_radius) > 0.0
        else 1.0
    )
    # A shaft end is expected to be much smaller than a hub/flange seating
    # face.  Relax only the end-face area recall threshold for that explicit
    # radius regime; all axis, contact, polarity and phase checks remain.
    minimum_face_area_ratio = 0.08 if radius_ratio <= 0.75 else 0.35
    try:
        result = recall_axial_compound_candidates(
            fixed_features,
            moving_features,
            minimum_face_area_ratio=minimum_face_area_ratio,
        )
        result["minimum_face_area_ratio_used"] = minimum_face_area_ratio
        result["main_radius_ratio"] = radius_ratio
    except Exception as exc:
        result = {
            "schema_version": "axial_compound_interface.v1",
            "status": "unavailable",
            "reason": f"axial compound recall failed: {exc}",
            "exception_type": type(exc).__name__,
            "candidates": [],
            "auto_accept": False,
        }
    _AXIAL_COMPOUND_RECALL_CACHE[key] = result
    return result


def _compound_physical_constraints_satisfied(
    validation: dict[str, Any],
    proposal: dict[str, Any],
) -> bool:
    """Separate physical compound closure from phase observability.

    A continuous SO(2) phase has no numerical phase constraint.  It may close
    the physical axis/end-face relation, but its proposal guard remains set so
    it cannot be auto-accepted.  A discrete C_n orbit must be satisfied.
    """

    checks = validation.get("checks") or {}
    base = all(bool(checks.get(name)) for name in (
        "proper_rotation",
        "coaxial_direction",
        "radial_axis_center",
        "end_face_contact",
        "opposed_end_face_normals",
    ))
    orbit = list(proposal.get("phase_orbit_degrees") or [])
    return bool(
        base
        and (not orbit or checks.get("phase_in_active_orbit") is True)
    )


def _compound_axis_phase_constraints_satisfied(
    validation: dict[str, Any],
    proposal: dict[str, Any],
) -> bool:
    """Validate coaxial/phase closure while intentionally ignoring axial stop."""

    checks = validation.get("checks") or {}
    orbit = list(proposal.get("phase_orbit_degrees") or [])
    return bool(
        checks.get("proper_rotation")
        and checks.get("coaxial_direction")
        and checks.get("radial_axis_center")
        and (not orbit or checks.get("phase_in_active_orbit") is True)
    )


def _has_paired_topological_key_slot_witness(
    row: dict[str, Any],
    features: dict[str, Any],
) -> bool:
    fixed_part = str(row.get("fixed_part") or "")
    moving_part = str(row.get("moving_part") or "")
    if not fixed_part or not moving_part:
        return False
    for part in (fixed_part, moving_part):
        sidecar = (features.get(part) or {}).get("brep_graph_sidecar") or {}
        if sidecar.get("hash_verified") is not True:
            return False
    witnesses = list(row.get("phase_witness") or [])
    if not witnesses:
        witnesses = list(
            (row.get("compound_proposal") or {}).get("phase_witness") or []
        )
    return any(
        witness.get("fixed_kind") == "topological_key_slot"
        and witness.get("moving_kind") == "topological_key_slot"
        and bool(str(witness.get("fixed_witness_id") or "").strip())
        and bool(str(witness.get("moving_witness_id") or "").strip())
        for witness in witnesses
    )


def _current_compound_validation(
    row: dict[str, Any],
    placements: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    fixed = row.get("fixed_part")
    moving = row.get("moving_part")
    interface = row.get("compound_candidate")
    proposal = row.get("compound_proposal")
    if (
        fixed not in placements
        or moving not in placements
        or not isinstance(interface, dict)
        or not isinstance(proposal, dict)
    ):
        return {"error": "compound_pose_inputs_missing", "checks": {}}, False
    try:
        fixed_world = placement_to_matrix(placements[fixed])
        moving_world = placement_to_matrix(placements[moving])
        relative = np.linalg.inv(fixed_world) @ moving_world
        validation = validate_axial_compound_pose(
            interface,
            proposal,
            collision_free=True,
            transform=relative,
        )
        validation["collision_free"] = None
        validation["collision_scope"] = "deferred_to_exact_occt"
        validation["is_closed"] = False
        return validation, _compound_physical_constraints_satisfied(
            validation, proposal
        )
    except Exception as exc:
        return {
            "compound_constraints_satisfied": False,
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "checks": {},
        }, False


def _axial_group_centering_diagnostics(
    support_row: dict[str, Any],
    shaft_row: dict[str, Any],
    placements: dict[str, Any],
    features: dict[str, Any],
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a symmetric two-support / one-shaft geometric arrangement.

    This is an anonymous B-Rep rule.  It only activates when two comparable
    large-radius axial bodies meet at a compound end-face interface, extend in
    opposite and similarly sized directions from that interface, and a third
    smaller-radius axial body is connected to one of them.  The small body must
    be centred on the support interface and overlap both supports.
    """

    result: dict[str, Any] = {
        "schema_version": "axial_group_centering.v1",
        "pattern_detected": False,
        "supported": False,
        "proposal_only": True,
        "review_required": True,
        "can_auto_accept": False,
    }
    support_parts = [
        str(support_row.get("fixed_part") or ""),
        str(support_row.get("moving_part") or ""),
    ]
    shaft_edge_parts = [
        str(shaft_row.get("fixed_part") or ""),
        str(shaft_row.get("moving_part") or ""),
    ]
    if (
        len(set(support_parts)) != 2
        or any(part not in features or part not in placements for part in support_parts)
    ):
        result["reason"] = "support_pair_missing"
        return result
    shared = set(support_parts) & set(shaft_edge_parts)
    shaft_parts = set(shaft_edge_parts) - set(support_parts)
    if len(shared) != 1 or len(shaft_parts) != 1:
        result["reason"] = "axial_edges_do_not_form_support_shaft_chain"
        return result
    shaft_part = next(iter(shaft_parts))
    if shaft_part not in features or shaft_part not in placements:
        result["reason"] = "shaft_part_missing"
        return result

    support_radii = [
        _main_cylinder_radius(features[part]) for part in support_parts
    ]
    shaft_radius = _main_cylinder_radius(features[shaft_part])
    if any(radius is None for radius in support_radii) or shaft_radius is None:
        result["reason"] = "dominant_cylinder_missing"
        return result
    support_radius_ratio = min(support_radii) / max(support_radii)
    shaft_radius_ratio = shaft_radius / min(support_radii)
    result.update({
        "support_parts": support_parts,
        "shaft_part": shaft_part,
        "support_connection_id": support_row.get("connection_id"),
        "connection_id": shaft_row.get("connection_id"),
        "support_candidate_id": support_row.get("candidate_id"),
        "support_proposal_id": support_row.get("proposal_id"),
        "shaft_candidate_id": shaft_row.get("candidate_id"),
        "shaft_proposal_id": shaft_row.get("proposal_id"),
        "support_radius_ratio": support_radius_ratio,
        "shaft_to_support_radius_ratio": shaft_radius_ratio,
    })
    if support_radius_ratio < 0.90 or shaft_radius_ratio > 0.75:
        result["reason"] = "radius_hierarchy_not_symmetric_support_plus_shaft"
        return result

    if graph is not None:
        shaft_connection = next((
            row for row in graph.get("selected") or []
            if str(row.get("connection_id"))
            == str(shaft_row.get("connection_id"))
        ), None)
        if (
            shaft_connection is None
            or CLEARANCE not in set(shaft_connection.get("relation_types") or [])
        ):
            result["reason"] = "shaft_edge_is_not_a_selected_clearance_relation"
            return result

    bore_tolerance = max(1e-6, 0.05 * shaft_radius)
    support_bore_gaps: dict[str, float] = {}
    for part in support_parts:
        support_axis = _main_cylinder_axis(features[part])
        if support_axis is None:
            result["reason"] = "support_axis_missing"
            return result
        support_axis_np = np.asarray(_unit(support_axis), dtype=float)
        compatible_gaps = []
        for cylinder in features[part].get("cylinders") or []:
            radius = float(cylinder.get("radius", 0.0))
            cylinder_axis = cylinder.get("axis")
            if (
                radius < shaft_radius
                or not cylinder_axis
                or cylinder.get("surface_polarity") != "concave"
            ):
                continue
            if abs(float(np.dot(
                np.asarray(_unit(cylinder_axis), dtype=float),
                support_axis_np,
            ))) < math.cos(math.radians(2.0)):
                continue
            compatible_gaps.append(abs(radius - shaft_radius))
        if not compatible_gaps:
            result["reason"] = "support_has_no_coaxial_shaft_bore"
            return result
        support_bore_gaps[part] = min(compatible_gaps)
    result.update({
        "support_bore_radius_gap_mm": support_bore_gaps,
        "support_bore_tolerance_mm": bore_tolerance,
    })
    if any(gap > bore_tolerance for gap in support_bore_gaps.values()):
        result["reason"] = "support_bore_not_clearance_matched_to_shaft"
        return result
    support_slot_phase_witness = _has_paired_topological_key_slot_witness(
        support_row, features
    )
    shaft_slot_phase_witness = _has_paired_topological_key_slot_witness(
        shaft_row, features
    )
    result.update({
        "support_slot_phase_witness": support_slot_phase_witness,
        "shaft_slot_phase_witness": shaft_slot_phase_witness,
    })
    if not (support_slot_phase_witness and shaft_slot_phase_witness):
        result["reason"] = "complete_topological_key_slot_phase_loop_missing"
        return result

    support_validation, support_closed = _current_compound_validation(
        support_row, placements
    )
    if not support_closed:
        result["reason"] = "support_pair_compound_contact_not_closed"
        result["support_pose_validation"] = support_validation
        return result

    support_interface = support_row.get("compound_candidate") or {}
    fixed_face = support_interface.get("fixed_end_face") or {}
    moving_face = support_interface.get("moving_end_face") or {}
    if not fixed_face.get("position") or not moving_face.get("position"):
        result["reason"] = "support_contact_faces_missing"
        return result
    fixed_name = str(support_row["fixed_part"])
    moving_name = str(support_row["moving_part"])
    fixed_point = np.asarray(transform_point(
        fixed_face["position"], placements[fixed_name]
    ), dtype=float)
    moving_point = np.asarray(transform_point(
        moving_face["position"], placements[moving_name]
    ), dtype=float)
    axis_local = _main_cylinder_axis(features[fixed_name])
    if axis_local is None:
        result["reason"] = "support_axis_missing"
        return result
    axis_world = _unit(transform_vector(axis_local, placements[fixed_name]))
    axis_np = np.asarray(axis_world, dtype=float)
    target_coordinate = float(np.dot(0.5 * (fixed_point + moving_point), axis_np))
    support_intervals = [
        _axial_support_face_interval(
            features[part], placements[part], axis_world
        )
        for part in support_parts
    ]
    if any(interval is None for interval in support_intervals):
        result["reason"] = "support_axial_face_interval_missing"
        return result
    support_bbox_intervals = [
        _world_bbox_axis_interval(features[part], placements[part], axis_world)
        for part in support_parts
    ]

    side_rows = []
    for part, interval in zip(support_parts, support_intervals):
        low, high = interval
        interval_tolerance = max(1e-6, 0.02 * (high - low))
        if (
            target_coordinate < low - interval_tolerance
            or target_coordinate > high + interval_tolerance
        ):
            result["reason"] = "support_contact_plane_outside_support_extent"
            return result
        negative_extent = max(0.0, target_coordinate - low)
        positive_extent = max(0.0, high - target_coordinate)
        sign = -1 if negative_extent > positive_extent else 1
        major = max(negative_extent, positive_extent)
        minor = min(negative_extent, positive_extent)
        side_rows.append({
            "part": part,
            "interval": [low, high],
            "dominant_side": sign,
            "major_extent_mm": major,
            "minor_extent_mm": minor,
        })
    if side_rows[0]["dominant_side"] == side_rows[1]["dominant_side"]:
        result["reason"] = "supports_do_not_extend_to_opposite_sides"
        return result
    minimum_support_extent = max(1e-6, 0.05 * max(support_radii))
    if any(
        row["major_extent_mm"] < minimum_support_extent
        or row["minor_extent_mm"]
        > max(1e-6, 0.05 * row["major_extent_mm"])
        for row in side_rows
    ):
        result["reason"] = "support_side_dominance_is_ambiguous"
        return result
    major_ratio = min(row["major_extent_mm"] for row in side_rows) / max(
        row["major_extent_mm"] for row in side_rows
    )
    if major_ratio < 0.80:
        result["reason"] = "opposed_support_extents_are_not_comparable"
        return result
    union_low = min(interval[0] for interval in support_intervals)
    union_high = max(interval[1] for interval in support_intervals)
    union_center = 0.5 * (union_low + union_high)
    union_span = union_high - union_low
    union_center_tolerance = max(1e-6, 0.02 * union_span)
    union_center_residual = abs(target_coordinate - union_center)
    if union_center_residual > union_center_tolerance:
        result["reason"] = "support_contact_is_not_at_symmetric_union_center"
        return result

    shaft_interval = _axial_terminal_face_interval(
        features[shaft_part], placements[shaft_part], axis_world
    )
    if shaft_interval is None:
        result["reason"] = "shaft_terminal_face_interval_ambiguous"
        return result
    shaft_low, shaft_high = shaft_interval
    shaft_length = shaft_high - shaft_low
    shaft_center = 0.5 * (shaft_low + shaft_high)
    centering_residual = abs(shaft_center - target_coordinate)
    overlaps = [
        max(0.0, min(shaft_high, high) - max(shaft_low, low))
        for low, high in support_intervals
    ]
    minimum_overlap = max(1e-6, 0.15 * shaft_length)
    overlap_balance = min(overlaps) / max(overlaps) if max(overlaps) > 0 else 0.0

    shaft_validation, _ = _current_compound_validation(shaft_row, placements)
    shaft_axis_phase_supported = _compound_axis_phase_constraints_satisfied(
        shaft_validation, shaft_row.get("compound_proposal") or {}
    )
    centering_tolerance = max(1e-6, 0.002 * shaft_length)
    minimum_crossing_extent = max(1e-6, 0.01 * shaft_length)
    cross_interface_dependents = []
    if graph is not None:
        for connection in graph.get("selected") or []:
            parts = list(connection.get("parts") or [])
            relation_types = set(connection.get("relation_types") or [])
            if shaft_part not in parts or len(parts) != 2:
                continue
            if relation_types & {COAXIAL, CLEARANCE}:
                continue
            if not relation_types & {PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE}:
                continue
            dependent = parts[1] if parts[0] == shaft_part else parts[0]
            if dependent in support_parts or dependent not in placements:
                continue
            interval = _world_bbox_axis_interval(
                features.get(dependent, {}),
                placements[dependent],
                axis_world,
            )
            if interval is None:
                continue
            negative = target_coordinate - interval[0]
            positive = interval[1] - target_coordinate
            cross_interface_dependents.append({
                "part": dependent,
                "interval": list(interval),
                "negative_side_extent_mm": negative,
                "positive_side_extent_mm": positive,
                "crosses_support_contact_plane": bool(
                    negative >= minimum_crossing_extent
                    and positive >= minimum_crossing_extent
                ),
            })
    cross_interface_dependent_supported = any(
        row["crosses_support_contact_plane"]
        for row in cross_interface_dependents
    )
    result.update({
        "pattern_detected": True,
        "axis_world": axis_world,
        "target_center_coordinate": target_coordinate,
        "shaft_center_coordinate": shaft_center,
        "centering_offset_mm": target_coordinate - shaft_center,
        "centering_residual_mm": centering_residual,
        "centering_tolerance_mm": centering_tolerance,
        "shaft_terminal_interval": [shaft_low, shaft_high],
        "shaft_length_mm": shaft_length,
        "support_intervals": side_rows,
        "support_bbox_intervals": support_bbox_intervals,
        "support_union_center_coordinate": union_center,
        "support_union_center_residual_mm": union_center_residual,
        "support_union_center_tolerance_mm": union_center_tolerance,
        "two_sided_overlap_mm": overlaps,
        "minimum_overlap_mm": minimum_overlap,
        "overlap_balance": overlap_balance,
        "support_extent_balance": major_ratio,
        "shaft_axis_phase_supported": shaft_axis_phase_supported,
        "cross_interface_dependents": cross_interface_dependents,
        "minimum_crossing_extent_mm": minimum_crossing_extent,
        "cross_interface_dependent_supported": (
            cross_interface_dependent_supported
        ),
        "support_pose_validation": support_validation,
        "shaft_axis_phase_validation": shaft_validation,
    })
    result["supported"] = bool(
        shaft_axis_phase_supported
        and cross_interface_dependent_supported
        and centering_residual <= centering_tolerance
        and min(overlaps) >= minimum_overlap
        and overlap_balance >= 0.75
    )
    result["reason"] = (
        "symmetric_support_center_and_two_sided_insertion_satisfied"
        if result["supported"]
        else "symmetric_support_requires_centered_two_sided_shaft_insertion"
    )
    return result


def _axial_group_centering_candidates_for_candidate(
    source_candidate: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate a bounded group-centred shaft proposal from compound history."""

    history = list(source_candidate.get("axial_compound_history") or [])
    if len(history) < 2:
        return [], []
    generated: list[dict[str, Any]] = []
    required_connection_ids: list[str] = []
    seen = set()
    for support_row in history:
        for shaft_row in history:
            if support_row is shaft_row:
                continue
            diagnostics = _axial_group_centering_diagnostics(
                support_row,
                shaft_row,
                source_candidate.get("placements") or {},
                features,
                graph,
            )
            if diagnostics.get("pattern_detected") is not True:
                continue
            connection_id = str(diagnostics.get("connection_id"))
            required_connection_ids.append(connection_id)
            key = (
                str(diagnostics.get("support_connection_id")),
                connection_id,
                str(diagnostics.get("shaft_part")),
            )
            if key in seen:
                continue
            seen.add(key)
            offset = float(diagnostics.get("centering_offset_mm", 0.0))
            placements = json.loads(json.dumps(
                source_candidate.get("placements") or {}
            ))
            shaft_part = str(diagnostics["shaft_part"])
            current = placements.get(shaft_part)
            if current is None:
                continue
            shifted = _shift_placement_along_axis(
                current,
                list(diagnostics["axis_world"]),
                offset,
            )
            propagated = _set_placement_with_rigid_dependents(
                placements,
                shaft_part,
                shifted,
                graph,
                blocked=set(diagnostics.get("support_parts") or []),
            )
            current_diagnostics = _axial_group_centering_diagnostics(
                support_row, shaft_row, placements, features, graph
            )
            if current_diagnostics.get("supported") is not True:
                continue
            detail = {
                **current_diagnostics,
                "propagated_rigid_dependents": [
                    part for part in propagated if part != shaft_part
                ],
            }
            centering_history = list(
                source_candidate.get("axial_group_centering_history") or []
            ) + [detail]
            generated.append(_candidate_from_placements(
                features,
                placements,
                source_candidate,
                origin="axial_group_symmetric_centering",
                score_penalty=0.01,
                extra={
                    "proposal_only": True,
                    "review_required": True,
                    "can_auto_accept": False,
                    "axial_group_centering": detail,
                    "axial_group_centering_history": centering_history,
                    "axial_group_centering_required_connection_ids": sorted(
                        set(required_connection_ids)
                    ),
                },
            ))
    return generated, sorted(set(required_connection_ids))


def _axial_compound_candidates_for_connection(
    source_candidates: list[dict[str, Any]],
    connection: dict[str, Any],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int,
    refinement_phase: str,
) -> list[dict[str, Any]]:
    """Materialise a bounded compound axis/end-face frontier for one edge."""

    if max_candidates <= 0 or not source_candidates:
        return []
    parts = list(connection.get("parts") or [])
    if (
        len(parts) != 2
        or not set(connection.get("relation_types") or [])
        & {COAXIAL, CLEARANCE}
    ):
        return []
    axial_degree: dict[str, int] = defaultdict(int)
    for selected_connection in graph.get("selected") or []:
        if set(selected_connection.get("relation_types") or []) & {
            COAXIAL, CLEARANCE
        }:
            for part in selected_connection.get("parts") or []:
                axial_degree[str(part)] += 1
    central = [part for part in parts if axial_degree.get(part, 0) >= 2]
    if len(central) == 1:
        fixed = central[0]
        moving = parts[1] if parts[0] == fixed else parts[0]
    else:
        fixed, moving = parts
    recall = _cached_axial_compound_recall(
        features.get(fixed, {}),
        features.get(moving, {}),
    )
    proposal_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for interface in recall.get("candidates") or []:
        geometry = {
            key: value for key, value in interface.items()
            if key != "proposals"
        }
        for proposal in interface.get("proposals") or []:
            # The end face resolves the unoriented axis-line polarity.  The
            # other proper-rotation branch remains in the reusable recall
            # audit but is not a physical face-contact proposal.
            if proposal.get("end_face_orientation_compatible") is not True:
                continue
            proposal_rows.append((geometry, proposal))
    proposal_rows.sort(key=lambda row: (
        bool(row[1].get("proposal_only", True)),
        -float(row[0].get("end_face_area_ratio", 0.0)),
        abs(float(row[1].get("phase_degrees", 0.0))),
        float(row[1].get("phase_degrees", 0.0)),
        str(row[1].get("proposal_id")),
    ))
    if not proposal_rows:
        return []

    generated: list[dict[str, Any]] = []
    # The first source sees the full phase orbit before another source pose is
    # expanded.  This preserves symmetry representatives under a small fixed
    # quota instead of spending the quota on duplicate source variants.
    active_sources = source_candidates[: min(4, len(source_candidates))]
    for source_index, source_candidate in enumerate(active_sources):
        placements = source_candidate.get("placements") or {}
        if fixed not in placements or moving not in placements:
            continue
        fixed_world = placement_to_matrix(placements[fixed])
        for interface_geometry, proposal in proposal_rows:
            relative = np.asarray(proposal["transform"], dtype=float)
            moving_world = fixed_world @ relative
            new_placements = json.loads(json.dumps(placements))
            propagated_cluster = _set_placement_with_rigid_dependents(
                new_placements,
                moving,
                matrix_to_placement(moving_world),
                graph,
                blocked={fixed},
            )
            validation = validate_axial_compound_pose(
                interface_geometry,
                proposal,
                collision_free=True,
                transform=relative,
            )
            validation["collision_free"] = None
            validation["collision_scope"] = "deferred_to_exact_occt"
            validation["is_closed"] = False
            physical_closed = _compound_physical_constraints_satisfied(
                validation, proposal
            )
            if not physical_closed:
                continue
            source_guard = bool(
                source_candidate.get("proposal_only")
                or source_candidate.get("review_required")
                or source_candidate.get("can_auto_accept") is False
            )
            proposal_guard = bool(
                proposal.get("proposal_only")
                or proposal.get("review_required")
                or len(proposal.get("phase_orbit_degrees") or []) != 1
                or proposal.get("phase_status")
                != "resolved_by_asymmetric_witness"
            )
            review_required = source_guard or proposal_guard
            detail = {
                "connection_id": connection.get("connection_id"),
                "fixed_part": fixed,
                "moving_part": moving,
                "refinement_phase": refinement_phase,
                "source_candidate_index": source_index,
                "recall_status": recall.get("status"),
                "candidate_id": interface_geometry.get("candidate_id"),
                "proposal_id": proposal.get("proposal_id"),
                "axis_polarity": proposal.get("axis_polarity"),
                "phase_convention": proposal.get("phase_convention"),
                "symmetry_group": proposal.get("symmetry_group"),
                "interface_symmetry_order": proposal.get(
                    "interface_symmetry_order"
                ),
                "interface_phase_orbit_degrees": proposal.get(
                    "interface_phase_orbit_degrees"
                ),
                "phase_orbit_degrees": proposal.get("phase_orbit_degrees"),
                "whole_part_symmetry_order": proposal.get(
                    "whole_part_symmetry_order"
                ),
                "phase_status": proposal.get("phase_status"),
                "phase_degrees": proposal.get("phase_degrees"),
                "phase_witness": proposal.get("phase_witness") or [],
                "phase_witness_residual_deg": proposal.get(
                    "phase_residual_deg"
                ),
                "phase_residual_deg": validation.get("phase_residual_deg"),
                "compound_constraints_satisfied": physical_closed,
                "constraint_validation": validation,
                "propagated_rigid_dependents": [
                    part for part in propagated_cluster if part != moving
                ],
                "compound_candidate": interface_geometry,
                "compound_proposal": proposal,
                "proposal_only": review_required,
                "review_required": review_required,
                "can_auto_accept": False if review_required else None,
            }
            history = list(
                source_candidate.get("axial_compound_history") or []
            ) + [detail]
            refined_ids = list(dict.fromkeys(
                str(row.get("connection_id")) for row in history
                if row.get("connection_id") is not None
            ))
            area_error = 1.0 - float(
                interface_geometry.get("end_face_area_ratio", 0.0)
            )
            generated.append(_candidate_from_placements(
                features,
                new_placements,
                source_candidate,
                origin="axial_compound_interface_recall",
                score_penalty=0.08 + 0.03 * max(0.0, area_error),
                extra={
                    "proposal_only": review_required,
                    "review_required": review_required,
                    "can_auto_accept": (
                        False if review_required
                        else source_candidate.get("can_auto_accept")
                    ),
                    "axial_compound_interface": detail,
                    "axial_compound_history": history,
                    "axial_compound_refined_connection_ids": refined_ids,
                },
            ))
            if len(generated) >= max_candidates:
                return generated
    return generated


def _axial_compound_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    """Allocate a fixed compound budget and compose two central-axis leaves."""

    connections = [
        row for row in graph.get("selected") or []
        if len(row.get("parts") or []) == 2
        and bool(set(row.get("relation_types") or []) & {COAXIAL, CLEARANCE})
    ]
    if not connections or max_candidates <= 0:
        return []
    independent_budget = (
        max_candidates
        if len(connections) == 1
        else max(len(connections), (2 * max_candidates) // 3)
    )
    generated: list[dict[str, Any]] = []
    independent: list[list[dict[str, Any]]] = []
    for connection, quota in zip(
        connections, _fair_quotas(independent_budget, len(connections))
    ):
        rows = _axial_compound_candidates_for_connection(
            source_candidates,
            connection,
            graph,
            features,
            max_candidates=quota,
            refinement_phase="independent_edge_quota",
        )
        independent.append(rows)
        generated.extend(rows)

    remaining = max(0, max_candidates - len(generated))
    composition_jobs = []
    for left_index in range(len(connections)):
        for right_index in range(left_index + 1, len(connections)):
            for left in independent[left_index]:
                left_detail = left.get("axial_compound_interface") or {}
                for right in independent[right_index]:
                    right_detail = right.get("axial_compound_interface") or {}
                    fixed = left_detail.get("fixed_part")
                    if (
                        not fixed
                        or fixed != right_detail.get("fixed_part")
                        or not left_detail.get("moving_part")
                        or not right_detail.get("moving_part")
                        or left_detail.get("moving_part")
                        == right_detail.get("moving_part")
                    ):
                        continue
                    right_proposal = right_detail.get("compound_proposal") or {}
                    relative = np.asarray(
                        right_proposal.get("transform"), dtype=float
                    )
                    placements = json.loads(json.dumps(
                        left.get("placements") or {}
                    ))
                    if fixed not in placements or relative.shape != (4, 4):
                        continue
                    right_moving = str(right_detail["moving_part"])
                    moving_world = placement_to_matrix(placements[fixed]) @ relative
                    propagated_cluster = _set_placement_with_rigid_dependents(
                        placements,
                        right_moving,
                        matrix_to_placement(moving_world),
                        graph,
                        blocked={
                            str(fixed),
                            str(left_detail.get("moving_part") or ""),
                        },
                    )
                    composed_right_detail = {
                        **right_detail,
                        "composition_propagated_rigid_dependents": [
                            part for part in propagated_cluster
                            if part != right_moving
                        ],
                    }
                    history = list(
                        left.get("axial_compound_history") or []
                    ) + [composed_right_detail]
                    refined_ids = list(dict.fromkeys(
                        str(row.get("connection_id")) for row in history
                        if row.get("connection_id") is not None
                    ))
                    review_required = any(
                        bool(row.get("proposal_only") or row.get("review_required"))
                        for row in history
                    )
                    candidate = _candidate_from_placements(
                        features,
                        placements,
                        left,
                        origin="axial_compound_star_composition",
                        score_penalty=0.02,
                        extra={
                            "proposal_only": review_required,
                            "review_required": review_required,
                            "can_auto_accept": (
                                False
                                if review_required
                                else left.get("can_auto_accept")
                            ),
                            "axial_compound_interface": composed_right_detail,
                            "axial_compound_history": history,
                            "axial_compound_refined_connection_ids": refined_ids,
                        },
                    )
                    centered, required_ids = (
                        _axial_group_centering_candidates_for_candidate(
                            candidate, graph, features
                        )
                    )
                    if required_ids:
                        # A symmetric dual-support pattern makes an arbitrary
                        # one-end-face shaft stop a known degenerate pose.  The
                        # raw composition remains in the audit frontier, but it
                        # cannot claim closure without current group-centering
                        # evidence.
                        candidate[
                            "axial_group_centering_required_connection_ids"
                        ] = required_ids
                    for variant in [candidate, *centered]:
                        precheck = _pose_precheck(variant, graph, features)
                        closure = precheck["constraint_closure"]
                        overlap = precheck["overlap_objective"]
                        composition_jobs.append((
                            (
                                -int(closure.get("closed_connection_count", 0)),
                                -float(closure.get("closure_ratio", 0.0)),
                                int(overlap.get("severe_non_edge_overlap_count", 0)),
                                float(overlap.get("bbox_overlap_cost", 0.0)),
                                tuple(refined_ids),
                            ),
                            variant,
                        ))
    composition_jobs.sort(key=lambda item: item[0])
    seen_placements = {
        json.dumps(row.get("placements") or {}, sort_keys=True)
        for row in generated
    }
    for _, candidate in composition_jobs:
        if remaining <= 0:
            break
        signature = json.dumps(
            candidate.get("placements") or {}, sort_keys=True
        )
        if signature in seen_placements:
            continue
        seen_placements.add(signature)
        generated.append(candidate)
        remaining -= 1
    return generated[:max_candidates]


def _joinable_pose_parameter_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 80,
) -> list[dict[str, Any]]:
    """Search residual joint parameters without starving later graph edges.

    Half of the fixed budget is first distributed across all parameterizable
    selected edges.  The remainder composes two distinct edge refinements in a
    bounded coordinate step.  This does not increase the solver beam or bypass
    closure/collision/precision validation.
    """

    if max_candidates <= 0 or not source_candidates:
        return []
    supported = {COAXIAL, CLEARANCE, PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE}
    connections = [
        row for row in graph.get("selected") or []
        if len(row.get("parts") or []) == 2
        and bool(set(row.get("relation_types") or []) & supported)
        and bool(row.get("matches") or [])
    ]
    if not connections:
        return []

    if len(connections) == 1:
        independent_budget = max_candidates
    else:
        independent_budget = min(
            max_candidates,
            max(len(connections), max_candidates // 2),
        )
    generated: list[dict[str, Any]] = []
    independent_by_connection: list[list[dict[str, Any]]] = []
    for connection, quota in zip(
        connections, _fair_quotas(independent_budget, len(connections))
    ):
        rows = _joinable_pose_parameter_candidates_for_connection(
            source_candidates,
            connection,
            graph,
            features,
            max_candidates=quota,
            refinement_phase="independent_edge_quota",
        )
        independent_by_connection.append(rows)
        generated.extend(rows)

    remaining = max(0, max_candidates - len(generated))
    directions = [
        (source_index, target_index)
        for source_index, source_rows in enumerate(independent_by_connection)
        if source_rows
        for target_index in range(len(connections))
        if target_index != source_index
    ]
    for (source_index, target_index), quota in zip(
        directions, _fair_quotas(remaining, len(directions))
    ):
        if quota <= 0:
            continue
        # Width two is a fixed local coordinate frontier, not an expansion of
        # solve_small_assembly's beam.  Applying the target edge to an already
        # refined source-edge pose preserves both leaf transforms.
        source_rows = independent_by_connection[source_index][:2]
        rows = _joinable_pose_parameter_candidates_for_connection(
            source_rows,
            connections[target_index],
            graph,
            features,
            max_candidates=min(quota, max_candidates - len(generated)),
            refinement_phase="two_edge_coordinate_composition",
        )
        generated.extend(rows)
        if len(generated) >= max_candidates:
            break
    return generated[:max_candidates]


def _placed_obb(
    obb: dict[str, Any], placement: dict[str, Any]
) -> dict[str, Any]:
    return {
        **obb,
        "center": transform_point(obb.get("center", [0, 0, 0]), placement),
        "axes": [
            _unit(transform_vector(axis, placement))
            for axis in obb.get("axes") or []
        ],
    }


def _carrier_thin_side(
    features: dict[str, Any],
    placements: dict[str, Any],
    *,
    carrier: str,
    moving: str,
) -> dict[str, Any] | None:
    """Locate a placed component on one unambiguous side of a thin carrier.

    This deliberately uses only OBB geometry and the already proposed rigid
    placements.  The sign of an OBB axis is arbitrary, but comparing two
    moving parts against the *same* carrier makes that sign cancel out.  A
    near-centre result abstains rather than inventing an accessible side.
    """

    carrier_obb = (features.get(carrier) or {}).get("obb") or {}
    moving_obb = (features.get(moving) or {}).get("obb") or {}
    dimensions = list(carrier_obb.get("dimensions") or [])
    axes = list(carrier_obb.get("axes") or [])
    if (
        carrier not in placements
        or moving not in placements
        or len(dimensions) != 3
        or len(axes) != 3
        or not moving_obb
    ):
        return None
    try:
        thin_axis_index = int(np.argmin(np.asarray(dimensions, dtype=float)))
        carrier_world = _placed_obb(carrier_obb, placements[carrier])
        moving_world = _placed_obb(moving_obb, placements[moving])
        thin_axis = _unit(carrier_world["axes"][thin_axis_index])
        if thin_axis is None:
            return None
        offset = float(np.dot(
            np.asarray(moving_world["center"], dtype=float)
            - np.asarray(carrier_world["center"], dtype=float),
            thin_axis,
        ))
        # A component whose centre is within one quarter of the carrier's
        # thin extent cannot reliably tell us which exterior side it uses.
        minimum_margin = max(
            1e-6,
            0.25 * abs(float(dimensions[thin_axis_index])),
        )
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    if abs(offset) < minimum_margin:
        return {
            "side_sign": 0,
            "offset_mm": offset,
            "minimum_margin_mm": minimum_margin,
            "thin_axis_index": thin_axis_index,
            "thin_axis_world": np.asarray(thin_axis, dtype=float).tolist(),
            "reason": "moving_center_too_close_to_carrier_midplane",
        }
    return {
        "side_sign": 1 if offset > 0.0 else -1,
        "offset_mm": offset,
        "minimum_margin_mm": minimum_margin,
        "thin_axis_index": thin_axis_index,
        "thin_axis_world": np.asarray(thin_axis, dtype=float).tolist(),
        "reason": "unambiguous_thin_carrier_side",
    }


def _edge_slot_open_side(
    edge_detail: dict[str, Any],
    features: dict[str, Any],
    placements: dict[str, Any],
    *,
    carrier: str,
) -> dict[str, Any] | None:
    """Return the outward channel-mouth side from audited slot geometry.

    ``edge_slot_interface`` records a floor normal that is explicitly oriented
    from the internal channel floor toward its open mouth.  It is stronger than
    a card body's centre (which can straddle the carrier during insertion), so
    use it whenever composing a footprint partner on the same thin carrier.
    """

    carrier_obb = (features.get(carrier) or {}).get("obb") or {}
    dimensions = list(carrier_obb.get("dimensions") or [])
    axes = list(carrier_obb.get("axes") or [])
    insertion_axis = edge_detail.get("insertion_axis")
    if (
        carrier not in placements
        or len(dimensions) != 3
        or len(axes) != 3
        or not isinstance(insertion_axis, list)
        or len(insertion_axis) != 3
    ):
        return None
    try:
        thin_axis_index = int(np.argmin(np.asarray(dimensions, dtype=float)))
        carrier_world = _placed_obb(carrier_obb, placements[carrier])
        thin_axis = _unit(carrier_world["axes"][thin_axis_index])
        mouth_axis = _unit(transform_vector(insertion_axis, placements[carrier]))
        if thin_axis is None or mouth_axis is None:
            return None
        alignment = float(np.dot(mouth_axis, thin_axis))
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    # The reusable edge-slot detector already requires a floor normal aligned
    # with the carrier's thin axis.  Preserve a small numerical margin here
    # rather than treating an oblique channel as a carrier-side witness.
    if abs(alignment) < 0.80:
        return None
    return {
        "side_sign": 1 if alignment > 0.0 else -1,
        "thin_axis_index": thin_axis_index,
        "thin_axis_world": np.asarray(thin_axis, dtype=float).tolist(),
        "mouth_axis_world": np.asarray(mouth_axis, dtype=float).tolist(),
        "thin_axis_alignment": alignment,
        "reason": "audited_channel_mouth_on_carrier_thin_side",
    }


def _carrier_open_side_consistent_candidates(
    edge_slot_candidates: list[dict[str, Any]],
    planar_footprint_candidates: list[dict[str, Any]],
    features: dict[str, Any],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    """Compose a slot-mounted part with a same-side footprint-mounted part.

    A repeated bounded edge slot is stronger evidence of the actively used
    side of a thin carrier than a free planar footprint polarity.  When both
    interfaces share the carrier, retain the edge-slot placement verbatim and
    only compose planar proposals whose moving body lies on that same side.
    This is a review-only group-consistency refinement; it does not claim that
    co-sided parts necessarily belong to the same real-world assembly.
    """

    if max_candidates <= 0:
        return []
    jobs: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for edge_candidate in edge_slot_candidates:
        edge_history = list(edge_candidate.get("edge_slot_history") or [])
        if not edge_history and edge_candidate.get("edge_slot_interface"):
            edge_history = [edge_candidate["edge_slot_interface"]]
        for edge_detail in edge_history:
            carrier = str(edge_detail.get("stationary_part") or "")
            edge_moving = str(edge_detail.get("movable_part") or "")
            if not carrier or not edge_moving:
                continue
            edge_side = _edge_slot_open_side(
                edge_detail,
                features,
                edge_candidate.get("placements") or {},
                carrier=carrier,
            )
            if not edge_side:
                continue
            for footprint_candidate in planar_footprint_candidates:
                footprint_history = list(
                    footprint_candidate.get("planar_footprint_history") or []
                )
                if not footprint_history and footprint_candidate.get(
                    "planar_footprint"
                ):
                    footprint_history = [
                        footprint_candidate["planar_footprint"]
                    ]
                for footprint_detail in footprint_history:
                    if str(footprint_detail.get("stationary_part") or "") != carrier:
                        continue
                    footprint_moving = str(
                        footprint_detail.get("movable_part") or ""
                    )
                    if not footprint_moving or footprint_moving == edge_moving:
                        continue
                    footprint_placement = (
                        footprint_candidate.get("placements") or {}
                    ).get(footprint_moving)
                    if footprint_placement is None:
                        continue
                    placements = json.loads(json.dumps(
                        edge_candidate.get("placements") or {}
                    ))
                    placements[footprint_moving] = json.loads(json.dumps(
                        footprint_placement
                    ))
                    footprint_side = _carrier_thin_side(
                        features,
                        placements,
                        carrier=carrier,
                        moving=footprint_moving,
                    )
                    if (
                        not footprint_side
                        or int(footprint_side.get("side_sign", 0))
                        != int(edge_side["side_sign"])
                    ):
                        continue
                    detail = {
                        "carrier_part": carrier,
                        "reference_connection_id": edge_detail.get(
                            "connection_id"
                        ),
                        "target_connection_id": footprint_detail.get(
                            "connection_id"
                        ),
                        "reference_interface": "repeated_bounded_edge_slot",
                        "target_interface": "carrier_parallel_surface",
                        "carrier_thin_axis_index": edge_side["thin_axis_index"],
                        "carrier_thin_axis_world": edge_side["thin_axis_world"],
                        "reference_side_sign": edge_side["side_sign"],
                        "target_side_sign": footprint_side["side_sign"],
                        "reference_channel_mouth_alignment": edge_side[
                            "thin_axis_alignment"
                        ],
                        "target_offset_mm": footprint_side["offset_mm"],
                        "minimum_margin_mm": float(
                            footprint_side["minimum_margin_mm"]
                        ),
                        "supported": True,
                        "proposal_only": True,
                        "review_required": True,
                        "can_auto_accept": False,
                        "preserved_component_placements": [edge_moving],
                        "reason": (
                            "A bounded edge-slot placement and a planar "
                            "footprint placement occupy the same unambiguous "
                            "side of their shared thin carrier."
                        ),
                    }
                    refined_edge_ids = list(dict.fromkeys(
                        str(row.get("connection_id")) for row in edge_history
                        if row.get("connection_id") is not None
                    ))
                    refined_footprint_ids = list(dict.fromkeys(
                        str(row.get("connection_id"))
                        for row in footprint_history
                        if row.get("connection_id") is not None
                    ))
                    candidate = _candidate_from_placements(
                        features,
                        placements,
                        edge_candidate,
                        origin="carrier_open_side_composition",
                        score_penalty=0.0,
                        extra={
                            "proposal_only": True,
                            "review_required": True,
                            "can_auto_accept": False,
                            "edge_slot_interface": edge_detail,
                            "edge_slot_history": edge_history,
                            "edge_slot_refined_connection_ids": refined_edge_ids,
                            "planar_footprint": footprint_detail,
                            "planar_footprint_history": footprint_history,
                            "planar_footprint_refined_connection_ids": (
                                refined_footprint_ids
                            ),
                            "carrier_open_side_consistency": detail,
                            "carrier_open_side_consistency_history": [detail],
                        },
                    )
                    # The composition has all evidence retained by either
                    # source candidate, so it should not be artificially
                    # disadvantaged in the bounded exact-validation frontier.
                    candidate["total_score"] = max(
                        float(edge_candidate.get("total_score", 0.0)),
                        float(footprint_candidate.get("total_score", 0.0)),
                    )
                    jobs.append((
                        (
                            -abs(float(footprint_side["offset_mm"])),
                            str(edge_detail.get("connection_id") or ""),
                            str(footprint_detail.get("connection_id") or ""),
                            int(footprint_detail.get("support_polarity", 0)),
                            int(footprint_detail.get("normal_sign", 0)),
                        ),
                        candidate,
                    ))
    jobs.sort(key=lambda item: item[0])
    return [candidate for _, candidate in jobs[:max_candidates]]


def _cached_enclosure_bay_recall(
    stationary_features: dict[str, Any],
    moving_features: dict[str, Any],
) -> dict[str, Any]:
    """Cache pair-invariant, review-only repeated-bay proposals."""

    key = (id(stationary_features), id(moving_features))
    cached = _ENCLOSURE_BAY_RECALL_CACHE.get(key)
    if cached is not None:
        return cached
    # The enclosure provider cross-compares walls and rails.  Feeding every
    # cosmetic face of a vendor chassis into that combinatorial stage is both
    # unnecessary and unsafe.  Retain a deterministic, permissive geometry
    # ROI: a potential wall/rail needs at least one footprint extent plausibly
    # related to a moving OBB extent.  This is recall-only; the provider still
    # requires opposing walls, paired rails, and a functional-body fit.
    moving_obb = moving_features.get("obb") or {}
    moving_dimensions = [
        abs(float(value)) for value in moving_obb.get("dimensions") or []
        if abs(float(value)) > 1e-9
    ]
    raw_planes = list(stationary_features.get("planes") or [])
    roi_rows: list[tuple[tuple[float, int], dict[str, Any]]] = []
    if len(moving_dimensions) == 3:
        for raw_index, plane in enumerate(raw_planes):
            dimensions = plane.get("footprint_dimensions") or []
            if len(dimensions) < 2:
                continue
            try:
                extents = [abs(float(value)) for value in dimensions[:2]]
            except (TypeError, ValueError):
                continue
            if min(extents) <= 1e-9:
                continue
            relative_errors = [
                abs(math.log(extent / moving_extent))
                for extent in extents
                for moving_extent in moving_dimensions
                if moving_extent > 1e-9
            ]
            if not relative_errors:
                continue
            nearest = min(relative_errors)
            # A rail can be thin in one direction, but its long direction must
            # remain within this deliberately broad 5x scale band.  Full-size
            # chassis skins and tiny cosmetic details then stay out.
            if nearest > math.log(5.0):
                continue
            roi_rows.append(((nearest, raw_index), plane))
    # Bound the caller-side geometric frontier before the provider's pairwise
    # wall search.  Tie breaking by original index makes this reproducible.
    roi_rows.sort(key=lambda item: item[0])
    # 240 yields at most a modest pairwise wall frontier while still retaining
    # many more geometric alternatives than the downstream proposal budget.
    filtered_planes = [row for _, row in roi_rows[:240]]
    stationary_roi = {
        **stationary_features,
        "planes": filtered_planes,
    }
    try:
        result = propose_enclosure_bay_placements(
            stationary_roi,
            moving_features,
            maximum=16,
        )
    except Exception as exc:
        result = {
            "status": "abstain",
            "reason": f"enclosure bay recall failed: {exc}",
            "proposals": [],
            "audit": {"exception_type": type(exc).__name__},
            "proposal_only": True,
            "review_required": True,
            "can_auto_accept": False,
        }
    result.setdefault("audit", {}).update({
        "caller_plane_size_roi_enabled": True,
        "caller_raw_plane_count": len(raw_planes),
        "caller_roi_plane_count": len(filtered_planes),
        "caller_roi_maximum_plane_count": 240,
    })
    _ENCLOSURE_BAY_RECALL_CACHE[key] = result
    return result


def _enclosure_bay_candidates_for_connection(
    source_candidates: list[dict[str, Any]],
    connection: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int,
    refinement_phase: str,
) -> list[dict[str, Any]]:
    """Materialise repeated-bay transforms for exactly one selected edge."""

    if max_candidates <= 0:
        return []
    parts = list(connection.get("parts") or [])
    if len(parts) != 2:
        return []
    relation_types = set(connection.get("relation_types") or [])
    if not relation_types & {PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE}:
        return []
    diagonals = {
        part: _bbox_diagonal_from_features(features.get(part, {}))
        for part in parts
    }
    stationary = max(parts, key=lambda part: diagonals[part])
    movable = parts[1] if parts[0] == stationary else parts[0]
    recall = _cached_enclosure_bay_recall(
        features.get(stationary, {}),
        features.get(movable, {}),
    )
    proposals = [
        row for row in recall.get("proposals") or []
        if int(row.get("independent_evidence_count", 0)) >= 4
        and row.get("proposal_only") is True
        and row.get("review_required") is True
        and row.get("can_auto_accept") is False
        and row.get("transform_4x4") is not None
    ]
    if not proposals:
        return []

    generated = []
    for source_candidate in source_candidates[:1]:
        placements = source_candidate.get("placements") or {}
        if stationary not in placements or movable not in placements:
            continue
        stationary_world = placement_to_matrix(placements[stationary])
        for proposal in proposals[:max_candidates]:
            relative = np.asarray(proposal["transform_4x4"], dtype=float)
            if relative.shape != (4, 4):
                continue
            moving_world = stationary_world @ relative
            new_placements = json.loads(json.dumps(placements))
            new_placements[movable] = matrix_to_placement(moving_world)
            detail = {
                **proposal,
                "connection_id": connection.get("connection_id"),
                "stationary_part": stationary,
                "movable_part": movable,
                "refinement_phase": refinement_phase,
                "recall_status": recall.get("status"),
                "recall_reason": recall.get("reason"),
                "functional_body_envelope_audit": recall.get(
                    "functional_body_envelope_audit"
                ) or {},
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            }
            history = list(
                source_candidate.get("enclosure_bay_history") or []
            ) + [detail]
            refined_ids = list(dict.fromkeys(
                str(row.get("connection_id")) for row in history
                if row.get("connection_id") is not None
            ))
            generated.append(_candidate_from_placements(
                features,
                new_placements,
                source_candidate,
                origin="repeated_enclosure_bay_recall",
                score_penalty=(
                    0.10
                    + 0.05 * max(
                        0.0,
                        1.0 - float(proposal.get("proposal_score", 0.0)),
                    )
                ),
                extra={
                    "proposal_only": True,
                    "review_required": True,
                    "can_auto_accept": False,
                    "enclosure_bay": detail,
                    "enclosure_bay_history": history,
                    "enclosure_bay_refined_connection_ids": refined_ids,
                },
            ))
            if len(generated) >= max_candidates:
                break
    return generated[:max_candidates]


def _enclosure_bay_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    """Allocate one fixed review-only repeated-bay frontier."""

    connections = [
        row for row in graph.get("selected") or []
        if len(row.get("parts") or []) == 2
        and bool(set(row.get("relation_types") or []) & {
            PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE
        })
    ]
    generated = []
    for connection, quota in zip(
        connections, _fair_quotas(max_candidates, len(connections))
    ):
        generated.extend(_enclosure_bay_candidates_for_connection(
            source_candidates,
            connection,
            features,
            max_candidates=quota,
            refinement_phase="independent_edge_quota",
        ))
    return generated[:max_candidates]


def _cached_edge_slot_recall(
    stationary_features: dict[str, Any],
    moving_features: dict[str, Any],
) -> dict[str, Any]:
    """Cache pair-invariant repeated bounded edge-slot proposals."""

    key = (id(stationary_features), id(moving_features))
    cached = _EDGE_SLOT_RECALL_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        result = recall_edge_slot_interface_proposals(
            stationary_features,
            moving_features,
            maximum_proposals=16,
        )
    except Exception as exc:
        result = {
            "status": "abstain",
            "reason": f"edge-slot recall failed: {exc}",
            "proposals": [],
            "audit": {"exception_type": type(exc).__name__},
            "proposal_only": True,
            "review_required": True,
            "can_auto_accept": False,
        }
    _EDGE_SLOT_RECALL_CACHE[key] = result
    return result


def _edge_slot_candidates_for_connection(
    source_candidates: list[dict[str, Any]],
    connection: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int,
    refinement_phase: str,
) -> list[dict[str, Any]]:
    """Materialise repeated edge-slot transforms for one selected edge."""

    if max_candidates <= 0:
        return []
    parts = list(connection.get("parts") or [])
    if len(parts) != 2:
        return []
    if not set(connection.get("relation_types") or []) & {
        PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE
    }:
        return []
    diagonals = {
        part: _bbox_diagonal_from_features(features.get(part, {}))
        for part in parts
    }
    stationary = max(parts, key=lambda part: diagonals[part])
    movable = parts[1] if parts[0] == stationary else parts[0]
    recall = _cached_edge_slot_recall(
        features.get(stationary, {}),
        features.get(movable, {}),
    )
    proposals = [
        row for row in recall.get("proposals") or []
        if int(row.get("independent_evidence_count", 0)) >= 4
        and row.get("has_multi_evidence_support") is True
        and row.get("proposal_only") is True
        and row.get("review_required") is True
        and row.get("can_auto_accept") is False
        and row.get("transform_matrix") is not None
    ]
    if not proposals:
        return []

    generated = []
    for source_candidate in source_candidates[:1]:
        placements = source_candidate.get("placements") or {}
        if stationary not in placements or movable not in placements:
            continue
        stationary_world = placement_to_matrix(placements[stationary])
        for proposal in proposals[:max_candidates]:
            relative = np.asarray(proposal["transform_matrix"], dtype=float)
            if relative.shape != (4, 4):
                continue
            moving_world = stationary_world @ relative
            new_placements = json.loads(json.dumps(placements))
            new_placements[movable] = matrix_to_placement(moving_world)
            detail = {
                **proposal,
                "connection_id": connection.get("connection_id"),
                "stationary_part": stationary,
                "movable_part": movable,
                "refinement_phase": refinement_phase,
                "recall_status": recall.get("status"),
                "recall_reason": recall.get("reason"),
                "recall_audit": recall.get("audit") or {},
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            }
            history = list(
                source_candidate.get("edge_slot_history") or []
            ) + [detail]
            refined_ids = list(dict.fromkeys(
                str(row.get("connection_id")) for row in history
                if row.get("connection_id") is not None
            ))
            generated.append(_candidate_from_placements(
                features,
                new_placements,
                source_candidate,
                origin="repeated_bounded_edge_slot_recall",
                score_penalty=(
                    0.06
                    + 0.04 * float(proposal.get("length_relative_error", 0.0))
                    + 0.01 * float(
                        (proposal.get("floor_evidence") or {}).get(
                            "mirror_score", 0.0
                        )
                    )
                ),
                extra={
                    "proposal_only": True,
                    "review_required": True,
                    "can_auto_accept": False,
                    "edge_slot_interface": detail,
                    "edge_slot_history": history,
                    "edge_slot_refined_connection_ids": refined_ids,
                },
            ))
            if len(generated) >= max_candidates:
                break
    return generated[:max_candidates]


def _edge_slot_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    """Allocate a fixed review-only repeated edge-slot frontier."""

    connections = [
        row for row in graph.get("selected") or []
        if len(row.get("parts") or []) == 2
        and bool(set(row.get("relation_types") or []) & {
            PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE
        })
    ]
    generated = []
    for connection, quota in zip(
        connections, _fair_quotas(max_candidates, len(connections))
    ):
        generated.extend(_edge_slot_candidates_for_connection(
            source_candidates,
            connection,
            features,
            max_candidates=quota,
            refinement_phase="independent_edge_quota",
        ))
    return generated[:max_candidates]


def _cached_planar_footprint_recall(
    stationary_features: dict[str, Any],
    moving_features: dict[str, Any],
) -> dict[str, Any]:
    key = (id(stationary_features), id(moving_features))
    cached = _PLANAR_FOOTPRINT_RECALL_CACHE.get(key)
    if cached is not None:
        return cached
    # The downstream gate can only accept planes whose two extents match the
    # two non-thin moving OBB axes.  Apply that exact size ROI before NumPy
    # normalisation; a vendor board may contain >100k planar faces but only a
    # handful can possibly be a 72x75 socket or 31x133 slot.
    moving_obb = moving_features.get("obb") or {}
    moving_dimensions = list(moving_obb.get("dimensions") or [])
    target_footprint: list[float] = []
    if len(moving_dimensions) == 3:
        target_footprint = sorted(
            abs(float(value)) for value in moving_dimensions
        )[1:]
    raw_planes = list(stationary_features.get("planes") or [])
    filtered_planes = []
    if len(target_footprint) == 2:
        for plane in raw_planes:
            dimensions = plane.get("footprint_dimensions") or []
            if len(dimensions) < 2:
                continue
            candidate = sorted(abs(float(value)) for value in dimensions[:2])
            maximum_error = max(
                abs(candidate[index] - target_footprint[index])
                / max(candidate[index], target_footprint[index], 1e-9)
                for index in range(2)
            )
            if maximum_error <= 0.18:
                filtered_planes.append(plane)
    stationary_roi = {
        **stationary_features,
        "planes": filtered_planes,
    }
    try:
        result = recall_planar_footprint_proposals(
            stationary_roi,
            moving_features,
            maximum=64,
            # Radius/polarity/layer evidence remains supported by the reusable
            # module.  It is disabled in this large-case caller until a local
            # cylinder ROI is available; the all-pairs board×CPU loop is both
            # expensive and already proved to create no strict case4 match.
            enable_cylinder_layout_evidence=False,
        )
    except Exception as exc:
        result = {
            "schema_version": "planar_footprint.v1",
            "status": "unavailable",
            "reason": f"planar footprint recall failed: {exc}",
            "proposals": [],
            "audit": {"exception_type": type(exc).__name__},
            "proposal_only": True,
            "review_required": True,
            "can_auto_accept": False,
        }
    result.setdefault("audit", {}).update({
        "caller_plane_size_roi_enabled": True,
        "caller_raw_plane_count": len(raw_planes),
        "caller_roi_plane_count": len(filtered_planes),
        "caller_cylinder_layout_disabled_without_local_roi": True,
    })
    _PLANAR_FOOTPRINT_RECALL_CACHE[key] = result
    return result


def _planar_footprint_candidates_for_connection(
    source_candidates: list[dict[str, Any]],
    connection: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int,
    refinement_phase: str,
) -> list[dict[str, Any]]:
    """Materialise bounded multi-evidence footprint proposals for one edge."""

    if max_candidates <= 0:
        return []
    parts = list(connection.get("parts") or [])
    if len(parts) != 2:
        return []
    diagonals = {
        part: _bbox_diagonal_from_features(features.get(part, {}))
        for part in parts
    }
    stationary = max(parts, key=lambda part: diagonals[part])
    movable = parts[1] if parts[0] == stationary else parts[0]
    recall = _cached_planar_footprint_recall(
        features.get(stationary, {}),
        features.get(movable, {}),
    )
    proposals = [
        row for row in recall.get("proposals") or []
        if row.get("has_multi_evidence_support")
        and int(row.get("independent_evidence_count", 0)) >= 2
        and row.get("proposal_only") is True
        and row.get("can_auto_accept") is False
    ]
    if not proposals:
        return []

    generated = []
    active_sources = source_candidates[:1]
    for source_candidate, quota in zip(
        active_sources, _fair_quotas(max_candidates, len(active_sources))
    ):
        placements = source_candidate.get("placements") or {}
        if stationary not in placements or movable not in placements:
            continue
        stationary_world = placement_to_matrix(placements[stationary])
        for proposal in proposals[:quota]:
            relative = np.asarray(proposal["transform_matrix"], dtype=float)
            moving_world = stationary_world @ relative
            new_placements = json.loads(json.dumps(placements))
            new_placements[movable] = matrix_to_placement(moving_world)
            detail = {
                **proposal,
                "connection_id": connection.get("connection_id"),
                "stationary_part": stationary,
                "movable_part": movable,
                "refinement_phase": refinement_phase,
                "recall_status": recall.get("status"),
                "recall_audit": recall.get("audit") or {},
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            }
            history = list(
                source_candidate.get("planar_footprint_history") or []
            ) + [detail]
            refined_ids = list(dict.fromkeys(
                str(row.get("connection_id")) for row in history
                if row.get("connection_id") is not None
            ))
            size_error = max(
                (
                    float(value)
                    for evidence in proposal.get("evidence") or []
                    for value in evidence.get("relative_size_errors") or []
                ),
                default=0.0,
            )
            score_penalty = (
                0.08
                + 0.05 * size_error
                + (
                    0.0
                    if int(proposal.get("support_polarity", 0)) == 1
                    else 0.02
                )
            )
            generated.append(_candidate_from_placements(
                features,
                new_placements,
                source_candidate,
                origin="planar_footprint_socket_recall",
                score_penalty=score_penalty,
                extra={
                    "proposal_only": True,
                    "review_required": True,
                    "can_auto_accept": False,
                    "planar_footprint": detail,
                    "planar_footprint_history": history,
                    "planar_footprint_refined_connection_ids": refined_ids,
                },
            ))
            if len(generated) >= max_candidates:
                break
    return generated[:max_candidates]


def _planar_footprint_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    """Allocate a fixed footprint frontier, then compose safe star leaves."""

    connections = [
        row for row in graph.get("selected") or []
        if len(row.get("parts") or []) == 2
        and bool(set(row.get("relation_types") or []) & {
            PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE
        })
    ]
    if not connections or max_candidates <= 0:
        return []
    independent_budget = (
        max_candidates if len(connections) == 1
        else max(len(connections), (3 * max_candidates) // 4)
    )
    generated: list[dict[str, Any]] = []
    independent: list[list[dict[str, Any]]] = []
    for connection, quota in zip(
        connections, _fair_quotas(independent_budget, len(connections))
    ):
        rows = _planar_footprint_candidates_for_connection(
            source_candidates[:1],
            connection,
            features,
            max_candidates=quota,
            refinement_phase="independent_edge_quota",
        )
        independent.append(rows)
        generated.extend(rows)

    remaining = max(0, max_candidates - len(generated))
    composition_jobs = []
    for left_index in range(len(connections)):
        for right_index in range(left_index + 1, len(connections)):
            for left in independent[left_index]:
                left_detail = left.get("planar_footprint") or {}
                for right in independent[right_index]:
                    right_detail = right.get("planar_footprint") or {}
                    if (
                        left_detail.get("stationary_part")
                        != right_detail.get("stationary_part")
                        or not left_detail.get("movable_part")
                        or not right_detail.get("movable_part")
                        or left_detail.get("movable_part")
                        == right_detail.get("movable_part")
                    ):
                        continue
                    placements = json.loads(json.dumps(
                        left.get("placements") or {}
                    ))
                    right_movable = str(right_detail["movable_part"])
                    placements[right_movable] = json.loads(json.dumps(
                        (right.get("placements") or {})[right_movable]
                    ))
                    history = list(
                        left.get("planar_footprint_history") or []
                    ) + list(right.get("planar_footprint_history") or [])
                    refined_ids = list(dict.fromkeys(
                        str(row.get("connection_id")) for row in history
                        if row.get("connection_id") is not None
                    ))
                    candidate = _candidate_from_placements(
                        features,
                        placements,
                        left,
                        origin="planar_footprint_star_composition",
                        score_penalty=0.02,
                        extra={
                            "proposal_only": True,
                            "review_required": True,
                            "can_auto_accept": False,
                            "planar_footprint": right_detail,
                            "planar_footprint_history": history,
                            "planar_footprint_refined_connection_ids": refined_ids,
                        },
                    )
                    precheck = _pose_precheck(candidate, graph, features)
                    closure = precheck["constraint_closure"]
                    overlap = precheck["overlap_objective"]
                    independent_evidence = sum(
                        int(row.get("independent_evidence_count", 0))
                        for row in history
                    )
                    composition_jobs.append((
                        (
                            -int(closure.get("closed_connection_count", 0)),
                            -float(closure.get("closure_ratio", 0.0)),
                            int(overlap.get("severe_non_edge_overlap_count", 0)),
                            float(overlap.get("bbox_overlap_cost", 0.0)),
                            -independent_evidence,
                        ),
                        candidate,
                    ))
    composition_jobs.sort(key=lambda item: item[0])
    composition_buckets: dict[
        tuple[tuple[str, int, int], ...],
        list[tuple[tuple[Any, ...], dict[str, Any]]],
    ] = defaultdict(list)
    for item in composition_jobs:
        history = list(item[1].get("planar_footprint_history") or [])
        signature = tuple(sorted(
            (
                str(row.get("connection_id")),
                int(row.get("normal_sign", 0)),
                int(row.get("support_polarity", 0)),
            )
            for row in history
        ))
        composition_buckets[signature].append(item)
    bucket_order = sorted(
        composition_buckets,
        key=lambda signature: (
            -sum(row[2] == 1 for row in signature),
            tuple((row[1], row[2], row[0]) for row in signature),
        ),
    )
    selected_compositions = []
    cursor = 0
    while len(selected_compositions) < remaining:
        progressed = False
        for signature in bucket_order:
            rows = composition_buckets[signature]
            if cursor >= len(rows):
                continue
            selected_compositions.append(rows[cursor][1])
            progressed = True
            if len(selected_compositions) >= remaining:
                break
        if not progressed:
            break
        cursor += 1
    generated.extend(selected_compositions)
    return generated[:max_candidates]


def _obb_insertion_candidates_for_connection(
    source_candidates: list[dict[str, Any]],
    connection: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int,
    refinement_phase: str,
) -> list[dict[str, Any]]:
    """Generate role-stratified OBB insertion proposals for one edge."""

    if max_candidates <= 0:
        return []
    parts = list(connection.get("parts") or [])
    if len(parts) != 2:
        return []
    diagonals = {
        part: _bbox_diagonal_from_features(features.get(part, {}))
        for part in parts
    }
    stationary = max(parts, key=lambda part: diagonals[part])
    movable = parts[1] if parts[0] == stationary else parts[0]
    fixed_obb_local = features.get(stationary, {}).get("obb")
    moving_obb = features.get(movable, {}).get("obb")
    if not fixed_obb_local or not moving_obb:
        return []
    moving_dimensions = [
        float(value) for value in moving_obb.get("dimensions") or []
    ]
    moving_axes = moving_obb.get("axes") or []
    moving_center = moving_obb.get("center")
    if len(moving_dimensions) != 3 or len(moving_axes) != 3 or not moving_center:
        return []
    ordered_axes = sorted(range(3), key=lambda axis: moving_dimensions[axis])
    role_by_axis = {
        ordered_axes[0]: "shortest",
        ordered_axes[1]: "middle",
        ordered_axes[2]: "longest",
    }
    relation_priority = [PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE]
    matches = []
    for relation in relation_priority:
        matches.extend([
            row for row in connection.get("matches") or []
            if row.get("type") == relation
        ])
    if not matches:
        return []

    generated = []
    active_sources = source_candidates[:2]
    source_quotas = _fair_quotas(max_candidates, len(active_sources))
    for source_candidate, source_quota in zip(active_sources, source_quotas):
        if source_quota <= 0:
            continue
        placements = source_candidate.get("placements") or {}
        if stationary not in placements or movable not in placements:
            continue
        fixed_obb = _placed_obb(
            fixed_obb_local, placements.get(stationary, {})
        )
        try:
            frames = enumerate_axis_role_frames(
                fixed_obb, moving_obb, maximum=24
            )
        except ValueError:
            continue
        fixed_dimensions = [
            float(value) for value in fixed_obb.get("dimensions") or []
        ]
        jobs_by_role: dict[int, list[tuple[Any, ...]]] = {
            axis: [] for axis in ordered_axes
        }
        for match_rank, match in enumerate(matches[:2]):
            axis_data = _joinable_axis_data_for_match(
                match, stationary, movable, features, placements
            )
            if axis_data is None:
                continue
            (
                movable_feature_axis_local,
                movable_feature_anchor_local,
                anchor_axis,
                anchor_origin,
                _,
            ) = axis_data
            anchor_axis = _unit(anchor_axis)
            stationary_feature = None
            if match.get("type") in {PLANAR_MATE, PLANAR_ALIGN}:
                stationary_feature = _feature_for_match(
                    match, stationary, features, "planes"
                )
            anchor_kind = (
                "face_centroid"
                if stationary_feature and stationary_feature.get("centroid")
                else "weak_pocket"
                if match.get("type") == POCKET_MATE
                else "support_plane_origin"
            )
            for frame_rank, frame in enumerate(frames):
                rotation_axis_angle = frame.get("rotation_axis_angle")
                rotation_placement = {
                    "rotate_sequence": (
                        [{"axis_angle": rotation_axis_angle}]
                        if rotation_axis_angle else []
                    )
                }
                mapping = list(frame["axis_mapping"])
                for moving_axis in ordered_axes:
                    fixed_axis = int(mapping[moving_axis])
                    if fixed_axis >= len(fixed_dimensions):
                        continue
                    rotated_axis = _unit(transform_vector(
                        moving_axes[moving_axis], rotation_placement
                    ))
                    axis_compatibility = abs(_dot(rotated_axis, anchor_axis))
                    if axis_compatibility < 0.90:
                        # An axis-role proposal must materially align that OBB
                        # axis with the interface direction.  Merely relabeling
                        # the same rotation as short/middle/long would create
                        # fake diversity and starve real orientation variants.
                        continue
                    rotated_feature_axis = _unit(transform_vector(
                        movable_feature_axis_local, rotation_placement
                    ))
                    feature_axis_signed_dot = _dot(
                        rotated_feature_axis, anchor_axis
                    )
                    rotated_feature_anchor = transform_vector(
                        movable_feature_anchor_local, rotation_placement
                    )
                    depth_cap = min(
                        moving_dimensions[moving_axis],
                        fixed_dimensions[fixed_axis],
                    )
                    for support_polarity in (-1.0, 1.0):
                        insertion_axis = [
                            support_polarity * value for value in rotated_axis
                        ]
                        leading_local = [
                            float(moving_center[index])
                            + support_polarity
                            * 0.5
                            * moving_dimensions[moving_axis]
                            * float(moving_axes[moving_axis][index])
                            for index in range(3)
                        ]
                        rotated_leading = transform_vector(
                            leading_local, rotation_placement
                        )
                        for depth_index, depth_fraction in enumerate(
                            (0.0, 0.25, 0.5, 0.75, 1.0)
                        ):
                            sampled_depth = depth_fraction * depth_cap
                            for anchor_strategy_rank, (
                                anchor_strategy,
                                rotated_moving_anchor,
                            ) in enumerate((
                                (
                                    "matched_feature_centroid",
                                    rotated_feature_anchor,
                                ),
                                ("obb_leading_support", rotated_leading),
                            )):
                                translation = [
                                    float(anchor_origin[index])
                                    - rotated_moving_anchor[index]
                                    + sampled_depth * insertion_axis[index]
                                    for index in range(3)
                                ]
                                if (
                                    anchor_strategy
                                    == "matched_feature_centroid"
                                ):
                                    phase_priority = (
                                        0 if depth_index == 0
                                        else 10 + depth_index
                                    )
                                else:
                                    support_depth_priority = {
                                        1.0: 1,
                                        0.5: 2,
                                        0.0: 3,
                                        0.75: 4,
                                        0.25: 5,
                                    }
                                    phase_priority = support_depth_priority[
                                        float(depth_fraction)
                                    ]
                                priority = (
                                    phase_priority,
                                    match_rank,
                                    -feature_axis_signed_dot,
                                    -axis_compatibility,
                                    -float(frame.get(
                                        "dimension_order_score", 0.0
                                    )),
                                    depth_index,
                                    anchor_strategy_rank,
                                    int(support_polarity < 0.0),
                                    frame_rank,
                                )
                                jobs_by_role[moving_axis].append((
                                    priority,
                                    frame,
                                    rotation_placement,
                                    translation,
                                    insertion_axis,
                                    fixed_axis,
                                    sampled_depth,
                                    depth_fraction,
                                    anchor_kind,
                                    anchor_strategy,
                                    match,
                                    support_polarity,
                                    axis_compatibility,
                                    feature_axis_signed_dot,
                                ))
        for rows in jobs_by_role.values():
            rows.sort(key=lambda item: item[0])

        # Recall must cover insertion depth before repeatedly spending the
        # frontier on a zero-depth face-centroid contact.  The latter is a
        # useful seed, but it cannot assemble a slide-in module by itself.
        # Within each depth layer, retain the existing short/middle/long role
        # diversity; every generated row remains review-only.
        # Keep the zero-depth anchor as one of the early strata for ordinary
        # flush mates, but never let it consume the whole frontier.
        depth_layers = (1.0, 0.0, 0.5, 0.75, 0.25)
        rows_by_role_and_depth: dict[
            int, dict[float, list[tuple[Any, ...]]]
        ] = {
            axis: {depth: [] for depth in depth_layers}
            for axis in ordered_axes
        }
        for moving_axis, rows in jobs_by_role.items():
            for row in rows:
                depth_fraction = float(row[7])
                if depth_fraction in rows_by_role_and_depth[moving_axis]:
                    rows_by_role_and_depth[moving_axis][depth_fraction].append(
                        row
                    )
        for per_depth in rows_by_role_and_depth.values():
            for rows in per_depth.values():
                rows.sort(key=lambda item: item[0])
        cursor = 0
        generated_for_source = 0
        while (
            len(generated) < max_candidates
            and generated_for_source < source_quota
        ):
            progressed = False
            for depth_fraction in depth_layers:
                for moving_axis in ordered_axes:
                    rows = rows_by_role_and_depth[moving_axis][depth_fraction]
                    if cursor >= len(rows):
                        continue
                    progressed = True
                    (
                        _, frame, rotation_placement, translation,
                        insertion_axis, fixed_axis, sampled_depth,
                        _depth_fraction, anchor_kind, anchor_strategy, match,
                        support_polarity, axis_compatibility,
                        feature_axis_signed_dot,
                    ) = rows[cursor]
                    placement = dict(rotation_placement)
                    placement["translate"] = translation
                    new_placements = json.loads(json.dumps(placements))
                    new_placements[movable] = placement
                    detail = {
                    "connection_id": connection["connection_id"],
                    "stationary_part": stationary,
                    "movable_part": movable,
                    "moving_insertion_axis": moving_axis,
                    "moving_axis_role": role_by_axis[moving_axis],
                    "fixed_target_axis": fixed_axis,
                    "axis_mapping": frame["axis_mapping"],
                    "axis_signs": frame["axis_signs"],
                    "support_polarity": int(support_polarity),
                    "insertion_axis_world": insertion_axis,
                    "anchor_kind": anchor_kind,
                    "anchor_strategy": anchor_strategy,
                    "anchor_origin_world": anchor_origin,
                    "anchor_axis_world": anchor_axis,
                    "movable_feature_anchor_local": (
                        movable_feature_anchor_local
                    ),
                    "match_type": match.get("type"),
                    "match_feat_a_idx": match.get("feat_a_idx"),
                    "match_feat_b_idx": match.get("feat_b_idx"),
                    "sampled_depth_mm": sampled_depth,
                    "sampled_depth_fraction": depth_fraction,
                    "depth_verified": False,
                    "axis_compatibility": axis_compatibility,
                    "obb_axis_alignment_threshold": 0.90,
                    "obb_axis_aligned_interface_only": True,
                    "feature_axis_signed_dot": feature_axis_signed_dot,
                    "frame_dimension_order_score": frame.get(
                        "dimension_order_score"
                    ),
                    "rotation_axis_angle": frame.get(
                        "rotation_axis_angle"
                    ),
                    "translation": translation,
                    "source_candidate_origin": source_candidate.get(
                        "candidate_origin", "solver_primary"
                    ),
                    "refinement_phase": refinement_phase,
                    "review_required": True,
                    "can_auto_accept": False,
                }
                    history = list(
                        source_candidate.get("obb_insertion_history") or []
                    ) + [detail]
                    refined_ids = list(dict.fromkeys(
                        str(row.get("connection_id")) for row in history
                        if row.get("connection_id") is not None
                    ))
                    generated.append(_candidate_from_placements(
                    features,
                    new_placements,
                    source_candidate,
                    origin=(
                        "obb_two_edge_insertion_composition"
                        if len(refined_ids) >= 2
                        else "obb_insertion_axis_role_search"
                    ),
                    score_penalty=(
                        0.16
                        + 0.04 * (sampled_depth / max(depth_cap, 1.0))
                        + 0.03 * (1.0 - axis_compatibility)
                        + (0.02 if anchor_kind != "face_centroid" else 0.0)
                    ),
                    extra={
                        "proposal_only": True,
                        "review_required": True,
                        "can_auto_accept": False,
                        "obb_insertion": detail,
                        "obb_insertion_history": history,
                        "obb_refined_connection_ids": refined_ids,
                    },
                    ))
                    generated_for_source += 1
                    if (
                        len(generated) >= max_candidates
                        or generated_for_source >= source_quota
                    ):
                        break
                if (
                    len(generated) >= max_candidates
                    or generated_for_source >= source_quota
                ):
                    break
            if not progressed:
                break
            cursor += 1
        if len(generated) >= max_candidates:
            break
    return generated[:max_candidates]


def _obb_insertion_pose_candidates(
    source_candidates: list[dict[str, Any]],
    graph: dict[str, Any],
    features: dict[str, Any],
    *,
    max_candidates: int = 24,
) -> list[dict[str, Any]]:
    """Allocate OBB role proposals per edge, then compose two leaf moves."""

    connections = [
        row for row in graph.get("selected") or []
        if len(row.get("parts") or []) == 2
        and all(features.get(part, {}).get("obb") for part in row["parts"])
        and bool(set(row.get("relation_types") or []) & {
            PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE
        })
    ]
    if not connections or max_candidates <= 0:
        return []
    independent_budget = (
        max_candidates if len(connections) == 1
        else max(len(connections), (3 * max_candidates) // 4)
    )
    generated = []
    independent = []
    for connection, quota in zip(
        connections, _fair_quotas(independent_budget, len(connections))
    ):
        rows = _obb_insertion_candidates_for_connection(
            source_candidates[:1],
            connection,
            features,
            max_candidates=quota,
            refinement_phase="independent_edge_quota",
        )
        independent.append(rows)
        generated.extend(rows)
    remaining = max(0, max_candidates - len(generated))
    # A star assembly's leaf placements are independent in the carrier frame.
    # Compose the already-stratified single-edge proposals directly, then use
    # algebraic group closure only to rank this fixed Cartesian frontier.  No
    # additional solver beam or exact-collision work is introduced.
    composition_jobs = []
    for left_index in range(len(connections)):
        for right_index in range(left_index + 1, len(connections)):
            for left in independent[left_index]:
                left_detail = left.get("obb_insertion") or {}
                for right in independent[right_index]:
                    right_detail = right.get("obb_insertion") or {}
                    if (
                        left_detail.get("stationary_part")
                        != right_detail.get("stationary_part")
                        or not left_detail.get("movable_part")
                        or not right_detail.get("movable_part")
                        or left_detail.get("movable_part")
                        == right_detail.get("movable_part")
                    ):
                        continue
                    placements = json.loads(json.dumps(
                        left.get("placements") or {}
                    ))
                    right_movable = str(right_detail["movable_part"])
                    placements[right_movable] = json.loads(json.dumps(
                        (right.get("placements") or {})[right_movable]
                    ))
                    history = list(
                        left.get("obb_insertion_history") or []
                    ) + list(right.get("obb_insertion_history") or [])
                    refined_ids = list(dict.fromkeys(
                        str(row.get("connection_id")) for row in history
                        if row.get("connection_id") is not None
                    ))
                    candidate = _candidate_from_placements(
                        features,
                        placements,
                        left,
                        origin="obb_two_edge_insertion_composition",
                        score_penalty=0.02,
                        extra={
                            "proposal_only": True,
                            "review_required": True,
                            "can_auto_accept": False,
                            "obb_insertion": right_detail,
                            "obb_insertion_history": history,
                            "obb_refined_connection_ids": refined_ids,
                        },
                    )
                    precheck = _pose_precheck(candidate, graph, features)
                    closure = precheck["constraint_closure"]
                    overlap = precheck["overlap_objective"]
                    satisfied_count = sum(
                        len(row.get("satisfied_relation_types") or [])
                        for row in closure.get("connections") or []
                    )
                    candidate["obb_composition_precheck"] = {
                        "closed_connection_count": closure.get(
                            "closed_connection_count", 0
                        ),
                        "connection_count": closure.get(
                            "connection_count", 0
                        ),
                        "closure_ratio": closure.get("closure_ratio", 0.0),
                        "satisfied_relation_type_count": satisfied_count,
                        "severe_non_edge_overlap_count": overlap.get(
                            "severe_non_edge_overlap_count", 0
                        ),
                    }
                    composition_jobs.append((
                        (
                            -int(closure.get("closed_connection_count", 0)),
                            -float(closure.get("closure_ratio", 0.0)),
                            -int(satisfied_count),
                            int(overlap.get(
                                "severe_non_edge_overlap_count", 0
                            )),
                            float(overlap.get("bbox_overlap_cost", 0.0)),
                            -sum(
                                float(row.get("axis_compatibility", 0.0))
                                for row in history
                            ),
                        ),
                        candidate,
                    ))
    composition_jobs.sort(key=lambda item: item[0])
    for _, candidate in composition_jobs[:remaining]:
        generated.append(candidate)

    # Non-star chains cannot safely merge leaf transforms.  Retain the old
    # bounded coordinate fallback only when no common-carrier composition was
    # possible.
    remaining = max(0, max_candidates - len(generated))
    if remaining and not composition_jobs:
        directions = [
            (source_index, target_index)
            for source_index, rows in enumerate(independent) if rows
            for target_index in range(len(connections))
            if target_index != source_index
        ]
        for (source_index, target_index), quota in zip(
            directions, _fair_quotas(remaining, len(directions))
        ):
            if quota <= 0:
                continue
            rows = _obb_insertion_candidates_for_connection(
                independent[source_index][:2],
                connections[target_index],
                features,
                max_candidates=min(
                    quota, max_candidates - len(generated)
                ),
                refinement_phase="two_edge_coordinate_composition",
            )
            generated.extend(rows)
            if len(generated) >= max_candidates:
                break
    return generated[:max_candidates]


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
    joinable_parameter_candidates: list[dict[str, Any]] = []
    obb_insertion_candidates: list[dict[str, Any]] = []
    planar_footprint_candidates: list[dict[str, Any]] = []
    axial_compound_candidates: list[dict[str, Any]] = []
    enclosure_bay_candidates: list[dict[str, Any]] = []
    edge_slot_candidates: list[dict[str, Any]] = []
    carrier_open_side_candidates: list[dict[str, Any]] = []
    axial_compound_budget = 8 if large_step_case else 24
    joinable_parameter_budget = 24 if large_step_case else 200
    socket_budget = 0
    enclosure_bay_budget = 0
    edge_slot_budget = 0
    planar_footprint_budget = 0
    carrier_open_side_budget = 0
    obb_budget = 0
    if not large_step_case:
        candidates.extend(
            _axial_slide_pose_candidates(
                search, graph, features, max_candidates=160
            )
        )
        # Pair-wise SDF optima often place several satellites at the same
        # high-contact location on a central shaft.  Relax a bounded number of
        # composed group seeds along the shared axial DOF before exact group
        # collision validation.
        composed_sources = [
            row for row in candidates
            if row.get("candidate_origin")
            == "joinable_pair_pose_composition"
        ][:8]
        composed_relaxation_count = 0
        for source_candidate in composed_sources:
            local_search = {
                **search,
                **source_candidate,
                "placements": source_candidate["placements"],
            }
            generated = _axial_slide_pose_candidates(
                local_search, graph, features, max_candidates=24
            )
            for row in generated:
                row["candidate_origin"] = (
                    "joinable_group_axial_relaxation"
                )
                row["joinable_group_pose"] = source_candidate.get(
                    "joinable_group_pose"
                )
            candidates.extend(generated)
            composed_relaxation_count += len(generated)
        axial_compound_candidates = _axial_compound_pose_candidates(
            candidates,
            graph,
            features,
            max_candidates=axial_compound_budget,
        )
        candidates.extend(axial_compound_candidates)
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
        # Compound proposals share the pre-existing 200-row structured budget
        # with residual parameter search; enabling the provider cannot expand
        # the fixed frontier.
        joinable_parameter_budget = max(
            0, 200 - len(axial_compound_candidates)
        )
        joinable_parameter_candidates = _joinable_pose_parameter_candidates(
            axial_compound_candidates + candidates,
            graph,
            features,
            max_candidates=joinable_parameter_budget,
        )
        candidates.extend(joinable_parameter_candidates)
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
        # Keep the large-case structured frontier fixed at 48 candidates:
        # 24 proposals are shared by compound axial, repeated enclosure bays,
        # multi-plane footprint, and residual OBB recall; 24 retain parameter
        # search.  Adding a provider never expands this fixed frontier.
        pre_obb_candidates = list(candidates)
        axial_compound_candidates = _axial_compound_pose_candidates(
            pre_obb_candidates,
            graph,
            features,
            max_candidates=axial_compound_budget,
        )
        candidates.extend(axial_compound_candidates)
        socket_budget = max(0, 24 - len(axial_compound_candidates))
        edge_slot_budget = min(8, socket_budget)
        edge_slot_candidates = _edge_slot_pose_candidates(
            pre_obb_candidates,
            graph,
            features,
            max_candidates=edge_slot_budget,
        )
        candidates.extend(edge_slot_candidates)
        remaining_socket_budget = max(
            0, socket_budget - len(edge_slot_candidates)
        )
        enclosure_bay_budget = min(8, remaining_socket_budget)
        enclosure_bay_candidates = _enclosure_bay_pose_candidates(
            pre_obb_candidates,
            graph,
            features,
            max_candidates=enclosure_bay_budget,
        )
        candidates.extend(enclosure_bay_candidates)
        structured_remaining_budget = max(
            0,
            remaining_socket_budget - len(enclosure_bay_candidates),
        )
        # Reserve at most two of the pre-existing structured slots for a
        # group-level same-carrier composition.  If no such pairing is
        # supported, the unused slots return to the OBB fallback below; this
        # never expands the fixed large-STEP frontier.
        carrier_open_side_budget = min(2, structured_remaining_budget)
        planar_footprint_budget = max(
            0,
            structured_remaining_budget - carrier_open_side_budget,
        )
        structured_sources = (
            edge_slot_candidates
            + enclosure_bay_candidates
            + pre_obb_candidates
        )
        planar_footprint_candidates = _planar_footprint_pose_candidates(
            structured_sources,
            graph,
            features,
            max_candidates=planar_footprint_budget,
        )
        candidates.extend(planar_footprint_candidates)
        carrier_open_side_candidates = (
            _carrier_open_side_consistent_candidates(
                edge_slot_candidates,
                planar_footprint_candidates,
                features,
                max_candidates=carrier_open_side_budget,
            )
        )
        candidates.extend(carrier_open_side_candidates)
        obb_budget = max(
            0,
            structured_remaining_budget
            - len(planar_footprint_candidates)
            - len(carrier_open_side_candidates),
        )
        obb_insertion_candidates = _obb_insertion_pose_candidates(
            structured_sources, graph, features, max_candidates=obb_budget
        )
        candidates.extend(obb_insertion_candidates)
        joinable_parameter_candidates = _joinable_pose_parameter_candidates(
            axial_compound_candidates
            + edge_slot_candidates
            + enclosure_bay_candidates
            + planar_footprint_candidates
            + carrier_open_side_candidates
            + obb_insertion_candidates
            + pre_obb_candidates,
            graph,
            features,
            max_candidates=24,
        )
        candidates.extend(joinable_parameter_candidates)
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
    selected_connection_ids = [
        str(row.get("connection_id"))
        for row in graph.get("selected") or []
        if row.get("connection_id") is not None
    ]
    obb_eligible_connection_ids = {
        str(row.get("connection_id"))
        for row in graph.get("selected") or []
        if row.get("connection_id") is not None
        and len(row.get("parts") or []) == 2
        and all(
            features.get(part, {}).get("obb")
            for part in row.get("parts") or []
        )
        and bool(set(row.get("relation_types") or []) & {
            PLANAR_MATE, PLANAR_ALIGN, POCKET_MATE
        })
    }
    edge_coverage = []
    obb_edge_coverage = []
    footprint_edge_coverage = []
    compound_edge_coverage = []
    compound_eligible_connection_ids = {
        str(row.get("connection_id"))
        for row in graph.get("selected") or []
        if row.get("connection_id") is not None
        and len(row.get("parts") or []) == 2
        and bool(set(row.get("relation_types") or []) & {COAXIAL, CLEARANCE})
    }
    for connection_id in selected_connection_ids:
        independent_count = sum(
            row.get("joinable_pose_search", {}).get("connection_id")
            == connection_id
            and row.get("joinable_pose_search", {}).get("refinement_phase")
            == "independent_edge_quota"
            for row in joinable_parameter_candidates
        )
        composed_count = sum(
            connection_id
            in set(row.get("joinable_pose_refined_connection_ids") or [])
            and len(set(row.get("joinable_pose_refined_connection_ids") or [])) >= 2
            for row in joinable_parameter_candidates
        )
        edge_coverage.append({
            "connection_id": connection_id,
            "independent_candidate_count": int(independent_count),
            "two_edge_composed_candidate_count": int(composed_count),
            "received_independent_quota": bool(independent_count),
        })
        footprint_independent_count = sum(
            row.get("planar_footprint", {}).get("connection_id")
            == connection_id
            and row.get("planar_footprint", {}).get("refinement_phase")
            == "independent_edge_quota"
            for row in planar_footprint_candidates
        )
        footprint_composed_count = sum(
            connection_id
            in set(row.get(
                "planar_footprint_refined_connection_ids"
            ) or [])
            and len(set(row.get(
                "planar_footprint_refined_connection_ids"
            ) or [])) >= 2
            for row in planar_footprint_candidates
        )
        footprint_edge_coverage.append({
            "connection_id": connection_id,
            "independent_candidate_count": int(
                footprint_independent_count
            ),
            "two_edge_composed_candidate_count": int(
                footprint_composed_count
            ),
            "received_independent_quota": bool(
                footprint_independent_count
            ),
        })
        if connection_id in compound_eligible_connection_ids:
            compound_rows = [
                row for row in axial_compound_candidates
                if row.get("axial_compound_interface", {}).get(
                    "connection_id"
                ) == connection_id
            ]
            compound_edge_coverage.append({
                "connection_id": connection_id,
                "candidate_count": len(compound_rows),
                "received_independent_quota": bool(compound_rows),
                "review_only_count": sum(
                    bool(row.get("proposal_only")) for row in compound_rows
                ),
                "phase_orbits": [
                    row.get("axial_compound_interface", {}).get(
                        "phase_orbit_degrees"
                    )
                    for row in compound_rows
                ],
            })
        if connection_id in obb_eligible_connection_ids:
            obb_independent_count = sum(
                row.get("obb_insertion", {}).get("connection_id")
                == connection_id
                and row.get("obb_insertion", {}).get("refinement_phase")
                == "independent_edge_quota"
                for row in obb_insertion_candidates
            )
            obb_composed_count = sum(
                connection_id
                in set(row.get("obb_refined_connection_ids") or [])
                and len(set(row.get("obb_refined_connection_ids") or [])) >= 2
                for row in obb_insertion_candidates
            )
            obb_edge_coverage.append({
                "connection_id": connection_id,
                "independent_candidate_count": int(obb_independent_count),
                "two_edge_composed_candidate_count": int(obb_composed_count),
                "received_independent_quota": bool(obb_independent_count),
            })
    augmented["candidate_augmentation"] = {
        "identity_candidate_added": True,
        "planar_slide_search_enabled": True,
        "pocket_depth_search_enabled": True,
        "joinable_multi_axial_pose_search_enabled": True,
        "joinable_pose_parameter_search_enabled": True,
        "joinable_pose_parameter_budget": joinable_parameter_budget,
        "joinable_pose_parameter_generated_count": len(
            joinable_parameter_candidates
        ),
        "joinable_two_edge_composition_count": sum(
            row.get("candidate_origin") == "joinable_two_edge_pose_composition"
            for row in joinable_parameter_candidates
        ),
        "joinable_edge_quota_coverage": edge_coverage,
        "all_selected_edges_received_independent_quota": bool(edge_coverage)
        and all(row["received_independent_quota"] for row in edge_coverage),
        "obb_insertion_axis_role_search_enabled": bool(large_step_case),
        "obb_insertion_axis_role_budget": (
            obb_budget if large_step_case else 0
        ),
        "obb_insertion_axis_role_generated_count": len(
            obb_insertion_candidates
        ),
        "obb_two_edge_composition_count": sum(
            row.get("candidate_origin") == "obb_two_edge_insertion_composition"
            for row in obb_insertion_candidates
        ),
        "obb_edge_quota_coverage": obb_edge_coverage,
        "all_obb_eligible_edges_received_independent_quota": (
            bool(obb_edge_coverage)
            and all(
                row["received_independent_quota"]
                for row in obb_edge_coverage
            )
        ),
        "planar_footprint_search_enabled": bool(large_step_case),
        "planar_footprint_budget": (
            planar_footprint_budget if large_step_case else 0
        ),
        "carrier_open_side_composition_budget": (
            carrier_open_side_budget if large_step_case else 0
        ),
        "carrier_open_side_composition_generated_count": len(
            carrier_open_side_candidates
        ),
        "planar_footprint_generated_count": len(
            planar_footprint_candidates
        ),
        "planar_footprint_star_composition_count": sum(
            row.get("candidate_origin")
            == "planar_footprint_star_composition"
            for row in planar_footprint_candidates
        ),
        "planar_footprint_edge_coverage": footprint_edge_coverage,
        "enclosure_bay_search_enabled": bool(large_step_case),
        "enclosure_bay_budget": (
            enclosure_bay_budget if large_step_case else 0
        ),
        "enclosure_bay_generated_count": len(enclosure_bay_candidates),
        "enclosure_bay_review_only_count": sum(
            bool(row.get("proposal_only")) for row in enclosure_bay_candidates
        ),
        "enclosure_bay_connection_ids": sorted({
            str(row.get("enclosure_bay", {}).get("connection_id"))
            for row in enclosure_bay_candidates
            if row.get("enclosure_bay", {}).get("connection_id") is not None
        }),
        "edge_slot_search_enabled": bool(large_step_case),
        "edge_slot_budget": edge_slot_budget if large_step_case else 0,
        "edge_slot_generated_count": len(edge_slot_candidates),
        "edge_slot_review_only_count": sum(
            bool(row.get("proposal_only")) for row in edge_slot_candidates
        ),
        "edge_slot_connection_ids": sorted({
            str(row.get("edge_slot_interface", {}).get("connection_id"))
            for row in edge_slot_candidates
            if row.get("edge_slot_interface", {}).get("connection_id")
            is not None
        }),
        "axial_compound_interface_search_enabled": True,
        "axial_compound_interface_budget": axial_compound_budget,
        "axial_compound_interface_generated_count": len(
            axial_compound_candidates
        ),
        "axial_compound_star_composition_count": sum(
            row.get("candidate_origin")
            == "axial_compound_star_composition"
            for row in axial_compound_candidates
        ),
        "axial_group_symmetric_centering_count": sum(
            row.get("candidate_origin")
            == "axial_group_symmetric_centering"
            for row in axial_compound_candidates
        ),
        "axial_compound_review_only_count": sum(
            bool(row.get("proposal_only"))
            for row in axial_compound_candidates
        ),
        "axial_compound_edge_coverage": compound_edge_coverage,
        "all_axial_compound_eligible_edges_received_quota": (
            bool(compound_edge_coverage)
            and all(
                row["received_independent_quota"]
                for row in compound_edge_coverage
            )
        ),
        "large_step_structured_candidate_budget": (
            48 if large_step_case else None
        ),
        "large_step_case": large_step_case,
        "total_surface_count": total_surface_count,
        "total_pose_candidates_after_augmentation": len(unique),
        "joinable_group_axial_relaxation_count": (
            composed_relaxation_count if not large_step_case else 0
        ),
    }
    return augmented


def _topology_graph(
    graph: dict[str, Any],
    topology: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a graph view for one retained topology frontier row."""

    if not topology:
        return graph
    selected = [
        # The public contract distinguishes skeleton/support roles only.  The
        # active topology id/rank is audited separately on the pose candidate.
        {**row, "selection_role": "connected_skeleton"}
        for row in topology.get("rows") or []
    ]
    if not selected:
        return graph
    touched = {
        str(part)
        for row in selected
        for part in row.get("parts") or []
    }
    output = dict(graph)
    output["selected"] = selected
    output["connected"] = bool(topology.get("connected"))
    output["unresolved_parts"] = sorted(
        set(str(part) for part in graph.get("part_ids") or []) - touched
    )
    output["active_topology_id"] = topology.get("topology_id")
    output["active_topology_rank"] = topology.get("rank")
    return output


def _candidate_topology_graph(
    candidate: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    rows = candidate.get("_topology_selected")
    if not rows:
        return fallback
    output = dict(fallback)
    output["selected"] = rows
    output["connected"] = True
    output["unresolved_parts"] = []
    output["active_topology_id"] = candidate.get("topology_id")
    output["active_topology_rank"] = candidate.get("topology_rank")
    return output


def _solve_topology_pose_frontier(
    features: dict[str, Any],
    scored: list[dict[str, Any]],
    graph: dict[str, Any],
    *,
    beam_width: int,
    joinable_pose_dir: str | Path | None,
) -> dict[str, Any]:
    """Run bounded pose search for each retained graph topology.

    The pair-score rank orders work only.  Every retained topology reaches the
    same closure and exact-validation stages, so a rank-1 accidental edge
    cannot make a lower-ranked but physically coherent topology unreachable.
    """

    topologies = list(graph.get("topology_frontier") or [])
    if not topologies:
        topologies = [{
            "topology_id": "topology:legacy_selected",
            "rank": 1,
            "connected": bool(graph.get("connected")),
            "rows": list(graph.get("selected") or []),
        }]
    combined_candidates: list[dict[str, Any]] = []
    audits = []
    primary_search: dict[str, Any] | None = None
    for topology in topologies:
        local_graph = _topology_graph(graph, topology)
        selected_pairs = {
            canonical_pair(row["parts"]) for row in local_graph.get("selected") or []
        }
        solver_matches = [
            row for row in scored if canonical_pair(row["parts"]) in selected_pairs
        ]
        local_search = solve_small_assembly(
            features,
            solver_matches,
            beam_width=beam_width,
            target_branching=min(3, max(1, len(features) - 1)),
        )
        local_search = _inject_joinable_group_pose_seed(
            local_search, features, joinable_pose_dir
        )
        local_search = _augment_pose_candidates(
            local_search, local_graph, features
        )
        if primary_search is None:
            primary_search = local_search
        before = len(combined_candidates)
        for candidate in local_search.get("pose_candidates") or []:
            row = dict(candidate)
            row["topology_id"] = topology.get("topology_id")
            row["topology_rank"] = topology.get("rank")
            row["_topology_selected"] = list(local_graph.get("selected") or [])
            combined_candidates.append(row)
        audits.append({
            "topology_id": topology.get("topology_id"),
            "topology_rank": topology.get("rank"),
            "parts": [row.get("parts") for row in local_graph.get("selected") or []],
            "solver_match_count": len(solver_matches),
            "pose_candidate_count": len(combined_candidates) - before,
            "search_status": local_search.get("status"),
        })
    assert primary_search is not None
    combined = dict(primary_search)
    combined["pose_candidates"] = combined_candidates
    combined["complete_pose_candidate_count"] = len(combined_candidates)
    combined["topology_search_audit"] = {
        "schema_version": "known_group_topology_pose_frontier.v1",
        "topology_count": len(audits),
        "rows": audits,
        "selection_boundary": (
            "Pair scores order a bounded topology frontier; closure and exact "
            "validation select a pose, and no topology score can auto-accept."
        ),
    }
    augmentation = dict(combined.get("candidate_augmentation") or {})
    augmentation["topology_frontier_count"] = len(audits)
    augmentation["total_pose_candidates_after_topology_union"] = len(combined_candidates)
    combined["candidate_augmentation"] = augmentation
    return combined


def _exact_candidate_has_containment(
    precheck: dict[str, Any],
) -> bool:
    return any(
        row.get("pair_kind") == "selected_pair_containment"
        for row in precheck.get("overlap_objective", {}).get(
            "bbox_overlap_items", []
        )
    )


def _exact_candidate_quality(
    item: tuple[int, dict[str, Any], dict[str, Any]],
) -> tuple[float, float, float, int]:
    rank, candidate, precheck = item
    closure = precheck.get("constraint_closure") or {}
    return (
        float(closure.get("closure_ratio", 0.0)),
        float(precheck.get("group_pose_precheck_score", 0.0)),
        float(candidate.get("total_score", 0.0)),
        -int(rank),
    )


def _exact_candidate_topology_key(
    candidate: dict[str, Any],
) -> tuple[int, str]:
    raw_rank = candidate.get("topology_rank")
    try:
        topology_rank = int(raw_rank)
    except (TypeError, ValueError):
        topology_rank = 1_000_000_000
    return (
        topology_rank,
        str(candidate.get("topology_id") or "legacy"),
    )


def _obb_role_signature(candidate: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    history = list(candidate.get("obb_insertion_history") or [])
    if not history and candidate.get("obb_insertion"):
        history = [candidate["obb_insertion"]]
    rows = {
        (
            str(row.get("connection_id") or "unknown_connection"),
            str(row.get("moving_axis_role") or "unknown_role"),
        )
        for row in history
    }
    footprint_history = list(
        candidate.get("planar_footprint_history") or []
    )
    if not footprint_history and candidate.get("planar_footprint"):
        footprint_history = [candidate["planar_footprint"]]
    rows.update({
        (
            str(row.get("connection_id") or "unknown_connection"),
            "footprint:"
            + str(row.get("equivalence_class_id") or "unknown_anchor")
            + ":phase=" + str(row.get("phase_degrees"))
            + ":support=" + str(row.get("support_polarity"))
            + ":normal=" + str(row.get("normal_sign")),
            # Normal sign is a distinct proper orientation for an asymmetric
            # thin component and must reach the bounded exact frontier.
        )
        for row in footprint_history
    })
    enclosure_history = list(candidate.get("enclosure_bay_history") or [])
    if not enclosure_history and candidate.get("enclosure_bay"):
        enclosure_history = [candidate["enclosure_bay"]]
    rows.update({
        (
            str(row.get("connection_id") or "unknown_connection"),
            "enclosure_bay:slot=" + str(row.get("slot_index"))
            + ":polarity=" + str(row.get("depth_polarity"))
            + ":opening=" + str(row.get("opening_polarity")),
        )
        for row in enclosure_history
    })
    edge_slot_history = list(candidate.get("edge_slot_history") or [])
    if not edge_slot_history and candidate.get("edge_slot_interface"):
        edge_slot_history = [candidate["edge_slot_interface"]]
    rows.update({
        (
            str(row.get("connection_id") or "unknown_connection"),
            "edge_slot:family=" + str(row.get("slot_family_id"))
            + ":slot=" + str(row.get("slot_rank"))
            + ":long_sign=" + str(row.get("long_axis_sign")),
        )
        for row in edge_slot_history
    })
    compound_history = list(candidate.get("axial_compound_history") or [])
    if not compound_history and candidate.get("axial_compound_interface"):
        compound_history = [candidate["axial_compound_interface"]]
    rows.update({
        (
            str(row.get("connection_id") or "unknown_connection"),
            "axial_compound:polarity=" + str(row.get("axis_polarity"))
            + ":phase=" + str(row.get("phase_degrees"))
            + ":symmetry=" + str(row.get("whole_part_symmetry_order")),
        )
        for row in compound_history
    })
    centering_history = list(
        candidate.get("axial_group_centering_history") or []
    )
    if not centering_history and candidate.get("axial_group_centering"):
        centering_history = [candidate["axial_group_centering"]]
    rows.update({
        (
            str(row.get("connection_id") or "unknown_connection"),
            "axial_group_centering:support="
            + str(row.get("support_connection_id"))
            + ":shaft=" + str(row.get("shaft_part")),
        )
        for row in centering_history
    })
    ordered = sorted(rows)
    return tuple(ordered) if ordered else (("baseline", "baseline"),)


def _plan_exact_rank_budget(
    prechecked: list[tuple[int, dict[str, Any], dict[str, Any]]],
    *,
    budget: int,
    large_step_case: bool,
) -> dict[str, Any]:
    """Plan a bounded exact frontier with topology and OBB-role coverage."""

    limit = max(0, int(budget))
    ranked = sorted(
        prechecked,
        key=lambda item: (
            bool(item[2]["constraint_closure"].get("fully_closed")),
            *_exact_candidate_quality(item),
        ),
        reverse=True,
    )
    if limit == 0:
        return {
            "selected_ranks": [],
            "selection_reason_by_rank": {},
            "summary": {
                "budget": limit,
                "eligible_candidate_count": 0,
                "eligible_topologies": [],
                "covered_topologies": [],
                "uncovered_topologies": [],
                "covered_obb_role_signatures": [],
            },
        }
    if not large_step_case:
        selected = [int(rank) for rank, _, _ in ranked[:limit]]
        return {
            "selected_ranks": selected,
            "selection_reason_by_rank": {
                rank: "bounded_quality_frontier" for rank in selected
            },
            "summary": {
                "budget": limit,
                "eligible_candidate_count": len(ranked),
                "eligible_topologies": [],
                "covered_topologies": [],
                "uncovered_topologies": [],
                "covered_obb_role_signatures": [],
            },
        }

    # Non-closed poses are short-circuited before exact validation and must
    # not consume one of the three large-STEP Boolean slots.
    eligible = [
        item for item in ranked
        if item[2].get("constraint_closure", {}).get("fully_closed")
    ]
    buckets: dict[
        tuple[int, str],
        list[tuple[int, dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for item in eligible:
        buckets[_exact_candidate_topology_key(item[1])].append(item)
    topology_keys = sorted(buckets)
    selected: list[int] = []
    selected_set: set[int] = set()
    reason_by_rank: dict[int, str] = {}
    covered_roles: dict[
        tuple[int, str], set[tuple[tuple[str, str], ...]]
    ] = defaultdict(set)

    def add(
        item: tuple[int, dict[str, Any], dict[str, Any]],
        reason: str,
    ) -> bool:
        rank = int(item[0])
        if rank in selected_set or len(selected) >= limit:
            return False
        selected.append(rank)
        selected_set.add(rank)
        reason_by_rank[rank] = reason
        covered_roles[_exact_candidate_topology_key(item[1])].add(
            _obb_role_signature(item[1])
        )
        return True

    # Phase A: one fully-closed representative per retained topology, ordered
    # by topology rank.  Containment is preferred within each topology.
    for topology_key in topology_keys:
        if len(selected) >= limit:
            break
        representative = max(
            buckets[topology_key],
            key=lambda item: (
                _exact_candidate_has_containment(item[2]),
                *_exact_candidate_quality(item),
            ),
        )
        add(representative, "topology_coverage")

    # Phase B: if topology coverage leaves slots, cover an unseen OBB role in
    # round-robin order before spending exact work on another depth/sign
    # variant of a role already checked for that topology.
    progressed = True
    while len(selected) < limit and progressed:
        progressed = False
        for topology_key in topology_keys:
            if len(selected) >= limit:
                break
            unseen = [
                item for item in buckets[topology_key]
                if int(item[0]) not in selected_set
                and _obb_role_signature(item[1])
                not in covered_roles[topology_key]
                and _obb_role_signature(item[1])
                != (("baseline", "baseline"),)
            ]
            if unseen:
                add(
                    max(unseen, key=_exact_candidate_quality),
                    "obb_role_diversity",
                )
                progressed = True

    # Phase C: fill any remaining slots by the existing quality ordering.
    for item in eligible:
        if len(selected) >= limit:
            break
        add(item, "quality_fill")

    eligible_topologies = [
        {"topology_rank": key[0], "topology_id": key[1]}
        for key in topology_keys
    ]
    covered_topology_keys = {
        _exact_candidate_topology_key(item[1])
        for item in eligible
        if int(item[0]) in selected_set
    }
    return {
        "selected_ranks": selected,
        "selection_reason_by_rank": reason_by_rank,
        "summary": {
            "budget": limit,
            "eligible_candidate_count": len(eligible),
            "eligible_topologies": eligible_topologies,
            "covered_topologies": [
                row for row in eligible_topologies
                if (row["topology_rank"], row["topology_id"])
                in covered_topology_keys
            ],
            "uncovered_topologies": [
                row for row in eligible_topologies
                if (row["topology_rank"], row["topology_id"])
                not in covered_topology_keys
            ],
            "covered_obb_role_signatures": [
                [list(pair) for pair in signature]
                for signatures in covered_roles.values()
                for signature in sorted(signatures)
                if signature != (("baseline", "baseline"),)
            ],
        },
    }


def _select_exact_rank_budget(
    prechecked: list[tuple[int, dict[str, Any], dict[str, Any]]],
    *,
    budget: int,
    large_step_case: bool,
) -> set[int]:
    """Compatibility wrapper returning the selected exact-check ranks."""

    plan = _plan_exact_rank_budget(
        prechecked,
        budget=budget,
        large_step_case=large_step_case,
    )
    return {int(rank) for rank in plan["selected_ranks"]}


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
    # OCCT Boolean Common is the final validator, not an inner-loop objective.
    # Large vendor STEP files use a much smaller, fully-closed frontier and a
    # per-solid AABB broad phase; whole-compound Boolean checks previously
    # stalled or exhausted memory on these models.
    exact_check_budget = (
        min(len(candidates), 3)
        if large_step_case
        else min(len(candidates), 180)
    )
    prechecked = []
    for rank, candidate in enumerate(candidates, 1):
        precheck = _pose_precheck(
            candidate, _candidate_topology_graph(candidate, graph), features
        )
        prechecked.append((rank, candidate, precheck))

    exact_budget_plan = _plan_exact_rank_budget(
        prechecked,
        budget=exact_check_budget,
        large_step_case=large_step_case,
    )
    exact_rank_budget = {
        int(rank) for rank in exact_budget_plan["selected_ranks"]
    }
    search["exact_budget_audit"] = exact_budget_plan["summary"]

    for rank, candidate, precheck in prechecked:
        components = _portable_components(
            candidate["components"], case_dir, output_dir
        )
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
            exact = (
                exact_shape_collisions_solid_broadphase(
                    output_dir,
                    components,
                    maximum_solid_pair_checks=512,
                )
                if large_step_case
                else exact_shape_collisions(output_dir, components)
            )
        final_score = _group_pose_final_score(candidate, exact, precheck)
        row = {
            "rank": rank,
            "candidate_origin": candidate.get("candidate_origin", "solver_beam"),
            "proposal_only": candidate.get("proposal_only", False),
            "review_required": candidate.get("review_required", False),
            "can_auto_accept": candidate.get("can_auto_accept"),
            "topology_id": candidate.get("topology_id"),
            "topology_rank": candidate.get("topology_rank"),
            "exact_budget_selected": rank in exact_rank_budget,
            "exact_budget_selection_reason": exact_budget_plan[
                "selection_reason_by_rank"
            ].get(rank),
            "exact_budget_obb_role_signature": [
                list(pair) for pair in _obb_role_signature(candidate)
            ],
            "axial_slide": candidate.get("axial_slide"),
            "planar_slide": candidate.get("planar_slide"),
            "pocket_depth": candidate.get("pocket_depth"),
            "joinable_pose_search": candidate.get("joinable_pose_search"),
            "joinable_pose_refinement_history": candidate.get(
                "joinable_pose_refinement_history"
            ),
            "joinable_pose_refined_connection_ids": candidate.get(
                "joinable_pose_refined_connection_ids"
            ),
            "joinable_multi_axial_search": candidate.get("joinable_multi_axial_search"),
            "obb_insertion": candidate.get("obb_insertion"),
            "obb_insertion_history": candidate.get("obb_insertion_history"),
            "obb_refined_connection_ids": candidate.get(
                "obb_refined_connection_ids"
            ),
            "planar_footprint": candidate.get("planar_footprint"),
            "planar_footprint_history": candidate.get(
                "planar_footprint_history"
            ),
            "planar_footprint_refined_connection_ids": candidate.get(
                "planar_footprint_refined_connection_ids"
            ),
            "axial_compound_interface": candidate.get(
                "axial_compound_interface"
            ),
            "axial_compound_history": candidate.get(
                "axial_compound_history"
            ),
            "axial_compound_refined_connection_ids": candidate.get(
                "axial_compound_refined_connection_ids"
            ),
            "axial_group_centering": candidate.get(
                "axial_group_centering"
            ),
            "axial_group_centering_history": candidate.get(
                "axial_group_centering_history"
            ),
            "axial_group_centering_required_connection_ids": candidate.get(
                "axial_group_centering_required_connection_ids"
            ),
            "enclosure_bay": candidate.get("enclosure_bay"),
            "enclosure_bay_history": candidate.get(
                "enclosure_bay_history"
            ),
            "enclosure_bay_refined_connection_ids": candidate.get(
                "enclosure_bay_refined_connection_ids"
            ),
            "edge_slot_interface": candidate.get("edge_slot_interface"),
            "edge_slot_history": candidate.get("edge_slot_history"),
            "edge_slot_refined_connection_ids": candidate.get(
                "edge_slot_refined_connection_ids"
            ),
            "carrier_open_side_consistency": candidate.get(
                "carrier_open_side_consistency"
            ),
            "carrier_open_side_consistency_history": candidate.get(
                "carrier_open_side_consistency_history"
            ),
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
                -_exact_collision_risk(item[1])[0],
                -_exact_collision_risk(item[1])[1],
                -_exact_collision_risk(item[1])[2],
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


def _refine_with_searchsimplex(
    placements: dict[str, Any],
    features: dict[str, Any],
    selected_pairs: set[tuple[str, str]],
    matches: list[dict[str, Any]],
    case_dir: Path,
) -> dict[str, Any]:
    """Refine clearance pair placements using SearchSimplex Nelder-Mead.

    Exports STL meshes, extracts joint axes, and runs SearchSimplex
    for each clearance pair. Falls back to original placement on failure.
    """
    try:
        from search_simplex import SearchSimplex
    except ImportError:
        return placements

    result = dict(placements)

    # Export STL files for all parts (do this once)
    stl_dir = case_dir / "_tmp_stl"
    stl_dir.mkdir(exist_ok=True)
    stl_cache = {}
    for part_name in features:
        stl_path = stl_dir / f"{Path(part_name).stem}.stl"
        if not stl_path.exists():
            _export_part_stl(case_dir / part_name, stl_path)
        if stl_path.exists():
            stl_cache[part_name] = stl_path

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

        ref_stl = stl_cache.get(ref)
        tgt_stl = stl_cache.get(tgt)
        if not ref_stl or not tgt_stl:
            continue

        # Get joint axis from the reference cylinder
        ref_cyls = features.get(ref, {}).get("cylinders", [])
        if not ref_cyls:
            continue
        axis_origin = list(ref_cyls[0].get("origin", [0, 0, 0]))
        axis_dir = list(ref_cyls[0].get("axis", [0, 0, 1]))

        try:
            ss = SearchSimplex(
                ref_stl, tgt_stl,
                axis_origin, axis_dir,
                num_surface_samples=500, budget=30,
            )
            opt = ss.search()
            offset = opt.get("offset", 0.0)
            rotation = opt.get("rotation_deg", 0.0)
            flip = opt.get("flip", False)
            overlap = opt.get("overlap", 1.0)
            print(f"  SearchSimplex {Path(ref).stem}+{Path(tgt).stem}: "
                  f"offset={offset:.1f} rot={rotation:.1f}deg flip={flip} "
                  f"overlap={overlap:.3f}", flush=True)

            # Only apply if significantly better (low overlap)
            if overlap < 0.1:
                # Build placement from optimized params
                from coordinate_solver import _global_vector, _vec_norm
                import math as _math
                d = _vec_norm(axis_dir)
                # Rotation around joint axis
                angle = _math.radians(rotation)
                c = _math.cos(angle); s = _math.sin(angle)
                x, y, z = d
                R = [
                    [c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                    [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                    [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
                ]
                # Convert rotation matrix to axis-angle for placement
                trace = R[0][0] + R[1][1] + R[2][2]
                rot_angle = _math.degrees(_math.acos(max(-1, min(1, (trace-1)/2))))
                if rot_angle > 0.1:
                    rx = R[2][1] - R[1][2]
                    ry = R[0][2] - R[2][0]
                    rz = R[1][0] - R[0][1]
                    rnorm = _math.sqrt(rx*rx + ry*ry + rz*rz)
                    if rnorm > 1e-9:
                        rot_seq = [{'axis_angle': [rx/rnorm, ry/rnorm, rz/rnorm, rot_angle]}]
                    else:
                        rot_seq = []
                else:
                    rot_seq = []

                current = list(result.get(tgt, {}).get("translate", [0, 0, 0]))
                slide = [offset * d[i] for i in range(3)]
                new_tgt = dict(result.get(tgt, {}))
                new_tgt["translate"] = [current[i] + slide[i] for i in range(3)]
                if rot_seq:
                    existing_rot = list(new_tgt.get("rotate_sequence", []))
                    new_tgt["rotate_sequence"] = existing_rot + rot_seq
                if flip:
                    # Add 180° flip around direction
                    new_tgt_rot = list(new_tgt.get("rotate_sequence", []))
                    new_tgt_rot.append({'axis_angle': [d[0], d[1], d[2], 180.0]})
                    new_tgt["rotate_sequence"] = new_tgt_rot
                result[tgt] = new_tgt
        except Exception as e:
            print(f"  SearchSimplex failed for {ref}+{tgt}: {e}", flush=True)

    return result


def _export_part_stl(part_name: str, stl_path: Path) -> None:
    """Export a STEP part to STL using OCCT."""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.StlAPI import StlAPI_Writer

        step_path = Path("sw") / Path(part_name).parent / part_name if "/" in part_name or "\\" in part_name else None
        if step_path and step_path.exists():
            pass
        else:
            # part_name is just a filename — look in known locations
            for search_dir in [Path(part_name).parent]:
                candidate = search_dir / part_name
                if candidate.exists():
                    step_path = candidate
                    break
        if not step_path or not step_path.exists():
            return

        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return
        reader.TransferRoots()
        shape = reader.OneShape()
        solids = []
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            solids.append(exp.Current())
            exp.Next()
        if solids:
            BRepMesh_IncrementalMesh(solids[0], 0.5, True, 0.5).Perform()
            StlAPI_Writer().Write(solids[0], str(stl_path))
    except Exception:
        pass


def run_known_group_assembly(
    case_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    joinable_report: str | Path | None = None,
    joinable_pose_dir: str | Path | None = None,
    brep_graph_dir: str | Path | None = None,
    beam_width: int = 20,
    write_assembly_step: bool = True,
) -> dict[str, Any]:
    case_dir = Path(case_dir).resolve()
    _PLANAR_FOOTPRINT_RECALL_CACHE.clear()
    _AXIAL_COMPOUND_RECALL_CACHE.clear()
    _ENCLOSURE_BAY_RECALL_CACHE.clear()
    _EDGE_SLOT_RECALL_CACHE.clear()
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
    brep_graph_audit = _attach_brep_graph_sidecars(
        features, step_files, brep_graph_dir
    )
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
    search = _solve_topology_pose_frontier(
        features,
        scored,
        graph,
        beam_width=beam_width,
        joinable_pose_dir=joinable_pose_dir,
    )
    selected_pose, exact, pose_audit, pose_fully_closed, selected_pose_rank = _evaluate_pose_candidates(
        case_dir, output_dir, search, graph, features
    )
    selected_graph = _candidate_topology_graph(selected_pose, graph)
    components = _portable_components(
        selected_pose["components"], case_dir, output_dir
    )
    placements = selected_pose["placements"]
    # ── Cylinder axial stop ──
    # ──────────────────────────
    # ─────────────────────────────
    manifest = {
        "schema_version": "2.0.0",
        "assembly_name": case_dir.name,
        "global_units": "mm",
        "components": components,
    }
    _write(output_dir / "assembly_manifest.json", manifest)

    used = _selected_evidence_fingerprints(selected_pose)
    constraints = []
    connections = []
    selected_pose_audit_row = next(
        (
            row for row in pose_audit
            if row.get("rank") == selected_pose_rank
        ),
        {},
    )
    closure_by_connection = {
        str(row.get("connection_id")): row
        for row in (
            selected_pose_audit_row.get("constraint_closure", {}).get(
                "connections"
            ) or []
        )
    }
    for selected in selected_graph["selected"]:
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
        closure_row = closure_by_connection.get(
            str(selected.get("connection_id")), {}
        )
        satisfied_types = list(dict.fromkeys(
            [row["type"] for row in selected_rows]
            + list(closure_row.get("satisfied_relation_types") or [])
        ))
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
            "constraint_closed_in_selected_pose": bool(
                closure_row.get("closed", bool(selected_rows))
            ),
            "closure_evidence": closure_row.get("closure_evidence"),
            "review_required": bool(
                closure_row.get("review_required")
                or not closure_row.get("closed", bool(selected_rows))
            ),
            "axial_compound_evidence": closure_row.get(
                "axial_compound_evidence"
            ) or [],
            "axial_group_centering_evidence": closure_row.get(
                "axial_group_centering_evidence"
            ) or [],
            "enclosure_bay_evidence": closure_row.get(
                "enclosure_bay_evidence"
            ) or [],
            "edge_slot_evidence": closure_row.get(
                "edge_slot_evidence"
            ) or [],
            "providers": selected["providers"],
            "relative_transform_a_to_b": _relative_transform(a, b, placements),
            "joinable_interface_candidates": (
                (selected.get("joinable") or {}).get("top_interface_candidates", [])
            ),
        })

    localized_interference = _localized_interference_review(
        exact,
        next(
            (
                row.get("constraint_closure") or {}
                for row in pose_audit
                if row.get("rank") == selected_pose_rank
            ),
            {},
        ),
    )
    exact["localized_interference_review"] = localized_interference
    if exact["status"] == "success":
        if exact["collisions"]:
            pose_status = (
                "uncertain"
                if localized_interference["eligible_for_review"]
                else "failed"
            )
        else:
            pose_status = (
                "valid"
                if pose_fully_closed and selected_graph["connected"]
                else "uncertain"
            )
    else:
        pose_status = "uncertain"
    limitations = []
    if joinable_audit["status"] != "success":
        limitations.append("JoinABLe缓存未提供；本次仅使用解析几何接口候选。")
    if not selected_graph["connected"]:
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
        "assembly_connected": bool(selected_graph["connected"]),
        "pose_status": pose_status,
        "direct_connections": connections,
        "assembly_relations": constraints,
        "components": components,
        "unresolved_parts": selected_graph["unresolved_parts"],
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
            "topology_frontier_count": graph.get("topology_frontier_count", 0),
            "selected_topology_id": selected_pose.get("topology_id"),
            "selected_topology_rank": selected_pose.get("topology_rank"),
            "joinable": joinable_audit,
            "brep_graph_sidecars": brep_graph_audit,
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
        "topology_frontier": [
            {key: value for key, value in row.items() if key != "rows"}
            for row in graph.get("topology_frontier") or []
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
        "topology_search_audit": search.get("topology_search_audit"),
        "joinable_group_pose_composition": search.get(
            "joinable_group_pose_composition"
        ),
        "group_pose_optimizer": {
            "enabled": True,
            "exact_check_policy": (
                "bounded_top_precheck_candidates; large STEP uses a three-pose "
                "fully-closed frontier with per-solid AABB broad phase"
            ),
            "selected_pose_rank": selected_pose_rank,
            "exact_budget_audit": search.get("exact_budget_audit"),
        },
        "pose_audit": pose_audit,
        "selected_exact_collision": exact,
    })
    _write(output_dir / "assembly_relations.json", validated)
    _write(
        output_dir / "conservative_pose_output.json",
        _conservative_pose_output(validated, pose_audit),
    )
    # Persist all audit artefacts before invoking the large native STEP writer.
    # If an OCCT vendor model exhausts memory or crashes during export, the
    # candidate/pose diagnostics remain available for review and reproduction.
    if write_assembly_step:
        build_assembly(
            str(output_dir / "assembly_manifest.json"),
            str(output_dir / "assembly.step"),
        )
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--joinable-report")
    parser.add_argument("--joinable-pose-dir")
    parser.add_argument(
        "--brep-graph-dir",
        help=(
            "optional directory of hash-verified enriched *.brep_graph.json "
            "sidecars used only for local topological interface evidence"
        ),
    )
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument(
        "--skip-assembly-step",
        action="store_true",
        help="write all pose/audit JSON but skip the final native STEP export",
    )
    args = parser.parse_args()
    result = run_known_group_assembly(
        args.case_dir,
        output_dir=args.output_dir,
        joinable_report=args.joinable_report,
        joinable_pose_dir=args.joinable_pose_dir,
        brep_graph_dir=args.brep_graph_dir,
        beam_width=args.beam_width,
        write_assembly_step=not args.skip_assembly_step,
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
