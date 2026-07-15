"""Recall-oriented local interface regions for large B-Rep graphs.

The released JoinABLe model scores a Cartesian product of B-Rep entities.  On
large chassis and PCB models, repeated exterior faces can both dominate that
product and make inference impractical.  This module provides a geometry-only
shortlist of *regions of interest* (ROIs) before pair scoring.

The shortlist is deliberately proposal-only:

* it never uses part names, file names, assembly labels, or case IDs;
* a high ROI score is not an assembly verdict;
* no ROI can bypass pose closure, exact collision, or conservative gates.

Scores combine local topology, boundedness, geometric rarity, and interface
family coverage.  Area is only one weak feature, so a small decorative face is
not automatically preferred and a large functional flange is not discarded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from typing import Any, Iterable


SCHEMA_VERSION = "interface_roi.v1"
_SUPPORTED_SURFACES = {
    "plane",
    "cylinder",
    "cone",
    "sphere",
    "torus",
    "bspline",
    "bezier",
    "surface_of_revolution",
    "surface_of_extrusion",
    "offset",
    "unknown",
}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _area_bucket(area: float) -> int:
    """Log bucket used only to detect repeated same-scale geometry."""

    if area <= 1e-12:
        return -999
    return int(round(math.log10(area) * 5.0))


def _surface_type(node: dict[str, Any]) -> str:
    value = str(node.get("surface_type") or "unknown").lower()
    return value if value in _SUPPORTED_SURFACES else "unknown"


def _face_edges(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes = {
        str(node.get("node_id")): node
        for node in graph.get("nodes") or []
        if node.get("node_id") is not None
    }
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    # New audited graphs store the two incident faces directly on each edge.
    for edge in nodes.values():
        if edge.get("entity_type") != "edge":
            continue
        edge_id = str(edge["node_id"])
        for face_id in edge.get("adjacent_face_ids") or []:
            face_id = str(face_id)
            if (face_id, edge_id) not in seen:
                result[face_id].append(edge)
                seen.add((face_id, edge_id))

    # Older graph contracts retain only explicit face--edge adjacency rows.
    for link in graph.get("edges") or []:
        source, target = str(link.get("src")), str(link.get("dst"))
        if nodes.get(source, {}).get("entity_type") == "edge":
            source, target = target, source
        edge = nodes.get(target)
        if (
            nodes.get(source, {}).get("entity_type") == "face"
            and edge is not None
            and edge.get("entity_type") == "edge"
            and (source, target) not in seen
        ):
            result[source].append(edge)
            seen.add((source, target))
    return result


def _patch_extent(node: dict[str, Any]) -> float | None:
    points = node.get("patch_points") or []
    valid = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            continue
        row = tuple(_finite_float(value, math.nan) for value in point)
        if all(math.isfinite(value) for value in row):
            valid.append(row)
    if len(valid) < 2:
        return None
    ranges = [
        max(point[axis] for point in valid) - min(point[axis] for point in valid)
        for axis in range(3)
    ]
    return math.sqrt(sum(value * value for value in ranges))


def _interface_hints(
    surface: str,
    *,
    boundary_count: int,
    concave_count: int,
    circular_count: int,
) -> list[str]:
    hints: list[str] = []
    if surface == "plane" and boundary_count >= 3:
        hints.append("planar_seating")
    if surface == "plane" and concave_count >= 2:
        hints.append("bounded_channel_or_pocket")
    if surface == "plane" and circular_count:
        hints.append("planar_with_circular_locator")
    if surface == "cylinder":
        hints.append("cylindrical_insert_or_bore")
    if surface in {"cone", "sphere", "torus", "surface_of_revolution"}:
        hints.append("curved_locator")
    if not hints:
        hints.append("generic_local_surface")
    return hints


def _roi_score(
    *,
    rarity: float,
    local_scale: float,
    boundary_count: int,
    concave_count: int,
    circular_count: int,
    neighbor_type_count: int,
    topology_available: bool,
    surface: str,
    scale_reliability: float,
) -> float:
    score = 0.08
    score += 0.24 * rarity
    score += 0.08 * local_scale * scale_reliability
    score += 0.12 * min(1.0, boundary_count / 4.0)
    score += 0.22 * min(1.0, concave_count / 2.0)
    score += 0.14 * min(1.0, circular_count / 2.0)
    score += 0.07 * min(1.0, neighbor_type_count / 3.0)
    score += 0.03 if topology_available else 0.0
    score += 0.05 if surface == "cylinder" else 0.0
    # Extremely small fillets, threads, and tessellation remnants are common
    # in vendor STEP.  Keep them auditable, but do not let thousands of them
    # consume the useful ROI frontier merely because their areas are small.
    score -= 0.20 * (1.0 - scale_reliability)
    return round(min(1.0, max(0.0, score)), 6)


def rank_interface_rois(
    graph: dict[str, Any],
    *,
    maximum: int = 96,
    per_family_minimum: int = 8,
) -> dict[str, Any]:
    """Return a bounded, family-stratified local-interface frontier.

    ``maximum`` bounds serialized output, not the internal audit.  Selection
    first preserves a small quota for every observed interface hint and then
    fills remaining slots by score.  This prevents one repeated feature family
    from deleting all candidates of another family.
    """

    if maximum < 1:
        raise ValueError("maximum must be positive")
    faces = [
        node
        for node in graph.get("nodes") or []
        if node.get("entity_type") == "face" and node.get("node_id") is not None
    ]
    if not faces:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "B-Rep graph contains no face nodes.",
            "rois": [],
            "audit": {"face_count": 0, "selected_count": 0},
        }

    face_edges = _face_edges(graph)
    face_node_by_id = {str(node["node_id"]): node for node in faces}
    areas = [_finite_float(face.get("area")) for face in faces]
    maximum_area = max(max(areas), 1e-12)
    part_extent = max(
        1e-9,
        _finite_float(
            (graph.get("metadata") or {}).get("checkpoint_pair_normalization_extent"),
            math.sqrt(maximum_area),
        ),
    )
    signature_counts = Counter(
        (_surface_type(face), _area_bucket(_finite_float(face.get("area"))))
        for face in faces
    )
    surface_counts = Counter(_surface_type(face) for face in faces)
    rows: list[dict[str, Any]] = []
    topology_rows = 0

    for face in faces:
        face_id = str(face["node_id"])
        surface = _surface_type(face)
        area = max(0.0, _finite_float(face.get("area")))
        signature_frequency = signature_counts[(surface, _area_bucket(area))]
        family_size = max(1, surface_counts[surface])
        # Absolute inverse frequency is more useful than dividing by a very
        # large family size: 100 repeated screw/fillet faces should remain
        # common even when the part contains 20,000 faces.
        rarity = 1.0 / (1.0 + math.log2(max(1, signature_frequency)))
        relative_area = min(1.0, area / maximum_area)
        local_scale = 1.0 - math.sqrt(relative_area)
        characteristic_ratio = math.sqrt(max(area, 0.0)) / part_extent
        scale_reliability = min(
            1.0,
            max(0.0, (characteristic_ratio - 0.0002) / 0.0048),
        )

        incident = [
            edge for edge in face_edges.get(face_id, [])
            if not bool(edge.get("is_seam_edge"))
        ]
        boundary_count = len(incident)
        concave_count = sum(
            str(edge.get("convexity")).lower() == "concave" for edge in incident
        )
        convex_count = sum(
            str(edge.get("convexity")).lower() == "convex" for edge in incident
        )
        circular_count = sum(
            str(edge.get("curve_type")).lower() in {"circle", "ellipse"}
            for edge in incident
        )
        topology_available = any(
            edge.get("topology_feature_status") == "success" for edge in incident
        )
        topology_rows += int(topology_available)
        neighbor_ids = {
            str(other)
            for edge in incident
            for other in edge.get("adjacent_face_ids") or []
            if str(other) != face_id
        }
        neighbor_types = sorted({
            _surface_type(face_node_by_id[neighbor])
            for neighbor in neighbor_ids
            if neighbor in face_node_by_id
        })
        hints = _interface_hints(
            surface,
            boundary_count=boundary_count,
            concave_count=concave_count,
            circular_count=circular_count,
        )
        score = _roi_score(
            rarity=rarity,
            local_scale=local_scale,
            boundary_count=boundary_count,
            concave_count=concave_count,
            circular_count=circular_count,
            neighbor_type_count=len(neighbor_types),
            topology_available=topology_available,
            surface=surface,
            scale_reliability=scale_reliability,
        )
        independent = sum((
            rarity >= 0.5,
            concave_count >= 2,
            circular_count >= 1,
            boundary_count >= 3 and topology_available,
            bool(neighbor_types),
        ))
        rows.append({
            "roi_id": f"roi:{face_id}",
            "seed_face_id": face_id,
            "surface_type": surface,
            "score": score,
            "area_mm2": area,
            "centroid": face.get("centroid"),
            "normal": face.get("normal"),
            "axis_origin": face.get("axis_origin"),
            "axis_direction": face.get("axis_direction"),
            "radius_mm": face.get("radius"),
            "interface_hints": hints,
            "attached_edge_ids": [str(edge.get("node_id")) for edge in incident],
            "independent_evidence_count": int(independent),
            "review_required": True,
            "evidence": {
                "geometric_rarity": round(rarity, 6),
                "same_scale_family_frequency": signature_frequency,
                "relative_area": round(relative_area, 9),
                "local_scale_score": round(local_scale, 6),
                "characteristic_length_ratio": round(characteristic_ratio, 9),
                "scale_reliability": round(scale_reliability, 6),
                "boundary_edge_count": boundary_count,
                "concave_edge_count": concave_count,
                "convex_edge_count": convex_count,
                "circular_boundary_count": circular_count,
                "topology_available": topology_available,
                "neighbor_face_types": neighbor_types,
                "patch_extent_mm": _patch_extent(face),
            },
            "reason": (
                "Recall-oriented local B-Rep interface proposal; functional "
                "meaning and assembly validity remain unverified."
            ),
        })

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            -int(row["independent_evidence_count"]),
            str(row["seed_face_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    families = sorted({hint for row in ranked for hint in row["interface_hints"]})
    quota = max(1, min(int(per_family_minimum), maximum))
    for family in families:
        for row in ranked:
            if len(selected) >= maximum:
                break
            if family not in row["interface_hints"] or row["roi_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["roi_id"])
            if sum(family in item["interface_hints"] for item in selected) >= quota:
                break
    surface_cap = max(quota, int(math.ceil(maximum * 0.40)))
    signature_cap = max(4, int(math.ceil(maximum * 0.08)))
    selected_surfaces = Counter(str(row["surface_type"]) for row in selected)
    selected_signatures = Counter(
        (str(row["surface_type"]), _area_bucket(_finite_float(row["area_mm2"])))
        for row in selected
    )
    for row in ranked:
        if len(selected) >= maximum:
            break
        surface_key = str(row["surface_type"])
        signature_key = (surface_key, _area_bucket(_finite_float(row["area_mm2"])))
        if (
            row["roi_id"] in selected_ids
            or selected_surfaces[surface_key] >= surface_cap
            or selected_signatures[signature_key] >= signature_cap
        ):
            continue
        selected.append(row)
        selected_ids.add(row["roi_id"])
        selected_surfaces[surface_key] += 1
        selected_signatures[signature_key] += 1
    # A tiny or highly repetitive graph may not fill the bounded frontier
    # under diversity caps.  Preserve recall by relaxing caps only for the
    # remaining slots, never by declaring the shortlist complete early.
    for row in ranked:
        if len(selected) >= maximum:
            break
        if row["roi_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["roi_id"])
    selected.sort(
        key=lambda row: (
            -float(row["score"]),
            -int(row["independent_evidence_count"]),
            str(row["seed_face_id"]),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "success" if topology_rows == len(faces) else "partial",
        "reason": (
            "All selected faces have audited edge topology."
            if topology_rows == len(faces)
            else "Some faces lack audited edge topology; rarity and boundary evidence were retained."
        ),
        "rois": selected,
        "audit": {
            "face_count": len(faces),
            "topology_supported_face_count": topology_rows,
            "selected_count": len(selected),
            "selection_limit": maximum,
            "observed_interface_families": families,
            "part_names_or_case_ids_used": False,
            "acceptance_decision": False,
        },
    }


def build_roi_subgraph(
    graph: dict[str, Any],
    rois: Iterable[dict[str, Any]],
    *,
    neighborhood_hops: int = 1,
    maximum_nodes: int = 4096,
) -> dict[str, Any]:
    """Build a valid, re-indexed JoinABLe body graph around selected ROIs."""

    if neighborhood_hops < 0:
        raise ValueError("neighborhood_hops must be non-negative")
    if maximum_nodes < 2:
        raise ValueError("maximum_nodes must be at least two")
    nodes = {
        str(node.get("node_id")): node
        for node in graph.get("nodes") or []
        if node.get("node_id") is not None
    }
    adjacency: dict[str, set[str]] = defaultdict(set)
    for link in graph.get("edges") or []:
        source, target = str(link.get("src")), str(link.get("dst"))
        if source in nodes and target in nodes:
            adjacency[source].add(target)
            adjacency[target].add(source)
    seeds = [
        str(row.get("seed_face_id"))
        for row in rois
        if str(row.get("seed_face_id")) in nodes
    ]
    if not seeds:
        raise ValueError("ROI shortlist contains no face present in graph")
    selected = set(seeds)
    frontier = set(seeds)
    # One hop adds boundary edges; the second adds neighboring faces.  The
    # caller can request more, but the node budget always remains authoritative.
    for _ in range(neighborhood_hops * 2):
        expansion = {
            neighbor
            for node_id in frontier
            for neighbor in adjacency.get(node_id, set())
            if neighbor not in selected
        }
        ordered = sorted(
            expansion,
            key=lambda node_id: (
                int(nodes[node_id].get("joinable_node_index", 10**12)),
                node_id,
            ),
        )
        room = maximum_nodes - len(selected)
        accepted = set(ordered[: max(0, room)])
        selected.update(accepted)
        frontier = accepted
        if not frontier or len(selected) >= maximum_nodes:
            break

    ordered_nodes = sorted(
        (deepcopy(nodes[node_id]) for node_id in selected),
        key=lambda row: (int(row.get("joinable_node_index", 10**12)), str(row["node_id"])),
    )
    for index, node in enumerate(ordered_nodes):
        node["joinable_node_index"] = index
        if "checkpoint_node_index" in node:
            node["checkpoint_node_index"] = index
    retained_ids = {str(node["node_id"]) for node in ordered_nodes}
    retained_links = [
        deepcopy(link)
        for link in graph.get("edges") or []
        if str(link.get("src")) in retained_ids and str(link.get("dst")) in retained_ids
    ]
    if not retained_links:
        raise ValueError("ROI subgraph has no retained face-edge adjacency")
    metadata = deepcopy(graph.get("metadata") or {})
    metadata.update({
        "num_faces": sum(node.get("entity_type") == "face" for node in ordered_nodes),
        "num_edges": sum(node.get("entity_type") == "edge" for node in ordered_nodes),
        "num_face_edge_adjacencies": len(retained_links),
        "roi_subgraph": {
            "schema_version": SCHEMA_VERSION,
            "seed_face_ids": seeds,
            "source_node_count": len(nodes),
            "retained_node_count": len(ordered_nodes),
            "maximum_nodes": maximum_nodes,
            "neighborhood_hops": neighborhood_hops,
            "geometry_only": True,
        },
    })
    output = deepcopy(graph)
    output["nodes"] = ordered_nodes
    output["edges"] = retained_links
    output["metadata"] = metadata
    return output


def match_roi_pairs(
    fixed: dict[str, Any],
    moving: dict[str, Any],
    *,
    maximum: int = 64,
) -> list[dict[str, Any]]:
    """Create review-only compatible ROI pairs without semantic assumptions."""

    if maximum < 1:
        raise ValueError("maximum must be positive")
    rows: list[dict[str, Any]] = []
    for left in fixed.get("rois") or []:
        for right in moving.get("rois") or []:
            left_surface, right_surface = left.get("surface_type"), right.get("surface_type")
            if left_surface != right_surface:
                continue
            dimension_score = 0.5
            if left_surface == "cylinder":
                a = _finite_float(left.get("radius_mm"))
                b = _finite_float(right.get("radius_mm"))
                if a <= 0.0 or b <= 0.0:
                    continue
                dimension_score = min(a, b) / max(a, b)
                if dimension_score < 0.6:
                    continue
            elif left_surface == "plane":
                a = _finite_float(left.get("area_mm2"))
                b = _finite_float(right.get("area_mm2"))
                if a > 0.0 and b > 0.0:
                    # Unequal seating faces are common, so area is deliberately weak.
                    dimension_score = math.sqrt(min(a, b) / max(a, b))
            hint_overlap = sorted(set(left.get("interface_hints") or []).intersection(
                right.get("interface_hints") or []
            ))
            score = (
                0.35 * _finite_float(left.get("score"))
                + 0.35 * _finite_float(right.get("score"))
                + 0.20 * dimension_score
                + (0.10 if hint_overlap else 0.0)
            )
            rows.append({
                "fixed_roi_id": left.get("roi_id"),
                "moving_roi_id": right.get("roi_id"),
                "fixed_face_id": left.get("seed_face_id"),
                "moving_face_id": right.get("seed_face_id"),
                "surface_type": left_surface,
                "score": round(min(1.0, score), 6),
                "dimension_compatibility": round(dimension_score, 6),
                "shared_interface_hints": hint_overlap,
                "independent_evidence_count": int(bool(hint_overlap)) + int(dimension_score >= 0.8),
                "review_required": True,
                "reason": "Local geometry compatibility only; pose and functional validity are unverified.",
            })
    return sorted(
        rows,
        key=lambda row: (-float(row["score"]), str(row["fixed_face_id"]), str(row["moving_face_id"])),
    )[:maximum]
