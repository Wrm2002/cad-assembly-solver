"""Conservative, geometry-only flange-hole pose recovery.

This module deliberately does *not* encode a case-specific transform.  It
matches circular features only when they are topologically adjacent to a
planar face, estimates a full rigid transform from at least three centres,
and records every rejected/ambiguous alternative for review.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-8


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < EPS:
        raise ValueError("zero-length vector")
    return vector / norm


@dataclass
class Host:
    key: str
    normal: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    points: list[dict[str, Any]]

    @property
    def span(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def _key(hole: dict[str, Any]) -> tuple[float, ...]:
    """Group split B-rep faces by their common infinite plane.

    Sheet-metal flanges are frequently represented as many coplanar faces;
    grouping by each finite face bbox loses exactly the three-hole pattern we
    need.  The oriented plane normal plus offset preserves the interface while
    still separating opposite sides of a sheet.
    """
    normal = np.asarray(hole["host_normal"], dtype=float)
    centre = np.asarray(hole["host_centre"], dtype=float)
    return tuple(np.round(normal, 3)) + (round(float(np.dot(normal, centre)) * 2.0) / 2.0,)


def _hosts(payload: dict[str, Any], max_points_per_host: int | None = None) -> list[Host]:
    buckets: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for hole in payload["holes"]:
        buckets.setdefault(_key(hole), []).append(hole)

    result: list[Host] = []
    for raw_key, holes in buckets.items():
        normal = _unit(np.asarray(holes[0]["host_normal"], dtype=float))
        bmin = np.min(np.asarray([hole["host_bbox_min"] for hole in holes], dtype=float), axis=0)
        bmax = np.max(np.asarray([hole["host_bbox_max"] for hole in holes], dtype=float), axis=0)
        # Collapse a through hole plus its counterbore/countersink cylinder
        # stack to one centre, preserving all observed radii as evidence.
        clustered: list[dict[str, Any]] = []
        for hole in holes:
            centre = np.asarray(hole["centre"], dtype=float)
            plane_point = np.asarray(hole["host_centre"], dtype=float)
            centre = centre - normal * float(np.dot(centre - plane_point, normal))
            for item in clustered:
                if float(np.linalg.norm(centre - item["centre"])) <= 0.75:
                    item["radii"].append(float(hole["radius"]))
                    break
            else:
                clustered.append({"centre": centre, "radii": [float(hole["radius"])]})
        # The face needs enough spread to be a mounting pattern rather than a
        # multi-cylinder detail of one local feature.
        if len(clustered) < 3:
            continue
        coords = np.asarray([item["centre"] for item in clustered])
        if float(np.max(np.ptp(coords, axis=0))) < 8.0:
            continue
        for item in clustered:
            item["radii"] = sorted({round(radius, 3) for radius in item["radii"]})
        if max_points_per_host is not None and len(clustered) > max_points_per_host:
            # Keep a spatially diverse deterministic subset for proposal
            # generation.  This prevents decorative perforation fields from
            # turning RANSAC into an exhaustive O(n^4) search.  The full
            # original feature set remains in the extraction audit.
            chosen = [min(range(len(clustered)), key=lambda index: tuple(clustered[index]["centre"]))]
            while len(chosen) < max_points_per_host:
                candidate = max(
                    (index for index in range(len(clustered)) if index not in chosen),
                    key=lambda index: min(float(np.linalg.norm(clustered[index]["centre"] - clustered[existing]["centre"])) for existing in chosen),
                )
                chosen.append(candidate)
            clustered = [clustered[index] for index in chosen]
        result.append(
            Host(
                key="/".join(str(value) for value in raw_key),
                normal=normal,
                bbox_min=bmin,
                bbox_max=bmax,
                points=clustered,
            )
        )
    return result


def _proper_transform(a0: np.ndarray, a1: np.ndarray, normal_a: np.ndarray, b0: np.ndarray, b1: np.ndarray, normal_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map source face normal to the opposing carrier normal and one hole pair."""
    ua = _unit(a1 - a0)
    va = _unit(np.cross(normal_a, ua))
    ub = _unit(b1 - b0)
    # Contact means opposite face normals.  This basis construction produces a
    # proper (det=+1) rigid rotation, never an accidental reflection.
    target_n = -normal_b
    vb = _unit(np.cross(target_n, ub))
    source_basis = np.column_stack((ua, va, normal_a))
    target_basis = np.column_stack((ub, vb, target_n))
    rotation = target_basis @ source_basis.T
    if np.linalg.det(rotation) < 0.99:
        raise ValueError("improper transform")
    return rotation, b0 - rotation @ a0


def _compatible_radii(left: list[float], right: list[float]) -> bool:
    # Screw clearance and counterbore radii can differ; nevertheless some
    # observed cylindrical radius should be reasonably compatible.
    return any(abs(a - b) <= 1.05 for a in left for b in right)


def _boundary_proximity(host: Host, bbox: tuple[np.ndarray, np.ndarray]) -> float:
    """Return a soft exterior-face prior derived only from geometry."""
    point = (host.bbox_min + host.bbox_max) / 2.0
    coordinate = float(np.dot(host.normal, point))
    corners = np.asarray(list(itertools.product(*zip(bbox[0], bbox[1]))), dtype=float)
    projected = corners @ host.normal
    distance = min(abs(coordinate - float(projected.min())), abs(coordinate - float(projected.max())))
    return float(np.exp(-distance / 4.0))


def _score_transform(
    host_a: Host,
    host_b: Host,
    rotation: np.ndarray,
    translation: np.ndarray,
    carrier_bbox: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any] | None:
    bpoints = np.asarray([item["centre"] for item in host_b.points])
    matches: list[dict[str, Any]] = []
    for source_index, source in enumerate(host_a.points):
        transformed = rotation @ source["centre"] + translation
        distances = np.linalg.norm(bpoints - transformed, axis=1)
        target_index = int(np.argmin(distances))
        residual = float(distances[target_index])
        if residual <= 0.65 and _compatible_radii(source["radii"], host_b.points[target_index]["radii"]):
            matches.append({"source": source_index, "target": target_index, "residual_mm": residual})
    # Enforce one-to-one hole correspondences.
    selected: dict[int, dict[str, Any]] = {}
    for match in matches:
        previous = selected.get(match["target"])
        if previous is None or match["residual_mm"] < previous["residual_mm"]:
            selected[match["target"]] = match
    matches = sorted(selected.values(), key=lambda value: value["source"])
    if len(matches) < 3:
        return None
    source_match_points = np.asarray([host_a.points[match["source"]]["centre"] for match in matches])
    target_match_points = np.asarray([host_b.points[match["target"]]["centre"] for match in matches])
    source_pair_distances = np.linalg.norm(source_match_points[:, None, :] - source_match_points[None, :, :], axis=2)
    target_pair_distances = np.linalg.norm(target_match_points[:, None, :] - target_match_points[None, :, :], axis=2)
    nonzero_source = source_pair_distances[source_pair_distances > 1e-6]
    nonzero_target = target_pair_distances[target_pair_distances > 1e-6]
    # Mounting evidence must come from distinct physical holes.  Adjacent
    # counterbores, slots, or cosmetic perforations a few millimetres apart
    # cannot be promoted to a three-fastener pattern.
    if float(nonzero_source.min()) < 8.0 or float(nonzero_target.min()) < 8.0:
        return None
    residual = float(np.mean([match["residual_mm"] for match in matches]))
    source_centres = source_match_points
    # A collinear three-hole row is allowed (rack ears often use one), but it
    # is reported as weaker evidence than a two-dimensional bolt pattern.
    singular = np.linalg.svd(source_centres - source_centres.mean(axis=0), compute_uv=False)
    rank2_strength = float(singular[1] if len(singular) > 1 else 0.0)

    # Transform the source host bounding box to test whether the attachment is
    # on the carrier's interior side instead of hanging outside it.
    corners = np.array(
        list(itertools.product(*zip(host_a.bbox_min, host_a.bbox_max))), dtype=float
    )
    # This is a conservative local proxy for the part bbox check; the full
    # original part bbox is checked by the caller after candidate selection.
    moved = (rotation @ corners.T).T + translation
    carrier_lo, carrier_hi = carrier_bbox
    local_inside = float(np.mean(np.all((moved >= carrier_lo - 3.0) & (moved <= carrier_hi + 3.0), axis=1)))
    return {
        "matches": matches,
        "match_count": len(matches),
        "mean_residual_mm": residual,
        "rank2_strength_mm": rank2_strength,
        "local_host_inside_fraction": local_inside,
    }


def recover(source_payload: dict[str, Any], carrier_payload: dict[str, Any], source_bbox: tuple[np.ndarray, np.ndarray], carrier_bbox: tuple[np.ndarray, np.ndarray], max_points_per_host: int | None = None) -> dict[str, Any]:
    source_hosts = _hosts(source_payload, max_points_per_host)
    carrier_hosts = _hosts(carrier_payload, max_points_per_host)
    candidates: list[dict[str, Any]] = []
    for host_a in source_hosts:
        # Very large source faces contain unrelated sheet-metal details and
        # create excessive false correspondences; retain realistic bracket
        # interfaces while keeping all candidates in the audit.
        for host_b in carrier_hosts:
            # Sample point pairs; their distance fixes the in-plane rotation.
            for ia0, ia1 in itertools.combinations(range(len(host_a.points)), 2):
                a0, a1 = host_a.points[ia0]["centre"], host_a.points[ia1]["centre"]
                length_a = float(np.linalg.norm(a1 - a0))
                if length_a < 6.0:
                    continue
                for ib0, ib1 in itertools.permutations(range(len(host_b.points)), 2):
                    b0, b1 = host_b.points[ib0]["centre"], host_b.points[ib1]["centre"]
                    if abs(length_a - float(np.linalg.norm(b1 - b0))) > 0.35:
                        continue
                    try:
                        rotation, translation = _proper_transform(a0, a1, host_a.normal, b0, b1, host_b.normal)
                    except ValueError:
                        continue
                    scored = _score_transform(host_a, host_b, rotation, translation, carrier_bbox)
                    if scored is None:
                        continue
                    # Original attachment bbox must sit in/against the carrier
                    # volume.  This rejects the previously observed outside
                    # mirror solution without adding a case-specific pose.
                    lo, hi = source_bbox
                    corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=float)
                    moved = (rotation @ corners.T).T + translation
                    inside = float(np.mean(np.all((moved >= carrier_bbox[0] - 3.0) & (moved <= carrier_bbox[1] + 3.0), axis=1)))
                    scored["source_part_inside_fraction"] = inside
                    scored["rotation"] = rotation.round(10).tolist()
                    scored["translation"] = translation.round(6).tolist()
                    scored["source_host_key"] = host_a.key
                    scored["carrier_host_key"] = host_b.key
                    scored["source_host_normal"] = host_a.normal.round(8).tolist()
                    scored["carrier_host_normal"] = host_b.normal.round(8).tolist()
                    scored["source_host_boundary_proximity"] = round(_boundary_proximity(host_a, source_bbox), 4)
                    scored["carrier_host_boundary_proximity"] = round(_boundary_proximity(host_b, carrier_bbox), 4)
                    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
                    scored["rotation_angle_deg"] = round(float(np.degrees(angle)), 4)
                    # Primary evidence: three or more centre-aligned holes,
                    # contact plane orientation, and an interior placement.
                    scored["geometry_score"] = round(
                        min(1.0, 0.45 + 0.1 * scored["match_count"] + 0.15 * inside + 0.08 * min(1.0, scored["rank2_strength_mm"] / 8.0) - 0.2 * scored["mean_residual_mm"]),
                        4,
                    )
                    # This only orders otherwise valid geometry candidates.
                    # It does not turn a non-unique result into an acceptance:
                    # exterior flange contact and a small frame adjustment are
                    # useful weak evidence when source CAD frames were exported
                    # together, but are never sufficient on their own.
                    scored["selection_score"] = round(
                        scored["geometry_score"]
                        + 0.12 * scored["source_host_boundary_proximity"]
                        + 0.12 * scored["carrier_host_boundary_proximity"]
                        + 0.12 * (1.0 - angle / np.pi),
                        4,
                    )
                    candidates.append(scored)
    # Deduplicate by transform and keep every distinct competing hypothesis.
    unique: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (-item["geometry_score"], -item["match_count"], item["mean_residual_mm"])):
        transform = np.array(candidate["rotation"]).ravel().tolist() + candidate["translation"]
        if any(np.linalg.norm(np.asarray(transform) - np.asarray(np.array(item["rotation"]).ravel().tolist() + item["translation"])) < 0.1 for item in unique):
            continue
        unique.append(candidate)
        if len(unique) >= 1000:
            break
    accepted = [item for item in unique if item["match_count"] >= 3 and item["mean_residual_mm"] <= 0.35 and item["source_part_inside_fraction"] >= 0.75 and item["geometry_score"] >= 0.8]
    selected = max(unique, key=lambda item: item["selection_score"], default=None)
    return {
        "method": "planar-hosted-hole-ransac",
        "source_host_count": len(source_hosts),
        "carrier_host_count": len(carrier_hosts),
        "candidate_count": len(unique),
        "accepted_candidate_count": len(accepted),
        "selected_candidate": selected,
        "all_candidates": unique,
        "top_candidates": unique[:20],
        "decision": "valid" if len(accepted) == 1 else "review" if unique else "failed",
        "decision_reason": "A unique multi-hole, interior candidate was found." if len(accepted) == 1 else "No unique multi-evidence pose; retain candidates for human review.",
    }


def _bbox_from_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([hole["centre"] for hole in payload["holes"]], dtype=float)
    # This fallback is deliberately not used by the case runner, which passes
    # exact source bboxes.  It keeps the module independently testable.
    return values.min(axis=0), values.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_holes", type=Path)
    parser.add_argument("carrier_holes", type=Path)
    parser.add_argument("source_bbox", type=Path, help="JSON file with bbox_min/bbox_max")
    parser.add_argument("carrier_bbox", type=Path, help="JSON file with bbox_min/bbox_max")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-host-points", type=int, default=None, help="proposal-only spatial subsampling cap")
    args = parser.parse_args()
    source = json.loads(args.source_holes.read_text(encoding="utf-8"))
    carrier = json.loads(args.carrier_holes.read_text(encoding="utf-8"))
    sb = json.loads(args.source_bbox.read_text(encoding="utf-8"))
    cb = json.loads(args.carrier_bbox.read_text(encoding="utf-8"))
    result = recover(source, carrier, (np.asarray(sb["bbox_min"]), np.asarray(sb["bbox_max"])), (np.asarray(cb["bbox_min"]), np.asarray(cb["bbox_max"])), args.max_host_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("decision", "candidate_count", "accepted_candidate_count")}, indent=2))


if __name__ == "__main__":
    main()
