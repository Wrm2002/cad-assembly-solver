"""Geometry-only folded-ear insertion proposal.

This is intentionally a proposal generator.  It derives an insertion pose from
an exterior, hole-bearing folded face, the carrier's opposing side wall, and a
locally dense side-wall feature band used as a stop-region cue.  Hole centres
are not used to fit the pose; they only select the local mounting region.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _proper_signed_permutations, _rotation_axis_angle


def _bbox(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(data["bbox_min"], dtype=float), np.asarray(data["bbox_max"], dtype=float)


def _axis(vector: np.ndarray) -> int | None:
    index = int(np.argmax(np.abs(vector)))
    return index if abs(vector[index]) >= 0.98 else None


def _group_hole_hosts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for hole in payload["holes"]:
        normal = np.asarray(hole["host_normal"], dtype=float)
        centre = np.asarray(hole["host_centre"], dtype=float)
        key = tuple(np.round(normal, 3)) + (round(float(np.dot(normal, centre)) * 2.0) / 2.0,)
        groups.setdefault(key, []).append(hole)
    out = []
    for key, rows in groups.items():
        normal = np.asarray(rows[0]["host_normal"], dtype=float)
        normal /= np.linalg.norm(normal)
        centres = np.asarray([row["centre"] for row in rows], dtype=float)
        # Collapse counterbores that share a projected centre.
        unique: list[np.ndarray] = []
        for centre in centres:
            projected = centre - normal * np.dot(centre - np.asarray(rows[0]["host_centre"]), normal)
            if not any(np.linalg.norm(projected - prior) < 0.8 for prior in unique):
                unique.append(projected)
        if len(unique) >= 3:
            out.append({"key": key, "normal": normal, "points": np.asarray(unique), "count": len(unique)})
    return out


def _boundary_proximity(normal: np.ndarray, point: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> float:
    corners = np.asarray(list(itertools.product(*zip(bounds[0], bounds[1]))), dtype=float)
    values = corners @ normal
    return float(np.exp(-min(abs(np.dot(normal, point) - values.min()), abs(np.dot(normal, point) - values.max())) / 4.0))


def _dense_band(points: np.ndarray, axis: int, width: float) -> float:
    values = np.asarray(points[:, axis], dtype=float)
    if not len(values):
        raise ValueError("empty point set")
    candidates = []
    for value in values:
        nearby = values[np.abs(values - value) <= width / 2.0]
        candidates.append((len(nearby), float(np.mean(nearby))))
    return max(candidates)[1]


def _plane_stops(planes: list[dict[str, Any]], axis: int, near: float) -> list[float]:
    candidates = []
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=float)
        if _axis(normal) != axis:
            continue
        value = float(plane["origin"][axis])
        # A stop needs a non-trivial local footprint; exclude microscopic edge
        # facets while retaining folded sheet-metal ledges.
        if min(float(plane["extent_u"]), float(plane["extent_v"])) < 0.7:
            continue
        candidates.append(value)
    return sorted(candidates, key=lambda value: abs(value - near))


def propose(source_holes: dict[str, Any], carrier_holes: dict[str, Any], source_planes: dict[str, Any], carrier_planes: dict[str, Any]) -> dict[str, Any]:
    source_bounds = _bbox(source_planes)
    carrier_bounds = _bbox(carrier_planes)
    source_dims = source_bounds[1] - source_bounds[0]
    carrier_dims = carrier_bounds[1] - carrier_bounds[0]
    source_long = int(np.argmax(source_dims))
    carrier_long = int(np.argmax(carrier_dims))
    carrier_thin = int(np.argmin(carrier_dims))
    source_hosts = _group_hole_hosts(source_holes)
    carrier_hosts = _group_hole_hosts(carrier_holes)
    candidates = []
    for source in source_hosts:
        source_axis = _axis(source["normal"])
        if source_axis is None:
            continue
        source_boundary = _boundary_proximity(source["normal"], source["points"].mean(axis=0), source_bounds)
        if source_boundary < 0.35:
            continue
        for carrier in carrier_hosts:
            carrier_axis = _axis(carrier["normal"])
            if carrier_axis is None:
                continue
            carrier_boundary = _boundary_proximity(carrier["normal"], carrier["points"].mean(axis=0), carrier_bounds)
            if carrier_boundary < 0.35:
                continue
            for rotation in _proper_signed_permutations():
                mapped_normal = rotation @ source["normal"]
                if float(np.dot(mapped_normal, carrier["normal"])) > -0.98:
                    continue
                mapped_long = _axis(rotation[:, source_long])
                if mapped_long != carrier_long:
                    continue
                # Pair folded faces only with a carrier side wall; the long
                # source extent then becomes the insertion direction.
                source_contact = source["points"].mean(axis=0)
                carrier_contact = carrier["points"].mean(axis=0)
                translate = carrier_contact - rotation @ source_contact
                # Slide laterally to the carrier's thin-axis centre.  This is
                # a containment constraint, not a manually supplied offset.
                corners = np.asarray(list(itertools.product(*zip(source_bounds[0], source_bounds[1]))), dtype=float)
                moved = (rotation @ corners.T).T + translate
                translate[carrier_thin] += (carrier_bounds[0][carrier_thin] + carrier_bounds[1][carrier_thin]) / 2.0 - (moved[:, carrier_thin].min() + moved[:, carrier_thin].max()) / 2.0
                moved = (rotation @ corners.T).T + translate
                if moved[:, carrier_thin].min() < carrier_bounds[0][carrier_thin] - 1.5 or moved[:, carrier_thin].max() > carrier_bounds[1][carrier_thin] + 1.5:
                    continue
                # Use the densest local side-wall feature band only as a stop
                # *region*.  This is deliberately weaker than a hole pose.
                band = _dense_band(carrier["points"], carrier_long, max(16.0, source_dims[source_long] * 0.35))
                stops = _plane_stops(carrier_planes["planes"], carrier_long, band)
                if not stops:
                    continue
                for stop in stops[:4]:
                    for edge in ("min", "max"):
                        candidate_translate = translate.copy()
                        moved = (rotation @ corners.T).T + candidate_translate
                        bound = moved[:, carrier_long].min() if edge == "min" else moved[:, carrier_long].max()
                        candidate_translate[carrier_long] += stop - bound
                        moved = (rotation @ corners.T).T + candidate_translate
                        inside = float(np.mean(np.all((moved >= carrier_bounds[0] - 1.5) & (moved <= carrier_bounds[1] + 1.5), axis=1)))
                        if inside < 0.99:
                            continue
                        angle = float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
                        candidates.append({
                            "rotation": rotation.round(10).tolist(),
                            "translation": candidate_translate.round(6).tolist(),
                            "source_contact_normal": source["normal"].round(8).tolist(),
                            "carrier_contact_normal": carrier["normal"].round(8).tolist(),
                            "source_hole_feature_count": source["count"],
                            "carrier_hole_feature_count": carrier["count"],
                            "stop_coordinate": round(stop, 6),
                            "stop_edge": edge,
                            "inside_fraction": inside,
                            "rotation_angle_deg": round(angle, 4),
                            "score": round(0.45 * source_boundary + 0.45 * carrier_boundary + 0.10 * (1.0 - angle / 180.0), 4),
                        })
    unique = []
    for candidate in sorted(candidates, key=lambda item: (-item["score"], item["rotation_angle_deg"])):
        vector = np.asarray(candidate["rotation"]).ravel().tolist() + candidate["translation"]
        if any(np.linalg.norm(np.asarray(vector) - np.asarray(np.asarray(item["rotation"]).ravel().tolist() + item["translation"])) < 0.1 for item in unique):
            continue
        unique.append(candidate)
    return {"status": "review_only", "method": "folded_face_sidewall_stop", "candidate_count": len(unique), "candidates": unique[:40]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_holes", type=Path)
    parser.add_argument("carrier_holes", type=Path)
    parser.add_argument("source_planes", type=Path)
    parser.add_argument("carrier_planes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = propose(*[json.loads(path.read_text(encoding="utf-8")) for path in (args.source_holes, args.carrier_holes, args.source_planes, args.carrier_planes)])
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "candidate_count": result["candidate_count"]}, indent=2))


if __name__ == "__main__":
    main()
