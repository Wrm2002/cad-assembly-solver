"""Conservative composite interface solver for read-only STEP proxy audits.

The solver separates a small flange-like attachment from a larger inserted
module.  It never uses component names as geometric evidence:

* small attachment -> carrier: planar-side containment + repeated hole lines;
* main module -> carrier: enclosure/bay proposal supplied by the proxy solver;
* no attachment -> module constraint is allowed.

All results are review-only until original-geometry validation succeeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _box_corners, _proper_signed_permutations, _rotation_axis_angle


LINE_TOLERANCE_MM = 0.75
RADIUS_TOLERANCE_MM = 0.20
MIN_HOLE_LINES = 3
MIN_PATTERN_SPREAD_MM = 10.0


@dataclass(frozen=True)
class HoleLine:
    axis: int
    centre: np.ndarray
    radii: tuple[float, ...]
    face_indices: tuple[int, ...]


def _major_axis(vector: np.ndarray) -> int | None:
    index = int(np.argmax(np.abs(vector)))
    return index if abs(vector[index]) >= 0.985 else None


def _hole_lines(rows: list[dict[str, Any]]) -> list[HoleLine]:
    """Collapse counterbores/chamfers into one physical cylindrical line."""
    groups: list[dict[str, Any]] = []
    for row in rows:
        radius = float(row["radius"])
        extent = float(row["axial_extent"])
        if not (0.8 <= radius <= 6.0 and 0.08 <= extent <= 15.0):
            continue
        centre = np.asarray(row["centre"], dtype=float)
        axis = _major_axis(np.asarray(row["axis"], dtype=float))
        if axis is None:
            continue
        perpendicular = [dimension for dimension in range(3) if dimension != axis]
        target = None
        for group in groups:
            if group["axis"] != axis:
                continue
            if np.linalg.norm(centre[perpendicular] - group["centre"][perpendicular]) <= LINE_TOLERANCE_MM:
                target = group
                break
        if target is None:
            target = {"axis": axis, "centre": centre.copy(), "radii": [], "faces": []}
            groups.append(target)
        target["radii"].append(radius)
        target["faces"].append(int(row["face_index"]))
    result = []
    for group in groups:
        result.append(
            HoleLine(
                group["axis"],
                group["centre"],
                tuple(sorted(set(round(radius, 3) for radius in group["radii"]))),
                tuple(sorted(set(group["faces"]))),
            )
        )
    return result


def _shared_radius(first: HoleLine, second: HoleLine) -> float | None:
    matches = [
        (left + right) / 2.0
        for left in first.radii
        for right in second.radii
        if abs(left - right) <= RADIUS_TOLERANCE_MM
    ]
    return min(matches) if matches else None


def _candidate_attachment_pose(
    carrier_lo: np.ndarray,
    carrier_hi: np.ndarray,
    attachment_lo: np.ndarray,
    attachment_hi: np.ndarray,
    carrier_lines: list[HoleLine],
    attachment_lines: list[HoleLine],
    *,
    excluded_screw_axes: set[int] | None = None,
    allowed_screw_axes: set[int] | None = None,
) -> dict[str, Any] | None:
    """RANSAC-like matching of independent hole centre lines.

    Only coordinates perpendicular to a screw axis are used to establish a
    hole's centreline.  The remaining axial translation is chosen by placing
    the attachment entirely inside the corresponding carrier side.
    """
    best: dict[str, Any] | None = None
    attachment_corners = _box_corners(attachment_lo, attachment_hi)
    for rotation in _proper_signed_permutations():
        transformed_lines: list[tuple[HoleLine, int, np.ndarray]] = []
        for line in attachment_lines:
            axis_vector = rotation[:, line.axis]
            mapped_axis = _major_axis(axis_vector)
            if mapped_axis is not None:
                transformed_lines.append((line, mapped_axis, rotation @ line.centre))
        for screw_axis in range(3):
            if allowed_screw_axes is not None and screw_axis not in allowed_screw_axes:
                continue
            if excluded_screw_axes and screw_axis in excluded_screw_axes:
                continue
            candidates = [(line, point) for line, axis, point in transformed_lines if axis == screw_axis]
            carrier_side_lines = [
                line
                for line in carrier_lines
                if line.axis == screw_axis and min(abs(line.centre[screw_axis] - carrier_lo[screw_axis]), abs(line.centre[screw_axis] - carrier_hi[screw_axis])) <= 12.0
            ]
            if len(candidates) < MIN_HOLE_LINES or len(carrier_side_lines) < MIN_HOLE_LINES:
                continue
            perpendicular = [dimension for dimension in range(3) if dimension != screw_axis]
            hypotheses: dict[tuple[int, int], list[tuple[HoleLine, np.ndarray, HoleLine, np.ndarray]]] = {}
            for attachment, point in candidates:
                for carrier in carrier_side_lines:
                    if _shared_radius(attachment, carrier) is None:
                        continue
                    delta = carrier.centre - point
                    key = tuple(int(round(delta[dimension] / LINE_TOLERANCE_MM)) for dimension in perpendicular)
                    hypotheses.setdefault(key, []).append((attachment, point, carrier, delta))
            for entries in hypotheses.values():
                if len(entries) < MIN_HOLE_LINES:
                    continue
                # Establish the two in-plane translation components by robust
                # median, then greedily form a one-to-one set of physical lines.
                planar_delta = np.median(np.asarray([entry[3] for entry in entries]), axis=0)
                rotated_corners = attachment_corners @ rotation.T
                side_centres = np.asarray([line.centre[screw_axis] for line in carrier_side_lines])
                chosen_side = carrier_hi[screw_axis] if abs(np.median(side_centres) - carrier_hi[screw_axis]) < abs(np.median(side_centres) - carrier_lo[screw_axis]) else carrier_lo[screw_axis]
                translate = planar_delta.copy()
                if chosen_side == carrier_hi[screw_axis]:
                    translate[screw_axis] = carrier_hi[screw_axis] - rotated_corners[:, screw_axis].max()
                else:
                    translate[screw_axis] = carrier_lo[screw_axis] - rotated_corners[:, screw_axis].min()
                available = set(range(len(carrier_side_lines)))
                matches = []
                for attachment, point in candidates:
                    placed = point + translate
                    options = []
                    for index in available:
                        carrier = carrier_side_lines[index]
                        radius = _shared_radius(attachment, carrier)
                        if radius is None:
                            continue
                        residual = float(np.linalg.norm(placed[perpendicular] - carrier.centre[perpendicular]))
                        if residual <= LINE_TOLERANCE_MM:
                            options.append((residual, index, carrier, radius))
                    if options:
                        residual, index, carrier, radius = min(options)
                        available.remove(index)
                        matches.append((attachment, carrier, residual, radius, placed))
                if len(matches) < MIN_HOLE_LINES:
                    continue
                pattern = np.asarray([match[4][perpendicular] for match in matches])
                spread = float(np.max(np.linalg.norm(pattern[:, None] - pattern[None, :], axis=-1)))
                if spread < MIN_PATTERN_SPREAD_MM:
                    continue
                residual = float(sum(match[2] for match in matches) / len(matches))
                candidate = {
                    "rotation": rotation,
                    "translate": translate,
                    "screw_axis": screw_axis,
                    "carrier_side": "max" if chosen_side == carrier_hi[screw_axis] else "min",
                    "mean_in_plane_residual_mm": residual,
                    "pattern_spread_mm": spread,
                    "matches": matches,
                }
                if best is None or (len(matches), -residual, spread) > (len(best["matches"]), -best["mean_in_plane_residual_mm"], best["pattern_spread_mm"]):
                    best = candidate
    return best


def solve(proxy_audit_path: Path, output_manifest: Path) -> dict[str, Any]:
    audit = json.loads(proxy_audit_path.read_text(encoding="utf-8"))
    components = list(audit["components"])
    dimensions = [np.asarray(component["source_dimensions"], dtype=float) for component in components]
    carrier_index = max(range(len(components)), key=lambda index: float(np.linalg.norm(dimensions[index])))
    remaining = [index for index in range(len(components)) if index != carrier_index]
    attachment_index = min(remaining, key=lambda index: float(np.linalg.norm(dimensions[index])))
    module_index = next(index for index in remaining if index != attachment_index)
    carrier, attachment, module = components[carrier_index], components[attachment_index], components[module_index]
    cylinders = audit["cylindrical_interface_candidates"]
    # A flange's screw bores normally pass through its thinnest direction.  This
    # geometric prior prevents an unrelated internal connector pattern from
    # becoming a fictitious side-wall attachment.
    attachment_thickness_axis = int(np.argmin(np.asarray(attachment["source_dimensions"], dtype=float)))
    attachment_hole_lines = [
        line
        for line in _hole_lines(cylinders[attachment["proxy_file"]])
        if line.axis == attachment_thickness_axis
    ]
    solution = _candidate_attachment_pose(
        np.asarray(carrier["source_bbox_min"], dtype=float),
        np.asarray(carrier["source_bbox_max"], dtype=float),
        np.asarray(attachment["source_bbox_min"], dtype=float),
        np.asarray(attachment["source_bbox_max"], dtype=float),
        _hole_lines(cylinders[carrier["proxy_file"]]),
        attachment_hole_lines,
        # Rack ears attach to an end flange: its normal is the carrier's depth
        # axis (the largest enclosure dimension), not the thin top/bottom axis.
        allowed_screw_axes={int(np.argmax(np.asarray(carrier["source_dimensions"], dtype=float)))},
    )
    if solution is None:
        raise RuntimeError("no 3-hole, same-radius, in-plane-consistent attachment pose found")
    # Main-module pose is intentionally not invented here.  It remains at its
    # source pose until an independently validated rail/stop proposal is passed
    # in; this prevents a valid ear result from forcing a PSU false positive.
    manifest_components = []
    originals_dir = proxy_audit_path.parents[2] / "5"  # caller can rewrite sources; used only by case runner
    for index, component in enumerate(components):
        placement: dict[str, Any] = {"translate": [0.0, 0.0, 0.0]}
        if index == attachment_index:
            placement = {"translate": np.round(solution["translate"], 6).tolist(), "rotate_sequence": [{"axis_angle": _rotation_axis_angle(solution["rotation"])}]}
        manifest_components.append({"id": f"comp_{index + 1:02d}", "source_file": component["proxy_file"], "placement": placement})
    result = {
        "status": "review_only",
        "carrier_file": carrier["proxy_file"],
        "attachment_file": attachment["proxy_file"],
        "module_file": module["proxy_file"],
        "attachment_transform": manifest_components[attachment_index]["placement"],
        "screw_axis": solution["screw_axis"],
        "carrier_side": solution["carrier_side"],
        "mean_in_plane_residual_mm": solution["mean_in_plane_residual_mm"],
        "pattern_spread_mm": solution["pattern_spread_mm"],
        "independent_hole_matches": [
            {"attachment_faces": match[0].face_indices, "carrier_faces": match[1].face_indices, "shared_radius_mm": match[3], "in_plane_residual_mm": match[2]}
            for match in solution["matches"]
        ],
        "module_status": "unresolved_pending_independent_rail_stop_solution",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps({"components": manifest_components, "result": result}, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proxy_audit", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(solve(args.proxy_audit, args.output_json), indent=2))


if __name__ == "__main__":
    main()
