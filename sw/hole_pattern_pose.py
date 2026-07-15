"""Pose proposals from repeated cylindrical-hole patterns in read-only proxies.

This is deliberately a conservative proposer: it returns no pose unless at
least three non-identical small cylindrical interfaces agree after one rigid
transform.  Hole-pattern evidence replaces the unsafe assumption that a module
merely fitting in a carrier bounding box is an assembly relation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _box_corners, _proper_signed_permutations, _rotation_axis_angle


MIN_INDEPENDENT_HOLE_MATCHES = 3
MIN_HOLE_PATTERN_SPREAD_MM = 12.0


def _axis_compatible(first: np.ndarray, second: np.ndarray) -> bool:
    return abs(float(np.dot(first, second))) >= 0.985


def _radius_compatible(first: float, second: float) -> bool:
    return abs(first - second) <= max(0.25, min(first, second) * 0.12)


def _small_cylinders(rows: list[dict[str, Any]], *, max_count: int = 350) -> list[dict[str, Any]]:
    """Keep likely fastener bores, retaining diversity across radii/positions."""
    chosen: list[dict[str, Any]] = []
    # The extractor already removes large shells.  Very long cylindrical faces
    # are normally rails/tubes rather than mounting bores.
    buckets: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        if not (0.5 <= float(row["radius"]) <= 12.0 and 0.08 <= float(row["axial_extent"]) <= 18.0):
            continue
        buckets.setdefault(round(float(row["radius"]) * 4.0) / 4.0, []).append(row)
    # Round-robin radius bins so thousands of 0.5-mm decorative cylinders do
    # not erase the less frequent mounting-hole diameters.
    ordered = [sorted(bucket, key=lambda item: (float(item["axial_extent"]), item["face_index"])) for _, bucket in sorted(buckets.items())]
    while ordered and len(chosen) < max_count:
        remaining: list[list[dict[str, Any]]] = []
        for bucket in ordered:
            if not bucket or len(chosen) >= max_count:
                continue
            row = bucket.pop(0)
            centre = np.asarray(row["centre"], dtype=float)
            duplicate = False
            for existing in chosen:
                if abs(float(existing["radius"]) - float(row["radius"])) < 0.05 and np.linalg.norm(centre - np.asarray(existing["centre"], dtype=float)) < 0.25:
                    duplicate = True
                    break
            if not duplicate:
                chosen.append(row)
            if bucket:
                remaining.append(bucket)
        ordered = remaining
    return chosen


def _overlap_fraction(module_lo: np.ndarray, module_hi: np.ndarray, carrier_lo: np.ndarray, carrier_hi: np.ndarray) -> float:
    intersection = np.maximum(0.0, np.minimum(module_hi, carrier_hi) - np.maximum(module_lo, carrier_lo))
    return float(np.prod(intersection) / max(np.prod(module_hi - module_lo), 1e-9))


def _score_transform(
    rotation: np.ndarray,
    translate: np.ndarray,
    module_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
) -> tuple[int, float, list[dict[str, Any]]]:
    matches: list[dict[str, Any]] = []
    used_carrier: set[int] = set()
    for module in module_rows:
        centre = rotation @ np.asarray(module["centre"], dtype=float) + translate
        axis = rotation @ np.asarray(module["axis"], dtype=float)
        best: tuple[float, int, dict[str, Any]] | None = None
        for index, carrier in enumerate(carrier_rows):
            if index in used_carrier or not _radius_compatible(float(module["radius"]), float(carrier["radius"])):
                continue
            if not _axis_compatible(axis, np.asarray(carrier["axis"], dtype=float)):
                continue
            error = float(np.linalg.norm(centre - np.asarray(carrier["centre"], dtype=float)))
            if error <= 1.5 and (best is None or error < best[0]):
                best = (error, index, carrier)
        if best is not None:
            error, index, carrier = best
            used_carrier.add(index)
            matches.append(
                {
                    "module_face_index": module["face_index"],
                    "carrier_face_index": carrier["face_index"],
                    "radius": round(float(module["radius"]), 6),
                    "centre_residual_mm": round(error, 6),
                }
            )
    residual = float(sum(item["centre_residual_mm"] for item in matches))
    if len(matches) >= 2:
        points = []
        for item in matches:
            row = next(row for row in module_rows if row["face_index"] == item["module_face_index"])
            points.append(rotation @ np.asarray(row["centre"], dtype=float) + translate)
        spread = max(float(np.linalg.norm(first - second)) for first in points for second in points)
    else:
        spread = 0.0
    return len(matches), residual, matches, spread


def propose_hole_pattern_pose(
    proxy_audit: Path,
    original_dir: Path,
    output_manifest: Path,
    *,
    module_file: str | None = None,
) -> dict[str, Any]:
    audit = json.loads(proxy_audit.read_text(encoding="utf-8"))
    components = list(audit["components"])
    dimensions = [np.asarray(component["source_dimensions"], dtype=float) for component in components]
    carrier_index = max(range(len(components)), key=lambda index: float(np.linalg.norm(dimensions[index])))
    others = [index for index in range(len(components)) if index != carrier_index]
    if module_file is None:
        module_index = max(others, key=lambda index: float(np.linalg.norm(dimensions[index])))
    else:
        matching = [index for index in others if components[index]["proxy_file"] == module_file]
        if len(matching) != 1:
            raise ValueError("--module-file must name exactly one non-carrier proxy component")
        module_index = matching[0]
    carrier, module = components[carrier_index], components[module_index]
    cylinders = audit.get("cylindrical_interface_candidates", {})
    # This is a proposal-stage RANSAC-like recall pass, not an exhaustive all
    # hole-pair Cartesian product.  Radius-diverse sampling keeps it bounded
    # and lets repeated two-hole support reject accidental one-hole matches.
    carrier_rows = _small_cylinders(cylinders.get(carrier["proxy_file"], []), max_count=72)
    module_rows = _small_cylinders(cylinders.get(module["proxy_file"], []), max_count=48)
    if len(carrier_rows) < 2 or len(module_rows) < 2:
        raise RuntimeError("insufficient small cylindrical interface candidates")

    module_lo = np.asarray(module["source_bbox_min"], dtype=float)
    module_hi = np.asarray(module["source_bbox_max"], dtype=float)
    carrier_lo = np.asarray(carrier["source_bbox_min"], dtype=float)
    carrier_hi = np.asarray(carrier["source_bbox_max"], dtype=float)
    candidates: dict[tuple[tuple[float, ...], tuple[float, ...]], dict[str, Any]] = {}
    for rotation in _proper_signed_permutations():
        for module_row in module_rows:
            module_axis = rotation @ np.asarray(module_row["axis"], dtype=float)
            module_centre = rotation @ np.asarray(module_row["centre"], dtype=float)
            for carrier_row in carrier_rows:
                if not _radius_compatible(float(module_row["radius"]), float(carrier_row["radius"])):
                    continue
                if not _axis_compatible(module_axis, np.asarray(carrier_row["axis"], dtype=float)):
                    continue
                translate = np.asarray(carrier_row["centre"], dtype=float) - module_centre
                key = (tuple(rotation.reshape(-1)), tuple(np.round(translate, 1)))
                if key in candidates:
                    continue
                count, residual, matches, spread = _score_transform(rotation, translate, module_rows, carrier_rows)
                if count < MIN_INDEPENDENT_HOLE_MATCHES or spread < MIN_HOLE_PATTERN_SPREAD_MM:
                    continue
                transformed = _box_corners(module_lo, module_hi) @ rotation.T + translate
                bbox_lo, bbox_hi = transformed.min(axis=0), transformed.max(axis=0)
                overlap = _overlap_fraction(bbox_lo, bbox_hi, carrier_lo, carrier_hi)
                candidates[key] = {
                    "rotation": rotation,
                    "translate": translate,
                    "hole_match_count": count,
                    "hole_residual_mm": residual,
                    "hole_matches": matches,
                    "hole_pattern_spread_mm": spread,
                    "carrier_bbox_overlap_fraction": overlap,
                }
    if not candidates:
        raise RuntimeError(
            f"no transform is supported by {MIN_INDEPENDENT_HOLE_MATCHES}+ compatible cylindrical interfaces "
            f"with {MIN_HOLE_PATTERN_SPREAD_MM:g} mm pattern spread"
        )
    # Strongest repeated pattern first.  Bounding-box overlap is only a weak
    # tie-breaker and never substitutes for a second matching hole.
    best = sorted(
        candidates.values(),
        key=lambda item: (-item["hole_match_count"], item["hole_residual_mm"], -item["carrier_bbox_overlap_fraction"]),
    )[0]

    originals = {
        path.name: path
        for path in [*original_dir.glob("*.step"), *original_dir.glob("*.stp")]
        if path.stem.lower() != "assembly"
    }
    manifest_components = []
    for index, component in enumerate(components):
        source = originals[component["proxy_file"]]
        placement: dict[str, Any] = {"translate": [0.0, 0.0, 0.0]}
        if index == module_index:
            placement = {
                "translate": np.round(best["translate"], 6).tolist(),
                "rotate_sequence": [{"axis_angle": _rotation_axis_angle(best["rotation"])}],
            }
        manifest_components.append(
            {
                "id": f"comp_{index + 1:02d}",
                "source": Path(os.path.relpath(source, output_manifest.parent)).as_posix(),
                "label": Path(component["proxy_file"]).stem,
                "role": "component",
                "placement": placement,
            }
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "2.0.0",
        "assembly_name": "proxy_hole_pattern_review",
        "global_units": "mm",
        "components": manifest_components,
        "proxy_pose_transfer": {
            "status": "review_only",
            "reason": "At least two compatible cylindrical interfaces support the transform; original geometry still requires collision and functional review.",
        },
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "status": "review_only",
        "carrier": carrier["proxy_file"],
        "module": module["proxy_file"],
        "auxiliary_parts_excluded_from_module_mating": [components[index]["proxy_file"] for index in others if index != module_index],
        "candidate_hole_counts": {"carrier": len(carrier_rows), "module": len(module_rows)},
        "hole_match_count": best["hole_match_count"],
        "hole_residual_mm": best["hole_residual_mm"],
        "hole_pattern_spread_mm": best["hole_pattern_spread_mm"],
        "hole_matches": best["hole_matches"],
        "rotation_matrix": best["rotation"].round(6).tolist(),
        "axis_angle": _rotation_axis_angle(best["rotation"]),
        "translate": np.round(best["translate"], 6).tolist(),
        "carrier_bbox_overlap_fraction": best["carrier_bbox_overlap_fraction"],
        "accepted": False,
        "required_next_gate": "original_geometry_collision_and_functional_review",
    }
    output_manifest.with_name("hole_pattern_pose_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proxy_audit", type=Path)
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--module-file", help="optional component file selected for pairwise hole-pattern proposal")
    args = parser.parse_args()
    result = propose_hole_pattern_pose(args.proxy_audit, args.original_dir, args.output_manifest, module_file=args.module_file)
    print(json.dumps({key: result[key] for key in ("status", "hole_match_count", "hole_residual_mm", "translate", "axis_angle")}, indent=2))


if __name__ == "__main__":
    main()
