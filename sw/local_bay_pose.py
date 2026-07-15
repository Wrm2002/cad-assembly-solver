"""Find a locally constrained insertion bay from original planar geometry.

This candidate generator is intentionally role-agnostic.  For every proper
box orientation it searches pairs of opposing planar guide walls whose gap is
only slightly larger than the transformed module width, then requires an
independent support plane.  It returns review candidates only: local guiding
geometry establishes a plausible pose, not source provenance or final mating.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _box_corners, _proper_signed_permutations, _rotation_axis_angle


def _axis(vector: list[float]) -> int | None:
    index = int(np.argmax(np.abs(vector)))
    return index if abs(vector[index]) >= 0.985 else None


def _ranges(plane: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(plane["origin"], dtype=float)
    u = np.asarray(plane["axis_u"], dtype=float) * float(plane["extent_u"]) / 2.0
    v = np.asarray(plane["axis_v"], dtype=float) * float(plane["extent_v"]) / 2.0
    corners = np.asarray([origin + su * u + sv * v for su in (-1, 1) for sv in (-1, 1)])
    return corners.min(axis=0), corners.max(axis=0)


def _planar_by_axis(planes: list[dict[str, Any]], axis: int) -> list[dict[str, Any]]:
    out = []
    for plane in planes:
        if _axis(plane["normal"]) != axis:
            continue
        lo, hi = _ranges(plane)
        plane = dict(plane)
        plane["lo"], plane["hi"] = lo, hi
        out.append(plane)
    return out


def propose(module_bbox: dict[str, Any], carrier_bbox: dict[str, Any], carrier_planes: list[dict[str, Any]]) -> dict[str, Any]:
    def bounds(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(payload.get("min", payload.get("bbox_min")), dtype=float),
            np.asarray(payload.get("max", payload.get("bbox_max")), dtype=float),
        )

    mlo, mhi = bounds(module_bbox)
    clo, chi = bounds(carrier_bbox)
    carrier_thin_axis = int(np.argmin(chi - clo))
    corners = _box_corners(mlo, mhi)
    candidates: list[dict[str, Any]] = []
    for rotation in _proper_signed_permutations():
        rotated = corners @ rotation.T
        rlo, rhi = rotated.min(axis=0), rotated.max(axis=0)
        dims = rhi - rlo
        support_axis = int(np.argmin(dims))
        # The carrier's thinnest envelope dimension is the only globally
        # defensible support direction for an enclosure; otherwise a large
        # front/rear plate can be mistaken for a floor.
        if support_axis != carrier_thin_axis:
            continue
        remaining = [axis for axis in range(3) if axis != support_axis]
        # The smaller of the horizontal dimensions is normally guided by a
        # close wall pair; enumerate rather than assume a carrier orientation.
        lateral_axis = min(remaining, key=lambda axis: dims[axis])
        insertion_axis = next(axis for axis in remaining if axis != lateral_axis)
        walls = _planar_by_axis(carrier_planes, lateral_axis)
        supports = _planar_by_axis(carrier_planes, support_axis)
        for left, right in itertools.combinations(walls, 2):
            nl, nr = np.asarray(left["normal"]), np.asarray(right["normal"])
            if float(np.dot(nl, nr)) > -0.97:
                continue
            a, b = sorted((float(left["origin"][lateral_axis]), float(right["origin"][lateral_axis])))
            gap = b - a
            clearance = gap - dims[lateral_axis]
            if clearance < -1.0 or clearance > max(7.0, dims[lateral_axis] * 0.10):
                continue
            # Both guide walls must overlap along the anticipated insertion
            # direction for a material portion of the module length.
            overlap_lo = max(float(left["lo"][insertion_axis]), float(right["lo"][insertion_axis]))
            overlap_hi = min(float(left["hi"][insertion_axis]), float(right["hi"][insertion_axis]))
            coverage = max(0.0, overlap_hi - overlap_lo) / max(float(dims[insertion_axis]), 1e-9)
            # Folded rails often terminate before the module's full body;
            # retain a candidate when they guide a substantial majority of
            # the insertion span, but keep it review-only below full cover.
            if coverage < 0.40:
                continue
            for support in supports:
                coordinate = float(support["origin"][support_axis])
                normal = np.asarray(support["normal"])
                # A lower support must face inward/upward; signs are inferred
                # from the module location, and both carrier sides are tried.
                for lower in (True, False):
                    target_lo = np.zeros(3, dtype=float)
                    target_lo[lateral_axis] = a + max(0.0, clearance) / 2.0
                    target_lo[insertion_axis] = overlap_lo
                    target_lo[support_axis] = coordinate if lower else coordinate - dims[support_axis]
                    translate = target_lo - rlo
                    moved = (rotation @ corners.T).T + translate
                    inside = float(np.mean(np.all((moved >= clo - 1.5) & (moved <= chi + 1.5), axis=1)))
                    if inside < 0.99:
                        continue
                    # Support must overlap the module footprint in both in-plane axes.
                    support_overlap = 1.0
                    for axis in (lateral_axis, insertion_axis):
                        inter = max(0.0, min(moved[:, axis].max(), support["hi"][axis]) - max(moved[:, axis].min(), support["lo"][axis]))
                        support_overlap *= inter / max(float(dims[axis]), 1e-9)
                    if support_overlap < 0.45:
                        continue
                    score = 0.55 * min(1.0, coverage) + 0.30 * support_overlap + 0.15 * max(0.0, 1.0 - max(clearance, 0.0) / 7.0)
                    candidates.append({
                        "rotation": rotation.round(10).tolist(),
                        "axis_angle": _rotation_axis_angle(rotation),
                        "translation": translate.round(6).tolist(),
                        "support_axis": support_axis,
                        "lateral_axis": lateral_axis,
                        "insertion_axis": insertion_axis,
                        "guide_wall_faces": [left["face_index"], right["face_index"]],
                        "guide_wall_gap_mm": round(gap, 5),
                        "guide_clearance_mm": round(clearance, 5),
                        "guide_coverage": round(coverage, 5),
                        "support_face": support["face_index"],
                        "support_overlap": round(support_overlap, 5),
                        "inside_fraction": inside,
                        "score": round(score, 5),
                    })
    candidates.sort(key=lambda item: -item["score"])
    return {"status": "review_only", "method": "local_opposing_guides_plus_support", "candidate_count": len(candidates), "candidates": candidates[:40]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module_bbox", type=Path)
    parser.add_argument("carrier_bbox", type=Path)
    parser.add_argument("carrier_planes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    module_bbox = json.loads(args.module_bbox.read_text(encoding="utf-8"))
    carrier_bbox = json.loads(args.carrier_bbox.read_text(encoding="utf-8"))
    planes = json.loads(args.carrier_planes.read_text(encoding="utf-8"))["planes"]
    result = propose(module_bbox, carrier_bbox, planes)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "candidate_count": result["candidate_count"]}, indent=2))


if __name__ == "__main__":
    main()
