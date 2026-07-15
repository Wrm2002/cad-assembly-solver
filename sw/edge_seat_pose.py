"""Add an exterior coplanar seating constraint to a local bay proposal.

The module remains guided by the local bay candidate.  This refinement selects
an exterior broad plane of the transformed module and an exterior broad plane
of the carrier with the same outward normal, then makes them coplanar.  It is
geometry-only and always emits review status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from interface_geometry_proxy import extract_source_summary


def _axis(vector: np.ndarray) -> int | None:
    index = int(np.argmax(np.abs(vector)))
    return index if abs(vector[index]) >= 0.985 else None


def _plane_payload(summary: Any) -> list[dict[str, Any]]:
    return [patch.as_json() for patch in summary.planar_faces]


def refine(
    candidate: dict[str, Any],
    module_step: Path,
    carrier_bbox: dict[str, Any],
    carrier_planes: list[dict[str, Any]],
) -> dict[str, Any]:
    R = np.asarray(candidate["rotation"], dtype=float)
    base_t = np.asarray(candidate["translation"], dtype=float)
    source = extract_source_summary(module_step)
    clo = np.asarray(carrier_bbox.get("min", carrier_bbox.get("bbox_min")), dtype=float)
    chi = np.asarray(carrier_bbox.get("max", carrier_bbox.get("bbox_max")), dtype=float)
    corners = np.asarray(
        [[x, y, z] for x in (source.bbox_min[0], source.bbox_max[0]) for y in (source.bbox_min[1], source.bbox_max[1]) for z in (source.bbox_min[2], source.bbox_max[2])],
        dtype=float,
    )
    best: tuple[float, dict[str, Any]] | None = None
    for source_plane in _plane_payload(source):
        normal = R @ np.asarray(source_plane["normal"], dtype=float)
        axis = _axis(normal)
        if axis is None or float(source_plane["area_proxy"]) < 250.0:
            continue
        # The bay solver already identifies its transverse guide direction.
        # An exterior seating plane must close that same open side; otherwise
        # a top cover can be incorrectly selected as an installation edge.
        if axis != int(candidate["lateral_axis"]):
            continue
        source_coordinate = float((R @ np.asarray(source_plane["origin"], dtype=float) + base_t)[axis])
        for carrier_plane in carrier_planes:
            carrier_normal = np.asarray(carrier_plane["normal"], dtype=float)
            if _axis(carrier_normal) != axis or float(np.dot(normal, carrier_normal)) < 0.985:
                continue
            boundary = clo[axis] if carrier_normal[axis] < 0 else chi[axis]
            if abs(float(carrier_plane["origin"][axis]) - boundary) > 3.0:
                continue
            if float(carrier_plane["area_proxy"]) < 500.0:
                continue
            translate = base_t.copy()
            translate[axis] += float(carrier_plane["origin"][axis]) - source_coordinate
            moved = (R @ corners.T).T + translate
            if moved[:, axis].min() < clo[axis] - 1.5 or moved[:, axis].max() > chi[axis] + 1.5:
                continue
            # Prefer broad seating faces and the smallest necessary change;
            # guide-wall evidence remains attached from the base candidate.
            score = float(source_plane["area_proxy"]) + 0.02 * float(carrier_plane["area_proxy"]) - abs(float(translate[axis] - base_t[axis]))
            row = {
                "rotation": candidate["rotation"],
                "axis_angle": candidate["axis_angle"],
                "translation": translate.round(6).tolist(),
                "base_candidate": candidate,
                "seat_axis": axis,
                "module_seat_face": source_plane["face_index"],
                "carrier_seat_face": carrier_plane["face_index"],
                "seat_coordinate": round(float(carrier_plane["origin"][axis]), 6),
                "edge_seat_shift_mm": round(float(translate[axis] - base_t[axis]), 6),
                "status": "review_only",
            }
            if best is None or score > best[0]:
                best = (score, row)
    if best is None:
        return {"status": "failed", "reason": "no contained exterior coplanar seating pair"}
    return best[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_audit", type=Path)
    parser.add_argument("module_step", type=Path)
    parser.add_argument("carrier_bbox", type=Path)
    parser.add_argument("carrier_planes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-index", type=int, default=28)
    args = parser.parse_args()
    audit = json.loads(args.local_audit.read_text(encoding="utf-8"))
    candidate = audit["candidates"][args.candidate_index]
    result = refine(
        candidate,
        args.module_step,
        json.loads(args.carrier_bbox.read_text(encoding="utf-8")),
        json.loads(args.carrier_planes.read_text(encoding="utf-8"))["planes"],
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "seat_axis", "edge_seat_shift_mm", "module_seat_face", "carrier_seat_face")}, indent=2))


if __name__ == "__main__":
    main()
