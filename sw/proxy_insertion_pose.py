"""Propose a conservative insertion pose from interface-proxy metadata.

This is intentionally a *candidate generator*, not an assembly acceptance
step.  It uses the largest component as a geometric carrier and the largest
remaining component as an inserted module.  Small auxiliary components cannot
create a direct mate to the module, avoiding a common single-planar-contact
false positive in sparse proxy geometry.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _proper_signed_permutations() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=float)
        for col, row in enumerate(permutation):
            base[row, col] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if round(np.linalg.det(candidate)) == 1:
                matrices.append(candidate)
    return matrices


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.asarray([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)


def _rotation_axis_angle(matrix: np.ndarray) -> list[float]:
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return [0.0, 0.0, 1.0, 0.0]
    if abs(math.pi - angle) < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) / 2.0, 0.0))
        if matrix[0, 1] < 0:
            axis[1] *= -1
        if matrix[0, 2] < 0:
            axis[2] *= -1
    else:
        axis = np.array(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
            dtype=float,
        ) / (2.0 * math.sin(angle))
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return [float(axis[0]), float(axis[1]), float(axis[2]), float(math.degrees(angle))]


def _choose_rotation(carrier_dims: np.ndarray, module_lo: np.ndarray, module_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the best proper axis permutation by conservative containment.

    We maximize the smallest occupied carrier fraction.  This favours a long
    module aligning to the carrier's long insertion direction and avoids a
    degenerate orientation that only fits by being implausibly skinny in one
    carrier dimension.
    """
    corners = _box_corners(module_lo, module_hi)
    best: tuple[tuple[float, float, float], np.ndarray, np.ndarray, np.ndarray] | None = None
    for rotation in _proper_signed_permutations():
        rotated = corners @ rotation.T
        lo, hi = rotated.min(axis=0), rotated.max(axis=0)
        dimensions = hi - lo
        ratio = dimensions / carrier_dims
        if np.any(ratio > 1.0 + 1e-6):
            continue
        # Min ratio is primary; a small mean log-gap breaks ties without
        # relying on component identity or a predefined orientation.
        rank = (float(np.min(ratio)), float(np.mean(ratio)), -float(np.std(ratio)))
        if best is None or rank > best[0]:
            best = (rank, rotation, lo, hi)
    if best is None:
        raise RuntimeError("no proper bounding-box orientation fits inside carrier")
    return best[1], best[2], best[3]


def _planes_near_boundary(planes: list[dict[str, Any]], axis: int, boundary: float, scale: float) -> list[dict[str, Any]]:
    result = []
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=float)
        if abs(normal[axis]) < 0.95:
            continue
        coordinate = float(plane["origin"][axis])
        if abs(coordinate - boundary) <= max(3.0, 0.035 * scale):
            result.append(plane)
    return result


def propose_insertion_pose(proxy_audit: Path, original_dir: Path, output_manifest: Path) -> dict[str, Any]:
    audit = json.loads(proxy_audit.read_text(encoding="utf-8"))
    components = list(audit["components"])
    if len(components) < 2:
        raise ValueError("need at least carrier and inserted module")
    dimensions = [np.asarray(component["source_dimensions"], dtype=float) for component in components]
    carrier_index = max(range(len(components)), key=lambda index: float(np.linalg.norm(dimensions[index])))
    noncarrier = [index for index in range(len(components)) if index != carrier_index]
    module_index = max(noncarrier, key=lambda index: float(np.linalg.norm(dimensions[index])))
    carrier = components[carrier_index]
    module = components[module_index]
    carrier_lo = np.asarray(carrier["source_bbox_min"], dtype=float)
    carrier_hi = np.asarray(carrier["source_bbox_max"], dtype=float)
    module_lo = np.asarray(module["source_bbox_min"], dtype=float)
    module_hi = np.asarray(module["source_bbox_max"], dtype=float)
    carrier_dims = carrier_hi - carrier_lo
    rotation, rotated_lo, rotated_hi = _choose_rotation(carrier_dims, module_lo, module_hi)
    module_dims = rotated_hi - rotated_lo

    planes_by_moving = audit.get("carrier_interface_planes_by_moving_component", {})
    planes = planes_by_moving.get(module["proxy_file"], [])
    insertion_axis = int(np.argmax(carrier_dims))
    support_axis = int(np.argmin(carrier_dims))
    lateral_axis = ({0, 1, 2} - {insertion_axis, support_axis}).pop()

    # Place the module on a supported carrier side, at an actual near-boundary
    # stop if available.  The endpoints are selected from geometry; high/low is
    # not encoded per case.  Ties prefer the side with more support planes.
    target_lo = carrier_lo + (carrier_dims - module_dims) / 2.0
    support_low = _planes_near_boundary(planes, support_axis, carrier_lo[support_axis], carrier_dims[support_axis])
    support_high = _planes_near_boundary(planes, support_axis, carrier_hi[support_axis], carrier_dims[support_axis])
    if len(support_low) >= len(support_high):
        target_lo[support_axis] = carrier_lo[support_axis]
        chosen_support_side = "low"
    else:
        target_lo[support_axis] = carrier_hi[support_axis] - module_dims[support_axis]
        chosen_support_side = "high"

    stop_low = _planes_near_boundary(planes, insertion_axis, carrier_lo[insertion_axis], carrier_dims[insertion_axis])
    stop_high = _planes_near_boundary(planes, insertion_axis, carrier_hi[insertion_axis], carrier_dims[insertion_axis])
    if len(stop_high) >= len(stop_low):
        stop_coordinate = max([float(plane["origin"][insertion_axis]) for plane in stop_high], default=float(carrier_hi[insertion_axis]))
        target_lo[insertion_axis] = min(carrier_hi[insertion_axis] - module_dims[insertion_axis], stop_coordinate - module_dims[insertion_axis])
        chosen_stop_side = "high"
    else:
        stop_coordinate = min([float(plane["origin"][insertion_axis]) for plane in stop_low], default=float(carrier_lo[insertion_axis]))
        target_lo[insertion_axis] = max(carrier_lo[insertion_axis], stop_coordinate)
        chosen_stop_side = "low"
    translate = target_lo - rotated_lo

    originals = {
        path.name: path
        for path in [*original_dir.glob("*.step"), *original_dir.glob("*.stp")]
        if path.stem.lower() != "assembly"
    }
    manifest_components: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        name = component["proxy_file"]
        source = originals.get(name)
        if source is None:
            raise FileNotFoundError(f"missing original component {name}")
        placement: dict[str, Any] = {"translate": [0.0, 0.0, 0.0]}
        if index == module_index:
            placement = {
                "translate": np.round(translate, 6).tolist(),
                "rotate_sequence": [{"axis_angle": _rotation_axis_angle(rotation)}],
            }
        manifest_components.append({
            "id": f"comp_{index + 1:02d}",
            "source": Path(__import__("os").path.relpath(source, output_manifest.parent)).as_posix(),
            "label": Path(name).stem,
            "role": "component",
            "placement": placement,
        })
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "2.0.0",
        "assembly_name": "proxy_interface_insertion_review",
        "global_units": "mm",
        "components": manifest_components,
        "proxy_pose_transfer": {
            "status": "review_only",
            "reason": "Bounding-box containment, selected support planes, and selected stop planes propose an insertion transform. Original B-Rep validation is still required.",
        },
    }
    output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "status": "review_only",
        "carrier": carrier["proxy_file"],
        "module": module["proxy_file"],
        "auxiliary_parts_excluded_from_module_mating": [components[index]["proxy_file"] for index in noncarrier if index != module_index],
        "rotation_matrix": rotation.round(6).tolist(),
        "axis_angle": _rotation_axis_angle(rotation),
        "translate": np.round(translate, 6).tolist(),
        "rotated_module_dimensions": np.round(module_dims, 6).tolist(),
        "carrier_dimensions": np.round(carrier_dims, 6).tolist(),
        "insertion_axis": insertion_axis,
        "support_axis": support_axis,
        "lateral_axis": lateral_axis,
        "support_plane_count": {"low": len(support_low), "high": len(support_high), "selected": chosen_support_side},
        "stop_plane_count": {"low": len(stop_low), "high": len(stop_high), "selected": chosen_stop_side, "coordinate": stop_coordinate},
        "evidence": ["oriented_bbox_containment", "carrier_support_planes", "carrier_stop_planes"],
        "does_not_establish": ["source_provenance", "exact_original_geometry_collision_free", "functional_validity"],
    }
    output_manifest.with_name("proxy_insertion_pose_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proxy_audit", type=Path)
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("output_manifest", type=Path)
    args = parser.parse_args()
    result = propose_insertion_pose(args.proxy_audit, args.original_dir, args.output_manifest)
    print(json.dumps({key: result[key] for key in ("status", "carrier", "module", "translate", "axis_angle")}, indent=2))


if __name__ == "__main__":
    main()
