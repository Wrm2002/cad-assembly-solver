"""Add compound rigid-Pose hypotheses from repeated B-Rep interface axes.

Single-interface joint prediction is ambiguous for repeated holes.  This
module performs geometry-only RANSAC over same-radius cylindrical face sets.
Three or more mutually consistent axis correspondences define a rigid
transform; the remaining axial ambiguity is resolved at projected cylindrical
support contact events.  The result is a compound multi-evidence factor, not a
named flange/key/part-family rule.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cylinders(graph: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for node in graph.get("nodes") or []:
        radius, direction = node.get("radius"), node.get("axis_direction")
        if node.get("entity_type") != "face" or radius is None or direction is None:
            continue
        direction = np.asarray(direction, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        point = np.asarray(node.get("centroid") or node.get("axis_origin"), dtype=float)
        area = float(node.get("area") or 0.0)
        length = area / max(2.0 * math.pi * float(radius), 1e-12)
        result.append({
            "entity_id": str(node.get("node_id")),
            "point": point,
            "direction": direction,
            "radius": float(radius),
            "length": float(length),
        })
    return result


def _radius_groups(values: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for value in sorted(values, key=lambda row: row["radius"]):
        for group in groups:
            reference = float(np.mean([row["radius"] for row in group]))
            if abs(value["radius"] - reference) <= tolerance * max(reference, 1e-9):
                group.append(value)
                break
        else:
            groups.append([value])
    return groups


def _basis(axis: np.ndarray, tangent: np.ndarray) -> np.ndarray | None:
    tangent = tangent - axis * float(tangent @ axis)
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-8:
        return None
    tangent /= norm
    third = np.cross(axis, tangent)
    third /= max(float(np.linalg.norm(third)), 1e-12)
    return np.column_stack((axis, tangent, third))


def _pose_signature(transform: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(transform[:3, :4].reshape(-1), 3))


def _inverse_rigid(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _ransac_group(
    group_a: list[dict[str, Any]], group_b: list[dict[str, Any]], *, minimum_support: int,
    distance_tolerance_ratio: float, clearance_mm: float,
) -> list[dict[str, Any]]:
    points_a = np.stack([row["point"] for row in group_a])
    points_b = np.stack([row["point"] for row in group_b])
    axis_a = np.mean(np.stack([row["direction"] for row in group_a]), axis=0)
    axis_b = np.mean(np.stack([row["direction"] for row in group_b]), axis=0)
    axis_a /= max(float(np.linalg.norm(axis_a)), 1e-12)
    axis_b /= max(float(np.linalg.norm(axis_b)), 1e-12)
    pair_distances = [
        float(np.linalg.norm(points_a[i] - points_a[j]))
        for i in range(len(points_a)) for j in range(i + 1, len(points_a))
    ]
    pattern_scale = max(np.median([value for value in pair_distances if value > 1e-8]), 1.0)
    match_tolerance = max(.2, distance_tolerance_ratio * pattern_scale)
    candidates: dict[tuple[float, ...], dict[str, Any]] = {}
    for ia in range(len(points_a)):
        for ja in range(len(points_a)):
            if ia == ja:
                continue
            distance_a = float(np.linalg.norm(points_a[ja] - points_a[ia]))
            for ib in range(len(points_b)):
                for jb in range(len(points_b)):
                    if ib == jb:
                        continue
                    distance_b = float(np.linalg.norm(points_b[jb] - points_b[ib]))
                    if abs(distance_a - distance_b) > match_tolerance:
                        continue
                    basis_a = _basis(axis_a, points_a[ja] - points_a[ia])
                    if basis_a is None:
                        continue
                    for polarity in (1.0, -1.0):
                        mapped_axis_b = polarity * axis_b
                        basis_b = _basis(mapped_axis_b, points_b[jb] - points_b[ib])
                        if basis_b is None:
                            continue
                        rotation = basis_a @ basis_b.T
                        translation = points_a[ia] - rotation @ points_b[ib]
                        moved_b = points_b @ rotation.T + translation
                        distances = np.linalg.norm(points_a[:, None, :] - moved_b[None, :, :], axis=-1)
                        used_b: set[int] = set()
                        matches = []
                        for source_index in np.argsort(distances.min(axis=1)):
                            target_index = int(np.argmin(distances[source_index]))
                            if target_index in used_b or distances[source_index, target_index] > match_tolerance:
                                continue
                            used_b.add(target_index)
                            matches.append((int(source_index), target_index))
                        if len(matches) < minimum_support:
                            continue
                        # The matched cylindrical spans determine the two axial
                        # support events where their end faces just touch.
                        center_a = float(np.mean([points_a[i] @ axis_a for i, _ in matches]))
                        center_b = float(np.mean([moved_b[j] @ axis_a for _, j in matches]))
                        length_a = float(np.mean([group_a[i]["length"] for i, _ in matches]))
                        length_b = float(np.mean([group_b[j]["length"] for _, j in matches]))
                        offsets = (
                            center_a - .5 * length_a - (center_b + .5 * length_b) - clearance_mm,
                            center_a + .5 * length_a - (center_b - .5 * length_b) + clearance_mm,
                        )
                        for offset in offsets:
                            transform = np.eye(4)
                            transform[:3, :3] = rotation
                            transform[:3, 3] = translation + axis_a * float(offset)
                            signature = _pose_signature(transform)
                            residual = float(np.mean([distances[i, j] for i, j in matches]))
                            confidence = min(.995, .55 + .07 * len(matches) + .15 * math.exp(-residual / match_tolerance))
                            row = {
                                "transform": transform,
                                "confidence": confidence,
                                "support": len(matches),
                                "residual_mm": residual,
                                "radius": float(np.mean([value["radius"] for value in group_a])),
                                "offset_mm": float(offset),
                                # This is the intended residual separation at
                                # the axial support event.  Keeping it distinct
                                # from offset_mm prevents future validators from
                                # mistaking a placement translation for a gap.
                                "clearance_mm": float(clearance_mm),
                                "matches": [
                                    [group_a[i]["entity_id"], group_b[j]["entity_id"]] for i, j in matches
                                ],
                            }
                            previous = candidates.get(signature)
                            if previous is None or (row["support"], -row["residual_mm"]) > (previous["support"], -previous["residual_mm"]):
                                candidates[signature] = row
    return sorted(candidates.values(), key=lambda row: (-row["support"], row["residual_mm"], -row["confidence"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_result", type=Path)
    parser.add_argument("graph_a", type=Path)
    parser.add_argument("graph_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-support", type=int, default=3)
    parser.add_argument("--radius-tolerance", type=float, default=.02)
    parser.add_argument("--distance-tolerance-ratio", type=float, default=.03)
    parser.add_argument("--clearance-mm", type=float, default=.02)
    parser.add_argument("--maximum-additions", type=int, default=24)
    args = parser.parse_args()
    result = _read(args.base_result)
    groups_a = [g for g in _radius_groups(_cylinders(_read(args.graph_a)), args.radius_tolerance) if len(g) >= args.minimum_support]
    groups_b = [g for g in _radius_groups(_cylinders(_read(args.graph_b)), args.radius_tolerance) if len(g) >= args.minimum_support]
    hypotheses = []
    for group_a in groups_a:
        radius_a = float(np.mean([row["radius"] for row in group_a]))
        for group_b in groups_b:
            radius_b = float(np.mean([row["radius"] for row in group_b]))
            if abs(radius_a - radius_b) > args.radius_tolerance * max(radius_a, radius_b):
                continue
            hypotheses.extend(_ransac_group(
                group_a, group_b, minimum_support=args.minimum_support,
                distance_tolerance_ratio=args.distance_tolerance_ratio,
                clearance_mm=args.clearance_mm,
            ))
    additions = []
    for index, hypothesis in enumerate(hypotheses[: max(1, args.maximum_additions)]):
        transform = hypothesis.pop("transform")
        additions.append({
            "entity_a": f"compound_axis_set_a_{index:03d}",
            "entity_b": f"compound_axis_set_b_{index:03d}",
            "rank": index + 1,
            "manifold_type": "compound_multi_axis_rigid",
            "frame_a": np.eye(4).tolist(),
            "frame_b": _inverse_rigid(transform).tolist(),
            "initial_pose_b_in_a": transform.tolist(),
            "free_dof_mask": [0, 0, 0, 0, 0, 0],
            "confidence": float(hypothesis["confidence"]),
            "provenance": {
                "multi_interface_ransac": True,
                "independent_evidence_count": int(hypothesis["support"]),
                "geometry_only": True,
                "case_specific_override": False,
                **hypothesis,
            },
        })
    output = copy.deepcopy(result)
    output.setdefault("joint_hypotheses", {}).setdefault("rows", []).extend(additions)
    output["joint_hypotheses"]["multi_axis_ransac"] = {
        "added_rows": len(additions),
        "minimum_support": args.minimum_support,
        "geometry_only": True,
        "case_specific_override": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"added_rows": len(additions), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
