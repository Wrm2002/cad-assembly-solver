"""Generate folded-flange EAR poses from physical hole axes and contact faces.

The important distinction in this solver is between B-Rep cylindrical faces
and physical holes.  Counterbores, stepped bores, and the two sides of a thin
sheet may expose many cylindrical faces on one axis; those faces count as one
piece of evidence.  A pose is proposed only when multiple independent axes
agree with multiple opposing sheet-contact planes.

No stored Case-5 transform or absolute target coordinate is read.  Candidate
regions are derived from the component/carrier bounding envelopes and the
original STEP B-Rep measurements.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _rotation_axis_angle


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def unit(vector: Any) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    return result / max(float(np.linalg.norm(result)), 1e-12)


def transformed_bbox(payload: dict[str, Any], R: np.ndarray, t: np.ndarray):
    lo = np.asarray(payload["bbox_min"], dtype=float)
    hi = np.asarray(payload["bbox_max"], dtype=float)
    corners = np.asarray(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    points = corners @ R.T + t
    return points.min(axis=0), points.max(axis=0)


@dataclass
class PhysicalHole:
    axis: np.ndarray
    point: np.ndarray
    members: list[dict[str, Any]]

    @property
    def radius(self) -> float:
        # The smallest coaxial cylinder normally represents the through bore;
        # larger radii are counterbores/chamfers and do not add evidence.
        return min(float(member["radius"]) for member in self.members)


def physical_holes(
    rows: list[dict[str, Any]],
    *,
    maximum_radius: float = 3.0,
    line_tolerance_mm: float = 0.35,
) -> list[PhysicalHole]:
    """Collapse all coaxial cylindrical B-Rep faces into physical hole axes."""
    result: list[PhysicalHole] = []
    for row in rows:
        if float(row["radius"]) >= maximum_radius:
            continue
        axis = unit(row["axis"])
        first_nonzero = next((value for value in axis if abs(value) > 1e-8), 1.0)
        if first_nonzero < 0:
            axis = -axis
        point = np.asarray(row["centre"], dtype=float)
        match = None
        for hole in result:
            if abs(float(np.dot(axis, hole.axis))) < 0.995:
                continue
            line_distance = float(np.linalg.norm(np.cross(point - hole.point, hole.axis)))
            if line_distance < line_tolerance_mm:
                match = hole
                break
        if match is None:
            match = PhysicalHole(axis=axis, point=point, members=[])
            result.append(match)
        match.members.append(row)
    return result


def boundary_members(
    hole: PhysicalHole,
    *,
    normal_sign: int,
    boundary_coordinate: float,
    boundary_depth_mm: float,
) -> list[dict[str, Any]]:
    return [
        member
        for member in hole.members
        if float(member["host_normal"][0]) * normal_sign > 0.98
        and abs(float(member["host_centre"][0]) - boundary_coordinate) < boundary_depth_mm
    ]


def greedy_pairs(
    source: list[dict[str, Any]],
    target: list[dict[str, Any]],
    source_points: np.ndarray,
    target_points: np.ndarray,
    in_plane_translation: np.ndarray,
    contact_translation_x: float,
    *,
    distance_tolerance_mm: float = 1.0,
    radius_tolerance_mm: float = 0.70,
    contact_tolerance_mm: float = 1.20,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for source_index, source_hole in enumerate(source):
        for target_index, target_hole in enumerate(target):
            distance = float(
                np.linalg.norm(
                    source_points[source_index] + in_plane_translation - target_points[target_index]
                )
            )
            if distance >= distance_tolerance_mm:
                continue
            if abs(source_hole["hole"].radius - target_hole["hole"].radius) >= radius_tolerance_mm:
                continue
            contact_options: list[tuple[float, dict[str, Any], dict[str, Any], float]] = []
            for source_member in source_hole["members"]:
                for target_member in target_hole["members"]:
                    required_tx = float(
                        target_member["host_centre"][0]
                        - source_hole["rotation_x"] * source_member["host_centre"][0]
                    )
                    error = abs(required_tx - contact_translation_x)
                    if error < contact_tolerance_mm:
                        contact_options.append((error, source_member, target_member, required_tx))
            if not contact_options:
                continue
            _, source_member, target_member, required_tx = min(contact_options, key=lambda row: row[0])
            edges.append(
                {
                    "distance": distance,
                    "source_index": source_index,
                    "target_index": target_index,
                    "source_member": source_member,
                    "target_member": target_member,
                    "required_tx": required_tx,
                }
            )
    edges.sort(key=lambda row: row["distance"])
    selected: list[dict[str, Any]] = []
    used_source: set[int] = set()
    used_target: set[int] = set()
    for edge in edges:
        if edge["source_index"] in used_source or edge["target_index"] in used_target:
            continue
        used_source.add(edge["source_index"])
        used_target.add(edge["target_index"])
        selected.append(edge)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--carrier-side",
        choices=("left", "right", "both"),
        default="left",
        help="semantic carrier side to inspect; no pose coordinates are supplied",
    )
    args = parser.parse_args()
    output = args.folder

    ear_holes_raw = load(output / "ear_holes_raw.json")["holes"]
    chassis_holes_raw = load(output / "chassis_holes_raw.json")["holes"]
    ear_planes = load(output / "ear_planes_raw.json")["planes"]
    ear_bbox = load(output / "ear_bbox.json")
    chassis_bbox = load(output / "chassis_bbox.json")
    ear_lo = np.asarray(ear_bbox["bbox_min"], dtype=float)
    ear_hi = np.asarray(ear_bbox["bbox_max"], dtype=float)
    chassis_lo = np.asarray(chassis_bbox["bbox_min"], dtype=float)
    chassis_hi = np.asarray(chassis_bbox["bbox_max"], dtype=float)

    ear_physical = physical_holes(ear_holes_raw)
    chassis_physical = physical_holes(chassis_holes_raw)
    service_planes = [
        plane
        for plane in ear_planes
        if float(plane.get("area_proxy", 0.0)) > 500.0
        and abs(float(plane["normal"][2])) > 0.98
    ]

    carrier_sides = (-1, 1) if args.carrier_side == "both" else ((-1,) if args.carrier_side == "left" else (1,))
    candidates: list[dict[str, Any]] = []
    recall = {
        "raw_ear_cylindrical_face_count": len(ear_holes_raw),
        "ear_physical_hole_axis_count": len(ear_physical),
        "raw_chassis_cylindrical_face_count": len(chassis_holes_raw),
        "chassis_physical_hole_axis_count": len(chassis_physical),
        "source_interface_axis_count": 0,
        "target_interface_axis_count": 0,
        "seed_count": 0,
        "multi_physical_hole_candidate_count": 0,
        "three_hole_candidate_count": 0,
        "deduplicated_candidate_count": 0,
    }

    source_boundary_depth = max(5.0, 0.20 * float(ear_hi[0] - ear_lo[0]))
    target_boundary_depth = max(12.0, 0.03 * float(chassis_hi[0] - chassis_lo[0]))

    for carrier_side in carrier_sides:
        target_boundary = float(chassis_lo[0] if carrier_side < 0 else chassis_hi[0])
        for source_side in (-1, 1):
            source_boundary = float(ear_lo[0] if source_side < 0 else ear_hi[0])
            rotation_x = -carrier_side * source_side
            R = np.diag([rotation_x, 1.0, rotation_x])
            if not np.isclose(np.linalg.det(R), 1.0):
                continue

            source: list[dict[str, Any]] = []
            for hole in ear_physical:
                if abs(float(hole.axis[0])) < 0.98:
                    continue
                members = boundary_members(
                    hole,
                    normal_sign=source_side,
                    boundary_coordinate=source_boundary,
                    boundary_depth_mm=source_boundary_depth,
                )
                if members:
                    source.append(
                        {
                            "hole": hole,
                            "members": members,
                            "rotation_x": rotation_x,
                        }
                    )

            target: list[dict[str, Any]] = []
            for hole in chassis_physical:
                if abs(float(hole.axis[0])) < 0.98:
                    continue
                members = boundary_members(
                    hole,
                    normal_sign=carrier_side,
                    boundary_coordinate=target_boundary,
                    boundary_depth_mm=target_boundary_depth,
                )
                if members:
                    target.append({"hole": hole, "members": members})

            recall["source_interface_axis_count"] += len(source)
            recall["target_interface_axis_count"] += len(target)
            if len(source) < 2 or len(target) < 2:
                continue
            source_points = np.asarray(
                [[hole["hole"].point[1], rotation_x * hole["hole"].point[2]] for hole in source]
            )
            target_points = np.asarray([[hole["hole"].point[1], hole["hole"].point[2]] for hole in target])

            for source_index, source_hole in enumerate(source):
                for target_index, target_hole in enumerate(target):
                    if abs(source_hole["hole"].radius - target_hole["hole"].radius) >= 0.70:
                        continue
                    in_plane_seed = target_points[target_index] - source_points[source_index]
                    for source_member in source_hole["members"]:
                        for target_member in target_hole["members"]:
                            recall["seed_count"] += 1
                            tx_seed = float(
                                target_member["host_centre"][0]
                                - rotation_x * source_member["host_centre"][0]
                            )
                            pairs = greedy_pairs(
                                source,
                                target,
                                source_points,
                                target_points,
                                in_plane_seed,
                                tx_seed,
                            )
                            if len(pairs) < 2:
                                continue
                            # Refit the three translation coordinates from all
                            # independent correspondences, then validate again.
                            ty = float(
                                np.mean(
                                    [
                                        target_points[pair["target_index"]][0]
                                        - source_points[pair["source_index"]][0]
                                        for pair in pairs
                                    ]
                                )
                            )
                            tz = float(
                                np.mean(
                                    [
                                        target_points[pair["target_index"]][1]
                                        - source_points[pair["source_index"]][1]
                                        for pair in pairs
                                    ]
                                )
                            )
                            tx = float(np.mean([pair["required_tx"] for pair in pairs]))
                            fitted_in_plane = np.asarray([ty, tz])
                            pairs = greedy_pairs(
                                source,
                                target,
                                source_points,
                                target_points,
                                fitted_in_plane,
                                tx,
                            )
                            if len(pairs) < 2:
                                continue
                            recall["multi_physical_hole_candidate_count"] += 1
                            if len(pairs) >= 3:
                                recall["three_hole_candidate_count"] += 1

                            t = np.asarray([tx, ty, tz], dtype=float)
                            bbox_lo, bbox_hi = transformed_bbox(ear_bbox, R, t)
                            # The long EAR dimension must remain vertical and
                            # overlap the chassis height; small buttons/tabs may
                            # project a few millimetres outside the envelope.
                            vertical_overlap = max(
                                0.0,
                                min(float(bbox_hi[1]), float(chassis_hi[1]))
                                - max(float(bbox_lo[1]), float(chassis_lo[1])),
                            )
                            vertical_fraction = vertical_overlap / max(
                                float(bbox_hi[1] - bbox_lo[1]), 1e-9
                            )
                            if vertical_fraction < 0.88:
                                continue

                            outside_span = (
                                target_boundary - float(bbox_lo[0])
                                if carrier_side < 0
                                else float(bbox_hi[0]) - target_boundary
                            )
                            outside_fraction = max(
                                0.0,
                                min(
                                    1.0,
                                    outside_span / max(float(bbox_hi[0] - bbox_lo[0]), 1e-9),
                                ),
                            )
                            if outside_fraction < 0.45:
                                continue

                            service_candidates = [
                                plane
                                for plane in service_planes
                                if float(plane["normal"][2]) * rotation_x > 0.98
                            ]
                            service_plane = (
                                max(service_candidates, key=lambda plane: float(plane["area_proxy"]))
                                if service_candidates
                                else None
                            )
                            expected_end = float(chassis_hi[2] if rotation_x > 0 else chassis_lo[2])
                            service_z = (
                                float(rotation_x * service_plane["origin"][2] + tz)
                                if service_plane
                                else float(bbox_hi[2] if rotation_x > 0 else bbox_lo[2])
                            )
                            service_flush_error = abs(service_z - expected_end)
                            if service_flush_error > 35.0:
                                continue

                            hole_residuals = [
                                float(
                                    np.linalg.norm(
                                        source_points[pair["source_index"]]
                                        + fitted_in_plane
                                        - target_points[pair["target_index"]]
                                    )
                                )
                                for pair in pairs
                            ]
                            contact_offsets = [float(pair["required_tx"]) for pair in pairs]
                            mean_hole_residual = float(np.mean(hole_residuals))
                            contact_plane_std = float(np.std(contact_offsets))
                            score = (
                                0.30 * min(1.0, len(pairs) / 3.0)
                                + 0.22 * max(0.0, 1.0 - mean_hole_residual / 1.0)
                                + 0.18 * max(0.0, 1.0 - contact_plane_std / 1.2)
                                + 0.12 * outside_fraction
                                + 0.08 * vertical_fraction
                                + 0.10 * max(0.0, 1.0 - service_flush_error / 35.0)
                            )
                            candidates.append(
                                {
                                    "R": R.tolist(),
                                    "t_mm": t.round(6).tolist(),
                                    "axis_angle": _rotation_axis_angle(R),
                                    "carrier_side": "left" if carrier_side < 0 else "right",
                                    "source_flange_side": source_side,
                                    "matched_physical_holes": [
                                        {
                                            "ear_face": int(pair["source_member"]["face_index"]),
                                            "chassis_face": int(pair["target_member"]["face_index"]),
                                            "axis_distance_residual_mm": round(residual, 4),
                                            "required_contact_tx_mm": round(float(pair["required_tx"]), 4),
                                        }
                                        for pair, residual in zip(pairs, hole_residuals)
                                    ],
                                    "independent_physical_hole_count": len(pairs),
                                    "mean_hole_axis_residual_mm": round(mean_hole_residual, 4),
                                    "contact_plane_translation_std_mm": round(contact_plane_std, 4),
                                    "outside_fraction": round(outside_fraction, 4),
                                    "vertical_overlap_fraction": round(vertical_fraction, 4),
                                    "service_face": int(service_plane["face_index"]) if service_plane else None,
                                    "service_face_z_mm": round(service_z, 4),
                                    "carrier_opening_end_z_mm": round(expected_end, 4),
                                    "service_flush_error_mm": round(service_flush_error, 4),
                                    "bbox_mm": {
                                        "min": bbox_lo.round(4).tolist(),
                                        "max": bbox_hi.round(4).tolist(),
                                    },
                                    "score": round(float(score), 5),
                                }
                            )

    candidates.sort(
        key=lambda row: (
            -int(row["independent_physical_hole_count"]),
            -float(row["score"]),
            float(row["mean_hole_axis_residual_mm"]),
        )
    )
    unique_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        R = np.asarray(candidate["R"], dtype=float)
        t = np.asarray(candidate["t_mm"], dtype=float)
        if any(
            np.allclose(R, np.asarray(previous["R"], dtype=float), atol=1e-8)
            and np.linalg.norm(t - np.asarray(previous["t_mm"], dtype=float)) < 0.75
            for previous in unique_candidates
        ):
            continue
        unique_candidates.append(candidate)
    recall["deduplicated_candidate_count"] = len(unique_candidates)

    payload = {
        "method": "physical_hole_axis_clusters_plus_folded_contact_planes_plus_external_service_face",
        "candidate_count": len(unique_candidates),
        "recall_audit": recall,
        "candidates": unique_candidates[:20],
    }
    save(output / "ear_folded_flange_candidates.json", payload)

    if unique_candidates:
        repository = Path(__file__).resolve().parents[1]
        manifest_dir = output / "folded_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        def manifest_for(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
            return {
                "assembly_name": "case5 folded flange physical-hole review",
                "accepted": False,
                "pose_status": "geometry_candidate_requires_occt_and_human_review",
                "candidate_rank": rank,
                "candidate_evidence": {
                    "independent_physical_hole_count": candidate[
                        "independent_physical_hole_count"
                    ],
                    "mean_hole_axis_residual_mm": candidate[
                        "mean_hole_axis_residual_mm"
                    ],
                    "contact_plane_translation_std_mm": candidate[
                        "contact_plane_translation_std_mm"
                    ],
                    "service_flush_error_mm": candidate["service_flush_error_mm"],
                },
                "components": [
                    {
                        "id": "chassis_fixed",
                        "source": str(
                            (repository / "sw" / "5" / "01-ASSY-CHASSIS-MODULE-R6250H0.stp").resolve()
                        ),
                        "placement": {"translate": [0.0, 0.0, 0.0]},
                    },
                    {
                        "id": "EAR_folded_flange",
                        "source": str(
                            (repository / "sw" / "5" / "01-ASSY-CHASSIS-EAR-L-R620.stp").resolve()
                        ),
                        "placement": {
                            "rotate_sequence": [{"axis_angle": candidate["axis_angle"]}],
                            "translate": candidate["t_mm"],
                        },
                    },
                ],
            }

        for rank, candidate in enumerate(unique_candidates[:5], start=1):
            save(manifest_dir / f"physical_holes_rank_{rank:02d}.json", manifest_for(candidate, rank))
        save(output / "ear_folded_flange_manifest.json", manifest_for(unique_candidates[0], 1))

    print(
        json.dumps(
            {
                "candidate_count": len(unique_candidates),
                "recall_audit": recall,
                "top": unique_candidates[:3],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
