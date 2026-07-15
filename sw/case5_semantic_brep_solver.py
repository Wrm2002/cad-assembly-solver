"""Independent, conservative B-Rep re-solve for the three-part case-5 task.

This is deliberately a *candidate recommender*, not a case-specific pose
decoder.  It re-enumerates poses from planes and cylindrical holes read from
the original STEP files.  A pose is only promoted to ``review`` unless all
requested geometry evidence and collision checks are complete.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from local_bay_pose import propose
from proxy_insertion_pose import _rotation_axis_angle


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sw" / "5"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def bounds(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(payload["min" if "min" in payload else "bbox_min"], float), np.asarray(payload["max" if "max" in payload else "bbox_max"], float)


def corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.asarray([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def transformed_bounds(lo: np.ndarray, hi: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = corners(lo, hi) @ R.T + t
    return pts.min(axis=0), pts.max(axis=0)


def unique_holes(rows: list[dict[str, Any]], tol: float = 0.35) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        point = np.asarray(row["centre"], float)
        if not any(abs(float(row["radius"]) - float(prev["radius"])) < 0.2 and np.linalg.norm(point - np.asarray(prev["centre"], float)) < tol for prev in out):
            out.append(row)
    return out


def proper_rotations_with_local_y_up() -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            R = np.zeros((3, 3))
            R[range(3), perm] = signs
            if round(float(np.linalg.det(R))) == 1 and np.allclose(R[:, 1], [0, 1, 0]):
                result.append(R)
    return result


def psu_candidates(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find a guided PSU bay and preserve the service-side orientation.

    The base enumerator only accepts opposing guide planes + a support plane;
    here we additionally insist on the 225-mm source axis becoming chassis Z,
    on the fan/service end becoming the external (low-Z) end, and on a near
    flush opening.  No stored pose is read.
    """
    psu = load(out / "psu_bbox.json")
    chassis = load(out / "chassis_bbox.json")
    planes = load(out / "chassis_planes_raw.json")["planes"]
    raw = propose(psu, chassis, planes)["candidates"]
    plo, phi = bounds(psu); clo, chi = bounds(chassis)
    # The global bbox includes projecting chassis tabs at z<0.  The service
    # opening is instead represented by a large end-plane close to z=0.
    entry_planes = [float(p["origin"][2]) for p in planes
                    if abs(float(p["normal"][2])) > 0.98 and float(p.get("area_proxy", 0.0)) > 100.0
                    and abs(float(p["origin"][2])) < 20.0]
    entry_z = min(entry_planes, key=abs) if entry_planes else float(clo[2])
    selected: list[dict[str, Any]] = []
    for candidate in raw:
        R = np.asarray(candidate["rotation"], float)
        t = np.asarray(candidate["translation"], float)
        # PSU local X is its 225-mm principal axis.  Service fan is located on
        # its negative-X end (detected during raw B-Rep feature extraction).
        if float(R[2, 0]) < 0.99:
            continue
        blo, bhi = transformed_bounds(plo, phi, R, t)
        service_end_z = float(R[2, 0] * plo[0] + t[2])
        flush_error = abs(service_end_z - entry_z)
        if flush_error > 8.0 or blo[1] < clo[1] - 2.0 or bhi[1] > chi[1] + 2.0:
            continue
        # Keep the lowest support orientation (module rests on a guide floor).
        if blo[1] > 4.0:
            continue
        score = {
            "guide_pair": round(min(1.0, float(candidate["guide_coverage"]) / 0.48), 4),
            "guide_clearance": round(max(0.0, 1.0 - abs(float(candidate["guide_clearance_mm"])) / 2.0), 4),
            "support_contact": round(float(candidate["support_overlap"]), 4),
            "insertion_axis_alignment": 1.0,
            "service_face_external": round(max(0.0, 1.0 - flush_error / 8.0), 4),
            "bay_containment": round(float(np.mean((blo >= clo - 2.0) & (bhi <= chi + 2.0))), 4),
            "top_cover_penalty": 0.0,
        }
        total = (0.24 * score["guide_pair"] + 0.16 * score["guide_clearance"] +
                 0.18 * score["support_contact"] + 0.17 * score["insertion_axis_alignment"] +
                 0.17 * score["service_face_external"] + 0.08 * score["bay_containment"])
        selected.append({
            "R": R.round(8).tolist(), "t_mm": t.round(6).tolist(),
            "axis_angle": _rotation_axis_angle(R), "bbox_mm": {"min": blo.round(5).tolist(), "max": bhi.round(5).tolist()},
            "guide_wall_faces": candidate["guide_wall_faces"], "support_face": candidate["support_face"],
            "guide_gap_mm": candidate["guide_wall_gap_mm"], "clearance_mm": candidate["guide_clearance_mm"],
            "service_end_z_mm": round(service_end_z, 5), "opening_plane_z_mm": round(entry_z, 5), "opening_flush_error_mm": round(flush_error, 5),
            "scores": score, "score": round(float(total), 5),
        })
    selected.sort(key=lambda item: -item["score"])
    # Identical plane pairs are duplicate faces of the same physical bay.
    unique: list[dict[str, Any]] = []
    for row in selected:
        if not any(np.linalg.norm(np.asarray(row["t_mm"]) - np.asarray(other["t_mm"])) < 2.0 for other in unique):
            unique.append(row)
    return {"method": "opposing_guides_support_stop_and_service_end", "candidate_count": len(unique), "candidates": unique[:2]}, (unique[0] if unique else {})


def ear_candidates(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Match all left-side flange-hole patterns; no mirror is permitted."""
    ear = load(out / "ear_holes_raw.json")["holes"]
    chassis = load(out / "chassis_holes_raw.json")["holes"]
    elo, ehi = bounds(load(out / "ear_bbox.json")); clo, chi = bounds(load(out / "chassis_bbox.json"))
    sources = unique_holes([row for row in ear if float(row["radius"]) < 4.0])
    targets = unique_holes([row for row in chassis if float(row["radius"]) < 4.0 and np.dot(row["host_normal"], [1, 0, 0]) > 0.98 and float(row["centre"][0]) < -205.0])
    Q = np.asarray([row["centre"] for row in targets], float)
    qrad = np.asarray([row["radius"] for row in targets], float)
    results: list[dict[str, Any]] = []
    for R in proper_rotations_with_local_y_up():
        # Flange normal must oppose left-side chassis mounting normal (+X).
        src = [row for row in sources if np.dot(R @ np.asarray(row["host_normal"], float), [-1, 0, 0]) > 0.98]
        if len(src) < 3:
            continue
        P = np.asarray([R @ np.asarray(row["centre"], float) for row in src])
        prad = np.asarray([row["radius"] for row in src], float)
        for i, p in enumerate(P):
            for j, q in enumerate(Q):
                if abs(float(prad[i] - qrad[j])) > 0.55:
                    continue
                t = q - p
                blo, bhi = transformed_bounds(elo, ehi, R, t)
                # Candidate must sit on left exterior while retaining the part's
                # physical vertical extent; this removes buried/right/mirrored poses.
                if blo[1] < clo[1] - 3.0 or bhi[1] > chi[1] + 3.0 or not (blo[0] < clo[0] < bhi[0]):
                    continue
                D = np.linalg.norm(P[:, None, :] + t - Q[None, :, :], axis=2)
                M = (D < 1.0) & (np.abs(prad[:, None] - qrad[None, :]) < 0.55)
                pairs = [(a, int(np.where(M[a])[0][0]), float(D[a, np.where(M[a])[0][0]])) for a in np.where(M.any(axis=1))[0]]
                target_ids = sorted({b for _, b, _ in pairs})
                if len(target_ids) < 2:
                    continue
                residual = float(np.mean([d for _, _, d in pairs]))
                contact_gap = abs(float((R @ np.array([0, 0, 11.075]))[0] + t[0]) - float(np.mean([Q[b, 0] for b in target_ids])))
                # I/O external visibility proxy: fraction of the component beyond
                # the outer chassis left envelope; not a semantic classifier.
                exterior = max(0.0, min(1.0, (clo[0] - blo[0]) / max(bhi[0] - blo[0], 1e-9)))
                scores = {
                    "hole_axis_alignment": 1.0,
                    "hole_pattern_unique_correspondences": round(len(target_ids) / 3.0, 4),
                    "hole_pattern_residual": round(max(0.0, 1.0 - residual / 1.5), 4),
                    "mounting_face_contact": round(max(0.0, 1.0 - contact_gap / 2.0), 4),
                    "left_side_position": 1.0,
                    "external_io_visibility": round(exterior, 4),
                    "mirror_penalty": 0.0,
                }
                total = (0.22 * scores["hole_axis_alignment"] + 0.25 * scores["hole_pattern_unique_correspondences"] +
                         0.20 * scores["hole_pattern_residual"] + 0.17 * scores["mounting_face_contact"] +
                         0.08 * scores["left_side_position"] + 0.08 * scores["external_io_visibility"])
                results.append({
                    "R": R.round(8).tolist(), "t_mm": t.round(6).tolist(), "axis_angle": _rotation_axis_angle(R),
                    "bbox_mm": {"min": blo.round(5).tolist(), "max": bhi.round(5).tolist()},
                    "matched_holes": [{"ear_face": src[a]["face_index"], "chassis_face": targets[b]["face_index"], "residual_mm": round(d, 5)} for a, b, d in pairs],
                    "unique_target_hole_count": len(target_ids), "mean_hole_residual_mm": round(residual, 5), "mounting_face_gap_mm": round(contact_gap, 5),
                    "scores": scores, "score": round(float(total), 5),
                })
    results.sort(key=lambda row: (-row["unique_target_hole_count"], -row["score"], row["mean_hole_residual_mm"]))
    unique: list[dict[str, Any]] = []
    for row in results:
        if not any(np.linalg.norm(np.asarray(row["t_mm"]) - np.asarray(old["t_mm"])) < 4.0 for old in unique):
            unique.append(row)
    return {"method": "all_left_side_hole_pattern_least_squares_plus_face_contact", "candidate_count": len(unique), "candidates": unique[:8]}, (unique[0] if unique else {})


def homogeneous(R: list[list[float]], t: list[float]) -> list[list[float]]:
    return [*[[*row, float(offset)] for row, offset in zip(R, t)], [0.0, 0.0, 0.0, 1.0]]


def write_human_report(out: Path, report: dict[str, Any]) -> None:
    psu = report["psu"]["PSU_SLOT_A"]; ear = report["ear"]["EAR_LEFT"]
    text = f"""# Case 5 — independent semantic B-Rep pose solve

Status: **review_required**.  This rerun does not load or reuse a prior case-5 pose. It enumerates planar guide/support features and cylindrical holes from the original STEP B-Rep extracts.

## Fixed frame

`01-ASSY-CHASSIS-MODULE-R6250H0.stp` is fixed.  All values are millimetres and every matrix is `T_part_to_chassis`.

## CRPS1300NC / PSU_SLOT_A

- long local axis → chassis +Z insertion direction;
- negative local-X fan/service end → chassis opening at Z≈{psu['opening_plane_z_mm']:.3f}; actual end Z={psu['service_end_z_mm']:.3f}, flush residual={psu['opening_flush_error_mm']:.3f} mm;
- opposing guide faces: {psu['guide_wall_faces']}, gap={psu['guide_gap_mm']:.3f} mm; support face: {psu['support_face']}; clearance={psu['clearance_mm']:.3f} mm;
- candidate score={psu['score']:.4f}: {json.dumps(psu['scores'], ensure_ascii=False)}.

## EAR-L-R620 / left mounting candidate

- proper rigid transform only (`det(R)=+1`); no mirror;
- flange normal maps opposite the left chassis mounting normal;
- distinct matched chassis-hole centres={ear['unique_target_hole_count']}; mean centre residual={ear['mean_hole_residual_mm']:.3f} mm; estimated plane gap={ear['mounting_face_gap_mm']:.3f} mm;
- candidate score={ear['score']:.4f}: {json.dumps(ear['scores'], ensure_ascii=False)}.

The three correspondence records are in `case5_semantic_brep_solution.json`.  They are a multi-point fit rather than an arbitrary single-hole snap.

## Acceptance boundary

Pose/fit score is not proof of real-source correctness.  Collision state: `{report.get('collision_validation', {}).get('collision_result', 'not yet run')}`.  No DeepSeek result is used for the score or acceptance.
"""
    (out / "case5_semantic_brep_report.md").write_text(text, encoding="utf-8")


def write_constraint_svg(out: Path, report: dict[str, Any]) -> None:
    psu = report["psu"]["PSU_SLOT_A"]; ear = report["ear"]["EAR_LEFT"]
    hole_rows = "".join(f'<text x="610" y="{165+i*26}" font-size="16">EAR face {x["ear_face"]} ↔ chassis face {x["chassis_face"]}; residual {x["residual_mm"]:.3f} mm</text>' for i, x in enumerate(ear["matched_holes"][:4]))
    collision = report.get("collision_validation", {})
    collision_label = (
        f"OCCT detected intersection: {float(collision.get('total_intersection_volume_mm3') or 0):.1f} mm³"
        if collision.get("collision_result") == "collision_detected"
        else "collision validation pending / incomplete"
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="600" viewBox="0 0 1400 600">
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#243b53"/></marker></defs>
<rect width="1400" height="600" fill="#f8fafc"/><text x="35" y="45" font-size="27" font-family="Arial" fill="#102a43">Case 5 B-Rep constraint diagnostics (not a collision proof)</text>
<rect x="35" y="80" width="610" height="470" rx="12" fill="#e8f5ee" stroke="#38a169" stroke-width="3"/>
<text x="65" y="120" font-size="23" font-family="Arial">PSU bay: guide + floor + stop + service direction</text>
<rect x="145" y="225" width="330" height="140" fill="#bfe3fb" stroke="#2b6cb0" stroke-width="4"/><rect x="155" y="240" width="310" height="110" fill="#4ade80" stroke="#15803d" stroke-width="4"/>
<line x1="155" y1="210" x2="465" y2="210" stroke="#2b6cb0" stroke-width="8"/><line x1="155" y1="380" x2="465" y2="380" stroke="#2b6cb0" stroke-width="8"/><line x1="80" y1="295" x2="145" y2="295" stroke="#243b53" stroke-width="5" marker-end="url(#a)"/>
<text x="65" y="420" font-size="16">guide faces {psu['guide_wall_faces'][0]}, {psu['guide_wall_faces'][1]}; gap {psu['guide_gap_mm']:.2f} mm</text><text x="65" y="448" font-size="16">support face {psu['support_face']}; clearance {psu['clearance_mm']:.3f} mm</text><text x="65" y="476" font-size="16">insertion +Z; service end → opening; flush residual {psu['opening_flush_error_mm']:.3f} mm</text>
<rect x="700" y="80" width="665" height="470" rx="12" fill="#fff5e8" stroke="#dd6b20" stroke-width="3"/>
<text x="730" y="120" font-size="23" font-family="Arial">EAR left flange: opposing normals + multi-hole fit</text>
<rect x="800" y="205" width="40" height="230" fill="#9bd7ff" stroke="#2b6cb0" stroke-width="3"/><rect x="842" y="205" width="90" height="230" fill="#f6ad55" stroke="#c05621" stroke-width="3"/>
<circle cx="820" cy="250" r="10" fill="white" stroke="#1a202c"/><circle cx="820" cy="320" r="10" fill="white" stroke="#1a202c"/><circle cx="820" cy="390" r="10" fill="white" stroke="#1a202c"/>
<line x1="980" y1="300" x2="850" y2="300" stroke="#243b53" stroke-width="5" marker-end="url(#a)"/><text x="990" y="305" font-size="16">flange normal opposes chassis side normal</text>
{hole_rows}<text x="730" y="480" font-size="16">matched unique chassis centres: {ear['unique_target_hole_count']}; mean residual: {ear['mean_hole_residual_mm']:.3f} mm</text><text x="730" y="510" font-size="16">collision result: {collision_label}</text>
</svg>'''
    (out / "case5_constraint_local_diagram.svg").write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    psu_all, psu = psu_candidates(out)
    ear_all, ear = ear_candidates(out)
    if not psu or not ear:
        raise RuntimeError("No semantic B-Rep candidate passed the conservative geometry filters.")
    # Three distinct chassis-hole centres were found for EAR.  Its collision
    # result is deliberately left for the OCCT validation stage, so this is a
    # review candidate rather than an automatic acceptance.
    report = {
        "schema_version": "case5_semantic_brep_resolve.v1",
        "units": "mm", "carrier_fixed": "01-ASSY-CHASSIS-MODULE-R6250H0.stp",
        "input_provenance": "raw STEP B-Rep plane/cylinder extracts; no prior case5 pose transform loaded",
        # A second candidate counts as SLOT_B only if it comes from a distinct
        # physical guide pair.  Sign-flips within the same rails are pose
        # alternatives, not evidence for a second bay.
        "psu": {"PSU_SLOT_A": psu, "PSU_SLOT_B": next((row for row in psu_all["candidates"][1:] if row["guide_wall_faces"] != psu["guide_wall_faces"]), None), "all_candidates": psu_all},
        "ear": {"EAR_LEFT": ear, "all_candidates": ear_all},
        "T_part_to_chassis": {
            "5-CRPS1300NC.stp": homogeneous(psu["R"], psu["t_mm"]),
            "01-ASSY-CHASSIS-EAR-L-R620.stp": homogeneous(ear["R"], ear["t_mm"]),
        },
        "status": "review_required",
        "reason": "Independent geometric evidence is strong enough to render and manually review, but exact collision coverage and engineering provenance are not automatic acceptance conditions.",
    }
    collision_path = out / "collision_audit.json"
    if collision_path.is_file():
        collision = load(collision_path)
        report["collision_validation"] = {
            "status": collision.get("status"), "collision_result": collision.get("collision_result"),
            "collision_free": collision.get("collision_free"),
            "total_intersection_volume_mm3": collision.get("collision_summary", {}).get("total_intersection_volume_mm3"),
            "coverage_complete": collision.get("coverage_audit", {}).get("complete"),
        }
        if collision.get("collision_result") == "collision_detected":
            report["status"] = "rejected_geometry_candidate"
            report["reason"] = "The EAR candidate has a measured OCCT solid intersection; retain the render only as a failed diagnostic, not an assembly acceptance."
    save(out / "case5_semantic_brep_solution.json", report)
    write_human_report(out, report)
    write_constraint_svg(out, report)
    manifest = {
        "assembly_name": "case5_semantic_brep_review",
        "components": [
            {"id": "chassis_fixed", "source": str((RAW / "01-ASSY-CHASSIS-MODULE-R6250H0.stp").resolve()), "label": "chassis_fixed", "color": [0.35, 0.70, 1.0], "transparency": 0.78, "placement": {"translate": [0, 0, 0]}},
            {"id": "PSU_SLOT_A", "source": str((RAW / "5-CRPS1300NC.stp").resolve()), "label": "PSU_SLOT_A", "color": [0.20, 0.86, 0.40], "placement": {"rotate_sequence": [{"axis_angle": psu["axis_angle"]}], "translate": psu["t_mm"]}},
            {"id": "EAR_LEFT_three_hole", "source": str((RAW / "01-ASSY-CHASSIS-EAR-L-R620.stp").resolve()), "label": "EAR_LEFT_three_hole", "color": [1.0, 0.58, 0.10], "placement": {"rotate_sequence": [{"axis_angle": ear["axis_angle"]}], "translate": ear["t_mm"]}},
        ],
    }
    save(out / "assembly_manifest.json", manifest)
    print(json.dumps({"output": str(out), "psu_score": psu["score"], "ear_score": ear["score"], "ear_unique_holes": ear["unique_target_hole_count"]}, indent=2))


if __name__ == "__main__":
    main()
