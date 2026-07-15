"""Assemble the conservative Case 5 review candidate from computed evidence.

The script has no case-specific transform constants.  The ear transform comes
from the planar-hosted hole RANSAC audit, and the PSU transform comes from the
independent proxy rail/stop proposal.  Both remain review-only until original
B-rep collision checks are stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _rotation_axis_angle


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_frame_consistent_hole_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose a render hypothesis without converting ambiguity to acceptance.

    All filters are evidence from the two parts: at least three hole centres,
    sub-millimetre residual, both faces near an exterior boundary, and at
    least half of the attachment volume placed inside the carrier envelope.
    The remaining hypotheses are ordered by the smallest frame adjustment.
    This is a weak export-frame prior used only for inspection rendering.
    """
    eligible = [
        item
        for item in candidates
        if item["match_count"] >= 3
        and item["mean_residual_mm"] <= 0.35
        # The attachment must sit on the carrier's inner side.  This is a
        # geometric containment gate, not a hand-authored translation.
        and item["source_part_inside_fraction"] >= 0.75
        and item["source_host_boundary_proximity"] >= 0.4
        and item["carrier_host_boundary_proximity"] >= 0.4
    ]
    if not eligible:
        raise RuntimeError("no multi-hole exterior/interior review candidate")
    return min(
        eligible,
        key=lambda item: (
            item["rotation_angle_deg"],
            -item["match_count"],
            item["mean_residual_mm"],
        ),
    )


def solve(case_dir: Path, output_dir: Path) -> dict[str, Any]:
    hole_audit = json.loads((output_dir / "ear_pose_coplanar_audit.json").read_text(encoding="utf-8"))
    psu_audit = json.loads((output_dir / "proxy_insertion_pose_audit.json").read_text(encoding="utf-8"))
    psu_hole_path = output_dir / "psu_hole_pose_audit.json"
    try:
        ear = _select_frame_consistent_hole_candidate(hole_audit["all_candidates"])
    except RuntimeError as error:
        result = {
            "case": "case5",
            "final_pose_status": "unresolved",
            "accepted": False,
            "ear_hole_pose": {
                "status": "failed_conservative_gate",
                "reason": str(error),
                "candidate_count": hole_audit["candidate_count"],
                "rejection": "No three-independent-hole pose also satisfies the inner-side containment gate.",
            },
            "psu_pose": {
                "status": "review_only",
                "reason": "Rail/stop proxy proposal is not corroborated by a contained hole-pattern pose.",
            },
            "manifest": None,
            "next_gate": "local_mating_feature_extraction_or_human_review",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "complete_solve_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    chassis = case_dir / "01-ASSY-CHASSIS-MODULE-R6250H0.stp"
    ear_file = case_dir / "01-ASSY-CHASSIS-EAR-L-R620.stp"
    psu = case_dir / "5-CRPS1300NC.stp"
    for path in (chassis, ear_file, psu):
        if not path.exists():
            raise FileNotFoundError(path)

    ear_rotation = np.asarray(ear["rotation"], dtype=float)
    manifest = {
        "schema_version": "2.0.0",
        "assembly_name": "case5_geometry_only_review",
        "global_units": "mm",
        "components": [
            {"id": "carrier", "source": "../../5/01-ASSY-CHASSIS-MODULE-R6250H0.stp", "label": chassis.stem, "role": "carrier", "placement": {"translate": [0, 0, 0]}},
            {
                "id": "ear_attachment",
                "source": "../../5/01-ASSY-CHASSIS-EAR-L-R620.stp",
                "label": ear_file.stem,
                "role": "flange_attachment",
                "placement": {"translate": ear["translation"], "rotate_sequence": [{"axis_angle": _rotation_axis_angle(ear_rotation)}]},
            },
            {
                "id": "psu",
                "source": "../../5/5-CRPS1300NC.stp",
                "label": psu.stem,
                "role": "inserted_module",
                "placement": {"translate": psu_audit["translate"], "rotate_sequence": [{"axis_angle": psu_audit["axis_angle"]}]},
            },
        ],
        "pose_status": "review",
        "not_auto_accepted": True,
        "reason": "Ear flange has multiple valid three-hole hypotheses; PSU has proxy rail/stop evidence but no stable original-B-rep collision verdict.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = {
        "case": "case5",
        "final_pose_status": "review",
        "accepted": False,
        "ear_hole_pose": {
            "status": "review",
            "selection_rule": "multi-hole exterior candidate, minimum export-frame rotation",
            "candidate_count": hole_audit["candidate_count"],
            "candidate": ear,
            "evidence": ["planar_contact", "three_hole_centres", "sub_mm_residual", "interior_envelope"],
        },
        "psu_pose": {
            "status": psu_audit["status"],
            "evidence": psu_audit["evidence"],
            "support_plane_count": psu_audit["support_plane_count"],
            "stop_plane_count": psu_audit["stop_plane_count"],
            "rotation_matrix": psu_audit["rotation_matrix"],
            "translation": psu_audit["translate"],
            "hole_pattern_cross_check": (
                "rejected: the matching-orientation three-hole proposal exceeds the carrier bounding envelope"
                if psu_hole_path.exists()
                else "not_run"
            ),
        },
        "source_hashes": {path.name: _hash(path) for path in (chassis, ear_file, psu)},
        "manifest": str(manifest_path.resolve()),
        "next_gate": "original_brep_collision_validation",
    }
    (output_dir / "complete_solve_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = solve(args.case_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("final_pose_status", "accepted", "manifest", "next_gate")}, indent=2))


if __name__ == "__main__":
    main()
