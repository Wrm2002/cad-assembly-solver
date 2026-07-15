"""Build an auditable *review* manifest from a folded-insertion pose audit.

The transform is read from the geometry proposal; it is never copied from an
image or a case-specific placement table.  This helper deliberately marks the
result as review-only because a guide/stop hypothesis must still be checked
against the original B-Rep and the physical fastening interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from proxy_insertion_pose import _rotation_axis_angle


def build_manifest(
    audit_path: Path,
    chassis: Path,
    ear: Path,
    output: Path,
    candidate_index: int,
    module: Path | None = None,
    module_audit_path: Path | None = None,
    render_mode: str = "all",
    module_candidate_index: int = 0,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    candidates = list(audit.get("candidates") or [])
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise ValueError(f"candidate index {candidate_index} is unavailable")
    candidate = candidates[candidate_index]
    if candidate.get("inside_fraction", 0.0) < 0.99:
        raise ValueError("refusing to render a non-contained insertion proposal")
    rotation = np.asarray(candidate["rotation"], dtype=float)
    axis_angle = _rotation_axis_angle(rotation)
    components = [
        {
            "id": "chassis",
            "label": chassis.stem,
            "source": str(chassis.resolve()),
            "placement": {"translate": [0.0, 0.0, 0.0]},
        },
        {
            "id": "ear",
            "label": ear.stem,
            "source": str(ear.resolve()),
            "placement": {
                "rotate_axis_angle": [float(v) for v in axis_angle],
                "translate": [float(v) for v in candidate["translation"]],
            },
        },
    ]
    module_evidence: dict[str, Any] | None = None
    if (module is None) != (module_audit_path is None):
        raise ValueError("module and module audit must be supplied together")
    if module is not None and module_audit_path is not None:
        module_audit = json.loads(module_audit_path.read_text(encoding="utf-8"))
        audit_status = module_audit.get("status", module_audit.get("decision"))
        if audit_status not in {"review", "review_only"}:
            raise ValueError("expected a review-only module insertion audit")
        local_candidates = module_audit.get("candidates") or module_audit.get("top_candidates")
        if local_candidates is not None:
            if module_candidate_index < 0 or module_candidate_index >= len(local_candidates):
                raise ValueError("requested module candidate is unavailable")
            local = local_candidates[module_candidate_index]
            module_axis_angle = local.get("axis_angle") or _rotation_axis_angle(
                np.asarray(local["rotation"], dtype=float)
            )
            module_translate = local["translation"]
            module_evidence = {
                "method": module_audit.get("method"),
                "candidate_index": module_candidate_index,
            }
            for key in (
                "guide_wall_faces", "guide_wall_gap_mm", "guide_clearance_mm",
                "guide_coverage", "support_face", "support_overlap",
                "match_count", "mean_residual_mm", "geometry_score",
                "source_host_normal", "carrier_host_normal",
            ):
                if key in local:
                    module_evidence[key] = local[key]
        else:
            module_axis_angle = module_audit["axis_angle"]
            module_translate = module_audit.get("translate", module_audit.get("translation"))
            if module_translate is None:
                raise ValueError("module audit contains no translation")
            module_evidence = {
                "method": "proxy_rail_stop",
                "inside_support_axis": module_audit.get("support_axis"),
                "support_plane_count": module_audit.get("support_plane_count"),
                "stop_plane_count": module_audit.get("stop_plane_count"),
            }
        components.append({
            "id": "psu",
            "label": module.stem,
            "source": str(module.resolve()),
            "placement": {
                "rotate_axis_angle": [float(v) for v in module_axis_angle],
                "translate": [float(v) for v in module_translate],
            },
        })
    if render_mode == "ear":
        components = [component for component in components if component["id"] != "psu"]
    elif render_mode == "psu":
        components = [component for component in components if component["id"] != "ear"]
    elif render_mode != "all":
        raise ValueError("render mode must be all, ear, or psu")
    manifest: dict[str, Any] = {
        "assembly_name": "case5 folded-ear insertion — geometry review only",
        "pose_status": "review",
        "accepted": False,
        "review_reason": (
            "Derived from folded face, opposing side wall and stop region; "
            "fastener/slot correspondence is not yet independently verified."
        ),
        "geometry_evidence": {
            "method": audit.get("method"),
            "candidate_index": candidate_index,
            "inside_fraction": candidate["inside_fraction"],
            "source_hole_feature_count": candidate["source_hole_feature_count"],
            "carrier_hole_feature_count": candidate["carrier_hole_feature_count"],
            "stop_coordinate": candidate["stop_coordinate"],
            "score": candidate["score"],
        },
        "module_geometry_evidence": module_evidence,
        "components": components,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("chassis", type=Path)
    parser.add_argument("ear", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--module", type=Path)
    parser.add_argument("--module-audit", type=Path)
    parser.add_argument("--module-candidate-index", type=int, default=0)
    parser.add_argument("--render-mode", choices=("all", "ear", "psu"), default="all")
    args = parser.parse_args()
    manifest = build_manifest(
        args.audit, args.chassis, args.ear, args.output, args.candidate_index,
        args.module, args.module_audit, args.render_mode, args.module_candidate_index,
    )
    print(json.dumps({
        "status": manifest["pose_status"],
        "candidate_index": manifest["geometry_evidence"]["candidate_index"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
