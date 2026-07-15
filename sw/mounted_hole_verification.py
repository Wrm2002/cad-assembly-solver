"""Verify fastener-hole correspondence around an independently proposed pose.

This is deliberately not a pose fitter.  It accepts a geometry-derived rigid
transform, then asks whether planar-hosted circular features provide a second,
independent mounting evidence channel near that transform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _key(hole: dict[str, Any]) -> tuple[float, ...]:
    n = np.asarray(hole["host_normal"], dtype=float)
    p = np.asarray(hole["host_centre"], dtype=float)
    return tuple(np.round(n, 3)) + (round(float(n @ p) * 2.0) / 2.0,)


def _groups(payload: dict[str, Any]) -> list[list[dict[str, Any]]]:
    result: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for hole in payload["holes"]:
        result.setdefault(_key(hole), []).append(hole)
    return [rows for rows in result.values() if len(rows) >= 3]


def _radii_ok(a: float, b: float) -> bool:
    return abs(a - b) <= 1.05


def _transform_from_audit(payload: dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray]:
    row = (payload.get("candidates") or [payload])[index]
    return np.asarray(row["rotation"], dtype=float), np.asarray(row["translation"], dtype=float)


def verify(source: dict[str, Any], carrier: dict[str, Any], transform: dict[str, Any], candidate_index: int) -> dict[str, Any]:
    R, t = _transform_from_audit(transform, candidate_index)
    out: list[dict[str, Any]] = []
    for left in _groups(source):
        ln = R @ np.asarray(left[0]["host_normal"], dtype=float)
        lp = R @ np.asarray(left[0]["host_centre"], dtype=float) + t
        for right in _groups(carrier):
            rn = np.asarray(right[0]["host_normal"], dtype=float)
            # Sheet-metal sides can be recorded with either orientation.
            normal_alignment = abs(float(ln @ rn))
            if normal_alignment < 0.985:
                continue
            plane_gap = abs(float(rn @ (lp - np.asarray(right[0]["host_centre"], dtype=float))))
            if plane_gap > 1.25:
                continue
            used: set[int] = set()
            matches = []
            for i, hole in enumerate(left):
                centre = R @ np.asarray(hole["centre"], dtype=float) + t
                options = []
                for j, target in enumerate(right):
                    if j in used or not _radii_ok(float(hole["radius"]), float(target["radius"])):
                        continue
                    residual = float(np.linalg.norm(centre - np.asarray(target["centre"], dtype=float)))
                    if residual <= 1.25:
                        options.append((residual, j))
                if options:
                    residual, j = min(options)
                    used.add(j)
                    matches.append({"source_index": i, "target_index": j, "residual_mm": round(residual, 6)})
            independent = 0.0
            if len(matches) >= 2:
                points = [R @ np.asarray(left[m["source_index"]]["centre"], dtype=float) + t for m in matches]
                independent = max(float(np.linalg.norm(a - b)) for a in points for b in points)
            out.append({
                "source_host_normal": ln.round(8).tolist(),
                "carrier_host_normal": rn.round(8).tolist(),
                "plane_gap_mm": round(plane_gap, 6),
                "match_count": len(matches),
                "max_independent_span_mm": round(independent, 6),
                "mean_residual_mm": round(float(np.mean([m["residual_mm"] for m in matches])) if matches else 0.0, 6),
                "matches": matches,
            })
    out.sort(key=lambda row: (-row["match_count"], row["mean_residual_mm"], row["plane_gap_mm"]))
    strongest = out[0] if out else None
    confirmed = bool(strongest and strongest["match_count"] >= 3 and strongest["max_independent_span_mm"] >= 8.0)
    return {
        "status": "confirmed" if confirmed else "not_confirmed",
        "method": "fixed_pose_planar_hosted_hole_correspondence",
        "transform_candidate_index": candidate_index,
        "best_evidence": strongest,
        "checked_host_pairs": len(out),
        "all_evidence": out[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_holes", type=Path)
    parser.add_argument("carrier_holes", type=Path)
    parser.add_argument("transform_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate-index", type=int, default=0)
    args = parser.parse_args()
    result = verify(
        json.loads(args.source_holes.read_text(encoding="utf-8")),
        json.loads(args.carrier_holes.read_text(encoding="utf-8")),
        json.loads(args.transform_audit.read_text(encoding="utf-8")),
        args.candidate_index,
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "checked_host_pairs", "best_evidence")}, indent=2))


if __name__ == "__main__":
    main()
