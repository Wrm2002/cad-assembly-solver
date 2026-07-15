"""Compose geometry-only pair-Pose frontiers into review-only group poses.

This is a bounded alternative to a costly joint solve.  It combines only
pairwise hypotheses that have already passed exact OCCT collision validation,
then delegates the assembled candidate to a second exact multi-part check.
Part IDs are bookkeeping keys; no filename/name/case token is used as a pose
feature or as a score.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inverse(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _pair_valid(path: Path, limit: int) -> list[dict[str, Any]]:
    values = []
    for row in (_read(path).get("pose_search") or {}).get("results") or []:
        exact = row.get("exact_collision") or {}
        if exact.get("status") != "success" or exact.get("collisions"):
            continue
        transform = np.asarray(row.get("transform"), dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            continue
        score = float((row.get("evaluation") or {}).get("contact", 0.0))
        values.append({"transform": transform, "contact": score, "source": row})
    values.sort(key=lambda row: -row["contact"])
    return values[: max(1, limit)]


def _rigid_rows(path: Path, limit: int, offset: int = 0) -> list[dict[str, Any]]:
    values = []
    for row in (_read(path).get("joint_hypotheses") or {}).get("rows") or []:
        if row.get("manifold_type") != "compound_prismatic_insertion_rigid":
            continue
        transform = np.asarray(row.get("initial_pose_b_in_a"), dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            continue
        values.append({"transform": transform, "source": row})
    # Preserve all symmetric branches; the group-level OCCT check, not an
    # input-frame identity prior, resolves branches that affect collision.
    start = max(0, int(offset))
    return values[start:start + max(1, limit)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--pair-one", type=Path, required=True,
                        help="pair result for anchor -> first moving part")
    parser.add_argument("--pair-one-id", required=True)
    parser.add_argument("--pair-two", type=Path, required=True,
                        help="pair result for anchor -> second moving part")
    parser.add_argument("--pair-two-id", required=True)
    parser.add_argument("--pair-two-parent", default="",
                        help="If set to pair-one ID, compose pair-two relative Pose through that part.")
    parser.add_argument("--companion-pair", type=Path, required=True,
                        help="rigid interface result for companion -> pair-two")
    parser.add_argument("--companion-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-limit", type=int, default=4)
    parser.add_argument("--pair-one-limit", type=int,
                        help="Optional bounded limit for the first pair frontier.")
    parser.add_argument("--pair-two-limit", type=int,
                        help="Optional bounded limit for the second pair frontier.")
    parser.add_argument("--rigid-limit", type=int, default=32)
    parser.add_argument("--rigid-offset", type=int, default=0,
                        help="Audit a bounded symmetry slice without input-frame ranking.")
    parser.add_argument("--pair-two-axis", default="",
                        help="Optional world-axis for an unresolved axial manifold, as x,y,z.")
    parser.add_argument("--pair-two-offsets-mm", default="0",
                        help="Comma-separated bounded translations along --pair-two-axis.")
    args = parser.parse_args()
    first = _pair_valid(args.pair_one, args.pair_one_limit or args.pair_limit)
    second = _pair_valid(args.pair_two, args.pair_two_limit or args.pair_limit)
    rigid = _rigid_rows(args.companion_pair, args.rigid_limit, args.rigid_offset)
    offsets = [float(value) for value in args.pair_two_offsets_mm.split(",") if value.strip()]
    axis = None
    if args.pair_two_axis.strip():
        axis = np.asarray([float(value) for value in args.pair_two_axis.split(",")], dtype=float)
        if axis.shape != (3,) or float(np.linalg.norm(axis)) <= 1e-12:
            parser.error("--pair-two-axis must be a nonzero x,y,z vector")
        axis /= float(np.linalg.norm(axis))
    rows = []
    for index, (one, two, link, offset) in enumerate(itertools.product(first, second, rigid, offsets)):
        # Link maps pair-two coordinates into companion coordinates:
        # T_pair_two = T_companion * T_link.
        pair_two_pose = np.array(two["transform"], dtype=float, copy=True)
        if args.pair_two_parent:
            if args.pair_two_parent != args.pair_one_id:
                parser.error("currently --pair-two-parent must equal --pair-one-id")
            pair_two_pose = one["transform"] @ pair_two_pose
        if axis is not None:
            pair_two_pose[:3, 3] += axis * float(offset)
        companion_pose = pair_two_pose @ _inverse(link["transform"])
        rows.append({
            "hypothesis_id": f"pairwise_compose:{index:04d}",
            "part_poses": {
                args.anchor: np.eye(4).tolist(),
                args.pair_one_id: one["transform"].tolist(),
                args.pair_two_id: pair_two_pose.tolist(),
                args.companion_id: companion_pose.tolist(),
            },
            "prior_sum": float(one["contact"] + two["contact"]),
            "source_evidence": {
                "pair_one": {"contact": one["contact"], "occt_pair_valid": True},
                "pair_two": {"contact": two["contact"], "occt_pair_valid": True},
                "companion": {
                    "geometry_only": True,
                    "multi_interface_prismatic": True,
                    "occt_pair_valid": "pending_group_validation",
                },
                "pair_two_manifold_offset_mm": float(offset),
            },
            "accepted": False,
            "review_required": True,
            "exact_validation": {"status": "not_checked"},
        })
    result = {
        "schema_version": "pairwise_exact_frontier_composition.v1",
        "status": "review_required",
        "accepted": False,
        "review_required": True,
        "part_ids": [args.anchor, args.pair_one_id, args.pair_two_id, args.companion_id],
        "anchor_id": args.anchor,
        "hypothesis_count": len(rows),
        "hypotheses": rows,
        "feature_policy": "pair OCCT geometry, B-Rep interface frames, and topology-backed prism/slot evidence only",
        "acceptance_boundary": "A composed pose requires exact multi-part OCCT validation and connected-contact review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hypothesis_count": len(rows), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    raise SystemExit(main())
