"""Refine one bounded global-Pose hypothesis with multi-body SDF contact cost."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "sw"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from global_pose_solver import MultiBodyContactRefiner, make_occt_exact_validator  # noqa: E402


def _mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mapping requires PART_ID=PATH")
    part, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not part or not path.is_file():
        raise argparse.ArgumentTypeError("mapping requires an existing path")
    return part, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", action="append", type=_mapping, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=384)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--translation-bound-mm", type=float)
    parser.add_argument("--step", action="append", type=_mapping)
    args = parser.parse_args()

    mesh_paths = dict(args.mesh)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    hypothesis = next(
        (row for row in source.get("hypotheses") or []
        if row.get("hypothesis_id") == args.hypothesis_id),
        None,
    )
    if hypothesis is None:
        parser.error("hypothesis ID was not found in input")
    direct_edges = [
        (str(row["source"]), str(row["target"]))
        for row in hypothesis.get("tree_candidates") or []
    ]
    refiner = MultiBodyContactRefiner(
        mesh_paths, direct_edges, sample_count=args.sample_count
    )
    initial = {
        part: np.asarray(matrix, dtype=float)
        for part, matrix in hypothesis["part_poses"].items()
    }
    result = refiner.refine(
        initial,
        anchor_id=source.get("anchor_id"),
        translation_bound_mm=args.translation_bound_mm,
        maxiter=args.maxiter,
    )
    if args.step:
        validator = make_occt_exact_validator(dict(args.step))
        result["exact_validation"] = validator({
            part: np.asarray(matrix, dtype=float)
            for part, matrix in result["part_poses"].items()
        })
    else:
        result["exact_validation"] = {"status": "not_checked"}
    result["input_audit"] = {
        "source_global_pose": str(args.input.resolve()),
        "hypothesis_id": args.hypothesis_id,
        "direct_edges": [list(edge) for edge in direct_edges],
        "mesh_paths": {part: str(path.resolve()) for part, path in mesh_paths.items()},
        "part_ids_are_bookkeeping_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "initial_objective": result["initial_objective"],
        "final_objective": result["final_objective"],
        "exact_status": result["exact_validation"]["status"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
