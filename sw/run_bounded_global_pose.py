"""CLI for bounded, auditable multi-part pose hypothesis recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "sw"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from global_pose_solver import (  # noqa: E402
    build_pools_from_joinable_reports,
    make_occt_exact_validator,
    solve_bounded_global_pose,
)


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    part_id, path = value.split("=", 1)
    if not part_id or not path:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    return part_id, Path(path)


def _pair(value: str) -> dict[str, str]:
    fields = value.split(",", 2)
    if len(fields) != 3 or not all(fields):
        raise argparse.ArgumentTypeError(
            "--pair requires SOURCE_ID,TARGET_ID,JOINABLE_RESULT_JSON"
        )
    return {"source": fields[0], "target": fields[1], "result_path": fields[2]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", action="append", required=True, type=_part)
    parser.add_argument("--pair", action="append", type=_pair)
    parser.add_argument(
        "--pair-frontier-manifest", type=Path,
        help="Manifest emitted by run_joinable_pair_frontier.py.",
    )
    parser.add_argument(
        "--edge-manifest", type=Path,
        help=(
            "Optional geometry candidate-edge manifest (pair_candidates with "
            "parts). It gates which JoinABLe pair pools may become global "
            "constraints; it is not an assembly truth file."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pair-candidates", type=int, default=3)
    parser.add_argument("--max-topologies", type=int, default=8)
    parser.add_argument("--max-hypotheses", type=int, default=96)
    parser.add_argument("--exact-top-n", type=int, default=0)
    args = parser.parse_args()

    part_sources = dict(args.part)
    pair_records = list(args.pair or [])
    if args.pair_frontier_manifest:
        manifest = json.loads(
            args.pair_frontier_manifest.read_text(encoding="utf-8")
        )
        for row in manifest.get("records") or []:
            if row.get("status") == "success":
                pair_records.append({
                    "source": row["source"],
                    "target": row["target"],
                    "result_path": row["result_path"],
                })
    if not pair_records:
        parser.error("provide at least one --pair or --pair-frontier-manifest")
    allowed_pairs = None
    if args.edge_manifest:
        edge_data = json.loads(args.edge_manifest.read_text(encoding="utf-8"))
        source_name_to_id = {
            path.name: part for part, path in part_sources.items()
        }

        def edge_part_to_id(value: object) -> str:
            text = str(value)
            return text if text in part_sources else source_name_to_id.get(text, text)

        allowed_pairs = {
            tuple(sorted(edge_part_to_id(value) for value in (row.get("parts") or [])))
            for row in edge_data.get("pair_candidates") or []
            if len(row.get("parts") or []) == 2
        }
        pair_records = [
            row for row in pair_records
            if tuple(sorted((str(row["source"]), str(row["target"])))) in allowed_pairs
        ]
        if not pair_records:
            parser.error("edge manifest removed every provided pair record")
    pools, pair_audit = build_pools_from_joinable_reports(
        pair_records,
        maximum_candidates_per_pair=args.max_pair_candidates,
    )
    validator = (
        make_occt_exact_validator(part_sources)
        if args.exact_top_n > 0 else None
    )
    result = solve_bounded_global_pose(
        list(part_sources),
        pools,
        max_candidates_per_pair=args.max_pair_candidates,
        max_topologies=args.max_topologies,
        max_hypotheses=args.max_hypotheses,
        exact_validator=validator,
        validate_top_n=args.exact_top_n,
    )
    result["input_audit"] = {
        "part_sources": {part: str(path.resolve()) for part, path in part_sources.items()},
        "pair_pools": pair_audit,
        "edge_gate": {
            "enabled": allowed_pairs is not None,
            "allowed_pairs": sorted(map(list, allowed_pairs or set())),
            "edge_manifest": (
                str(args.edge_manifest.resolve()) if args.edge_manifest else None
            ),
        },
        "part_ids_are_bookkeeping_only": True,
        "semantic_acceptance_not_performed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "hypothesis_count": result["hypothesis_count"],
        "accepted": result["accepted"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
