"""Isolated physical-pose validation for one known real-world group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import GroupProposal
from geometry_pipeline import solve_and_validate_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool")
    parser.add_argument("group_id")
    parser.add_argument("parts_json")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pool = Path(args.pool).resolve()
    config = json.loads(
        Path(args.config).resolve().read_text(encoding="utf-8")
    )
    parts = json.loads(args.parts_json)
    proposal = GroupProposal(
        group_id=args.group_id,
        parts=parts,
        candidate_edges=[],
        geometry_score=0.0,
        connected=False,
        status="known_group_evaluation_only",
        reasons=["folder membership supplied as evaluation ground truth"],
    )
    result = solve_and_validate_group(
        pool,
        proposal,
        config,
        output_namespace="known_group_validation",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
