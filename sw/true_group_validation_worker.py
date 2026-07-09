"""Isolated GT-group validation worker used only for false-negative auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import GroupProposal
from geometry_pipeline import solve_and_validate_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir")
    parser.add_argument("truth_group_id")
    parser.add_argument("subject_id")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pool = Path(args.pool_dir).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    gt = json.loads((pool / "pool_gt.json").read_text(encoding="utf-8"))
    group = next(
        item for item in gt["true_groups"]
        if item["group_id"] == args.truth_group_id
    )
    proposal = GroupProposal(
        group_id=args.subject_id,
        parts=group["parts"],
        candidate_edges=[],
        geometry_score=1.0,
        connected=True,
        status="diagnostic_ground_truth_group",
        reasons=[
            "evaluation-only membership; prohibited from selection pipeline"
        ],
    )
    result = solve_and_validate_group(
        pool, proposal, config, output_namespace="oracle_validation"
    )
    print(json.dumps({"accepted": result["metrics"]["accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
