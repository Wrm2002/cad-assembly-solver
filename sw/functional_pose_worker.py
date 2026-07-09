"""Isolated pose/collision worker for one functional group proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import GroupProposal
from geometry_pipeline import solve_and_validate_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir", type=Path)
    parser.add_argument("proposal_file", type=Path)
    parser.add_argument("group_id")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    proposals = json.loads(args.proposal_file.read_text(encoding="utf-8"))
    proposal = next(
        GroupProposal.model_validate(row)
        for row in proposals
        if row["group_id"] == args.group_id
    )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = solve_and_validate_group(
        args.pool_dir.resolve(),
        proposal,
        config,
        output_namespace=str(args.output_root.resolve()),
    )
    print(
        json.dumps(
            {
                "group_id": args.group_id,
                "physical_pose_valid": result["metrics"][
                    "physical_pose_valid"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

