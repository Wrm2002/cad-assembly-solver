"""Isolated worker for one CAD group; protects the controller from OCCT faults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contracts import GroupProposal
from geometry_pipeline import solve_and_validate_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir")
    parser.add_argument("group_id")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pool = Path(args.pool_dir).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    proposals = json.loads(
        (pool / "grouping" / "group_proposals.json").read_text(encoding="utf-8")
    )
    proposal = next(
        GroupProposal.model_validate(item)
        for item in proposals
        if item["group_id"] == args.group_id
    )
    result = solve_and_validate_group(pool, proposal, config)
    print(
        json.dumps(
            {
                "group_id": proposal.group_id,
                "accepted": result["metrics"]["accepted"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
