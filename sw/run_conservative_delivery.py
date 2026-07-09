"""One-command reproducible conservative D3.5-D7 delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_human_semantic_review_pack import run as build_human_pack
from candidate_recall_audit import run_audit
from conservative_pipeline import run as run_conservative


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(here / "mixed_pools_v1"))
    parser.add_argument(
        "--results", default=str(here / "data" / "results")
    )
    parser.add_argument(
        "--human-pack",
        help="New, empty folder for blinded STL/PNG review artifacts.",
    )
    parser.add_argument(
        "--deepseek-mode",
        choices=("live", "cache_only", "off"),
        default="cache_only",
    )
    args = parser.parse_args()
    pipeline_config = here / "configs" / "pool_pipeline.json"
    conservative_config = (
        here / "configs" / "conservative_pipeline.json"
    )
    audit = run_audit(args.root, args.results)
    conservative = run_conservative(
        args.root,
        args.results,
        pipeline_config_path=pipeline_config,
        conservative_config_path=conservative_config,
    )
    human = None
    if args.human_pack:
        human = build_human_pack(
            args.root,
            args.results,
            args.human_pack,
            pipeline_config,
            deepseek_mode=args.deepseek_mode,
        )
    print(
        json.dumps(
            {
                "candidate_recall": audit,
                "conservative_metrics": conservative,
                "human_semantic_pack": human,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
