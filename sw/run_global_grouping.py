"""Run deterministic global grouping for every mixed pool and aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from global_grouping import run


METRIC_KEYS = (
    "exact_group_precision",
    "exact_group_recall",
    "exact_group_f1",
    "copart_pair_precision",
    "copart_pair_recall",
    "copart_pair_f1",
)


def run_all(root: Path, config: Path) -> dict:
    results = []
    for pool in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (pool / "pool_gt.json").is_file():
            continue
        result = run(pool, config)
        gt = json.loads((pool / "pool_gt.json").read_text(encoding="utf-8"))
        proposals = json.loads(
            (pool / "grouping" / "group_proposals.json").read_text(
                encoding="utf-8"
            )
        )
        proposed_sets = {frozenset(item["parts"]) for item in proposals}
        truth_sets = {
            frozenset(item["parts"]) for item in gt.get("true_groups", [])
        }
        covered = len(proposed_sets & truth_sets)
        results.append(
            {
                "pool_id": result["pool_id"],
                "proposal_count": len(proposals),
                "truth_groups_covered_by_proposals": covered,
                "proposal_truth_group_recall": (
                    covered / len(truth_sets) if truth_sets else 0.0
                ),
                **result["metrics"],
            }
        )

    aggregate = {
        "schema_version": "1.0.0",
        "pool_count": len(results),
        "macro_metrics": {
            key: sum(item[key] for item in results) / len(results)
            if results
            else 0.0
            for key in METRIC_KEYS
        },
        "proposal_truth_group_recall": (
            sum(item["truth_groups_covered_by_proposals"] for item in results)
            / sum(item["true_groups"] for item in results)
            if results
            else 0.0
        ),
        "pools": results,
    }
    output = root / "grouping_benchmark.json"
    output.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).parent / "mixed_pools_v1"),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    report = run_all(Path(args.root).resolve(), Path(args.config).resolve())
    print(json.dumps(report["macro_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
