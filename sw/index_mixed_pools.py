"""Index every frozen mixed pool and aggregate recall/reduction diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pool_index import index_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pools_root")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    root = Path(args.pools_root).resolve()
    rows = []
    for pool in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (pool / "pool_gt.json").is_file():
            continue
        outputs = index_pool(pool / "parts", pool / "index", args.config)
        quality = outputs.get("index_quality.json", {})
        screening = outputs["screening_audit.json"]
        row = {
            "pool_id": pool.name,
            "num_parts": len(outputs["part_features.json"]),
            "total_pairs": screening["total_pairs"],
            "accepted_pairs": screening["accepted_pairs"],
            "pair_reduction_rate": (
                screening["rejected_pairs"] / screening["total_pairs"]
                if screening["total_pairs"] else 0.0
            ),
            "prescreen_pair_recall": quality.get("prescreen_pair_recall"),
            "generated_typed_edge_recall": quality.get("generated_typed_edge_recall"),
            "pruned_typed_edge_recall": quality.get("pruned_typed_edge_recall"),
            "generated_candidates": len(outputs["geometry_candidates.json"]),
            "kept_candidates": len(outputs["pruned_candidates.json"]),
        }
        rows.append(row)
        print(
            f"{pool.name}: pairs={row['accepted_pairs']}/{row['total_pairs']} "
            f"prescreen_recall={row['prescreen_pair_recall']}",
            flush=True,
        )
    weighted_truth = 0
    prescreen_hits = generated_hits = pruned_hits = 0.0
    for pool in sorted(path for path in root.iterdir() if path.is_dir()):
        quality_path = pool / "index" / "index_quality.json"
        if not quality_path.is_file():
            continue
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        count = quality["ground_truth_edges"]
        weighted_truth += count
        prescreen_hits += quality["prescreen_pair_recall"] * count
        generated_hits += quality["generated_typed_edge_recall"] * count
        pruned_hits += quality["pruned_typed_edge_recall"] * count
    report = {
        "schema_version": "1.0.0",
        "num_pools": len(rows),
        "pools": rows,
        "aggregate": {
            "ground_truth_edges": weighted_truth,
            "prescreen_pair_recall": (
                prescreen_hits / weighted_truth if weighted_truth else None
            ),
            "generated_typed_edge_recall": (
                generated_hits / weighted_truth if weighted_truth else None
            ),
            "pruned_typed_edge_recall": (
                pruned_hits / weighted_truth if weighted_truth else None
            ),
            "mean_pair_reduction_rate": (
                sum(row["pair_reduction_rate"] for row in rows) / len(rows)
                if rows else None
            ),
        },
    }
    output = root / "index_benchmark.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
