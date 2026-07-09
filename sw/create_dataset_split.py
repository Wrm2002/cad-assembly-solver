"""Create deterministic, stratified train/calibration/test case splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def create_split(dataset_root: str | Path, seed: int = 20260702) -> dict:
    root = Path(dataset_root).resolve()
    strata = defaultdict(list)
    metadata = {}
    for case in sorted(path for path in root.iterdir() if path.is_dir()):
        gt_path = case / "gt.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        key = (int(gt["group_size"]), str(gt.get("template", "unknown")))
        strata[key].append(case.name)
        metadata[case.name] = {
            "group_size": key[0],
            "template": key[1],
        }
    assignments = {}
    stratum_summary = {}
    for key, cases in sorted(strata.items()):
        ranked = sorted(
            cases,
            key=lambda case_id: hashlib.sha256(
                f"{seed}|{case_id}".encode("utf-8")
            ).hexdigest(),
        )
        total = len(ranked)
        train_count = int(total * 0.6)
        calibration_count = int(total * 0.2)
        if total >= 5:
            train_count = max(1, train_count)
            calibration_count = max(1, calibration_count)
        boundaries = (
            train_count,
            train_count + calibration_count,
        )
        for index, case_id in enumerate(ranked):
            split = (
                "train"
                if index < boundaries[0]
                else (
                    "calibration"
                    if index < boundaries[1]
                    else "test"
                )
            )
            assignments[case_id] = {
                **metadata[case_id],
                "split": split,
            }
        stratum_summary[f"{key[0]}:{key[1]}"] = {
            split: sum(
                assignments[case_id]["split"] == split
                for case_id in ranked
            )
            for split in ("train", "calibration", "test")
        }
    return {
        "schema_version": "1.0.0",
        "dataset_root": str(root),
        "seed": seed,
        "policy": "stratified_by_group_size_and_template_60_20_20",
        "assignments": assignments,
        "strata": stratum_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()
    result = create_split(args.dataset_root, args.seed)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = {
        split: sum(
            item["split"] == split
            for item in result["assignments"].values()
        )
        for split in ("train", "calibration", "test")
    }
    print(json.dumps(counts))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
