"""Evaluate pose-validation false negatives using GT groups, never for selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
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
    root = Path(args.root).resolve()
    config = Path(args.config).resolve()
    rows = []
    for pool in sorted(path for path in root.iterdir() if path.is_dir()):
        gt_path = pool / "pool_gt.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        for group in gt["true_groups"]:
            subject = "GT_" + hashlib.sha256(
                "|".join(sorted(group["parts"])).encode("utf-8")
            ).hexdigest()[:12]
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "true_group_validation_worker.py"),
                    str(pool),
                    group["group_id"],
                    subject,
                    "--config",
                    str(config),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            result_path = (
                pool / "oracle_validation" / subject / "validation_result.json"
            )
            if process.returncode == 0 and result_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                accepted = bool(result["metrics"]["accepted"])
                reason = result["warnings"]
            else:
                accepted = False
                reason = [
                    f"isolated worker failed with returncode={process.returncode}",
                    process.stderr[-2000:],
                ]
            rows.append(
                {
                    "pool_id": gt["pool_id"],
                    "truth_group_id": group["group_id"],
                    "subject_id": subject,
                    "part_count": len(group["parts"]),
                    "accepted": accepted,
                    "diagnostic_only": True,
                    "reasons": reason,
                }
            )
            print(
                f"{gt['pool_id']}/{group['group_id']}: "
                f"{'accepted' if accepted else 'rejected'}",
                flush=True,
            )
    accepted_count = sum(row["accepted"] for row in rows)
    report = {
        "schema_version": "1.0.0",
        "diagnostic_only": True,
        "selection_leakage_prohibited": True,
        "truth_group_count": len(rows),
        "accepted_truth_group_count": accepted_count,
        "truth_group_validation_recall": (
            accepted_count / len(rows) if rows else 0.0
        ),
        "groups": rows,
    }
    (root / "true_group_validation_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "truth_group_count": len(rows),
                "accepted_truth_group_count": accepted_count,
                "truth_group_validation_recall": report[
                    "truth_group_validation_recall"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
