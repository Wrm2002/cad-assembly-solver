"""Run the geometry pipeline in isolated pool processes and aggregate results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METRIC_KEYS = (
    "exact_group_precision",
    "exact_group_recall",
    "exact_group_f1",
    "copart_pair_precision",
    "copart_pair_recall",
    "copart_pair_f1",
)


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
    results = []
    failures = []
    for pool in sorted(path for path in root.iterdir() if path.is_dir()):
        if not (pool / "pool_gt.json").is_file():
            continue
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "geometry_pipeline.py"),
                str(pool),
                "--config",
                str(config),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode != 0:
            failures.append(
                {
                    "pool_id": pool.name,
                    "returncode": process.returncode,
                    "stdout_tail": process.stdout[-2000:],
                    "stderr_tail": process.stderr[-4000:],
                }
            )
            print(f"{pool.name}: failed ({process.returncode})", flush=True)
            continue
        report = json.loads(
            (pool / "validation" / "validation_summary.json").read_text(
                encoding="utf-8"
            )
        )
        results.append(report)
        print(
            f"{pool.name}: attempts={report['attempt_count']} "
            f"converged={report['converged']} "
            f"pair_f1={report['metrics']['copart_pair_f1']:.3f}",
            flush=True,
        )

    benchmark = {
        "schema_version": "1.0.0",
        "pool_count": len(results),
        "failure_count": len(failures),
        "converged_pool_count": sum(item["converged"] for item in results),
        "macro_metrics": {
            key: (
                sum(item["metrics"][key] for item in results) / len(results)
                if results
                else 0.0
            )
            for key in METRIC_KEYS
        },
        "pools": [
            {
                "pool_id": item["pool_id"],
                "attempt_count": item["attempt_count"],
                "converged": item["converged"],
                **item["metrics"],
            }
            for item in results
        ],
        "failures": failures,
    }
    (root / "geometry_validation_benchmark.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(benchmark["macro_metrics"], ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
