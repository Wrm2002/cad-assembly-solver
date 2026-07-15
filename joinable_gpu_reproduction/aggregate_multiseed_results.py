"""Aggregate bounded STEP-domain adaptation runs without test-set selection."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_metrics(section: dict[str, Any]) -> dict[str, float]:
    exact = section["exact"]
    return {
        f"top_{k}_recall": float(exact[f"top_{k}_recall"])
        for k in (1, 5, 10, 20)
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for path in args.reports:
        report = read_json(path)
        baseline_validation = exact_metrics(
            report["baseline_validation"]
        )
        best_validation = exact_metrics(report["best_validation"])
        baseline_test = exact_metrics(report["baseline_test"])
        best_test = exact_metrics(report["best_test"])
        runs.append(
            {
                "report": str(path.resolve()),
                "seed": int(report["configuration"]["seed"]),
                "best_epoch": int(report["best_epoch"]),
                "baseline_validation": baseline_validation,
                "best_validation": best_validation,
                "validation_delta": {
                    key: best_validation[key] - baseline_validation[key]
                    for key in baseline_validation
                },
                "baseline_test": baseline_test,
                "best_test": best_test,
                "test_delta": {
                    key: best_test[key] - baseline_test[key]
                    for key in baseline_test
                },
            }
        )

    validation_deltas = [row["validation_delta"] for row in runs]
    test_deltas = [row["test_delta"] for row in runs]
    mean_validation_delta = mean_metrics(validation_deltas)
    mean_test_delta = mean_metrics(test_deltas)
    positive_validation_top10_runs = sum(
        row["validation_delta"]["top_10_recall"] > 0 for row in runs
    )
    non_degraded_test_top10_runs = sum(
        row["test_delta"]["top_10_recall"] >= 0 for row in runs
    )
    result = {
        "schema_version": "1.0.0",
        "selection_policy": (
            "checkpoints selected on validation exact Top-10/Top-5/Top-1; "
            "test is evaluation-only"
        ),
        "run_count": len(runs),
        "runs": runs,
        "mean_validation_delta": mean_validation_delta,
        "mean_test_delta": mean_test_delta,
        "positive_validation_top10_run_count": (
            positive_validation_top10_runs
        ),
        "non_degraded_test_top10_run_count": (
            non_degraded_test_top10_runs
        ),
        "conclusion": {
            "validation_signal_consistent": (
                positive_validation_top10_runs >= max(1, len(runs) - 1)
            ),
            "independent_test_signal_consistent": (
                non_degraded_test_top10_runs >= max(1, len(runs) - 1)
                and mean_test_delta["top_10_recall"] >= 0
            ),
            "safe_to_promote_adapted_checkpoint": (
                positive_validation_top10_runs >= max(1, len(runs) - 1)
                and non_degraded_test_top10_runs >= max(1, len(runs) - 1)
                and mean_test_delta["top_10_recall"] >= 0
            ),
        },
        "failure_reasons": [],
        "unavailable_fields": [
            "functional_assembly_validity",
            "mixed_pool_grouping_quality",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["conclusion"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
