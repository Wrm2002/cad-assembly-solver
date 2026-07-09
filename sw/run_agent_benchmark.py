"""Run frozen Agent evaluation, respecting the semantic calibration gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_controller import run_agent_pool, semantic_application_allowed


METRICS = (
    "exact_group_precision",
    "exact_group_recall",
    "exact_group_f1",
    "copart_pair_precision",
    "copart_pair_recall",
    "copart_pair_f1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="mixed_pools_v1")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument(
        "--calibration",
        default=str(
            Path(__file__).parent
            / "configs"
            / "semantic_calibration.json"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    calibration_path = Path(args.calibration).resolve()
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    allowed = semantic_application_allowed(calibration)
    calibration_pools = set(calibration["calibration_pools"])
    pools = [
        path for path in sorted(root.iterdir())
        if path.is_dir()
        and (path / "pool_gt.json").is_file()
        and path.name not in calibration_pools
    ]
    rows = []
    # A failed calibration gate prohibits holdout API calls and semantic
    # influence. This preserves an untouched evaluation set.
    semantic_mode = "live" if allowed else "off"
    for pool in pools:
        report = run_agent_pool(
            pool,
            args.config,
            semantic_mode=semantic_mode,
            calibration_path=calibration_path,
        )
        metrics = report["validation"]["metrics"]
        rows.append(
            {
                "pool_id": pool.name,
                "semantic_mode": semantic_mode,
                "semantic_applied": report["semantic_applied"],
                "converged": report["validation"]["converged"],
                **metrics,
            }
        )
        print(
            f"{pool.name}: semantic={semantic_mode} "
            f"pair_f1={metrics['copart_pair_f1']:.3f}",
            flush=True,
        )
    result = {
        "schema_version": "1.0.0",
        "calibration_gate_passed": allowed,
        "calibration_pools": sorted(calibration_pools),
        "holdout_pool_count": len(rows),
        "holdout_api_calls_permitted": allowed,
        "holdout_semantic_mode": semantic_mode,
        "semantic_applied_pool_count": sum(
            row["semantic_applied"] for row in rows
        ),
        "macro_metrics": {
            key: (
                sum(row[key] for row in rows) / len(rows)
                if rows
                else 0.0
            )
            for key in METRICS
        },
        "pools": rows,
    }
    (root / "agent_benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["macro_metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
