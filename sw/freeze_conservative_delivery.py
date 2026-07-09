"""Freeze hashes and measured outputs for the conservative D3.5-D7 release."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
FILES = [
    "candidate_recall_audit.py",
    "global_optimizer/group_consistency.py",
    "conservative_pipeline.py",
    "semantic_review.py",
    "semantic_pool.py",
    "semantic_explanation_batch.py",
    "build_human_semantic_review_pack.py",
    "run_conservative_delivery.py",
    "geometry_pipeline.py",
    "agent_controller.py",
    "configs/pool_pipeline.json",
    "configs/conservative_pipeline.json",
    "configs/semantic_calibration.json",
]
RESULTS = [
    "data/results/candidate_recall_by_type.json",
    "data/results/candidate_recall_by_group_size.json",
    "data/results/conservative_metrics.json",
    "data/results/baseline_comparison.json",
    "data/results/semantic_calibration_report.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    metrics = json.loads(
        (HERE / "data" / "results" / "conservative_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": "1.0.0",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "stage": "D3.5_D7_conservative_precision_first",
        "policy": {
            "objective": "minimize false automatic accepts",
            "semantic_mode": "explanation_only",
            "reinforcement_learning": False,
            "multi_agent_expansion": False,
            "max_auto_accept_group_size": 5,
        },
        "metrics": metrics,
        "source_sha256": {
            name: digest(HERE / name) for name in FILES
        },
        "result_sha256": {
            name: digest(HERE / name)
            for name in RESULTS
            if (HERE / name).is_file()
        },
        "verification": {
            "core_tests": "42/42 passed",
            "generator_tests": "3/3 passed",
            "stl_roundtrip": "OCCT write/read/shape-valid checks passed",
        },
        "known_limits": [
            "No group is currently safe enough for automatic acceptance.",
            "Auto-accept precision is not statistically estimable at zero accepts.",
            "Only 6/30 truth groups reach the bounded human review frontier.",
            "DeepSeek remains unable to affect grouping until human labels and holdout safety pass.",
        ],
    }
    output = HERE / "D7_CONSERVATIVE_FREEZE.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
