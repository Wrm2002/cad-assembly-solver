"""Bounded CAD Agent controller with calibrated semantic gating."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts import AgentEvent
from geometry_pipeline import run_pool
from global_grouping import run as run_grouping
from semantic_pool import review_pool


def semantic_application_allowed(
    calibration: dict[str, Any] | None,
    semantic_config: dict[str, Any] | None = None,
) -> bool:
    configured_mode = (
        (semantic_config or {}).get("application_mode")
        or (calibration or {}).get("semantic_application_mode")
    )
    return bool(
        calibration
        and calibration.get("semantic_reranking_enabled") is True
        and configured_mode == "rerank"
    )


def run_agent_pool(
    pool_dir: str | Path,
    config_path: str | Path,
    *,
    semantic_mode: str = "off",
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    pool = Path(pool_dir).resolve()
    config_path = Path(config_path).resolve()
    calibration = None
    if calibration_path and Path(calibration_path).is_file():
        calibration = json.loads(
            Path(calibration_path).read_text(encoding="utf-8")
        )
    events = []

    def event(state, action, outcome, message, evidence=()):
        events.append(
            AgentEvent(
                event_id=f"E_{pool.name}_{len(events):03d}",
                timestamp=datetime.now(timezone.utc),
                run_id=f"agent_{pool.name}",
                sequence=len(events),
                state=state,
                action=action,
                tool=None,
                parameters={},
                outcome=outcome,
                evidence_refs=list(evidence),
                retry_count=0,
                message=message,
            ).model_dump(mode="json")
        )

    run_grouping(pool, config_path)
    event(
        "geometry_grouping",
        "generate_partition",
        "success",
        "Generated deterministic geometry proposals and baseline partition.",
        ["grouping/group_assignment.json"],
    )

    semantic_report = None
    overrides = {}
    semantic_applied = False
    if semantic_mode != "off":
        semantic_report = review_pool(
            pool, config_path, mode=semantic_mode
        )
        allowed = semantic_application_allowed(
            calibration, config.get("semantic_review")
        )
        if allowed:
            overrides = semantic_report["utility_overrides"]
            semantic_applied = bool(overrides)
            outcome = "applied" if semantic_applied else "no_reviews"
            message = "Calibrated semantic tie-break utilities were applied."
        else:
            outcome = "gated_off"
            message = (
                "Semantic reviews were retained for audit but calibration "
                "did not permit them to affect grouping."
            )
        event(
            "semantic_review",
            "review_ambiguous_proposals",
            outcome,
            message,
            ["semantic/semantic_review_report.json"],
        )
    else:
        event(
            "semantic_review",
            "skip_semantic_review",
            "disabled",
            "Geometry-only mode requested; no API call was made.",
        )

    validation = run_pool(
        pool,
        config_path,
        utility_overrides=overrides,
        output_namespace="agent_validation",
    )
    event(
        "pose_validation",
        "solve_validate_and_retry",
        "success" if validation["converged"] else "retry_exhausted",
        "Completed bounded pose solving and exact collision validation.",
        ["agent_validation/validation_summary.json"],
    )
    event(
        "complete",
        "emit_final_assignment",
        "success",
        "Emitted final assignment; geometry failures remain non-overridable.",
        ["agent_validation/validated_group_assignment.json"],
    )

    report = {
        "schema_version": "1.0.0",
        "pool_id": pool.name,
        "semantic_mode": semantic_mode,
        "semantic_calibration_available": calibration is not None,
        "semantic_application_allowed": semantic_application_allowed(
            calibration, config.get("semantic_review")
        ),
        "semantic_applied": semantic_applied,
        "semantic_review_count": (
            semantic_report["review_count"] if semantic_report else 0
        ),
        "validation": validation,
        "events": events,
    }
    output = pool / "agent"
    output.mkdir(parents=True, exist_ok=True)
    (output / "agent_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "agent_events.json").write_text(
        json.dumps(events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool_dir")
    parser.add_argument(
        "--semantic",
        choices=("off", "live", "cache_only"),
        default="off",
    )
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
    result = run_agent_pool(
        args.pool_dir,
        args.config,
        semantic_mode=args.semantic,
        calibration_path=args.calibration,
    )
    print(
        json.dumps(
            {
                "pool_id": result["pool_id"],
                "semantic_applied": result["semantic_applied"],
                "converged": result["validation"]["converged"],
                "metrics": result["validation"]["metrics"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
