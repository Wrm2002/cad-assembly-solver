"""One-command CAD pool pipeline; geometry mode requires no labels or API key."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from agent_controller import run_agent_pool
from geometry_pipeline import run_pool
from pool_index import index_pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="folder of STEP parts")
    parser.add_argument("--output", required=True, help="results directory")
    parser.add_argument(
        "--semantic",
        choices=("off", "deepseek"),
        default="off",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    config = Path(args.config).resolve()
    source_files = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    )
    if not 2 <= len(source_files) <= 12:
        raise ValueError(
            f"geometry MVP expects 2..12 STEP parts, found {len(source_files)}"
        )

    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    source_names = {path.name for path in source_files}
    stale = {
        path.name
        for path in parts.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".step", ".stp"}
        and path.name not in source_names
    }
    if stale:
        raise RuntimeError(
            "output/parts contains stale STEP files; choose a clean output "
            f"directory or remove explicitly: {sorted(stale)}"
        )
    for path in source_files:
        target = parts / path.name
        if path != target:
            shutil.copy2(path, target)

    index_pool(parts, output / "index", config)
    if args.semantic == "deepseek":
        agent = run_agent_pool(
            output,
            config,
            semantic_mode="live",
            calibration_path=(
                Path(__file__).parent
                / "configs"
                / "semantic_calibration.json"
            ),
        )
        report = agent["validation"]
        semantic_applied = agent["semantic_applied"]
    else:
        report = run_pool(output, config)
        semantic_applied = False
    summary = {
        "schema_version": "1.0.0",
        "mode": (
            "agent_semantic_review"
            if args.semantic == "deepseek"
            else "geometry_only"
        ),
        "api_key_used": args.semantic == "deepseek",
        "semantic_applied": semantic_applied,
        "part_count": len(source_files),
        "converged": report["converged"],
        "attempt_count": report["attempt_count"],
        "final_assignment": str(
            output / "validation" / "validated_group_assignment.json"
        ),
        "evaluation": (
            "available" if report["metrics"] is not None else "not_requested"
        ),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
