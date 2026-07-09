"""Isolated worker for pose-validating one generated known-group case."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from contracts import GroupProposal
from geometry_pipeline import _pipeline_fingerprint, solve_and_validate_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir")
    parser.add_argument("output_case_dir")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = Path(args.case_dir).resolve()
    output = Path(args.output_case_dir).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    gt = json.loads((source / "gt.json").read_text(encoding="utf-8"))

    parts_dir = output / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for name in gt["parts"]:
        shutil.copy2(source / "step" / name, parts_dir / name)
    proposal = GroupProposal(
        group_id=f"GT_{gt['case_id']}",
        parts=gt["parts"],
        candidate_edges=[],
        geometry_score=1.0,
        connected=True,
        status="diagnostic_ground_truth_group",
        reasons=[
            "evaluation-only known membership; prohibited from pool selection"
        ],
    )
    result = solve_and_validate_group(
        output,
        proposal,
        config,
        output_namespace="pose_validation",
    )
    case_result = {
        "schema_version": "1.0.0",
        "case_id": gt["case_id"],
        "group_size": int(gt["group_size"]),
        "template": gt.get("template"),
        "accepted": bool(result["metrics"]["accepted"]),
        "status": result["status"],
        "collision_count": result["collision_count"],
        "max_constraint_residual": result["max_constraint_residual"],
        "warnings": result["warnings"],
        "selected_pose_candidate_rank": result["metrics"].get(
            "selected_pose_candidate_rank"
        ),
        "exact_pose_candidates_checked": result["metrics"].get(
            "exact_pose_candidates_checked"
        ),
        "expanded_states": result["metrics"].get("expanded_states"),
        "pipeline_fingerprint": _pipeline_fingerprint(config),
        "elapsed_seconds": time.perf_counter() - started,
        "validation_result": result,
    }
    (output / "case_result.json").write_text(
        json.dumps(case_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_id": gt["case_id"],
                "accepted": case_result["accepted"],
                "elapsed_seconds": case_result["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
