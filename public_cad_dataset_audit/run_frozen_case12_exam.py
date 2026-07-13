"""Run case1/2 only after Fusion training has completed.

The SolidWorks cases are fixed evaluation material.  This script never reads
them while creating tensors, choosing training epochs, or selecting an early
stopping checkpoint.  It calls the existing generic B-Rep inference/global
pose/OCCT/render runner and records both successes and failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "return_code": result.returncode,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dir", type=Path)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--solver-python", type=Path)
    parser.add_argument("--occt-python", type=Path, required=True)
    args = parser.parse_args()
    variants = {
        "pair_pose": args.training_dir / "pose_proposal" / "best.pt",
        "interface_score": args.training_dir / "interface_score" / "best.pt",
    }
    cases = {
        "case1": ROOT / "sw" / "exam_brep_manifold_20260710" / "case_1",
        "case2": ROOT / "sw" / "exam_brep_manifold_20260710" / "case_2",
    }
    report: dict[str, Any] = {
        "schema_version": "frozen_case12_post_training_exam.v1",
        "training_data": "Fusion360 only",
        "evaluation_policy": "Frozen SolidWorks cases are evaluation-only and do not affect model selection.",
        "runs": [],
    }
    for variant, checkpoint in variants.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing_trained_checkpoint:{checkpoint}")
        for case_name, case_root in cases.items():
            manifest = case_root / "pair_frontier" / "pair_frontier_manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(f"missing_frozen_pair_frontier:{manifest}")
            output = case_root / "equivalent_pose_brep_v1" / variant
            command = [
                str(args.runtime_python), "sw/run_strong_contact_exam.py",
                str(manifest), str(checkpoint), str(output),
                "--runtime-python", str(args.runtime_python),
                "--solver-python", str(args.solver_python or args.runtime_python),
                "--occt-python", str(args.occt_python),
                "--candidate-limit", "20", "--modes-per-candidate", "6",
                "--learned-pose-prior-weight", "0.35",
            ]
            record = {"case": case_name, "variant": variant, "checkpoint": str(checkpoint.resolve()),
                      "output_dir": str(output.resolve()), "command": command, **_run(command)}
            summary_path = output / "exam_summary.json"
            if summary_path.is_file():
                record["exam_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            report["runs"].append(record)
            print(json.dumps({"case": case_name, "variant": variant, "return_code": record["return_code"]}, ensure_ascii=False), flush=True)
    report["successful_runs"] = sum(row["return_code"] == 0 for row in report["runs"])
    report["failed_runs"] = len(report["runs"]) - report["successful_runs"]
    args.training_dir.mkdir(parents=True, exist_ok=True)
    (args.training_dir / "case12_exam_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Individual failures must not hide results from the other frozen case.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
