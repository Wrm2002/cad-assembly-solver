"""Evaluate every within-case part pair with bounded predictor/OCCT workers."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREDICTOR = ROOT / "tools" / "joinable_interface_predictor" / "rule_interface_predictor.py"
VALIDATOR = ROOT / "tools" / "occt_pose_validator" / "occt_pair_pose_validator.py"


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], timeout: int) -> tuple[str, str | None]:
    try:
        environment = os.environ.copy()
        environment["PYTHONWARNINGS"] = "ignore"
        completed = subprocess.run(command, timeout=timeout, env=environment)
        return (
            ("success" if completed.returncode == 0 else "partial_or_failed"),
            (None if completed.returncode == 0 else f"worker_exit_{completed.returncode}"),
        )
    except subprocess.TimeoutExpired:
        return "failed", f"worker_timeout_{timeout}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--positive-pair-report")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--validate-top", type=int, default=3)
    parser.add_argument("--validation-timeout", type=int, default=180)
    args = parser.parse_args()

    inventory = load(Path(args.inventory))
    positive_cache = {}
    if args.positive_pair_report and Path(args.positive_pair_report).exists():
        positive_report = load(Path(args.positive_pair_report))
        positive_cache = {
            (str(row["case_id"]), tuple(sorted(row["parts"]))): row
            for row in positive_report.get("pairs", [])
        }
    data_root = Path(args.data_root)
    descriptor_root = data_root / "descriptors"
    prediction_root = data_root / "all_pair_predictions"
    validation_root = data_root / "all_pair_pose_validation"
    cases = []
    for case in inventory["cases"]:
        case_id = str(case["case_id"])
        parts = [
            {"name": row["name"], "path": Path(row["source_path"])}
            for row in case["part_step_files"]
        ]
        pair_rows = []
        for pair_index, (part_a, part_b) in enumerate(itertools.combinations(parts, 2), 1):
            descriptor_a = (
                descriptor_root / f"case_{case_id}" / f"{part_a['path'].stem}.entities.json"
            )
            descriptor_b = (
                descriptor_root / f"case_{case_id}" / f"{part_b['path'].stem}.entities.json"
            )
            prediction = (
                prediction_root / f"case_{case_id}" / f"pair_{pair_index:02d}.json"
            )
            if descriptor_a.exists() and descriptor_b.exists():
                prediction_status, prediction_reason = run(
                    [
                        sys.executable,
                        str(PREDICTOR),
                        "--predict",
                        "--part-a-descriptors",
                        str(descriptor_a),
                        "--part-b-descriptors",
                        str(descriptor_b),
                        "--output",
                        str(prediction),
                        "--top-k",
                        str(args.top_k),
                    ],
                    120,
                )
            else:
                prediction_status, prediction_reason = "failed", "descriptor_missing"
            prediction_data = load(prediction) if prediction.exists() else {}
            large_pair = max(
                part_a["path"].stat().st_size, part_b["path"].stat().st_size
            ) > 5_000_000
            validate_count = min(
                len(prediction_data.get("candidates", [])),
                1 if large_pair else args.validate_top,
            )
            validations = []
            cached = positive_cache.get(
                (case_id, tuple(sorted([part_a["name"], part_b["name"]])))
            )
            if cached:
                validations = [
                    {
                        "rank": row["candidate_rank"],
                        "worker_status": row["worker_status"],
                        "worker_failure_reason": row["worker_failure_reason"],
                        "pose_status": row["final_pose_status"],
                        "output": row["output"],
                        "reused_from_positive_pair_benchmark": True,
                    }
                    for row in cached.get("pose_validation", [])
                ]
            for rank in ([] if cached else range(1, validate_count + 1)):
                output = (
                    validation_root
                    / f"case_{case_id}"
                    / f"pair_{pair_index:02d}_rank_{rank:02d}.json"
                )
                existing = load(output) if output.exists() else {}
                if existing.get("final_pose_status") in ("valid", "failed", "uncertain"):
                    status, reason = "success", None
                else:
                    status, reason = run(
                        [
                            sys.executable,
                            str(VALIDATOR),
                            "--part-a",
                            str(part_a["path"]),
                            "--part-b",
                            str(part_b["path"]),
                            "--prediction",
                            str(prediction),
                            "--rank",
                            str(rank),
                            "--output",
                            str(output),
                        ],
                        args.validation_timeout,
                    )
                if not output.exists():
                    dump(
                        output,
                        {
                            "schema_version": "1.0.0",
                            "candidate_rank": rank,
                            "final_pose_status": "uncertain",
                            "failure_reasons": [reason or "worker_no_output"],
                            "unavailable_fields": ["pose_checks"],
                        },
                    )
                result = load(output)
                validations.append(
                    {
                        "rank": rank,
                        "worker_status": status,
                        "worker_failure_reason": reason,
                        "pose_status": result.get("final_pose_status", "uncertain"),
                        "output": str(output.resolve()),
                    }
                )
            pair_rows.append(
                {
                    "pair_id": f"case_{case_id}_pair_{pair_index:02d}",
                    "parts": [part_a["name"], part_b["name"]],
                    "prediction_status": prediction_status,
                    "prediction_failure_reason": prediction_reason,
                    "prediction_path": str(prediction.resolve()),
                    "validated_candidate_count": len(validations),
                    "validations": validations,
                    "any_valid_pose": any(
                        row["pose_status"] == "valid" for row in validations
                    ),
                    "any_uncertain_pose": any(
                        row["pose_status"] == "uncertain" for row in validations
                    ),
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "parts": [part["name"] for part in parts],
                "pair_count": len(pair_rows),
                "pairs": pair_rows,
            }
        )
    report = {
        "schema_version": "1.0.0",
        "predictor_kind": "deterministic_rule_baseline_not_pretrained_joinable",
        "case_count": len(cases),
        "part_count": sum(len(case["parts"]) for case in cases),
        "pair_count": sum(case["pair_count"] for case in cases),
        "valid_pose_pair_count": sum(
            pair["any_valid_pose"] for case in cases for pair in case["pairs"]
        ),
        "cases": cases,
        "failure_reasons": [
            pair["prediction_failure_reason"]
            for case in cases
            for pair in case["pairs"]
            if pair["prediction_failure_reason"]
        ],
        "unavailable_fields": [
            "learned_joinable_probability",
            "designer_mate_labels_for_unannotated_pairs",
        ],
    }
    dump(Path(args.report), report)
    print(
        f"all-pair graph: {report['pair_count']} pairs, "
        f"{report['valid_pose_pair_count']} with bounded valid pose"
    )
    return 0 if not report["failure_reasons"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
