"""Run the isolated real-STEP pair-interface and pose-validation benchmark."""
from __future__ import annotations

import argparse
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


def truth_rank(prediction: dict, truth: dict, limit: int) -> int | None:
    sides = []
    for suffix in ("a", "b"):
        candidate = truth.get(f"candidate_interface_{suffix}", {})
        sides.append(
            {
                "face": set(int(value) for value in candidate.get("face_ids", [])),
                "edge": set(int(value) for value in candidate.get("edge_ids", [])),
            }
        )
    if not all(side["face"] or side["edge"] for side in sides):
        return None
    for candidate in prediction.get("candidates", [])[:limit]:
        matched = True
        for position, suffix in enumerate(("a", "b")):
            entity = candidate[f"part_{suffix}_entity"]
            matched = matched and (
                int(entity["topology_index"])
                in sides[position].get(entity["entity_type"], set())
            )
        if matched:
            return int(candidate["rank"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--manual-interfaces", required=True)
    parser.add_argument("--pair-truth", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--validate-top", type=int, default=5)
    parser.add_argument("--extract-timeout", type=int, default=900)
    parser.add_argument("--validation-timeout", type=int, default=180)
    args = parser.parse_args()

    inventory = load(Path(args.inventory))
    case_parts = {
        str(case["case_id"]): {
            part["name"]: Path(part["source_path"])
            for part in case["part_step_files"]
        }
        for case in inventory["cases"]
    }
    data_root = Path(args.data_root)
    descriptor_root = data_root / "descriptors"
    prediction_root = data_root / "interface_predictions"
    validation_root = data_root / "pose_validation"
    rows = []
    descriptor_rows = []

    for case_id, parts in case_parts.items():
        for name, source in parts.items():
            output = descriptor_root / f"case_{case_id}" / f"{source.stem}.entities.json"
            valid_existing = False
            if output.exists():
                try:
                    valid_existing = bool(load(output).get("entities"))
                except Exception:
                    valid_existing = False
            if valid_existing:
                status, reason = "success", None
            else:
                status, reason = run(
                    [
                        sys.executable,
                        str(PREDICTOR),
                        "--extract",
                        "--source",
                        str(source),
                        "--output",
                        str(output),
                        "--per-bucket",
                        "12",
                    ],
                    args.extract_timeout,
                )
            descriptor_rows.append(
                {
                    "case_id": case_id,
                    "part": name,
                    "status": status,
                    "failure_reason": reason,
                    "output": str(output.resolve()),
                }
            )

    for template_path in sorted(Path(args.manual_interfaces).glob("case_*.json")):
        template = load(template_path)
        case_id = str(template["case_id"])
        for pair_index, pair in enumerate(template["positive_part_pairs"], 1):
            part_a_name, part_b_name = pair["parts"]
            part_a = case_parts[case_id][part_a_name]
            part_b = case_parts[case_id][part_b_name]
            descriptor_a = descriptor_root / f"case_{case_id}" / f"{part_a.stem}.entities.json"
            descriptor_b = descriptor_root / f"case_{case_id}" / f"{part_b.stem}.entities.json"
            prediction_path = prediction_root / f"case_{case_id}" / f"pair_{pair_index:02d}.json"
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
                        str(prediction_path),
                        "--top-k",
                        str(args.top_k),
                    ],
                    120,
                )
            else:
                prediction_status, prediction_reason = "failed", "descriptor_missing"

            truth_path = (
                Path(args.pair_truth)
                / f"case_{case_id}"
                / f"pair_{pair_index:02d}.json"
            )
            truth = load(truth_path) if truth_path.exists() else {}
            prediction = load(prediction_path) if prediction_path.exists() else {}
            reference_status = truth.get("pose_reference_status", "unavailable")
            rank_5 = truth_rank(prediction, truth, 5)
            rank_10 = truth_rank(prediction, truth, 10)
            validation_rows = []
            # Large parts are still sent to OCCT, but only two ranked hypotheses;
            # timeout is an auditable uncertain result, never an automatic pass.
            large_pair = max(part_a.stat().st_size, part_b.stat().st_size) > 10_000_000
            validate_count = min(
                len(prediction.get("candidates", [])),
                1 if large_pair else args.validate_top,
            )
            for rank in range(1, validate_count + 1):
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
                            str(part_a),
                            "--part-b",
                            str(part_b),
                            "--prediction",
                            str(prediction_path),
                            "--rank",
                            str(rank),
                            "--output",
                            str(output),
                        ],
                        args.validation_timeout,
                    )
                if reason and not output.exists():
                    dump(
                        output,
                        {
                            "schema_version": "1.0.0",
                            "candidate_rank": rank,
                            "final_pose_status": "uncertain",
                            "failure_reasons": [reason],
                            "unavailable_fields": ["pose_checks"],
                        },
                    )
                result = load(output) if output.exists() else {}
                validation_rows.append(
                    {
                        "candidate_rank": rank,
                        "worker_status": status,
                        "worker_failure_reason": reason,
                        "final_pose_status": result.get(
                            "final_pose_status", "uncertain"
                        ),
                        "output": str(output.resolve()),
                    }
                )
            rows.append(
                {
                    "case_id": case_id,
                    "pair_index": pair_index,
                    "parts": [part_a_name, part_b_name],
                    "relation_type": pair.get("relation_type"),
                    "evidence_level": pair.get("evidence_level"),
                    "prediction_status": prediction_status,
                    "prediction_failure_reason": prediction_reason,
                    "prediction_path": str(prediction_path.resolve()),
                    "pose_reference_status": reference_status,
                    "truth_rank_at_5": rank_5,
                    "truth_rank_at_10": rank_10,
                    "truth_evaluable": reference_status == "contact_or_overlap"
                    and bool(
                        truth.get("candidate_interface_a", {}).get("face_ids")
                        or truth.get("candidate_interface_a", {}).get("edge_ids")
                    )
                    and bool(
                        truth.get("candidate_interface_b", {}).get("face_ids")
                        or truth.get("candidate_interface_b", {}).get("edge_ids")
                    ),
                    "pose_validation": validation_rows,
                    "any_valid_pose": any(
                        item["final_pose_status"] == "valid"
                        for item in validation_rows
                    ),
                }
            )

    evaluable = [row for row in rows if row["truth_evaluable"]]
    report = {
        "schema_version": "1.0.0",
        "predictor_kind": "deterministic_rule_baseline_not_pretrained_joinable",
        "descriptor_parts": descriptor_rows,
        "pair_count": len(rows),
        "truth_evaluable_pair_count": len(evaluable),
        "top_5_interface_recall": (
            sum(row["truth_rank_at_5"] is not None for row in evaluable) / len(evaluable)
            if evaluable
            else None
        ),
        "top_10_interface_recall": (
            sum(row["truth_rank_at_10"] is not None for row in evaluable) / len(evaluable)
            if evaluable
            else None
        ),
        "pair_pose_success_count": sum(row["any_valid_pose"] for row in rows),
        "pairs": rows,
        "failure_reasons": [
            reason
            for row in descriptor_rows
            for reason in [row["failure_reason"]]
            if reason
        ]
        + [
            row["prediction_failure_reason"]
            for row in rows
            if row["prediction_failure_reason"]
        ],
        "unavailable_fields": [
            "designer_interface_labels_for_all_pairs",
            "learned_joinable_probability",
        ],
    }
    dump(Path(args.report), report)
    print(
        f"pair benchmark: {len(rows)} pairs; "
        f"top5={report['top_5_interface_recall']}; "
        f"pose_success={report['pair_pose_success_count']}"
    )
    return 0 if all(row["prediction_status"] == "success" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
