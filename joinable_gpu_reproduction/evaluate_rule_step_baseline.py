"""Evaluate the deterministic STEP rule baseline on JoinABLe sample20 truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def truth_sets(
    row: dict[str, Any], exact_only: bool
) -> dict[str, set[tuple[str, int]]]:
    result: dict[str, set[tuple[str, int]]] = {"a": set(), "b": set()}
    for mapping in row["mapping_records"]:
        if exact_only and not mapping.get("is_exact_designer_entity"):
            continue
        side = mapping.get("side")
        entity_type = mapping.get("source_entity_type")
        if side not in result or entity_type not in {"face", "edge"}:
            continue
        for match in mapping.get("matches", []):
            result[side].add(
                (entity_type, int(match["occt_topology_index"]))
            )
    return result


def candidate_rank(
    prediction: dict[str, Any],
    truth: dict[str, set[tuple[str, int]]],
    limit: int,
) -> int | None:
    if not truth["a"] or not truth["b"]:
        return None
    for candidate in prediction.get("candidates", [])[:limit]:
        a = candidate["part_a_entity"]
        b = candidate["part_b_entity"]
        key_a = (str(a["entity_type"]), int(a["topology_index"]))
        key_b = (str(b["entity_type"]), int(b["topology_index"]))
        if key_a in truth["a"] and key_b in truth["b"]:
            return int(candidate["rank"])
    return None


def recall(rows: list[dict[str, Any]], field: str, k: int) -> float | None:
    evaluable = [row for row in rows if row[field.replace("_rank", "_evaluable")]]
    if not evaluable:
        return None
    return sum(
        row[field] is not None and row[field] <= k for row in evaluable
    ) / len(evaluable)


def metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    evaluable_field = f"{prefix}_evaluable"
    rank_field = f"{prefix}_rank"
    evaluable = [row for row in rows if row[evaluable_field]]
    result: dict[str, Any] = {
        f"{prefix}_evaluable_joint_count": len(evaluable)
    }
    for k in (1, 5, 10, 20):
        result[f"{prefix}_top_{k}_recall"] = (
            sum(
                row[rank_field] is not None and row[rank_field] <= k
                for row in evaluable
            )
            / len(evaluable)
            if evaluable
            else None
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-audit",
        type=Path,
        default=ROOT / "official_step_transfer_audit.json",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=ROOT / "rule_step_baseline" / "predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "rule_vs_pretrained_step_report.json",
    )
    args = parser.parse_args()

    transfer = read_json(args.transfer_audit)
    rows = []
    failures = []
    for joint in transfer["joints"]:
        prediction_path = (
            args.prediction_dir
            / joint["joint_set"].replace(".json", ".json")
        )
        if not prediction_path.is_file():
            failures.append(
                f"prediction_missing:{prediction_path}"
            )
            prediction = {"candidates": []}
        else:
            prediction = read_json(prediction_path)
        exact_truth = truth_sets(joint, exact_only=True)
        equivalent_truth = truth_sets(joint, exact_only=False)
        exact_evaluable = bool(exact_truth["a"] and exact_truth["b"])
        equivalent_evaluable = bool(
            equivalent_truth["a"] and equivalent_truth["b"]
        )
        rows.append(
            {
                "joint_set": joint["joint_set"],
                "joint_index": joint["joint_index"],
                "truth_entity_pair_type": joint["truth_entity_pair_type"],
                "pretrained_inference_status": joint["inference_status"],
                "rule_exact_evaluable": exact_evaluable,
                "rule_exact_rank": (
                    candidate_rank(prediction, exact_truth, 20)
                    if exact_evaluable
                    else None
                ),
                "rule_equivalent_evaluable": equivalent_evaluable,
                "rule_equivalent_rank": (
                    candidate_rank(prediction, equivalent_truth, 20)
                    if equivalent_evaluable
                    else None
                ),
                "pretrained_equivalent_evaluable": joint[
                    "equivalent_evaluable"
                ],
                "pretrained_equivalent_rank": joint["equivalent_rank"],
                "prediction_candidate_count": len(
                    prediction.get("candidates", [])
                ),
                "candidate_reduction_fraction": prediction.get(
                    "candidate_reduction_fraction"
                ),
            }
        )

    paired = [
        row for row in rows
        if row["pretrained_inference_status"] == "success"
        and row["rule_equivalent_evaluable"]
        and row["pretrained_equivalent_evaluable"]
    ]
    comparison: dict[str, Any] = {"paired_joint_count": len(paired)}
    for k in (1, 5, 10, 20):
        comparison[f"rule_top_{k}_recall"] = sum(
            row["rule_equivalent_rank"] is not None
            and row["rule_equivalent_rank"] <= k
            for row in paired
        ) / len(paired)
        comparison[f"pretrained_top_{k}_recall"] = sum(
            row["pretrained_equivalent_rank"] is not None
            and row["pretrained_equivalent_rank"] <= k
            for row in paired
        ) / len(paired)
        comparison[f"pretrained_minus_rule_top_{k}"] = (
            comparison[f"pretrained_top_{k}_recall"]
            - comparison[f"rule_top_{k}_recall"]
        )
    for rule_k, pretrained_k in ((5, 5), (10, 10), (20, 20)):
        key = f"union_rule_{rule_k}_pretrained_{pretrained_k}"
        comparison[f"{key}_candidate_budget_upper_bound"] = (
            rule_k + pretrained_k
        )
        comparison[f"{key}_recall"] = sum(
            (
                row["rule_equivalent_rank"] is not None
                and row["rule_equivalent_rank"] <= rule_k
            )
            or (
                row["pretrained_equivalent_rank"] is not None
                and row["pretrained_equivalent_rank"] <= pretrained_k
            )
            for row in paired
        ) / len(paired)

    comparison["by_entity_pair_type"] = {}
    for pair_type in sorted(
        {row["truth_entity_pair_type"] for row in paired}
    ):
        subset = [
            row for row in paired
            if row["truth_entity_pair_type"] == pair_type
        ]
        comparison["by_entity_pair_type"][pair_type] = {
            "count": len(subset),
            "rule_top_10_recall": sum(
                row["rule_equivalent_rank"] is not None
                and row["rule_equivalent_rank"] <= 10
                for row in subset
            )
            / len(subset),
            "pretrained_top_10_recall": sum(
                row["pretrained_equivalent_rank"] is not None
                and row["pretrained_equivalent_rank"] <= 10
                for row in subset
            )
            / len(subset),
        }

    reductions = [
        float(row["candidate_reduction_fraction"])
        for row in rows
        if row["candidate_reduction_fraction"] is not None
    ]
    report = {
        "schema_version": "1.0.0",
        "purpose": (
            "Same-STEP, same-truth comparison of deterministic rule ranking "
            "and official pretrained JoinABLe"
        ),
        "rule_all_mapped_joints": {
            **metrics(rows, "rule_exact"),
            **metrics(rows, "rule_equivalent"),
        },
        "paired_comparison": comparison,
        "mean_rule_candidate_reduction_fraction": (
            sum(reductions) / len(reductions) if reductions else None
        ),
        "rows": rows,
        "failure_reasons": failures,
        "unavailable_fields": [
            "functional_assembly_validity",
            "final_pose_without_external_solver",
        ],
    }
    write_json(args.output, report)
    print(json.dumps(report["paired_comparison"], ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
