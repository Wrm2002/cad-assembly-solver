"""评估功能接口闭环特征；真值仅用于离线计分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from functional_interface_features import compute_functional_interface_features


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def evaluate(
    pools_root: Path,
    proposal_experiment: Path,
    pose_experiment: Path,
) -> dict[str, Any]:
    records = []
    for pool in sorted(
        row for row in pools_root.iterdir()
        if row.is_dir() and (row / "pool_gt.json").is_file()
    ):
        truth = {
            frozenset(group["parts"])
            for group in _load(pool / "pool_gt.json").get("true_groups", [])
        }
        pair_edges = _load(
            proposal_experiment / "pools" / pool.name / "pair_edges.json"
        )
        roles = _load(
            proposal_experiment / "pools" / pool.name / "role_hypotheses.json"
        )
        pose_rows = [
            row for row in _load(pose_experiment / "pose_validation_records.json")
            if row["pool_id"] == pool.name
            and row["pose_validation"]["final_pose_status"] == "valid"
        ]
        for row in pose_rows:
            features = compute_functional_interface_features(
                row, pair_edges, roles
            )
            records.append(
                {
                    "pool_id": pool.name,
                    "group_id": row["group_id"],
                    "assembly_family": row["assembly_family"],
                    "is_true_group": frozenset(row["parts"]) in truth,
                    **features,
                }
            )
    labels = [int(row["is_true_group"]) for row in records]
    scores = [float(row["functional_closure_score"]) for row in records]
    true_scores = [score for label, score in zip(labels, scores) if label]
    false_scores = [score for label, score in zip(labels, scores) if not label]
    return {
        "pose_valid_group_count": len(records),
        "pose_valid_true_count": sum(labels),
        "pose_valid_false_count": len(records) - sum(labels),
        "functional_closure_auc": _auc(labels, scores),
        "true_score_min": min(true_scores, default=None),
        "true_score_mean": sum(true_scores) / len(true_scores) if true_scores else None,
        "false_score_max": max(false_scores, default=None),
        "false_score_mean": sum(false_scores) / len(false_scores) if false_scores else None,
        "perfect_separation": bool(
            true_scores and false_scores and min(true_scores) > max(false_scores)
        ),
        "records": records,
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=here / "data/functional_error_analysis_v3",
    )
    args = parser.parse_args()
    ordinary = evaluate(
        here / "data/functional_mixed_pools_topology_v1",
        here / "data/functional_grouping_experiment_v2",
        here / "data/functional_frontier_pose_experiment_v2",
    )
    harder = evaluate(
        here / "data/functional_cad_holdout_pools_v1",
        here / "data/harder_functional_grouping_experiment_v2",
        here / "data/harder_frontier_pose_experiment_v2",
    )
    gate_passed = bool(
        ordinary["functional_closure_auc"] is not None
        and ordinary["functional_closure_auc"] >= 0.70
        and ordinary["perfect_separation"]
        and harder["perfect_separation"]
    )
    report = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only_calibration",
        "ordinary": ordinary,
        "harder_holdout_provisional": harder,
        "functional_interface_gate_passed": gate_passed,
        "production_reranking_enabled": False,
        "decision": (
            "keep_explanation_and_review_features_only"
            if not gate_passed else "eligible_for_separate_signed_calibration"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "functional_interface_feature_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 功能接口闭环特征评估",
        "",
        f"- 普通集 AUC：{ordinary['functional_closure_auc']}",
        f"- 普通集完全分离：{ordinary['perfect_separation']}",
        f"- harder holdout 完全分离：{harder['perfect_separation']}",
        f"- 功能接口门通过：{gate_passed}",
        "- 生产重排启用：False",
        "",
        "未通过时，这些特征只用于解释和人工复核，不影响 accepted/rejected。",
    ]
    (args.output / "functional_interface_feature_evaluation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "ordinary_auc": ordinary["functional_closure_auc"],
        "ordinary_perfect_separation": ordinary["perfect_separation"],
        "harder_perfect_separation": harder["perfect_separation"],
        "functional_interface_gate_passed": gate_passed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

