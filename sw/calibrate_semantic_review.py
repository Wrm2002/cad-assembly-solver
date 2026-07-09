"""Calibrate whether semantic scores add signal before enabling reranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_pool import review_pool


def _auc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _brier(labels, scores):
    if not labels:
        return None
    return sum(
        (float(score) - int(label)) ** 2
        for label, score in zip(labels, scores)
    ) / len(labels)


def calibration_gate(
    *,
    semantic_auc,
    semantic_brier,
    geometry_brier,
    verdicts,
    holdout,
) -> bool:
    """Return true only when semantic evidence is calibrated and harmless."""
    return bool(
        semantic_auc is not None
        and semantic_auc >= 0.70
        and semantic_brier is not None
        and geometry_brier is not None
        and semantic_brier < geometry_brier
        and {"accept", "reject", "abstain"}.issubset(set(verdicts))
        and holdout.get("auto_accept_precision_not_decreased") is True
        and holdout.get("false_positive_count_not_increased") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="mixed_pools_v1")
    parser.add_argument(
        "--pools",
        nargs="+",
        default=["pool_001", "pool_002", "pool_003"],
    )
    parser.add_argument(
        "--mode",
        choices=("live", "cache_only"),
        default="live",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "pool_pipeline.json"),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parent / "configs" / "semantic_calibration.json"
        ),
    )
    parser.add_argument(
        "--holdout-report",
        help=(
            "Optional JSON containing "
            "auto_accept_precision_not_decreased and "
            "false_positive_count_not_increased."
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = []
    for pool_name in args.pools:
        pool = root / pool_name
        report = review_pool(pool, args.config, mode=args.mode)
        gt = json.loads(
            (pool / "pool_gt.json").read_text(encoding="utf-8")
        )
        truth = {
            frozenset(group["parts"]) for group in gt["true_groups"]
        }
        proposal_parts = {
            item["group_id"]: frozenset(item["parts"])
            for item in json.loads(
                (pool / "grouping" / "group_proposals.json").read_text(
                    encoding="utf-8"
                )
            )
        }
        for review in report["reviews"]:
            decision = review["decision"]
            rows.append(
                {
                    "pool_id": pool_name,
                    "proposal_id": decision["proposal_id"],
                    "label": int(
                        proposal_parts[decision["proposal_id"]] in truth
                    ),
                    "geometry_score": review["request_summary"][
                        "geometry_group_score"
                    ],
                    "semantic_score": decision["plausibility_score"],
                    "semantic_confidence": decision["confidence"],
                    "verdict": decision["verdict"],
                    "cache_hit": review["cache_hit"],
                    "tokens": review.get("usage", {}).get(
                        "total_tokens", 0
                    ),
                }
            )
        print(f"{pool_name}: reviews={report['review_count']}", flush=True)

    labels = [row["label"] for row in rows]
    geometry_scores = [row["geometry_score"] for row in rows]
    semantic_scores = [row["semantic_score"] for row in rows]
    geometry_auc = _auc(labels, geometry_scores)
    semantic_auc = _auc(labels, semantic_scores)
    brier = _brier(labels, semantic_scores)
    geometry_brier = _brier(labels, geometry_scores)
    verdicts = {row["verdict"] for row in rows}
    score_range = (
        max(semantic_scores) - min(semantic_scores)
        if semantic_scores
        else 0.0
    )
    holdout = {}
    if args.holdout_report:
        holdout_path = Path(args.holdout_report).resolve()
        if holdout_path.is_file():
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    required_verdicts = {"accept", "reject", "abstain"}
    enabled = calibration_gate(
        semantic_auc=semantic_auc,
        semantic_brier=brier,
        geometry_brier=geometry_brier,
        verdicts=verdicts,
        holdout=holdout,
    )
    report = {
        "schema_version": "1.0.0",
        "calibration_pools": args.pools,
        "holdout_pools": [
            path.name
            for path in sorted(root.iterdir())
            if path.is_dir() and path.name not in args.pools
        ],
        "review_count": len(rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "geometry_auc": geometry_auc,
        "semantic_auc": semantic_auc,
        "semantic_brier_score": brier,
        "geometry_brier_score": geometry_brier,
        "semantic_score_range": score_range,
        "verdicts": sorted(verdicts),
        "total_tokens": sum(row["tokens"] for row in rows),
        "semantic_reranking_enabled": enabled,
        "semantic_application_mode": (
            "rerank" if enabled else "explanation_only"
        ),
        "holdout_safety": holdout,
        "gate_rules": {
            "minimum_semantic_auc": 0.70,
            "brier_must_improve_over_geometry": True,
            "required_verdicts": sorted(required_verdicts),
            "holdout_auto_accept_precision_not_decreased": True,
            "holdout_false_positive_count_not_increased": True,
        },
        "rows": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "review_count",
                    "positive_count",
                    "negative_count",
                    "geometry_auc",
                    "semantic_auc",
                    "semantic_brier_score",
                    "semantic_score_range",
                    "verdicts",
                    "total_tokens",
                    "semantic_reranking_enabled",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
