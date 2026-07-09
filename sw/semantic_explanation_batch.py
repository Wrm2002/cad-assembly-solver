"""Generate bounded DeepSeek explanations without changing group decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from contracts import GroupProposal
from semantic_pool import build_summary
from semantic_review import DeepSeekReviewer


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _semantic_validity(decision: dict[str, Any]) -> str:
    if decision["verdict"] == "abstain" or decision["confidence"] < 0.5:
        return "unknown"
    if decision["verdict"] == "accept":
        return "high" if decision["plausibility_score"] >= 0.75 else "medium"
    return "low"


def run(
    pools_root: str | Path,
    results_dir: str | Path,
    pipeline_config_path: str | Path,
    conservative_config_path: str | Path,
    *,
    mode: str = "live",
) -> dict[str, Any]:
    root = Path(pools_root).resolve()
    results = Path(results_dir).resolve()
    pipeline_config = _load(Path(pipeline_config_path).resolve())
    conservative = _load(Path(conservative_config_path).resolve())
    review_groups = _load(results / "final_review_groups.json")
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in review_groups:
        if item.get("pose", {}).get("final_pose_status") == "valid":
            by_pool[item["pool_id"]].append(item)
    limit = int(conservative["maximum_semantic_explanations_per_pool"])
    inputs, reviews = [], []
    reviewer = DeepSeekReviewer(
        pipeline_config["semantic_review"],
        results / "deepseek_cache",
    )
    for pool_id, items in sorted(by_pool.items()):
        pool = root / pool_id
        part_features = _load(pool / "index" / "part_features.json")
        part_map = {item["part_id"]: item for item in part_features}
        edges = _load(pool / "index" / "pruned_candidates.json")
        edge_map = {item["candidate_id"]: item for item in edges}
        selected = sorted(
            items,
            key=lambda item: (
                float(item["geometry_score"]),
                float(
                    item["consistency"]["group_consistency_score"]
                ),
            ),
            reverse=True,
        )[:limit]
        for item in selected:
            proposal = GroupProposal.model_validate(
                {
                    key: item[key]
                    for key in (
                        "schema_version",
                        "group_id",
                        "parts",
                        "candidate_edges",
                        "geometry_score",
                        "connected",
                        "status",
                        "reasons",
                    )
                }
            )
            summary = build_summary(
                proposal, part_map, edge_map, len(part_map)
            )
            summary["hard_geometry_status"] = (
                "physical_pose_valid_but_provenance_unknown"
            )
            summary["group_consistency"] = item["consistency"]
            summary["decision_boundary"] = {
                "mode": "explanation_only",
                "may_change_grouping": False,
                "may_change_acceptance": False,
                "human_label_required": True,
            }
            inputs.append({"pool_id": pool_id, **summary})
            print(
                f"DeepSeek explanation: {pool_id}/{item['group_id']}",
                flush=True,
            )
            record = reviewer.review(summary, mode=mode)
            decision = record["decision"]
            reviews.append(
                {
                    "schema_version": "1.0.0",
                    "pool_id": pool_id,
                    "candidate_id": item["group_id"],
                    "semantic_validity": _semantic_validity(decision),
                    "semantic_score": decision["plausibility_score"],
                    "functional_reason": decision["explanation"],
                    "possible_system": "unknown_without_reliable_metadata",
                    "risk": decision.get("risk_flags", []),
                    "is_geometrically_feasible_but_semantically_invalid": (
                        decision["verdict"] == "reject"
                    ),
                    "review_required": True,
                    "suggested_action": "review",
                    "raw_verdict": decision["verdict"],
                    "semantic_confidence": decision["confidence"],
                    "reason_codes": decision["reason_codes"],
                    "application_mode": "explanation_only",
                    "affected_final_decision": False,
                    "cache_hit": record.get("cache_hit", False),
                    "usage": record.get("usage", {}),
                }
            )
    _write(results / "semantic_inputs.json", inputs)
    _write(results / "semantic_reviews.json", reviews)
    old_calibration = _load(
        Path(__file__).parent / "configs" / "semantic_calibration.json"
    )
    calibration_report = {
        **old_calibration,
        "semantic_reranking_enabled": False,
        "semantic_application_mode": "explanation_only",
        "current_gate_rules": {
            "minimum_semantic_auc": 0.70,
            "brier_must_improve_over_geometry": True,
            "required_verdicts": ["accept", "reject", "abstain"],
            "holdout_auto_accept_precision_not_decreased": True,
            "holdout_false_positive_count_not_increased": True,
        },
        "gate_failure_reasons": [
            "historical semantic AUC is below 0.70",
            "historical verdict set lacks reject",
            "historical Brier score does not beat geometry",
            "new human semantic labels are not available yet",
        ],
        "explanation_batch_count": len(reviews),
    }
    _write(results / "semantic_calibration_report.json", calibration_report)
    gate_report = "\n".join(
        [
            "# Semantic Gate Decision",
            "",
            "**Decision: CLOSED — explanation-only.**",
            "",
            f"- Explanations generated: {len(reviews)}",
            f"- Historical AUC: {old_calibration.get('semantic_auc')}",
            f"- Historical Brier: "
            f"{old_calibration.get('semantic_brier_score')}",
            "- Human labels from the visual review pack are required before "
            "recalibration.",
            "- No semantic score changed ranking, acceptance, or rejection.",
            "",
        ]
    )
    (results / "semantic_gate_decision.md").write_text(
        gate_report, encoding="utf-8"
    )
    return {
        "semantic_inputs": len(inputs),
        "semantic_reviews": len(reviews),
        "semantic_reranking_enabled": False,
        "application_mode": "explanation_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="mixed_pools_v1")
    parser.add_argument(
        "--results",
        default=str(Path(__file__).parent / "data" / "results"),
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(
            Path(__file__).parent / "configs" / "pool_pipeline.json"
        ),
    )
    parser.add_argument(
        "--conservative-config",
        default=str(
            Path(__file__).parent
            / "configs"
            / "conservative_pipeline.json"
        ),
    )
    parser.add_argument(
        "--mode", choices=("live", "cache_only", "off"), default="live"
    )
    args = parser.parse_args()
    result = run(
        args.root,
        args.results,
        args.pipeline_config,
        args.conservative_config,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
