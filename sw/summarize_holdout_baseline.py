"""Freeze the first no-tuning evaluation of the locked CAD holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-root",
        default=str(here / "data" / "functional_cad_holdout_v1"),
    )
    parser.add_argument(
        "--pools-root",
        default=str(here / "data" / "functional_cad_holdout_pools_v1"),
    )
    parser.add_argument(
        "--results-root",
        default=str(here / "data" / "functional_cad_holdout_results_v1"),
    )
    args = parser.parse_args()
    holdout = Path(args.holdout_root).resolve()
    pools = Path(args.pools_root).resolve()
    results = Path(args.results_root).resolve()
    lock_bytes = (holdout / "holdout_lock.json").read_bytes()
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    pool_manifest = _load(pools / "mixed_pool_manifest.json")
    if pool_manifest["holdout_lock_sha256"] != lock_sha:
        raise ValueError("holdout lock mismatch; baseline is not auditable")

    recall = _load(results / "candidate_recall_by_type.json")
    truth_interfaces = sum(
        row["truth_interfaces"] for row in recall.values()
    )
    generated_interfaces = sum(
        row["generated"] for row in recall.values()
    )
    kept_interfaces = sum(
        row["kept_after_pruning"] for row in recall.values()
    )
    recall_summary = {
        "truth_interfaces": truth_interfaces,
        "generated": generated_interfaces,
        "kept_after_pruning": kept_interfaces,
        "generated_recall": (
            generated_interfaces / truth_interfaces
            if truth_interfaces
            else None
        ),
        "post_pruning_recall": (
            kept_interfaces / truth_interfaces
            if truth_interfaces
            else None
        ),
        "by_type": recall,
    }
    group = _load(results / "functional_group_metrics.json")
    pose = _load(results / "true_group_pose_audit.json")
    negatives = _load(results / "forced_hard_negative_audit.json")
    conservative = _load(results / "conservative_metrics.json")
    signoff = _load(holdout / "engineering_signoff.json")
    payload = {
        "schema_version": "1.0.0",
        "artifact_role": "locked_holdout_first_evaluation_no_tuning",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_lock_sha256": lock_sha,
        "used_for_rule_tuning": False,
        "truth_status": (
            "provisional_pending_qualified_mechanical_engineer_signoff"
        ),
        "candidate_recall": recall_summary,
        "group_metrics": {
            "group_proposal_recall": group["group_proposal_recall"],
            "group_recall_at_k": group["group_recall_at_k"],
            "review_frontier_recall": group["review_frontier_recall"],
            "review_frontier_precision": group[
                "review_frontier_precision"
            ],
        },
        "pose_metrics": {
            "true_group_count": pose["true_group_count"],
            "pose_valid_count": pose["pose_valid_count"],
            "pose_failed_count": pose["pose_failed_count"],
            "true_group_pose_recall": pose["true_group_pose_recall"],
        },
        "hard_negative_metrics": {
            "hard_negative_count": negatives["hard_negative_count"],
            "production_edge_available_count": negatives[
                "production_edge_available_count"
            ],
            "d5_executed_count": negatives["d5_executed_count"],
            "auto_accepted_functional_false_positive_count": negatives[
                "auto_accepted_functional_false_positive_count"
            ],
            "decision_counts": negatives["decision_counts"],
        },
        "conservative_metrics": conservative,
        "deepseek_enabled": False,
        "mechanical_engineer_signoff_gate_passed": signoff[
            "gate_passed"
        ],
        "interpretation": [
            "Interface recall generalized to the locked modeled topologies.",
            "Group ranking and pose reconstruction did not generalize strongly.",
            "No holdout hard negative was automatically accepted.",
            "Metrics remain provisional until qualified engineer sign-off.",
            "This holdout must not be used for further rule tuning.",
        ],
    }
    output = results / "locked_holdout_baseline.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (results / "locked_holdout_baseline.md").write_text(
        "\n".join(
            [
                "# Locked CAD Holdout Baseline",
                "",
                "- Policy: first evaluation, no tuning",
                f"- Holdout lock: `{lock_sha}`",
                "- Candidate interface recall: "
                f"{recall_summary['generated_recall']:.2%} generated / "
                f"{recall_summary['post_pruning_recall']:.2%} "
                "post-pruning",
                "- Group proposal recall: "
                f"{group['group_proposal_recall']:.2%}",
                "- Review frontier recall: "
                f"{group['review_frontier_recall']:.2%}",
                "- True-group pose recall: "
                f"{pose['true_group_pose_recall']:.2%}",
                "- Hard-negative auto accepts: "
                f"{negatives['auto_accepted_functional_false_positive_count']}",
                "- DeepSeek: disabled",
                "- Mechanical-engineer sign-off: pending",
                "",
                "These metrics are provisional until the 12-row engineering "
                "review gate passes. Do not tune on this holdout.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
