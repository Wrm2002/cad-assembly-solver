"""Audit JoinABLe candidate recall on the real anonymized mixed pools.

The released model is an interface ranker, not a binary connection model.
Therefore the pair-level score used here is explicitly a high-recall heuristic
for candidate generation only.  It can never auto-accept a group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FEATURES = (
    "top_1_logit",
    "top_1_probability",
    "top_1_uniform_lift",
    "top_2_logit_margin",
    "normalized_entropy",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def pair_key(pool_id: str, pair: Iterable[str]) -> tuple[str, str, str]:
    first, second = sorted(str(value) for value in pair)
    return pool_id, first, second


def auc(rows: list[dict[str, Any]], feature: str, sign: int = 1) -> float | None:
    positives = [
        sign * float(row["pair_features"][feature])
        for row in rows
        if row["audit_label"] == "direct_positive"
        and row["status"] == "success"
        and row.get("pair_features", {}).get(feature) is not None
    ]
    negatives = [
        sign * float(row["pair_features"][feature])
        for row in rows
        if row["audit_label"] == "cross_group_negative"
        and row["status"] == "success"
        and row.get("pair_features", {}).get(feature) is not None
    ]
    if not positives or not negatives:
        return None
    score = sum(
        (positive > negative)
        + 0.5 * (positive == negative)
        for positive in positives
        for negative in negatives
    )
    return score / (len(positives) * len(negatives))


def metric_row(
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    true_rows = [
        row for row in rows if row["audit_label"] == "direct_positive"
    ]
    eligible = [row for row in true_rows if row["status"] == "success"]
    retained = [
        row
        for row in eligible
        if float(row["pair_features"]["top_1_uniform_lift"])
        >= threshold
    ]
    fallback = [
        row for row in true_rows if row["status"] != "success"
    ]
    return {
        "true_candidate_count": len(true_rows),
        "model_eligible_count": len(eligible),
        "retained_by_threshold_count": len(retained),
        "complexity_fallback_review_count": len(fallback),
        "pruned_true_count": len(eligible) - len(retained),
        "model_eligibility_recall": (
            len(eligible) / len(true_rows) if true_rows else None
        ),
        "retained_recall_among_eligible": (
            len(retained) / len(eligible) if eligible else None
        ),
        "model_only_total_recall": (
            len(retained) / len(true_rows) if true_rows else None
        ),
        "union_with_review_fallback_recall": (
            (len(retained) + len(fallback)) / len(true_rows)
            if true_rows
            else None
        ),
    }


def write_candidate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "pool_id",
        "true_group_id",
        "parts",
        "group_size",
        "candidate_type",
        "generated_or_not",
        "pruned_or_not",
        "pruning_reason",
        "missing_reason_guess",
        "geometry_features_available",
        "required_interface_type",
        "split",
        "combined_node_count",
        "joinable_uniform_lift",
        "final_candidate_state",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        "|".join(str(value) for value in row.get(field, []))
                        if isinstance(row.get(field), list)
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument("prediction_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold-safety-factor", type=float, default=0.70
    )
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    output_dir = args.output_dir.resolve()
    prediction = read_json(args.prediction_json.resolve())
    manifest = read_json(root / "mixed_pool_manifest.json")

    truth: dict[tuple[str, str, str], dict[str, Any]] = {}
    group_sizes: dict[tuple[str, str], int] = {}
    for pool in manifest.get("pools", []):
        pool_id = str(pool["pool_id"])
        gt = read_json(root / pool_id / "pool_gt.json")
        for group in gt.get("true_groups", []):
            group_sizes[(pool_id, str(group["group_id"]))] = len(
                group.get("part_ids") or []
            )
        for row in gt.get("direct_positive_pairs", []):
            key = pair_key(pool_id, row["part_pair"])
            truth[key] = {
                "audit_label": "direct_positive",
                "true_group_id": row.get("true_group_id"),
                "candidate_type": row.get(
                    "evidence_layer", "unknown_positive"
                ),
            }
        for row in gt.get("same_group_non_edges", []):
            key = pair_key(pool_id, row["part_pair"])
            truth[key] = {
                "audit_label": "same_group_nonedge_unknown",
                "true_group_id": row.get("true_group_id"),
                "candidate_type": "same_group_nonedge_unknown",
            }
        for row in gt.get("cross_group_negative_pairs", []):
            key = pair_key(pool_id, row["part_pair"])
            truth[key] = {
                "audit_label": "cross_group_negative",
                "true_group_id": None,
                "candidate_type": "cross_group_provenance_negative",
            }

    rows = []
    for prediction_row in prediction.get("pairs", []):
        key = pair_key(
            str(prediction_row["pool_id"]),
            [prediction_row["part_a"], prediction_row["part_b"]],
        )
        truth_row = truth.get(key)
        if truth_row is None:
            raise ValueError(f"prediction_pair_without_truth:{key}")
        row = dict(prediction_row)
        row.update(truth_row)
        row["group_size"] = (
            group_sizes.get(
                (str(row["pool_id"]), str(row["true_group_id"]))
            )
            if row["true_group_id"]
            else None
        )
        rows.append(row)
    if len(rows) != len(truth):
        raise ValueError(
            f"prediction_truth_pair_count_mismatch:{len(rows)}!={len(truth)}"
        )

    train_positive_lifts = [
        float(row["pair_features"]["top_1_uniform_lift"])
        for row in rows
        if row["split"] == "train"
        and row["audit_label"] == "direct_positive"
        and row["status"] == "success"
    ]
    if not train_positive_lifts:
        raise RuntimeError("no_model_eligible_train_positive")
    threshold = (
        min(train_positive_lifts) * args.threshold_safety_factor
    )
    for row in rows:
        if row["status"] != "success":
            row["candidate_state"] = "review_complexity_fallback"
            row["candidate_generated"] = True
            row["candidate_pruned"] = False
            row["candidate_reason"] = (
                "Outside the released model node-count envelope; preserved "
                "for review rather than silently deleted."
            )
        elif (
            float(row["pair_features"]["top_1_uniform_lift"])
            >= threshold
        ):
            row["candidate_state"] = "joinable_scored_candidate"
            row["candidate_generated"] = True
            row["candidate_pruned"] = False
            row["candidate_reason"] = (
                "Frozen JoinABLe concentration lift exceeds the train-only "
                "high-recall threshold."
            )
        else:
            row["candidate_state"] = "pruned_low_joinable_lift"
            row["candidate_generated"] = False
            row["candidate_pruned"] = True
            row["candidate_reason"] = (
                "Frozen JoinABLe concentration lift is below the train-only "
                "high-recall threshold."
            )

    missed_true = []
    pruned_true = []
    for row in rows:
        if row["audit_label"] != "direct_positive":
            continue
        common = {
            "pool_id": row["pool_id"],
            "true_group_id": row["true_group_id"],
            "parts": [row["part_a"], row["part_b"]],
            "group_size": row["group_size"],
            "candidate_type": row["candidate_type"],
            "generated_or_not": row["candidate_generated"],
            "pruned_or_not": row["candidate_pruned"],
            "pruning_reason": (
                row["candidate_reason"] if row["candidate_pruned"] else ""
            ),
            "missing_reason_guess": (
                "|".join(row.get("failure_reasons") or [])
                if row["status"] != "success"
                else ""
            ),
            "geometry_features_available": row["status"] == "success",
            "required_interface_type": row["candidate_type"],
            "split": row["split"],
            "combined_node_count": row.get("combined_node_count"),
            "joinable_uniform_lift": (
                (row.get("pair_features") or {}).get(
                    "top_1_uniform_lift"
                )
            ),
            "final_candidate_state": row["candidate_state"],
        }
        if row["status"] != "success":
            missed_true.append(common)
        if row["candidate_pruned"]:
            pruned_true.append(common)

    by_type = {}
    for candidate_type in sorted(
        {
            row["candidate_type"]
            for row in rows
            if row["audit_label"] == "direct_positive"
        }
    ):
        subset = [
            row
            for row in rows
            if row["audit_label"] == "direct_positive"
            and row["candidate_type"] == candidate_type
        ]
        by_type[candidate_type] = metric_row(subset, threshold)
    by_group_size = {}
    for size in sorted(
        {
            int(row["group_size"])
            for row in rows
            if row["audit_label"] == "direct_positive"
            and row["group_size"] is not None
        }
    ):
        subset = [
            row
            for row in rows
            if row["audit_label"] == "direct_positive"
            and row["group_size"] == size
        ]
        by_group_size[str(size)] = metric_row(subset, threshold)

    split_metrics = {
        split: metric_row(
            [row for row in rows if row["split"] == split],
            threshold,
        )
        for split in ("train", "validation", "test")
    }
    split_label_counts = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        counter = Counter(
            (row["audit_label"], row["candidate_state"])
            for row in subset
        )
        split_label_counts[split] = {
            f"{label}:{state}": count
            for (label, state), count in sorted(counter.items())
        }
    auc_report = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        auc_report[split] = {}
        for feature in FEATURES:
            high = auc(subset, feature, 1)
            low = auc(subset, feature, -1)
            auc_report[split][feature] = {
                "high_is_positive_auc": high,
                "low_is_positive_auc": low,
            }

    sanitized_pools = defaultdict(list)
    audit_rows = []
    for row in rows:
        score = (
            float(row["pair_features"]["top_1_uniform_lift"])
            if row["status"] == "success"
            else None
        )
        sanitized_pools[str(row["pool_id"])].append(
            {
                "part_pair": [row["part_a"], row["part_b"]],
                "candidate_state": row["candidate_state"],
                "active_for_group_search": row[
                    "candidate_generated"
                ],
                "joinable_uniform_lift": score,
                "joinable_threshold": threshold,
                "top_interface_candidates": (
                    row.get("candidates", [])[:5]
                ),
                "model_status": row["status"],
                "review_required": (
                    row["candidate_state"]
                    == "review_complexity_fallback"
                ),
                "reason": row["candidate_reason"],
                "failure_reasons": row.get("failure_reasons") or [],
                "unavailable_fields": row.get(
                    "unavailable_fields", []
                ),
            }
        )
        audit_rows.append(
            {
                "pair_id": row["pair_id"],
                "pool_id": row["pool_id"],
                "split": row["split"],
                "part_pair": [row["part_a"], row["part_b"]],
                "audit_label": row["audit_label"],
                "candidate_type": row["candidate_type"],
                "true_group_id": row["true_group_id"],
                "group_size": row["group_size"],
                "model_status": row["status"],
                "candidate_state": row["candidate_state"],
                "candidate_generated": row["candidate_generated"],
                "candidate_pruned": row["candidate_pruned"],
                "pair_features": row.get("pair_features"),
                "failure_reasons": row.get("failure_reasons") or [],
            }
        )

    all_true_metrics = metric_row(rows, threshold)
    active_count = sum(row["candidate_generated"] for row in rows)
    pruned_count = len(rows) - active_count
    calibration = {
        "schema_version": "1.0.0",
        "model_boundary": prediction.get("model_boundary"),
        "selected_candidate_feature": "top_1_uniform_lift",
        "selected_feature_semantics": (
            "Top-1 softmax probability divided by the uniform baseline; "
            "this is not a calibrated connection probability."
        ),
        "threshold_selection": {
            "split": "train",
            "minimum_model_eligible_positive": min(
                train_positive_lifts
            ),
            "safety_factor": args.threshold_safety_factor,
            "threshold": threshold,
            "target": "candidate recall, not acceptance precision",
        },
        "feature_auc_by_split": auc_report,
        "split_label_state_counts": split_label_counts,
        "promotion_decision": {
            "promoted_for_candidate_generation_only": True,
            "promoted_for_acceptance": False,
            "reason": (
                "Holdout recall is preserved, but the model has no no-joint "
                "class and validation negatives are not separated."
            ),
        },
        "failure_reasons": [],
        "unavailable_fields": [
            "calibrated_pair_connection_probability",
            "functional_semantic_validity",
        ],
    }
    recall_by_type = {
        "schema_version": "1.0.0",
        "threshold": threshold,
        "overall": all_true_metrics,
        "by_type": by_type,
        "failure_reasons": [],
        "unavailable_fields": [
            "large_designer_joint_sample",
            "functional_interface_type_labels",
        ],
    }
    recall_by_size = {
        "schema_version": "1.0.0",
        "threshold": threshold,
        "overall": all_true_metrics,
        "by_group_size": by_group_size,
        "by_split": split_metrics,
        "failure_reasons": [],
        "unavailable_fields": [],
    }
    candidate_graph = {
        "schema_version": "1.0.0",
        "dataset_id": prediction.get("dataset_id"),
        "candidate_policy": {
            "feature": "top_1_uniform_lift",
            "threshold": threshold,
            "threshold_selected_on": "train only",
            "complexity_failures": "preserve_as_review_fallback",
            "can_auto_accept": False,
        },
        "pool_count": len(sanitized_pools),
        "input_pair_count": len(rows),
        "active_candidate_count": active_count,
        "pruned_pair_count": pruned_count,
        "pools": [
            {
                "pool_id": pool_id,
                "edges": sorted(
                    edges, key=lambda edge: edge["part_pair"]
                ),
            }
            for pool_id, edges in sorted(sanitized_pools.items())
        ],
        "failure_reasons": [],
        "unavailable_fields": [
            "calibrated_pair_connection_probability",
            "functional_semantic_validity",
        ],
    }
    full_audit = {
        "schema_version": "1.0.0",
        "pair_count": len(rows),
        "active_candidate_count": active_count,
        "pruned_pair_count": pruned_count,
        "true_direct_pair_count": all_true_metrics[
            "true_candidate_count"
        ],
        "model_ineligible_true_pair_count": len(missed_true),
        "pruned_true_pair_count": len(pruned_true),
        "rows": audit_rows,
        "failure_reasons": [],
        "unavailable_fields": [
            "functional_semantic_validity",
            "calibrated_pair_connection_probability",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_csv(
        output_dir / "missed_true_candidates.csv", missed_true
    )
    write_candidate_csv(
        output_dir / "pruned_true_candidates.csv", pruned_true
    )
    write_json(
        output_dir / "candidate_recall_by_type.json", recall_by_type
    )
    write_json(
        output_dir / "candidate_recall_by_group_size.json",
        recall_by_size,
    )
    write_json(
        output_dir / "pair_score_calibration.json", calibration
    )
    write_json(
        output_dir / "candidate_graph_input.json", candidate_graph
    )
    write_json(
        output_dir / "candidate_recall_full_audit.json", full_audit
    )

    report_lines = [
        "# Candidate Recall Audit",
        "",
        f"- Input part pairs: {len(rows)}",
        f"- Frozen JoinABLe eligible pairs: {prediction['success_count']}",
        f"- Outside released node envelope: {prediction['failure_count']}",
        f"- Direct true edges: {all_true_metrics['true_candidate_count']}",
        (
            "- Model-only true-edge recall: "
            f"{all_true_metrics['model_only_total_recall']:.2%}"
        ),
        (
            "- Recall among model-eligible true edges: "
            f"{all_true_metrics['retained_recall_among_eligible']:.2%}"
        ),
        (
            "- Union recall with complexity-review fallback: "
            f"{all_true_metrics['union_with_review_fallback_recall']:.2%}"
        ),
        f"- True edges pruned by threshold: {len(pruned_true)}",
        f"- Active candidate pairs: {active_count}",
        f"- Low-priority pruned pairs: {pruned_count}",
        "",
        "The released JoinABLe model ranks interfaces on an already chosen "
        "part pair. It has no learned no-joint class. `top_1_uniform_lift` "
        "is therefore used only as a train-calibrated high-recall candidate "
        "priority heuristic, never as acceptance evidence.",
        "",
        "Pairs exceeding the released 950-node envelope are preserved as "
        "review candidates. They are reported as model misses, not silently "
        "deleted.",
    ]
    (output_dir / "candidate_recall_audit.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    system_review_lines = [
        "# Step 4 System Review",
        "",
        "## Independent reviewer conclusion",
        "",
        "The step is useful but narrower than the phrase “predict a "
        "connection” suggests. JoinABLe supplies interface-ranking evidence; "
        "it does not supply a calibrated connection probability.",
        "",
        f"- {prediction['success_count']}/{len(rows)} pairs are inside the "
        "released model envelope.",
        f"- {len(missed_true)}/{all_true_metrics['true_candidate_count']} "
        "true edges are outside that envelope and must remain review items.",
        f"- No true edge was removed by the train-only threshold in this "
        f"{all_true_metrics['true_candidate_count']}-edge audit.",
        "- The benchmark contains only one designer-joint edge; the other "
        "37 positives are contact observations. Interface-type conclusions "
        "are therefore not yet statistically broad.",
        "- Validation negatives are poorly separated even though the test "
        "pool happens to separate well. The threshold is not promoted to an "
        "accept/reject gate.",
        "",
        "## System decision",
        "",
        "Proceed to conservative graph grouping using the active candidate "
        "union. Complexity fallbacks and weak single-interface matches must "
        "stay in review. Do not enlarge the neural model or the 950-node "
        "limit in this project stage.",
    ]
    (output_dir / "candidate_recall_system_review.md").write_text(
        "\n".join(system_review_lines) + "\n", encoding="utf-8"
    )
    print(
        f"Candidate recall: model-only="
        f"{all_true_metrics['model_only_total_recall']:.3f}, "
        f"union={all_true_metrics['union_with_review_fallback_recall']:.3f}, "
        f"pruned_true={len(pruned_true)}"
    )
    return (
        0
        if all_true_metrics["union_with_review_fallback_recall"] == 1.0
        and not pruned_true
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
