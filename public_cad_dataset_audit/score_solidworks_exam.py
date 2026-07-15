"""Step 5 scoring script: evaluate solidworks_assembly_plan.json against human_labels.json.

CRITICAL: This script reads human_labels.json ONLY for scoring, NOT for inference.
It must be run AFTER the Step 3+4 pipeline has completed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CASES = ["1", "2", "3", "4", "5"]
OUTPUT_DIR = Path("public_cad_dataset_audit/outputs/step34_solidworks_plan")
LABELS_DIR = Path("sw/phase5_annotation_pack")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_case(case_id: str) -> dict[str, Any]:
    """Score one case by comparing system output with human_labels."""
    # Load system output
    plan_path = OUTPUT_DIR / f"case_{case_id}" / "solidworks_assembly_plan.json"
    if not plan_path.exists():
        return {"case_id": case_id, "error": "plan_not_found"}

    plan = load_json(plan_path)

    # Load human labels (scoring only!)
    labels_path = LABELS_DIR / f"case_{case_id}" / "human_labels.json"
    if not labels_path.exists():
        return {"case_id": case_id, "error": "human_labels_not_found"}

    human = load_json(labels_path)
    true_pairs = set()
    true_relations: dict[tuple[str, str], list[str]] = {}
    for rel in human.get("pass_1_direct_relations", []):
        pair = tuple(sorted(rel["parts"]))
        true_pairs.add(pair)
        true_relations[pair] = rel.get("relation_types", [])

    system_parts = [p["file_name"] for p in plan["input_parts"]]

    # Compare accepted edges
    accepted_edges = plan.get("accepted_edges", [])
    review_edges = plan.get("review_edges", [])
    rejected_edges = plan.get("rejected_edges", [])

    # True positives: accepted edges that match true pairs
    tp = 0
    fp = 0
    tp_details = []
    fp_details = []

    for edge in accepted_edges:
        pair = tuple(sorted(edge["parts"]))
        if pair in true_pairs:
            tp += 1
            tp_details.append({
                "pair": list(pair),
                "system_relation": edge.get("relation_type"),
                "true_relations": true_relations.get(pair, []),
                "relation_match": edge.get("relation_type") in true_relations.get(pair, []),
            })
        else:
            fp += 1
            fp_details.append({"pair": list(pair), "system_relation": edge.get("relation_type")})

    # False negatives: true pairs not in accepted edges
    fn = 0
    fn_details = []
    system_accepted_pairs = {tuple(sorted(e["parts"])) for e in accepted_edges}
    for true_pair in true_pairs:
        if true_pair not in system_accepted_pairs:
            fn += 1
            fn_details.append({
                "pair": list(true_pair),
                "in_review": true_pair in {tuple(sorted(e["parts"])) for e in review_edges},
                "in_rejected": true_pair in {tuple(sorted(e["parts"])) for e in rejected_edges},
            })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "case_id": case_id,
        "part_count": len(system_parts),
        "true_direct_edges": len(true_pairs),
        "accepted_count": len(accepted_edges),
        "review_count": len(review_edges),
        "rejected_count": len(rejected_edges),
        "unresolved_count": len(plan.get("unresolved_parts", [])),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_details": tp_details,
        "fp_details": fp_details,
        "fn_details": fn_details,
        "auto_accept_precision": precision,
        "false_positive_count": fp,
        "review_rate": len(review_edges) / max(len(accepted_edges) + len(review_edges) + len(rejected_edges), 1),
        "unresolved_parts_count": len(plan.get("unresolved_parts", [])),
    }


def main() -> int:
    print("=" * 60)
    print("Step 5 Scoring Report")
    print("human_labels.json read ONLY for scoring, NOT for inference")
    print("=" * 60)

    results = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_accepted = 0
    total_review = 0
    total_rejected = 0

    for case_id in CASES:
        result = score_case(case_id)
        results[case_id] = result

        if "error" in result:
            print(f"\nCase {case_id}: ERROR - {result['error']}")
            continue

        print(f"\n--- Case {case_id} ---")
        print(f"  Parts: {result['part_count']}")
        print(f"  True direct edges: {result['true_direct_edges']}")
        print(f"  Accepted: {result['accepted_count']} | Review: {result['review_count']} | Rejected: {result['rejected_count']}")
        print(f"  TP={result['true_positive']} FP={result['false_positive']} FN={result['false_negative']}")
        print(f"  Precision={result['precision']:.3f} Recall={result['recall']:.3f} F1={result['f1']:.3f}")

        if result["fp_details"]:
            print(f"  FALSE POSITIVES: {len(result['fp_details'])}")
        if result["fn_details"]:
            print(f"  FALSE NEGATIVES: {len(result['fn_details'])}")
            for fn in result["fn_details"]:
                status = "in_review" if fn["in_review"] else ("in_rejected" if fn["in_rejected"] else "missing")
                print(f"    - {fn['pair']}: {status}")

        total_tp += result["true_positive"]
        total_fp += result["false_positive"]
        total_fn += result["false_negative"]
        total_accepted += result["accepted_count"]
        total_review += result["review_count"]
        total_rejected += result["rejected_count"]

    # Overall
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print("OVERALL")
    print(f"  Total TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Precision={overall_precision:.3f} Recall={overall_recall:.3f} F1={overall_f1:.3f}")
    print(f"  Accepted={total_accepted} Review={total_review} Rejected={total_rejected}")
    print(f"  Auto-accept precision: {overall_precision:.1%}")
    print(f"  False positive count: {total_fp}")
    print(f"  Review rate: {total_review / max(total_accepted + total_review + total_rejected, 1):.1%}")

    # Save
    report = {
        "scoring_mode": "post_inference_only",
        "human_labels_used_for_inference": False,
        "cases": results,
        "overall": {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "accepted_count": total_accepted,
            "review_count": total_review,
            "rejected_count": total_rejected,
        },
    }

    report_path = OUTPUT_DIR / "scoring_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
