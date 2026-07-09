"""Audit controlled functional hard negatives against conservative decisions.

The audit treats functional validity as truth.  A negative pair is considered
exposed by a decision when both of its parts occur in that decision, including
larger candidate groups.  This catches a bad relation hidden inside a superset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DECISION_FILES = {
    "accepted": "final_accepted_groups.json",
    "review": "final_review_groups.json",
    "rejected": "final_rejected_groups.json",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _decision_index(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for decision, filename in DECISION_FILES.items():
        path = results_dir / filename
        for row in _load_json(path) if path.exists() else []:
            pool_id = str(row.get("pool_id", ""))
            indexed.setdefault(pool_id, []).append(
                {
                    "decision": decision,
                    "candidate_id": row.get("group_id") or row.get("candidate_id"),
                    "parts": set(row.get("parts", [])),
                    "decision_reasons": row.get("decision_reasons", []),
                }
            )
    return indexed


def audit(pool_root: Path, results_dir: Path) -> list[dict[str, Any]]:
    decisions = _decision_index(results_dir)
    rows: list[dict[str, Any]] = []
    priority = {"accepted": 3, "review": 2, "rejected": 1}

    for pool_dir in sorted(path for path in pool_root.iterdir() if path.is_dir()):
        gt_path = pool_dir / "pool_gt.json"
        if not gt_path.exists():
            continue
        gt = _load_json(gt_path)
        pool_id = str(gt["pool_id"])
        for negative in gt.get("functional_negative_groups", []):
            negative_parts = set(negative["parts"])
            exposed = [
                row
                for row in decisions.get(pool_id, [])
                if negative_parts.issubset(row["parts"])
            ]
            exact = [row for row in exposed if row["parts"] == negative_parts]
            accepted_exposure = [
                row for row in exposed if row["decision"] == "accepted"
            ]
            review_exposure = [
                row for row in exposed if row["decision"] == "review"
            ]
            rejected_exposure = [
                row for row in exposed if row["decision"] == "rejected"
            ]
            if accepted_exposure:
                strongest = accepted_exposure[0]
                outcome = "accepted"
            elif exact:
                strongest = max(
                    exact, key=lambda item: priority[item["decision"]]
                )
                outcome = strongest["decision"]
            elif review_exposure:
                strongest = review_exposure[0]
                outcome = "review_embedded"
            elif rejected_exposure:
                strongest = rejected_exposure[0]
                outcome = "rejected_embedded"
            else:
                strongest = None
                outcome = "not_proposed"

            rows.append(
                {
                    "pool_id": pool_id,
                    "split": gt.get("split"),
                    "negative_id": negative["negative_id"],
                    "negative_type": negative["negative_type"],
                    "parts": sorted(negative_parts),
                    "geometry_feasible": bool(negative.get("geometry_feasible")),
                    "functional_validity": negative.get("functional_validity"),
                    "weak_evidence_only": bool(negative.get("weak_evidence_only", False)),
                    "final_outcome": outcome,
                    "auto_accepted_false_positive": bool(
                        accepted_exposure
                    ),
                    "exact_candidate_seen": bool(exact),
                    "containing_candidate_count": len(exposed),
                    "representative_candidate_id": (
                        strongest["candidate_id"] if strongest else None
                    ),
                    "reason": negative.get("reason"),
                }
            )
    return rows


def write_outputs(rows: list[dict[str, Any]], results_dir: Path) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "functional_hard_negative_audit.csv"
    json_path = results_dir / "functional_hard_negative_audit.json"
    md_path = results_dir / "functional_hard_negative_audit.md"

    fieldnames = [
        "pool_id",
        "split",
        "negative_id",
        "negative_type",
        "parts",
        "geometry_feasible",
        "functional_validity",
        "weak_evidence_only",
        "final_outcome",
        "auto_accepted_false_positive",
        "exact_candidate_seen",
        "containing_candidate_count",
        "representative_candidate_id",
        "reason",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["parts"] = "|".join(row["parts"])
            writer.writerow(csv_row)

    outcome_counts = Counter(row["final_outcome"] for row in rows)
    by_type: dict[str, Counter[str]] = {}
    for row in rows:
        by_type.setdefault(row["negative_type"], Counter())[row["final_outcome"]] += 1
    summary = {
        "schema_version": "1.0.0",
        "truth_basis": "functional_validity",
        "negative_count": len(rows),
        "auto_accepted_false_positive_count": sum(
            bool(row["auto_accepted_false_positive"]) for row in rows
        ),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "outcomes_by_negative_type": {
            key: dict(sorted(value.items())) for key, value in sorted(by_type.items())
        },
        "records": rows,
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Functional Hard-Negative Audit",
        "",
        "- Truth basis: functional validity (not source ID)",
        f"- Controlled negatives: {len(rows)}",
        (
            "- Auto-accepted functional false positives: "
            f"{summary['auto_accepted_false_positive_count']}"
        ),
        f"- Outcomes: {dict(sorted(outcome_counts.items()))}",
        "",
        "| Pool | Negative type | Outcome | Exact candidate | Containing candidates |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pool_id']} | {row['negative_type']} | "
            f"{row['final_outcome']} | {row['exact_candidate_seen']} | "
            f"{row['containing_candidate_count']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.pool_root.resolve(), args.results_dir.resolve())
    summary = write_outputs(rows, args.results_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
