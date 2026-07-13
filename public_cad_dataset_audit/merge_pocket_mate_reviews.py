"""Merge human-reviewed pocket_mate labels back into the Fusion360 benchmark.

Reads the filled review_sheet.csv, matches candidates by sample_id,
and updates the JSONL benchmark files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    base = Path("public_cad_dataset_audit/outputs")
    review_csv = base / "pocket_mate_review_pack_50" / "review_sheet.csv"
    benchmark_dir = base / "fusion360_assembly_benchmark_a00_a01"

    # Load review labels
    reviews: dict[str, str] = {}
    with open(review_csv, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sample_id = row.get("sample_id", "").strip()
            label = row.get("review_label", "").strip()
            if sample_id and label:
                reviews[sample_id] = label

    print(f"Loaded {len(reviews)} review labels")
    label_counts = {"true_pocket_mate": 0, "false_pocket_mate": 0, "uncertain": 0}
    for v in reviews.values():
        label_counts[v] = label_counts.get(v, 0) + 1
    print(f"  true_pocket_mate:  {label_counts['true_pocket_mate']}")
    print(f"  false_pocket_mate: {label_counts['false_pocket_mate']}")
    print(f"  uncertain:         {label_counts['uncertain']}")

    # Process each split file
    updated_total = 0
    for split_name in ["train", "dev", "test"]:
        jsonl_path = benchmark_dir / f"fusion360_{split_name}.jsonl"
        if not jsonl_path.exists():
            print(f"  SKIP: {jsonl_path} not found")
            continue

        rows = load_jsonl(jsonl_path)
        updated = 0
        for row in rows:
            sid = row.get("sample_id", "")
            if sid not in reviews:
                continue

            label = reviews[sid]
            if label == "true_pocket_mate":
                # Ensure pocket_mate is in mapped_relation_types
                if "pocket_mate" not in (row.get("mapped_relation_types") or []):
                    row.setdefault("mapped_relation_types", []).append("pocket_mate")
                row["weak_label"] = False
                row["mapping_confidence"] = "high"
                row["mapping_reasons"] = (row.get("mapping_reasons") or []) + [
                    "human_review:confirmed_true_pocket_mate"
                ]
                if "pocket_mate_human_audit" in (row.get("unavailable_fields") or []):
                    row["unavailable_fields"].remove("pocket_mate_human_audit")
                updated += 1

            elif label == "false_pocket_mate":
                # Remove pocket_mate from mapped_relation_types
                if "pocket_mate" in (row.get("mapped_relation_types") or []):
                    row["mapped_relation_types"].remove("pocket_mate")
                row["weak_label"] = False
                row["mapping_confidence"] = "high"
                row["mapping_reasons"] = (row.get("mapping_reasons") or []) + [
                    "human_review:confirmed_false_pocket_mate"
                ]
                if "pocket_mate_human_audit" in (row.get("unavailable_fields") or []):
                    row["unavailable_fields"].remove("pocket_mate_human_audit")
                updated += 1

            elif label == "uncertain":
                row["mapping_reasons"] = (row.get("mapping_reasons") or []) + [
                    "human_review:uncertain_pocket_mate"
                ]
                updated += 1

        write_jsonl(jsonl_path, rows)
        print(f"  {split_name}: updated {updated}/{len(rows)} rows")
        updated_total += updated

    print(f"\nTotal updated: {updated_total} rows across all splits")

    # Update label_mapping_report.md
    report_path = benchmark_dir / "label_mapping_report.md"
    if report_path.exists():
        old = report_path.read_text(encoding="utf-8")
        # Append update note
        update_note = (
            f"\n\n## Human Review Update (2026-07-08)\n\n"
            f"Reviewed {len(reviews)} pocket_mate candidates via visual inspection.\n\n"
            f"- true_pocket_mate: {label_counts['true_pocket_mate']}\n"
            f"- false_pocket_mate: {label_counts['false_pocket_mate']}\n"
            f"- uncertain: {label_counts['uncertain']}\n\n"
            f"Verified candidates have been upgraded from weak_label to high confidence.\n"
            f"False candidates have had pocket_mate removed from their relation labels.\n"
        )
        report_path.write_text(old + update_note, encoding="utf-8")
        print("Updated label_mapping_report.md")

    print("\nDone. Benchmark files updated.")


if __name__ == "__main__":
    main()
