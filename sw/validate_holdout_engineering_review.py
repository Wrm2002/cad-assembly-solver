"""Validate qualified mechanical-engineer sign-off for the CAD holdout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_VERDICT = "confirm"
YES_VALUES = {"yes", "true", "1", "confirmed"}


def validate(holdout_root: str | Path) -> dict:
    root = Path(holdout_root).resolve()
    form = root / "ENGINEERING_REVIEW_FORM.csv"
    with form.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    if len(rows) != 12:
        failures.append(f"expected_12_review_rows_found_{len(rows)}")
    confirmed = 0
    reviewers = set()
    for row_number, row in enumerate(rows, start=2):
        missing = [
            field
            for field in (
                "engineer_verdict",
                "functional_validity_confirmed",
                "reviewer_name",
                "mechanical_engineering_qualification",
                "review_date",
                "signature",
            )
            if not str(row.get(field, "")).strip()
        ]
        if missing:
            failures.append(
                f"row_{row_number}_missing:{'|'.join(missing)}"
            )
            continue
        verdict = str(row["engineer_verdict"]).strip().lower()
        validity = str(
            row["functional_validity_confirmed"]
        ).strip().lower()
        recognizable = str(
            row.get("recognizable_engineering_system", "")
        ).strip().lower()
        if verdict != REQUIRED_VERDICT:
            failures.append(
                f"row_{row_number}_verdict_not_confirm:{verdict}"
            )
            continue
        if validity not in YES_VALUES:
            failures.append(
                f"row_{row_number}_functional_validity_not_confirmed"
            )
            continue
        if (
            row["sample_id"] == "POSITIVE"
            and recognizable not in YES_VALUES
        ):
            failures.append(
                f"row_{row_number}_positive_not_recognizable"
            )
            continue
        confirmed += 1
        reviewers.add(
            (
                row["reviewer_name"].strip(),
                row["mechanical_engineering_qualification"].strip(),
                row["signature"].strip(),
            )
        )
    passed = not failures and confirmed == 12
    payload = {
        "schema_version": "1.0.0",
        "status": (
            "passed_qualified_mechanical_engineer_review"
            if passed
            else "pending_or_failed_mechanical_engineer_review"
        ),
        "required_review_rows": 12,
        "confirmed_rows": confirmed,
        "reviewer_count": len(reviewers),
        "gate_passed": passed,
        "failure_reasons": failures,
    }
    (root / "engineering_signoff.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-root",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "functional_cad_holdout_v1"
        ),
    )
    args = parser.parse_args()
    result = validate(args.holdout_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
