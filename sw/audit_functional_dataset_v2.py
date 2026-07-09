"""Strict metadata audit for the functional assembly dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from functional_dataset_generator import FAMILIES, validate_metadata


def run(root: Path) -> dict:
    rows = []
    invalid = []
    families = Counter()
    generic_positive_cases = []
    for path in sorted(root.glob("*/metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_metadata(metadata, path.parent)
        family = metadata.get("assembly_family")
        families[family] += 1
        engineering_name = str(metadata.get("engineering_name", "")).strip()
        roles = [str(part.get("part_role", "")) for part in metadata.get("parts", [])]
        functional_mates = metadata.get("functional_mates", [])
        weak_mates = [
            index for index, mate in enumerate(functional_mates)
            if len(mate.get("independent_evidence", [])) < 2
        ]
        forbidden = bool(
            metadata.get("functional_positive_eligible") is False
            or metadata.get("dataset_intended_use") == "geometry_smoke_only"
        )
        if forbidden:
            generic_positive_cases.append(path.parent.name)
        row = {
            "case_id": metadata.get("case_id", path.parent.name),
            "assembly_family": family,
            "engineering_name": engineering_name,
            "part_count": len(roles),
            "roles": roles,
            "functional_mate_count": len(functional_mates),
            "weak_mate_indices": weak_mates,
            "negative_types": sorted(
                group.get("negative_type")
                for group in metadata.get("negative_groups", [])
            ),
            "source_id_is_production_truth": metadata.get(
                "source_id_is_production_truth"
            ),
            "errors": errors,
        }
        rows.append(row)
        if errors or weak_mates or not engineering_name or any(not role for role in roles):
            invalid.append(row)
    expected_negative_types = {
        "easy_negative", "geometric_hard_negative", "semantic_hard_negative"
    }
    negative_coverage_ok = all(
        set(row["negative_types"]) == expected_negative_types for row in rows
    )
    report = {
        "schema_version": "1.0.0",
        "dataset_root": str(root),
        "case_count": len(rows),
        "family_counts": dict(sorted(families.items())),
        "expected_families": sorted(FAMILIES),
        "all_families_present": set(families) == set(FAMILIES),
        "invalid_case_count": len(invalid),
        "negative_type_coverage_ok": negative_coverage_ok,
        "generic_geometry_stack_positive_count": len(generic_positive_cases),
        "generic_geometry_stack_positive_cases": generic_positive_cases,
        "source_id_used_as_production_truth": any(
            row["source_id_is_production_truth"] is True for row in rows
        ),
        "status": "pass" if (
            rows
            and not invalid
            and negative_coverage_ok
            and not generic_positive_cases
            and set(families) == set(FAMILIES)
        ) else "fail",
        "cases": rows,
    }
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, nargs="?", default=here / "data/functional_dataset_v1"
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "data/functional_dataset_v1/functional_dataset_audit_v2.json",
    )
    args = parser.parse_args()
    report = run(args.root.resolve())
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in (
        "case_count", "family_counts", "invalid_case_count",
        "generic_geometry_stack_positive_count", "status"
    )}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

