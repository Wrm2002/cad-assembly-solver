"""Convert Fusion/JoinABLe joint records to the common interface schema."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from joinable_common import (
    all_joint_records,
    build_common_sample,
    discover_joint_files,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--out_dir", default="converted_joint_samples"
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--failure-output", default="conversion_failures.json"
    )
    parser.add_argument(
        "--summary-output", default="conversion_summary.json"
    )
    args = parser.parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_joint_files(data_root)
    attempted = 0
    converted = []
    failures = []
    missing: Counter[str] = Counter()
    joint_types: Counter[str] = Counter()
    interface_count = 0
    contact_count = 0
    holes_count = 0
    transform_count = 0
    axis_count = 0
    for joint_file, joint_index in all_joint_records(files):
        if attempted >= max(1, args.limit):
            break
        attempted += 1
        try:
            sample = build_common_sample(joint_file, joint_index)
            safe_name = sample["sample_id"].replace(":", "_")
            destination = out_dir / f"{safe_name}.json"
            write_json(destination, sample)
            converted.append({
                "sample_id": sample["sample_id"],
                "output": str(destination.resolve()),
                "failure_reasons": sample["failure_reasons"],
                "unavailable_fields": sample["unavailable_fields"],
            })
            joint_types[sample["relation"]["joint_type"]] += 1
            for field in sample["unavailable_fields"]:
                missing[field] += 1
            if (
                sample["interface_a"]["entity_ids"]
                and sample["interface_b"]["entity_ids"]
            ):
                interface_count += 1
            if sample["contacts"]:
                contact_count += 1
            if sample["holes"]:
                holes_count += 1
            if sample["relation"]["transform_a_to_b"] is not None:
                transform_count += 1
            if (
                sample["relation"]["axis_origin"] is not None
                and sample["relation"]["axis_direction"] is not None
            ):
                axis_count += 1
        except Exception as exc:
            failures.append({
                "source_file": str(joint_file.resolve()),
                "joint_index": joint_index,
                "failure_reason": (
                    f"{type(exc).__name__}:{exc}"
                ),
                "failure_reasons": [
                    f"{type(exc).__name__}:{exc}"
                ],
                "unavailable_fields": ["common_schema_sample"],
            })
    failure_report = {
        "schema_version": "1.0.0",
        "failure_count": len(failures),
        "failures": failures,
        "failure_reasons": [
            row["failure_reason"] for row in failures
        ],
        "unavailable_fields": sorted({
            field for row in failures
            for field in row["unavailable_fields"]
        }),
    }
    write_json(Path(args.failure_output), failure_report)
    summary = {
        "schema_version": "1.0.0",
        "data_root": str(data_root.resolve()),
        "joint_set_file_count": len(files),
        "requested_sample_count": max(1, args.limit),
        "attempted_count": attempted,
        "success_count": len(converted),
        "failure_count": len(failures),
        "converted": converted,
        "missing_field_counts": dict(sorted(missing.items())),
        "joint_type_distribution": dict(sorted(joint_types.items())),
        "samples_with_interface_entity_ids": interface_count,
        "samples_with_contact_face_labels": contact_count,
        "samples_with_holes_labels": holes_count,
        "samples_with_transform": transform_count,
        "samples_with_axis_origin_direction": axis_count,
        "acceptance_met": len(converted) >= max(1, args.limit),
        "failure_reasons": (
            [] if len(converted) >= max(1, args.limit)
            else [
                f"only_{len(converted)}_of_"
                f"{max(1, args.limit)}_samples_converted"
            ]
        ),
        "unavailable_fields": sorted(missing),
    }
    write_json(Path(args.summary_output), summary)
    print(
        f"Converted {summary['success_count']}/"
        f"{summary['requested_sample_count']} samples"
    )
    return 0 if summary["acceptance_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
