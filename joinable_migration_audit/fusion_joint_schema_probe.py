"""Probe real Fusion 360 Joint Data records used by JoinABLe."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from joinable_common import (
    all_joint_records,
    build_common_sample,
    discover_joint_files,
    joint_type,
    load_json,
    write_json,
)


def _present(value: Any) -> bool:
    return value is not None and value != [] and value != {}


def probe(data_root: Path, limit: int) -> dict[str, Any]:
    files = discover_joint_files(data_root)
    samples = []
    failures = []
    joint_types: Counter[str] = Counter()
    presence: Counter[str] = Counter()
    for path, index in all_joint_records(files):
        if len(samples) >= limit:
            break
        try:
            data = load_json(path)
            joint = data["joints"][index]
            sample = build_common_sample(path, index)
        except Exception as exc:
            failures.append({
                "source_file": str(path.resolve()),
                "joint_index": index,
                "failure_reason": (
                    f"{type(exc).__name__}:{exc}"
                ),
                "unavailable_fields": ["normalized_joint_sample"],
            })
            continue
        joint_types[joint_type(joint)] += 1
        for field, value in {
            "assembly_id": sample["assembly_id"],
            "explicit_assembly_id": (
                data.get("assembly_id") or data.get("design_id")
            ),
            "inferred_assembly_id": sample["assembly_id"],
            "body_one": data.get("body_one"),
            "body_two": data.get("body_two"),
            "joints": data.get("joints"),
            "contacts": data.get("contacts"),
            "holes": data.get("holes"),
            "geometry_or_origin_one": joint.get(
                "geometry_or_origin_one"
            ),
            "geometry_or_origin_two": joint.get(
                "geometry_or_origin_two"
            ),
            "joint_type": sample["relation"]["joint_type"],
            "joint_axis_line": sample["relation"]["axis_origin"],
            "transform": sample["relation"]["transform_a_to_b"],
            "brep_entity_a": sample["interface_a"]["entity_ids"],
            "brep_entity_b": sample["interface_b"]["entity_ids"],
            "contact_face_pair": sample["contacts"],
            "hole_or_cylindrical_feature": sample["holes"],
        }.items():
            if _present(value):
                presence[field] += 1
        samples.append({
            "source_file": str(path.resolve()),
            "joint_index": index,
            "assembly_id": sample["assembly_id"],
            "assembly_id_semantics": sample["metadata"][
                "assembly_id_semantics"
            ],
            "raw_top_level_fields": sorted(data),
            "raw_joint_fields": sorted(joint),
            "body_one": data.get("body_one"),
            "body_two": data.get("body_two"),
            "joint_type": sample["relation"]["joint_type"],
            "geometry_or_origin_one": joint.get(
                "geometry_or_origin_one"
            ),
            "geometry_or_origin_two": joint.get(
                "geometry_or_origin_two"
            ),
            "joint_axis": {
                "part_a_origin": sample["relation"]["axis_origin"],
                "part_a_direction": sample["relation"][
                    "axis_direction"
                ],
                "part_b_origin": sample["relation"]["axis_origin_b"],
                "part_b_direction": sample["relation"][
                    "axis_direction_b"
                ],
            },
            "transform_a_to_b": sample["relation"][
                "transform_a_to_b"
            ],
            "interface_a": sample["interface_a"],
            "interface_b": sample["interface_b"],
            "contacts": sample["contacts"],
            "holes": sample["holes"],
            "failure_reasons": sample["failure_reasons"],
            "unavailable_fields": sample["unavailable_fields"],
        })
    unavailable = []
    if not files:
        unavailable.extend([
            "fusion_joint_files",
            "joint_schema_sample",
        ])
    if len(samples) < limit:
        unavailable.append(
            f"requested_{limit}_samples_only_{len(samples)}_available"
        )
    unavailable.extend(
        field for sample in samples
        for field in sample["unavailable_fields"]
    )
    return {
        "schema_version": "1.0.0",
        "audit_status": (
            "success" if samples
            else "failed_missing_or_unreadable_data"
        ),
        "data_root": str(data_root.resolve()),
        "expected_file_pattern": "joint_set_00000.json",
        "joint_set_file_count": len(files),
        "requested_sample_count": limit,
        "parsed_sample_count": len(samples),
        "field_presence_count": dict(sorted(presence.items())),
        "joint_type_distribution": dict(sorted(joint_types.items())),
        "samples": samples,
        "failures": failures,
        "failure_reasons": (
            [] if samples
            else [
                "No readable Fusion Joint Data records were discovered."
            ]
        ),
        "unavailable_fields": sorted(set(unavailable)),
        "required_download": {
            "name": "Fusion 360 Gallery Assembly Joint Data j1.0.0",
            "url": (
                "https://fusion-360-gallery-dataset.s3.us-west-2."
                "amazonaws.com/assembly/j1.0.0/j1.0.0.7z"
            ),
            "size_gb": 2.8,
            "expected_layout": (
                "<data_root>/joint_set_*.json plus body .json/.obj files"
            ),
        },
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Fusion Joint Schema Probe Report",
        "",
        f"- 状态：`{result['audit_status']}`",
        f"- joint-set 文件：{result['joint_set_file_count']}",
        (
            f"- 解析 joint samples："
            f"{result['parsed_sample_count']} / "
            f"{result['requested_sample_count']}"
        ),
        "",
        "## 字段实测",
        "",
        "| 字段 | 出现样本数 |",
        "|---|---:|",
    ]
    for field, count in result["field_presence_count"].items():
        lines.append(f"| `{field}` | {count} |")
    lines.extend([
        "",
        "## Joint type",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
    ])
    for name, count in result["joint_type_distribution"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend([
        "",
        "## 失败与不可用",
        "",
    ])
    if result["failure_reasons"]:
        lines.extend(
            f"- {reason}" for reason in result["failure_reasons"]
        )
    if result["unavailable_fields"]:
        lines.extend(
            f"- 不可用：`{field}`"
            for field in result["unavailable_fields"]
        )
    if not result["failure_reasons"] and not result["unavailable_fields"]:
        lines.append("- 本次样本解析无结构性失败。")
    lines.extend([
        "",
        "复现命令：",
        "",
        "```powershell",
        (
            "python fusion_joint_schema_probe.py "
            "--data_root D:\\path\\to\\j1.0.0 --limit 20"
        ),
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--sample-output",
        default="fusion_joint_schema_sample.json",
    )
    parser.add_argument(
        "--report-output",
        default="fusion_joint_schema_probe_report.md",
    )
    args = parser.parse_args()
    result = probe(Path(args.data_root), max(1, args.limit))
    write_json(Path(args.sample_output), result)
    Path(args.report_output).write_text(
        report_markdown(result), encoding="utf-8"
    )
    print(
        f"Parsed {result['parsed_sample_count']}/"
        f"{result['requested_sample_count']} joint samples"
    )
    return 0 if result["parsed_sample_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
