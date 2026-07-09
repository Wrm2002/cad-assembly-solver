"""对 pose-valid 但功能错误的候选组进行评估侧分类审计。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(parts: list[str]) -> frozenset[str]:
    return frozenset(parts)


def _semantic_table(pool: Path) -> dict[str, dict[str, Any]]:
    path = pool / "part_semantics.json"
    return _load(path) if path.is_file() else {}


def _assigned_roles(row: dict[str, Any]) -> dict[str, str]:
    result = {}
    for role, value in row.get("role_assignment", {}).items():
        values = value if isinstance(value, list) else [value]
        for part in values:
            if part:
                result[str(part)] = str(role)
    return result


ROLE_EQUIVALENCE = {
    "stepped_shaft": "shaft",
    "flanged_hub": "hub",
    "cartridge_housing": "housing",
    "bearing_retainer": "bearing_retainer",
    "locating_pin": "locating_pin",
}


def _canonical_role(role: str) -> str:
    return ROLE_EQUIVALENCE.get(role, role)


def classify(
    pool: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    gt = _load(pool / "pool_gt.json")
    truth_groups = gt.get("true_groups", [])
    semantics = _semantic_table(pool)
    parts = _key(row["parts"])
    closest = max(
        truth_groups,
        key=lambda group: len(parts & _key(group["parts"])) / len(
            parts | _key(group["parts"])
        ),
    )
    closest_parts = _key(closest["parts"])
    overlap = parts & closest_parts
    added = sorted(parts - closest_parts)
    missing = sorted(closest_parts - parts)
    jaccard = len(overlap) / len(parts | closest_parts)
    assigned = _assigned_roles(row)
    role_mismatches = []
    semantic_families = set()
    distractors = []
    for part in row["parts"]:
        semantic = semantics.get(part, {})
        actual_role = _canonical_role(str(semantic.get("part_role", "unknown")))
        predicted_role = _canonical_role(assigned.get(part, "unknown"))
        family = str(semantic.get("assembly_family", "unknown"))
        if family not in {"unknown", "unassigned_distractor", ""}:
            semantic_families.add(family)
        if family == "unassigned_distractor" or not semantic:
            distractors.append(part)
        if actual_role not in {"unknown", predicted_role}:
            role_mismatches.append(
                {
                    "part_id": part,
                    "assigned_role": predicted_role,
                    "evaluation_role": actual_role,
                }
            )

    if parts < closest_parts:
        category = "真实组的不完整子组"
    elif closest_parts < parts:
        category = "向真实组错误吸附额外零件"
    elif added and missing and len(added) == len(missing):
        category = "相似零件替换了真实零件"
    elif len(semantic_families) >= 2:
        category = "跨装配族混合"
    elif distractors:
        category = "吸附干扰零件"
    elif role_mismatches:
        category = "几何角色误判"
    else:
        category = "同族几何可行但功能关系错误"

    missing_evidence = []
    if row["assembly_family"] == "shaft_hub_key":
        missing_evidence.extend(
            ["轴键槽—键—轮毂键槽三方闭环", "扭矩传递关系"]
        )
    elif row["assembly_family"] == "bearing_housing":
        missing_evidence.extend(
            ["轴承内外圈双侧配合", "轴向定位/保持链"]
        )
    elif row["assembly_family"] == "cover_base":
        missing_evidence.extend(
            ["止口或孔阵列注册", "定位件同时约束底座与盖板"]
        )
    return {
        "dataset": pool.parent.name,
        "pool_id": pool.name,
        "group_id": row["group_id"],
        "assembly_family_hypothesis": row["assembly_family"],
        "parts": row["parts"],
        "review_rank": row.get("review_rank"),
        "geometry_score": row.get("geometry_score"),
        "pose_status": row["pose_validation"]["final_pose_status"],
        "closest_true_group_id": closest["group_id"],
        "closest_true_family": closest["assembly_family"],
        "closest_true_jaccard": round(jaccard, 8),
        "overlap_parts": sorted(overlap),
        "added_parts": added,
        "missing_parts": missing,
        "distractor_parts": distractors,
        "semantic_families": sorted(semantic_families),
        "role_mismatches": role_mismatches,
        "primary_failure_category": category,
        "missing_functional_evidence": missing_evidence,
        "production_decision_affected": False,
        "audit_note": "真值和语义字段只用于离线错误分类，不参与生产推理。",
    }


def run(specs: list[tuple[Path, Path]], output: Path) -> dict[str, Any]:
    records = []
    for pools_root, experiment_root in specs:
        rows = _load(experiment_root / "pose_validation_records.json")
        truth = {
            (pool.name, _key(group["parts"]))
            for pool in pools_root.iterdir()
            if pool.is_dir() and (pool / "pool_gt.json").is_file()
            for group in _load(pool / "pool_gt.json").get("true_groups", [])
        }
        for row in rows:
            is_true = (row["pool_id"], _key(row["parts"])) in truth
            if row["pose_validation"]["final_pose_status"] != "valid" or is_true:
                continue
            records.append(classify(pools_root / row["pool_id"], row))
    counts = Counter(row["primary_failure_category"] for row in records)
    by_family = Counter(row["assembly_family_hypothesis"] for row in records)
    report = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only_error_analysis",
        "pose_valid_false_group_count": len(records),
        "category_counts": dict(sorted(counts.items())),
        "family_counts": dict(sorted(by_family.items())),
        "production_decision_affected": False,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pose_valid_false_group_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "dataset", "pool_id", "group_id", "assembly_family_hypothesis",
        "parts", "review_rank", "geometry_score", "closest_true_group_id",
        "closest_true_family", "closest_true_jaccard", "added_parts",
        "missing_parts", "distractor_parts", "semantic_families",
        "role_mismatches", "primary_failure_category",
        "missing_functional_evidence",
    ]
    with (output / "pose_valid_false_group_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            item = {key: row.get(key) for key in fields}
            for key in (
                "parts", "added_parts", "missing_parts", "distractor_parts",
                "semantic_families", "missing_functional_evidence",
            ):
                item[key] = "|".join(item[key])
            item["role_mismatches"] = json.dumps(
                item["role_mismatches"], ensure_ascii=False
            )
            writer.writerow(item)
    lines = [
        "# Pose 可行但功能错误候选审计",
        "",
        f"共分类 {len(records)} 个 pose-valid 假组。真值只用于离线评估。",
        "",
        "## 错误类型",
        "",
    ]
    lines.extend(f"- {name}：{count} 个" for name, count in sorted(counts.items()))
    lines.extend(["", "## 建议优先补充的证据", ""])
    lines.extend(
        [
            "- shaft_hub_key：轴键槽—键—轮毂键槽三方闭环和扭矩传递证据。",
            "- bearing_housing：轴承内外圈双侧配合及完整轴向定位链。",
            "- cover_base：止口/孔阵列注册以及定位件同时约束底座和盖板。",
        ]
    )
    (output / "pose_valid_false_group_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=here / "data/functional_error_analysis_v3",
    )
    args = parser.parse_args()
    report = run(
        [
            (
                here / "data/functional_mixed_pools_topology_v1",
                here / "data/functional_frontier_pose_experiment_v2",
            ),
            (
                here / "data/functional_cad_holdout_pools_v1",
                here / "data/harder_frontier_pose_experiment_v2",
            ),
        ],
        args.output.resolve(),
    )
    print(json.dumps({
        "pose_valid_false_group_count": report["pose_valid_false_group_count"],
        "category_counts": report["category_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

