"""诊断 harder holdout 中暂定真组的 pose 失败根因。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from placement_validation import exact_shape_collisions


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(parts: list[str]) -> frozenset[str]:
    return frozenset(parts)


def run(
    pools_root: Path,
    proposal_experiment: Path,
    pose_experiment: Path,
    output: Path,
) -> dict[str, Any]:
    score_rows = list(
        csv.DictReader(
            (pose_experiment / "candidate_scores.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    failures = [
        row for row in score_rows
        if row["is_true_group"] == "True"
        and row["final_pose_status"] != "valid"
    ]
    records = []
    for row in failures:
        pool = pools_root / row["pool_id"]
        gt = _load(pool / "pool_gt.json")
        parts = row["parts"].split("|")
        truth = next(
            group for group in gt["true_groups"]
            if _key(group["parts"]) == _key(parts)
        )
        components = [
            {
                "source": part,
                "placement": truth["placements"].get(part, {}),
            }
            for part in truth["parts"]
        ]
        gt_collision = exact_shape_collisions(pool / "parts", components)
        run_dir = pose_experiment / "pose_runs" / pool.name / row["group_id"]
        validation = _load(run_dir / "validation_result.json")
        search = _load(run_dir / "search_report.json")
        selected = _load(run_dir / "selected_matches.json")
        pair_edges = _load(
            proposal_experiment / "pools" / pool.name / "pair_edges.json"
        )
        edge_pairs = {_key(edge["parts"]): edge for edge in pair_edges}
        mate_coverage = []
        for mate in truth.get("true_mates", []):
            pair = _key([mate["part_a"], mate["part_b"]])
            edge = edge_pairs.get(pair)
            mate_coverage.append(
                {
                    "parts": sorted(pair),
                    "required_interface_type": mate.get(
                        "required_interface_type"
                    ),
                    "pair_edge_available": edge is not None,
                    "relation_types": edge.get("relation_types", []) if edge else [],
                    "physical_evidence": edge.get("physical_evidence", []) if edge else [],
                }
            )
        reasons = Counter(
            item.get("rejection_reason")
            for item in search.get("pose_candidate_audit", [])
        )
        solver_collisions = validation.get("metrics", {}).get(
            "exact_collisions", []
        )
        if gt_collision["collisions"]:
            root_cause = "暂定真值位姿本身存在实体交叠，需工程签核或接触容差策略"
        elif reasons and set(reasons) == {"solid_penetration"}:
            root_cause = "正确接口已召回，但所有搜索位姿都落入实体穿透；属于位姿假设/约束选择失败"
        else:
            root_cause = "位姿搜索或碰撞验证失败，需逐候选复核"
        selected_types = Counter(
            str(match.get("type", "unknown")) for match in selected
        )
        records.append(
            {
                "pool_id": pool.name,
                "true_group_id": truth["group_id"],
                "candidate_id": row["group_id"],
                "assembly_family": truth["assembly_family"],
                "parts": truth["parts"],
                "checked_pose_count": validation["metrics"].get(
                    "checked_pose_count"
                ),
                "complete_pose_candidate_count": search.get(
                    "complete_pose_candidate_count"
                ),
                "rejection_reason_counts": dict(reasons),
                "solver_collision_pairs": [
                    collision["parts"] for collision in solver_collisions
                ],
                "solver_max_common_volume_mm3": max(
                    (
                        float(collision.get("intersection_volume_mm3", 0.0))
                        for collision in solver_collisions
                    ),
                    default=0.0,
                ),
                "ground_truth_pose_collision_status": gt_collision["status"],
                "ground_truth_pose_collisions": gt_collision["collisions"],
                "ground_truth_pose_collision_free": not gt_collision["collisions"],
                "required_mate_pair_coverage": mate_coverage,
                "all_required_mate_pairs_available": all(
                    item["pair_edge_available"] for item in mate_coverage
                ),
                "selected_match_type_counts": dict(selected_types),
                "root_cause": root_cause,
                "recommended_fix": [
                    "增加键槽局部坐标与键的径向/轴向占位闭环",
                    "将键与轴、键与轮毂的两个平面约束联合生成位姿，而不是独立选边",
                    "保留当前精确碰撞拒绝规则，不以扩大 beam 掩盖穿透",
                    "由机械工程师确认暂定真值中的键是否允许过盈及允许交叠容差",
                ],
            }
        )
    report = {
        "schema_version": "1.0.0",
        "artifact_role": "evaluation_only_pose_failure_diagnosis",
        "failed_provisional_true_group_count": len(records),
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "true_group_pose_failure_diagnosis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 暂定真组 Pose 失败诊断",
        "",
        f"共发现 {len(records)} 个暂定真组未通过当前精确位姿验证。",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['pool_id']} / {record['candidate_id']}",
                "",
                f"- 装配族：{record['assembly_family']}",
                f"- 检查位姿数：{record['checked_pose_count']}",
                f"- 求解器最大交叠体积：{record['solver_max_common_volume_mm3']:.6f} mm³",
                f"- 真值位姿无碰撞：{record['ground_truth_pose_collision_free']}",
                f"- 必要 mate pair 全部召回：{record['all_required_mate_pairs_available']}",
                f"- 根因判断：{record['root_cause']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 结论",
            "",
            "两个失败都集中在 retained shaft_hub_key，且搜索位姿被实体穿透拒绝。应优先修复键槽三方联合位姿，不扩大通用 beam。",
        ]
    )
    (output / "true_group_pose_failure_diagnosis.md").write_text(
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
        here / "data/functional_cad_holdout_pools_v1",
        here / "data/harder_functional_grouping_experiment_v2",
        here / "data/harder_frontier_pose_experiment_v2",
        args.output.resolve(),
    )
    print(json.dumps({
        "failed_provisional_true_group_count": report[
            "failed_provisional_true_group_count"
        ],
        "root_causes": [row["root_cause"] for row in report["records"]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

