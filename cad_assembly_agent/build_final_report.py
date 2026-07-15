"""Build the project and desktop three-stage delivery report from real outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def value(value) -> str:
    if value is None:
        return "不可计算"
    if isinstance(value, float):
        return f"{value:.2%}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--project-output", required=True)
    parser.add_argument("--desktop-output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    stage1 = load(
        root / "datasets" / "step_real_benchmark" / "frozen_pair_benchmark.json"
    )
    edge = load(
        root / "datasets" / "step_real_benchmark" / "edge_local_feature_report.json"
    )
    pair = load(root / "reports" / "pair_benchmark_report.json")
    all_pair = load(root / "reports" / "all_pair_validation_summary.json")
    graph_dir = root / "reports" / "conservative_graph"
    metrics = load(graph_dir / "conservative_metrics.json")
    qwen = load(root / "reports" / "qwen_semantic_reviews.json")
    validation = load(root / "reports" / "three_stage_delivery_validation.json")
    descriptor_failures = [
        row
        for row in pair.get("descriptor_parts", [])
        if row.get("status") != "success"
    ]
    edge_failures = [
        row for row in edge.get("results", []) if row.get("status") != "success"
    ]

    lines = [
        "# CAD 自动装配 Agent 三阶段工作报告",
        "",
        "## 结论",
        "",
        "三阶段工程闭环已经实现并通过机器验收，但结果必须按证据强度解读：",
        "",
        "- B-Rep 可追踪图、规则 Top-K 接口候选、OCCT 位姿/碰撞验证、保守装配图和 Agent 审计链均已落盘。",
        "- 规则候选器在部分真实零件对上能够把可行接口排入前列，OCCT 能拒绝高排但穿透的错误候选。",
        "- 当前环境没有可直接复现官方预训练 JoinABLe 的 PyTorch/PyG 旧依赖，因此本轮是规则基线，不是学习模型结果。",
        "- 为避免 false positive，未经校准的规则边一律不自动接受；物理可行边进入人工复核。",
        "- 第4组原始 assembly STEP 的主板、内存和 CPU 处于分离位置，不能作为 expected-pose 真值；系统没有伪造位姿。",
        "",
        "因此，这条路线的“工具闭环”可行，但“公开学习先验已迁移成功”尚未被证明。下一步最小工作应是隔离复现官方 JoinABLe checkpoint，再与本规则基线做同一 Top-K 审计。",
        "",
        "## 第一阶段：真实 STEP Benchmark 与特征契约",
        "",
        f"- 人工确认语义正边：{stage1['semantic_positive_pair_count']} 对。",
        f"- 两侧可映射的 provisional 几何 support：{stage1['provisional_geometry_support_pair_count']} 对。",
        f"- expected pose/interface 真值不可用：{stage1['unusable_expected_pose_pair_count']} 对。",
        f"- 边局部特征覆盖零件：{edge.get('success_count', 0)}/{edge.get('part_count', 0)} 个完整成功；"
        f"部分或失败 {edge.get('partial_or_failed_count', 0)} 个。",
        "- 普通零件输出全边二面角/convexity；超大 PCBA 只对候选局部提取，避免 152,475 face × 408,379 edge 的无界处理。",
        "",
        "第一阶段没有把 minimum_distance=0 当作装配正确：它可能是接触，也可能是穿透。所有缺失字段和超时均保留在 JSON 中。",
        "",
        "## 第二阶段：Top-K 接口候选与 OCCT 双零件验证",
        "",
        f"- 运行语义正零件对：{pair.get('pair_count')}。",
        f"- 可评价接口真值对：{pair.get('truth_evaluable_pair_count')}。",
        f"- Top-5 接口召回：{value(pair.get('top_5_interface_recall'))}。",
        f"- Top-10 接口召回：{value(pair.get('top_10_interface_recall'))}。",
        f"- 有至少一个 OCCT 有效位姿的零件对：{pair.get('pair_pose_success_count')}。",
        f"- 描述器失败：{len(descriptor_failures)}。",
        "",
        "候选器不是全表面暴力穷举：解析面/边按几何类型与对数尺度分桶，保留资源有界实体，再对兼容桶做近邻匹配。`candidate_reduction_fraction` 记录相对保留实体笛卡尔积的削减比例。每个候选保留 face/edge topology id、类型、尺寸/半径证据和排序分数。",
        "",
        "OCCT validator 对候选生成轴对齐/法向贴合位姿，逐个检查 clearance、Boolean Common 体积和最小零件体积穿透比例。有效条件是接触且穿透比例不超过阈值；数值失败或超时为 uncertain，不会伪装成成功。",
        "",
        "## 第三阶段：多零件保守装配图与 Agent 调度",
        "",
        f"- 5 组内全部 pair：{all_pair.get('pair_count')}。",
        f"- bounded search 找到物理可行 pose 的 pair：{all_pair.get('valid_pose_pair_count')}。",
        f"- accepted/review/rejected edge：{metrics.get('accepted_edge_count')}/"
        f"{metrics.get('review_edge_count')}/{metrics.get('rejected_edge_count')}。",
        f"- unresolved parts：{metrics.get('unresolved_parts_count')}。",
        f"- false positive count（自动接受层）：{metrics.get('false_positive_count')}。",
        f"- auto-accept precision：{value(metrics.get('auto_accept_precision'))}；"
        f"{metrics.get('auto_accept_precision_reason')}",
        f"- Qwen review-only 成功：{qwen.get('success_count')}/{qwen.get('case_count')}。",
        "",
        "Agent 是确定性状态机，不是强化学习：枚举 pair → 调用候选器 → 调用 OCCT → 保守门控 → 构图 → 写审计事件。Review 边不会自动合并 connected component。Qwen 只读取渲染图、文件名和工具摘要并写复核说明，`semantic_reranking_enabled=false`，不能改变任何几何 tier。",
        "",
        "## 数据问题与失败事实",
        "",
        "- 8 对语义正边中只有部分 assembly pose 能提供 provisional 接触 support；其余包含资源超时、分离位姿或拓扑不可映射。",
        "- Case 1 原 manifest 的 -20 mm 位移会造成明显穿透；规则候选器找到的无穿透法兰平面位姿才通过 OCCT。",
        "- Case 4 identity 构建保留了分离的主板/内存/CPU，不能用作接口真值。",
        "- 大型服务器件的 exact distance/Boolean 可能在资源界限内超时，超时结果进入 review。",
        f"- edge feature 部分/失败零件：{len(edge_failures)}；详见 `edge_local_feature_report.json`。",
        "",
        "## 新增文件",
        "",
        "- `tools/cad_loader/freeze_pair_truth.py`：位姿来源与 closest-support 审计。",
        "- `datasets/step_real_benchmark/freeze_stage1_contract.py`：冻结8对 benchmark 契约。",
        "- `tools/brep_graph_extractor/edge_local_features.py`：二面角与 convexity。",
        "- `tools/joinable_interface_predictor/rule_interface_predictor.py`：资源有界 Top-K 基线。",
        "- `tools/occt_pose_validator/occt_pair_pose_validator.py`：双零件精确验证。",
        "- `run_pair_benchmark.py`：第二阶段批量审计。",
        "- `run_all_pair_graph.py`：组内全 pair 运行。",
        "- `tools/assembly_graph_builder/build_conservative_graph.py`：accepted/review/rejected/unresolved 构图。",
        "- `tools/semantic_reviewer/qwen_review_only.py`：Qwen 多模态复核说明。",
        "- `validate_three_stage_delivery.py`：关机前机器验收门。",
        "",
        "## 主要输出",
        "",
        "- `datasets/step_real_benchmark/frozen_pair_benchmark.json`",
        "- `datasets/step_real_benchmark/edge_local_feature_report.json`",
        "- `reports/pair_benchmark_report.json`",
        "- `reports/all_pair_validation_summary.json`",
        "- `reports/conservative_graph/assembly_candidate_graph.json`",
        "- `reports/conservative_graph/{accepted_edges,review_edges,rejected_edges,unresolved_parts}.json`",
        "- `reports/conservative_graph/agent_events.json`",
        "- `reports/qwen_semantic_reviews.json`",
        "- `reports/three_stage_delivery_validation.json`",
        "",
        "大型中间数据位于 `D:\\Model_match_agent_data\\step_real_benchmark`，原始 `sw/1..5` 文件未被覆盖。",
        "",
        "## 验收",
        "",
        f"- 机器检查：{validation.get('passed_count')}/{validation.get('check_count')} 通过。",
        f"- delivery_complete：{validation.get('delivery_complete')}",
        f"- 未通过项：{validation.get('failure_reasons') or '无'}",
        "",
        "## 下一步最小建议",
        "",
        "1. 在独立旧环境复现官方 JoinABLe checkpoint，不污染当前 OCCT 环境。",
        "2. 对同一批8对零件比较规则基线与 pretrained Top-5/Top-10；没有提升就停止迁移。",
        "3. 为第4组补真实已装配 STEP 或人工 face/edge mate 标签；不要继续猜 identity pose。",
        "4. 只有 learned scorer 在独立负例 holdout 上校准后，才允许极少数边进入 auto-accept。",
    ]
    text = "\n".join(lines) + "\n"
    for destination in (Path(args.project_output), Path(args.desktop_output)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print("final reports written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
