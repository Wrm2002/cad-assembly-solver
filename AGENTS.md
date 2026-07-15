你现在接手当前 SolidWorks / STEP 零件自动匹配系统。请不要继续扩大系统规模，不要引入 RL，不要继续做复杂多 Agent，不要继续盲目调 DeepSeek prompt。

当前实验已经说明：

1. 已知真组 pose reconstruction 已经有进步；
2. 2~5 零件组的 pose 验证能力相对可用；
3. mixed-pool 自动分组效果仍然弱；
4. 物理可装配不等于真实来源正确；
5. DeepSeek 在匿名几何摘要上的校准失败，当前不能作为分组裁判；
6. 当前最危险的问题是 false positive：系统把几何可行但实际不该成组的零件错误自动接受；
7. 现有 synthetic dataset 质量不足，3/4/5/6-part 正样本很多只是 cone/ring/block/plate 的几何堆叠，不像现实世界中存在的功能装配，因此不能继续用这种数据证明 semantic verification 或 functional grouping 的有效性。

因此，后续路线必须从“尽可能自动完整分组”改成：

成功率优先的保守候选推荐系统
+
功能装配数据集修复

最高原则：

1. 第一目标是降低错误自动接受。
2. 第二目标是把不确定候选交给人工复核。
3. 第三目标才是提高覆盖率。
4. 允许 unresolved_parts 增加。
5. 允许 review_candidates 增加。
6. 不允许为了覆盖率牺牲 accepted_groups 的正确率。
7. 不要把 pose 成功、collision_free 或 geometry_score 高当成“装配正确”的充分条件。
8. 不要把 source_id / generator case id 当成唯一工程真值。
9. 正样本必须看起来像现实工程装配，而不是随机几何堆叠。

============================================================
D0：Functional Dataset Repair 功能装配数据集修复
================================================

当前 synthetic dataset 的工程语义质量不足。请先修复数据集，不要继续盲目扩展 D6/D7。

目标：

生成少量但真实的功能装配模板，而不是大量随机几何堆叠。

第一版只支持 3 个 assembly family：

1. cover_base
   底座 + 盖板 + 定位销/螺柱/螺钉孔
2. shaft_hub_key
   轴 + 轮毂/齿轮简化件 + 键 + 可选轴向限位件
3. bearing_housing
   壳体 + 轴承 + 轴 + 端盖

每个 generated assembly 必须满足：

1. 能用一个现实工程名称描述；
2. 每个零件有 part_role；
3. 每个接口有 interface_type；
4. 每个 mate 有 functional_relation；
5. 每个正样本不是随机堆叠，而是符合该 assembly family 的功能结构；
6. 每个负样本不能只按 source_id 判断，而应按 functional validity 判断；
7. 如果零件几何和功能上可互换，必须标记为 interchangeable，而不能算作错误匹配。

每个 case 输出 metadata.json，格式包括：

{
  "case_id": "...",
  "assembly_family": "shaft_hub_key",
  "parts": [
    {
      "part_id": "P01",
      "file": "P01.step",
      "part_role": "shaft",
      "interface_types": ["cylindrical_shaft", "keyway"]
    }
  ],
  "functional_mates": [
    {
      "parts": ["shaft", "hub"],
      "mate_type": "coaxial_insert",
      "functional_relation": "shaft transmits torque to hub"
    }
  ],
  "valid_groups": [
    ["P01", "P02", "P03"]
  ],
  "optional_parts": [],
  "interchangeable_parts": [],
  "invalid_role_combinations": [
    ["shaft", "cover"],
    ["bearing", "flange"]
  ]
}

同时生成 hard negatives：

1. easy_negative：
   几何上不能装；
2. geometric_hard_negative：
   几何上能装但只靠弱接口，例如单平面贴合、单孔偶然匹配；
3. semantic_hard_negative：
   几何上能装但功能角色不合理，例如 shaft + cover、bearing + random plate、hub + chassis lid。

请不要再生成匿名 cone/ring/block/plate 的随机堆叠作为功能装配正样本。

D0 验收标准：

1. 每个正样本人工看图能说出它像什么装配；
2. 每个零件有明确 role；
3. 每个正样本至少有两个独立几何/功能证据；
4. DeepSeek 输入里包含 part_role、interface_type、assembly_family、functional_relation；
5. 数据集可以区分 geometry-valid-but-functionally-wrong 的 hard negative；
6. source_id 不再作为唯一真值。

============================================================
D3.5：候选召回修复 Candidate Recall Audit
=========================================

在进入 D4 前，先处理候选质量问题：

精细候选类型召回率目前约 79.4%，剪枝后约 63.2%。这说明真实候选在候选生成或剪枝阶段已经丢失。

请先实现候选召回审计，不要直接进入全局分组。

目标：

1. 找出哪些真值候选没有被生成；
2. 找出哪些真值候选在 pruning 阶段被删掉；
3. 按候选类型统计召回率；
4. 按 group size 统计召回率；
5. 输出可以人工检查的报告。

需要输出：

data/results/
├── missed_true_candidates.csv
├── pruned_true_candidates.csv
├── candidate_recall_by_type.json
├── candidate_recall_by_group_size.json
└── candidate_recall_audit.md

需要分析的字段至少包括：

- pool_id
- true_group_id
- parts
- group_size
- candidate_type
- generated_or_not
- pruned_or_not
- pruning_reason
- missing_reason_guess
- geometry_features_available
- required_interface_type

注意：

候选阶段应该偏高召回，接受阶段才严格。不要在候选阶段过早删除真值。

============================================================
D4：保守几何候选分层，不再强行完整分组
======================================

原 D4 如果是“纯几何全局分组”，请改掉。

D4 新目标：

不是把混合零件池强行划分完，而是把几何候选分成三类：

1. accepted_geometry_candidates
2. review_geometry_candidates
3. rejected_geometry_candidates

D4 不再要求所有零件都被分组。

D4 必须新增或强化以下逻辑：

一、multi-evidence gate

自动接受的几何候选必须至少有两个独立几何证据。

例如 flange-flange 不能只靠一个平面贴合，至少需要：

- 平面贴合；
- 孔阵列一致；
- 主轴或管道轴线近似共线；
- 无碰撞。

例如 shaft-hole 不能只靠半径接近，至少需要：

- 半径匹配；
- 轴线方向合理；
- 插入深度合理；
- 装配后无干涉。

如果候选只靠单个接口成立，不能自动接受，必须进入 review。

二、group_consistency.py

请新增一个轻量模块：

global_optimizer/group_consistency.py

职责不是做复杂推理，只做组级一致性检查。

输出字段：

{
  "group_consistency_score": 0.0,
  "evidence_count": 0,
  "independent_evidence_count": 0,
  "weak_single_interface_match": true,
  "has_multi_evidence_support": false,
  "has_central_part_structure": false,
  "blocks_larger_better_group": false,
  "review_required": true,
  "reason": "Only one weak planar contact supports this candidate."
}

检查内容：

1. evidence_count：几何证据数量；
2. independent_evidence_count：独立几何证据数量；
3. interface_coverage：关键接口是否被合理使用；
4. central_part_structure：是否存在合理的主零件-从属零件结构；
5. overfit_single_contact：是否只是单孔/单面偶然匹配；
6. conflict_with_larger_group：当前二元组是否阻断更完整的三元/四元组。

三、保守门控规则

D4 只允许高置信几何候选进入 accepted_geometry_candidates。

建议初始规则：

if collision_free == false:
    reject

if geometry_score < 0.80:
    reject

if independent_evidence_count < 2:
    review

if weak_single_interface_match == true:
    review

if group_size >= 6:
    review

if group_consistency_score < 0.70:
    review

只有满足：

- collision_free == true
- geometry_score >= 0.80
- independent_evidence_count >= 2
- weak_single_interface_match == false
- group_consistency_score >= 0.70
- no obvious global conflict

才允许进入 accepted_geometry_candidates。

D4 输出：

data/results/
├── accepted_geometry_candidates.json
├── review_geometry_candidates.json
├── rejected_geometry_candidates.json
└── geometry_candidate_tiering.md

============================================================
D5：Pose 验证只证明物理可行，不能证明装配正确
=============================================

保留当前 complete-pose backtracking 和 OCCT exact collision validation。

但是请改变 D5 的职责边界：

D5 只回答：

这个候选组是否存在物理可行 pose？

D5 不能回答：

这个候选组是否真实属于同一个来源装配？

因此：

1. pose 成功不能直接进入 final_groups；
2. pose 成功只能作为 accepted 的必要条件之一；
3. pose 失败可以 reject；
4. pose 不稳定或搜索边界失败进入 review；
5. 6-part 以上当前默认 review，不继续盲目扩大 beam；
6. 不要用更大 beam width 硬救六零件组，之前实验证明效果不好。

D5 输出三类：

data/results/
├── pose_validated_candidates.json
├── pose_failed_candidates.json
├── pose_uncertain_candidates.json
└── pose_validation_report.md

请在输出中记录：

- candidate_id
- checked_pose_count
- best_pose_rank
- rejection_reason_per_rank
- selected_constraint_residual
- collision_result
- occt_common_volume
- worker_status
- final_pose_status: valid | failed | uncertain

规则：

if pose_status == failed:
    reject

if pose_status == uncertain:
    review

if group_size >= 6:
    review

if pose_status == valid:
    只能进入下一层 gate，不能直接自动接受。

============================================================
D6：DeepSeek 降级为语义证据准备和校准门
=======================================

当前 DeepSeek 在匿名几何摘要上的校准结果失败：

- plausibility AUC = 0.5000
- Brier score = 0.7322
- 对许多 anonymous, geometrically feasible but provenance-wrong groups 给出高 plausibility

因此，DeepSeek 当前不能作为分组裁判。

请将 D6 改为：

D6：Semantic Evidence Preparation and Calibration Gate

D6 只做三件事：

1. 收集和整理可用语义字段；
2. 构造结构化 semantic review input；
3. 做 calibration gate。

DeepSeek 只有在 calibration gate 通过后才允许影响 grouping / reranking。

默认状态：

semantic_reranking_enabled = false

如果 calibration 失败：

1. DeepSeek 不能修改 final_score；
2. DeepSeek 不能改变 accepted/rejected；
3. DeepSeek 不能 rerank；
4. DeepSeek 只能为 review_candidates 写解释；
5. DeepSeek 可以帮助生成人工复核说明，但不能做自动裁判。

校准通过门槛建议：

- semantic AUC >= 0.70
- Brier score 明显优于 geometry-only baseline
- verdicts 至少包含 accept / reject / abstain 三类
- holdout 上不降低 auto_accept_precision
- 不增加 false_positive_count

如果没有真实语义字段，不要继续 prompt engineering。

需要优先收集的语义字段包括：

- part_name
- file_name
- part_role
- interface_type
- assembly_family
- functional_relation
- BOM hint
- CAD metadata
- known mechanical family label if available
- source template only for evaluation, never for production decision

DeepSeek 输入必须是结构化 JSON，不允许凭空从零件列表判断装配。

输出格式：

{
  "semantic_validity": "high | medium | low | unknown",
  "semantic_score": 0.0,
  "functional_reason": "...",
  "possible_system": "...",
  "risk": "...",
  "is_geometrically_feasible_but_semantically_invalid": true,
  "review_required": true,
  "suggested_action": "accept | reject | abstain | review"
}

但是在 calibration gate 失败时，以上输出只能进入报告，不得影响最终分组。

D6 输出：

data/results/
├── semantic_inputs.json
├── semantic_reviews.json
├── semantic_calibration_report.json
└── semantic_gate_decision.md

============================================================
D7：成功率优先的工程交付
========================

D7 不再以“完整自动分组 F1 最大化”为第一目标。

D7 新目标：

交付一个高精度、可人工复核的工程工具。

最终输出必须分成四类：

1. accepted_groups
   系统高度确信，可以自动接受。
2. review_groups
   几何可能成立，但证据不足、语义不明或存在冲突，需要人工复核。
3. rejected_groups
   明确几何失败、碰撞、单接口偶然匹配、低置信度或全局冲突。
4. unresolved_parts
   当前系统无法可靠分组的零件。

D7 评价指标改为：

1. auto_accept_precision
   自动接受结果的正确率。
2. false_positive_count
   错误自动接受数量。
3. review_rate
   进入人工复核的比例。
4. unresolved_parts_count
   未解决零件数量。
5. workload_reduction
   相比人工全查，减少了多少候选检查量。
6. accepted_group_count
   自动接受的高置信组数量。
7. rejected_reason_coverage
   被拒绝候选是否都有明确原因。

注意：

当前阶段允许 recall 低。
当前阶段允许 unresolved 多。
当前阶段允许 review 多。
当前阶段不允许 false positive 高。

D7 初始验收目标：

- auto_accept_precision >= 90%
- false_positive_count 明显下降
- review_candidates 覆盖主要不确定情况
- final report 能让工程师判断每个结果为什么被接受、复核或拒绝

D7 输出：

data/results/
├── final_accepted_groups.json
├── final_review_groups.json
├── final_rejected_groups.json
├── unresolved_parts.json
├── conservative_metrics.json
├── candidate_scores.csv
└── assembly_report.md

assembly_report.md 必须包含：

1. 输入零件数量；
2. 候选生成召回审计；
3. 几何候选数量；
4. accepted / review / rejected 数量；
5. pose validation 结果；
6. semantic gate 是否启用；
7. 自动接受组及证据；
8. 复核组及复核原因；
9. 拒绝组及拒绝原因；
10. 未解决零件；
11. false positive 风险分析；
12. 下一步建议。

============================================================
禁止事项
========

当前阶段禁止：

1. 不要使用 Minimum Vertex Cover 作为装配优化目标。
2. 不要引入 Reinforcement Learning。
3. 不要训练大模型。
4. 不要继续扩大多 Agent 架构。
5. 不要让 DeepSeek 在校准失败时影响结果。
6. 不要简单按 final_score 贪心选择。
7. 不要为了覆盖率强行完整分组。
8. 不要把 collision_free 当成装配正确。
9. 不要盲目扩大 beam width 来解决六零件组。
10. 不要只输出 final_groups，必须输出 accepted / review / rejected / unresolved。
11. 不要继续生成无现实功能意义的 cone/ring/block/plate 堆叠作为正样本。

============================================================
推荐最终接受逻辑
================

请把最终自动接受条件写成多条件门控，而不是单一 final_score：

accepted = (
    collision_free == true
    and pose_status == "valid"
    and geometry_score >= geometry_threshold
    and independent_evidence_count >= 2
    and group_consistency_score >= group_consistency_threshold
    and weak_single_interface_match == false
    and review_required == false
    and no_global_conflict == true
    and group_size <= 5
)

如果 semantic_reranking_enabled == true，则额外要求：

semantic_score >= semantic_threshold

如果 semantic_reranking_enabled == false，则 DeepSeek 输出不参与 accepted 判断。

建议初始阈值：

geometry_threshold: 0.80
group_consistency_threshold: 0.70
semantic_threshold: 0.75
max_auto_accept_group_size: 5

final_score 只能用于排序，不能单独决定 accepted。

============================================================
请先做什么
==========

请不要马上重写全系统。请先做以下最小修改：

1. 增加 D0 functional dataset repair；
2. 新增或修正 functional assembly templates：cover_base、shaft_hub_key、bearing_housing；
3. 为每个生成 case 输出 metadata.json；
4. 停止把随机 cone/ring/block/plate 堆叠作为功能装配正样本；
5. 增加 false_positive_audit.csv；
6. 增加 missed_true_candidates.csv 和 pruned_true_candidates.csv；
7. 增加 group_consistency.py；
8. 把 final output 改成 accepted / review / rejected / unresolved；
9. 把 DeepSeek 默认设置为 explanation-only，除非 calibration gate 通过；
10. 重新跑 frozen 12-pool benchmark 或新的 small functional benchmark；
11. 输出 conservative_metrics.json；
12. 对比修改前后的 false_positive_count、auto_accept_precision、review_rate、unresolved_parts_count。

最终请给我：

1. 修改了哪些文件；
2. 新增了哪些文件；
3. 当前 functional dataset 是否仍有无意义堆叠；
4. 当前 accepted precision；
5. 当前 false positive 数量；
6. 当前 review 数量；
7. 当前 unresolved parts 数量；
8. 哪些错误被成功转入 review；
9. 哪些错误仍然被自动接受；
10. 下一步最小修改建议。
