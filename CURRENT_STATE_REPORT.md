# CURRENT_STATE_REPORT

生成时间：2026-07-08  
项目根目录：`C:\Users\11049\Desktop\Model_match`

## 0. 按新 Prompt 对齐后的当前结论

当前项目应定义为“确定性 CAD 装配关系恢复”，不是 mixed-pool group discovery，也不是全自动装配 Agent。

现在的输入假设是：给定一组已知属于同一个机械装配体、且存在唯一确定装配关系的 CAD / STEP / SolidWorks 零件。

系统需要恢复：

1. same-assembly 零件对中的直接装配边 `direct assembly edge`；
2. 每条直接边的关系类型；
3. 对应接口实体；
4. pairwise pose / transform；
5. 全局一致的 assembly graph；
6. 每个可放置零件的 global placement；
7. 最终可被 SolidWorks runner 读取的 `solidworks_assembly_plan.json`。

最关键的现实状态（2026-07-08 审计更新）：

1. **Step 1 已大幅推进**：a00_a01 扩展版 benchmark 已有 **1,068 assemblies、15,039 positive direct edges、488,525 pair samples**。五类 relation label 均有覆盖，已超过"几百个 assembly、几千条 positive edges"的门槛。
2. **数据组织已到位**：`fusion360_assembly_benchmark.jsonl`（assembly-level）+ `fusion360_pair_relation_benchmark.jsonl`（pair-level）+ train/dev/test 严格按 assembly_id 隔离。
3. **planar_align 已修复**：通过 split repair 确保 dev/test 也有样本（train: 11, dev: 1, test: 1）。
4. **pocket_mate 仍是瓶颈**：206 条候选全部标记为 weak_label，需要人工审核 `review_sheet.csv` 后才能升级为可信训练样本。旧轻量模型 pocket_mate recall 仍为 0。
5. Step 2 轻量 pair relation head 已在新 benchmark 上初步训练（旧小数据），但需要用新 a00_a01 数据重新训练。
6. Step 3 几何验证已有基础，可复用，但需要稳定输出 `pairwise_transform`。
7. Step 4 有 accepted/review/rejected 保守输出雏形，但缺少明确的 assembly graph selection、pose graph propagation、cycle consistency check 和 `solidworks_assembly_plan.json` 生成器。
8. Step 5 SolidWorks 五组外部考试标签已准备，但必须保证推理阶段不读取 `human_labels.json`。

## 0.1 最新审计结论（2026-07-08 二次审计）

### Step 1 数据工程状态：显著推进

a00_a01 扩展版 benchmark 已构建完成，关键数字：

| 指标 | 旧版 (a00 only) | 新版 (a00_a01) | 验收门槛 |
|------|----------------|----------------|---------|
| assemblies | 508 | **1,068** | ≥ 几百 |
| positive edges | 6,548 | **15,039** | ≥ 几千 |
| pair samples | 257,031 | **488,525** | - |
| coaxial | 3,424 | **7,485** | ✅ |
| planar_mate | 4,261 | **10,376** | ✅ |
| clearance | 78 | **153** | ✅ |
| pocket_mate | 113 | **206** | ⚠️ 全 weak |
| planar_align | 2 (train only) | **13** (train:11, dev:1, test:1) | ✅ 已修复 |

输出文件位置：`public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/`

### pocket_mate 审核状态：等待人工

- `review_sheet.csv` 已生成：`public_cad_dataset_audit/outputs/pocket_mate_review_pack_50/review_sheet.csv`
- 50 个候选 pair 的 STEP/PNG 已复制到 `candidates/` 子目录
- 当前 `review_label` 列为空，**需要人工在 SolidWorks 中查看并填写**
- 填写完成后可运行脚本将人工标签合并回 benchmark，pocket_mate 从 weak_label 升级为 verified

### 当前阻塞项

1. **pocket_mate 人工审核** — 206 条候选全为 weak_label，Step 2 模型 pocket_mate recall 为 0
2. **Step 2 需用新 a00_a01 数据重新训练** — 旧模型只在旧小数据上训练
3. **Step 3/4 接入链路** — 需要标准化 Step 2→Step 3→Step 4 的数据流

---

## 1. 当前 repo 关键文件

### 1.1 Fusion360 数据转换

已有文件：

- `public_cad_dataset_audit/fusion360_common.py`
- `public_cad_dataset_audit/convert_fusion360_to_assembly_graph.py`
- `public_cad_dataset_audit/map_fusion360_to_solidworks_exam.py`
- `public_cad_dataset_audit/audit_fusion360_assembly_dataset.py`
- `public_cad_dataset_audit/build_pair_dataset.py`
- `joinable_migration_audit/convert_fusion_joint_to_common_schema.py`
- `joinable_migration_audit/fusion_joint_schema_probe.py`
- `joinable_migration_audit/joint_interface_schema.md`

已有输出：

- `public_cad_dataset_audit/outputs/fusion360_assembly_graphs_60/`
- `public_cad_dataset_audit/outputs/fusion360_to_solidworks_exam_mapping_60/`

旧小规模统计：

```json
{
  "assembly_count": 34,
  "sample_count": 10072,
  "positive_count": 396,
  "negative_count": 9676,
  "mapped_label_counts": {
    "clearance": 4,
    "coaxial": 186,
    "planar_mate": 303,
    "pocket_mate": 13
  },
  "step_missing_count": 520
}
```

判断：

- `fusion360_common.py` 可复用，但要把输出从“孤立 pair benchmark”提升为“assembly-level benchmark + pair-level labels”。
- `map_fusion360_to_solidworks_exam.py` 里有标签映射逻辑可复用，但它也读取 SolidWorks exam label，后续要严格拆分开发数据构建和考试判卷。
- 仅靠 Joint Dataset 不能构建 same-assembly non-edge，因为 Joint Dataset 是成对 joint 数据；same-assembly non-edge 需要 Assembly Dataset。

### 1.2 Assembly-level benchmark ✅ 已完成

已有产出（a00_a01 扩展版）：

- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/fusion360_assembly_benchmark.jsonl`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/fusion360_pair_relation_benchmark.jsonl`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/fusion360_train.jsonl`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/fusion360_dev.jsonl`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/fusion360_test.jsonl`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/fusion360_pocket_mate_candidates.json`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/label_mapping_report.md`
- `public_cad_dataset_audit/outputs/fusion360_assembly_benchmark_a00_a01/split_report.md`

构建脚本：`public_cad_dataset_audit/build_fusion360_assembly_benchmark.py`

每个 assembly 样本包含：`assembly_id`, `parts`, `all_candidate_pairs`, `positive_edges`, `same_assembly_non_edges`, `relation labels`, `interface face/edge`, `joint/contact`, `pairwise_transform`, `geometry file paths`。

缺口：pocket_mate 206 条仍为 weak_label，需要人工审核后升级。

### 1.3 Pair relation benchmark

已有基础：

- `public_cad_dataset_audit/build_pair_dataset.py`
- `public_cad_dataset_audit/map_fusion360_to_solidworks_exam.py`
- `public_cad_dataset_audit/outputs/fusion360_to_solidworks_exam_mapping_60/fusion360_relation_benchmark.jsonl`
- `public_cad_dataset_audit/build_fusion360_joint_relation_benchmark.py`（上一轮新增，尚未成功完整产出）

判断：

- Pair-level benchmark 方向对，但新 Prompt 下它必须从 assembly-level benchmark 派生，不能替代 assembly-level benchmark。
- `direct_edge_score` 的语义必须改成：同一 assembly 内两个零件是否存在直接装配边，而不是两个零件是否属于同组。

### 1.4 JoinABLe / pair prediction

已有文件：

- `cad_assembly_agent/tools/joinable_interface_predictor/pretrained_joinable_predictor.py`
- `cad_assembly_agent/tools/joinable_interface_predictor/rule_interface_predictor.py`
- `cad_assembly_agent/tools/joinable_interface_predictor/merge_candidate_providers.py`
- `cad_assembly_agent/tools/joinable_interface_predictor/JOINABLE_TOOL_CONTRACT.md`
- `sw/joinable_candidate_provider.py`
- `sw/audit_joinable_candidate_provider.py`
- `sw/audit_joinable_holdout.py`
- `public_cad_dataset_audit/train_pair_relation_head.py`
- `joinable_gpu_reproduction/`
- `joinable_migration_audit/`

已有 Step 2 原型结果：

- `public_cad_dataset_audit/outputs/step123_pair_relation_head/`
- 模型：`lightweight_pair_relation_head`
- 旧小数据 test relation micro-F1 约 0.6565
- `pocket_mate` test recall 为 0

判断：

- 轻量模型可作为 Step 2 baseline。
- JoinABLe 目前更适合作为 interface candidate/localization 工具或思路，不应宣称已完成生产级关系模型。
- Step 2 输出必须是：
  - `direct_edge_score`
  - `relation_type_scores`
  - `interface_candidates`
  - evidence / failure reasons
- Step 2 不能输出 accepted。

### 1.5 几何验证

已有文件：

- `sw/known_group_assembly.py`
- `sw/pair_edge.py`
- `sw/placement_validation.py`
- `sw/functional_pose_worker.py`
- `sw/real_known_group_worker.py`
- `sw/true_group_validation_worker.py`
- `sw/group_validation_worker.py`
- `sw/audit_dataset_pose_validation.py`
- `sw/audit_true_group_pose.py`
- `sw/audit_pose_valid_false_groups.py`
- `cad_assembly_agent/tools/occt_pose_validator/occt_pair_pose_validator.py`
- `public_cad_dataset_audit/run_step3_fusion_pose_smoke.py`
- `public_cad_dataset_audit/run_real_pose_validation.py`

已有 smoke 结果：

```json
{
  "selected_count": 5,
  "success_count": 5,
  "pose_valid_count": 3,
  "pose_failed_count": 1,
  "pose_uncertain_count": 1
}
```

判断：

- 几何验证基础可复用。
- 新 Prompt 下 Step 3 必须稳定输出：
  - `pose_status`
  - `pairwise_transform`
  - `selected_constraint_residual`
  - `collision_result`
  - `geometry_evidence`
  - `failure_reasons`
- `pose_status=valid` 只能证明物理可行，不能直接证明装配正确。

### 1.6 Pose solver

已有相关文件：

- `sw/known_group_assembly.py`
- `sw/functional_pose_worker.py`
- `sw/dataset_pose_case_worker.py`
- `sw/run_frontier_pose_experiment.py`
- `cad_assembly_agent/tools/occt_pose_validator/occt_pair_pose_validator.py`

缺口：

- pairwise transform 的统一 schema 尚不稳定。
- 还没有面向 Step 4 pose graph propagation 的标准输出。

### 1.7 Conservative gate

已有文件：

- `sw/conservative_pipeline.py`
- `sw/run_conservative_delivery.py`
- `sw/global_optimizer/group_consistency.py`
- `cad_assembly_agent/tools/assembly_graph_builder/build_conservative_graph.py`
- `public_cad_dataset_audit/evaluate_conservative_real_benchmark.py`
- `public_cad_dataset_audit/data/results/`
- `cad_assembly_agent/reports/conservative_graph/`

已有保守结果示例：

```json
{
  "accepted_group_count": 0,
  "false_positive_count": 0,
  "review_group_count": 1190,
  "review_rate": 0.9991603694374476,
  "unresolved_parts_count": 49,
  "semantic_reranking_enabled": false
}
```

判断：

- 保守方向正确，但新 Prompt 下输出对象应从 `accepted_groups` 转为 `accepted_edges`。
- 自动接受条件需要加入 `pose_graph_consistency == true` 和 `no_global_conflict == true`。

### 1.8 Pose graph propagation

审计发现：

- 有旧的 placement / assembly manifest 相关代码：
  - `sw/compute_manifest.py`
  - `sw/build_assembly.py`
  - `cad_assembly_agent/tools/cad_loader/freeze_pair_truth.py`
  - `sw/*/assembly_manifest.json`
- 但没有发现完整、独立、面向新 Prompt 的 pose graph propagation 模块。

缺口：

- 需要新增或整理模块完成：
  - 从 `accepted_edges` 构建 assembly graph；
  - root part 固定为单位矩阵；
  - 沿 `pairwise_transform` 传播；
  - 输出每个零件 `transform_world_from_part`；
  - 检查 cycle consistency；
  - 检查同一零件多路径位姿冲突；
  - 冲突时相关 edge/component 进入 review。

### 1.9 SolidWorks JSON 生成

审计发现：

- 未发现完整的 `solidworks_assembly_plan.json` 生成链路。
- 现有 assembly manifest / build assembly 代码可作为参考，但不满足新 Prompt 的最终 JSON schema。

缺口：

- 必须新增 `solidworks_assembly_plan.json` 生成器。
- 最小 schema 至少包含：
  - `schema_version`
  - `case_id`
  - `input_parts`
  - `placements`
  - `accepted_edges`
  - `review_edges`
  - `rejected_edges`
  - `unresolved_parts`
  - `generation_policy`
- `generation_policy.used_human_labels` 必须为 false。

### 1.10 SolidWorks test cases

已有目录：

- `sw/1`
- `sw/2`
- `sw/3`
- `sw/4`
- `sw/5`
- `sw/phase5_annotation_pack/`

已确认答案文件：

- `sw/phase5_annotation_pack/case_1/human_labels.json`
- `sw/phase5_annotation_pack/case_2/human_labels.json`
- `sw/phase5_annotation_pack/case_3/human_labels.json`
- `sw/phase5_annotation_pack/case_4/human_labels.json`
- `sw/phase5_annotation_pack/case_5/human_labels.json`

判断：

- SolidWorks 五组可作为 Step 5 外部考试。
- 推理阶段禁止读取 `human_labels.json`。
- 判卷阶段才允许读取。

### 1.11 human_labels 判卷脚本

相关文件：

- `public_cad_dataset_audit/map_fusion360_to_solidworks_exam.py`
- `public_cad_dataset_audit/audit_solidworks_external_mapping.py`
- `public_cad_dataset_audit/evaluate_conservative_real_benchmark.py`
- `cad_assembly_agent/tools/cad_loader/freeze_pair_truth.py`

风险：

- 这些文件中存在读取答案/manifest 的逻辑。
- 后续必须拆出明确的 inference runner 和 scoring runner。
- `human_labels.json` 只能在 scoring runner 里读取。

## 2. 已有功能

1. Fusion360 Assembly JSON 到统一 graph schema 的初版转换。
2. Fusion360 / Joint / SolidWorks 五类关系标签空间的初版映射。
3. 小规模 pair-level relation head baseline。
4. JoinABLe / rule interface candidate provider 的冻结工具和审计入口。
5. STEP / OCCT 几何验证基础。
6. known-group 装配关系输出入口。
7. group consistency / conservative gate 雏形。
8. accepted/review/rejected/unresolved 风格的保守输出雏形。
9. SolidWorks 1..5 外部考试标签包。

## 3. 缺失功能

1. `fusion360_assembly_benchmark.jsonl`。
2. assembly-level sample schema：
   - `parts`
   - `all_candidate_pairs`
   - `positive_edges`
   - `same_assembly_non_edges`
   - `pairwise_transform`
3. 从 assembly-level benchmark 派生的 `fusion360_pair_relation_benchmark.jsonl`。
4. 严格按 `assembly_id` 隔离的 train/dev/test split。
5. 扩大版 `pocket_mate_candidates` 和人工抽查入口。
6. 稳定的 Step 2 `direct_edge_score` / `relation_type_scores` 输出。
7. Step 2 输出到 Step 3 几何验证的 runner。
8. Step 3 标准化 `pairwise_transform` 输出。
9. pose graph propagation。
10. cycle consistency check。
11. `solidworks_assembly_plan.json` 生成器。
12. SolidWorks inference/scoring 阶段隔离报告。

## 4. 风险点（2026-07-08 更新）

1. **pocket_mate 人工审核阻塞**：
   - 206 条 pocket_mate 候选全为 weak_label。
   - 50 条已导出到 `review_sheet.csv` 等待审核。
   - 审核完成前 Step 2 模型的 pocket_mate recall 无法提升。

2. **数据泄漏**：
   - `human_labels.json` 必须只在最终判卷阶段读取。

3. **任务偏移**：
   - 不应再做 group discovery。
   - 输入零件默认属于同一 assembly。

4. **负样本语义正确**：
   - ✅ 当前 negative 已写为 `same_assembly_non_edge`。

5. **planar_align 稀疏**：
   - dev/test 各仅 1 条，模型可能无法稳定学习该标签。

6. **Step 4 缺口大**：
   - 当前还没有完整的 global placement / SolidWorks plan JSON 生成。

7. **Git 元数据异常**：
   - 根目录能看到 `.git`，但 `git rev-parse --show-toplevel` 返回 not a git repository。

## 5. 可复用代码

优先复用：

1. `public_cad_dataset_audit/fusion360_common.py`
2. `public_cad_dataset_audit/convert_fusion360_to_assembly_graph.py`
3. `public_cad_dataset_audit/train_pair_relation_head.py`
4. `sw/known_group_assembly.py`
5. `sw/global_optimizer/group_consistency.py`
6. `cad_assembly_agent/tools/occt_pose_validator/occt_pair_pose_validator.py`
7. `cad_assembly_agent/tools/joinable_interface_predictor/*`
8. `sw/build_assembly.py` 和 `sw/compute_manifest.py` 中的旧 placement/assembly 逻辑，可作为 SolidWorks plan JSON 的参考，但不能直接当最终方案。

## 6. 不应继续扩展的方向

1. mixed-pool group discovery。
2. Reinforcement Learning。
3. 复杂多 Agent 架构。
4. 让 DeepSeek/Qwen 做最终裁判。
5. 用 SolidWorks case_id、文件名、编号规则做答案硬编码。
6. 用 `collision_free` 或 `pose_valid` 直接推出装配正确。
7. 用 `final_score` 单独决定 accepted。
8. 用随机 cone/ring/block/plate 堆叠证明功能装配能力。

## 7. 最小可执行修改计划（2026-07-08 更新）

### 状态：Phase A（Step 1 数据工程）已基本完成

a00_a01 扩展版 benchmark 已构建，输出文件齐全。**唯一阻塞项是 pocket_mate 人工审核**。

### Phase A'：pocket_mate 人工审核（当前优先）

1. 人工打开 `review_sheet.csv` 对应的 STEP 文件
2. 为 50 个候选填写 `review_label`（true_pocket_mate / false_pocket_mate / uncertain）
3. 运行脚本将审核结果合并回 benchmark
4. 更新 `label_mapping_report.md`

### Phase B：Step 2 在新 a00_a01 数据上重新训练

目标：用 1,068 assemblies 的数据重新训练轻量 pair relation head。

最小改动：

1. 修改 `train_pair_relation_head.py` 指向 a00_a01 数据路径
2. 适配新字段：`direct_edge_score`, `relation_type_scores`, `interface_candidates`
3. 特别关注 pocket_mate recall（如果人工审核已完成）
4. 输出：`relation_model_predictions.jsonl`, `model_metrics.json`, `relation_model_report.md`, `confusion_matrix.csv`

注意：
- 必须使用 a00_a01 benchmark（1,068 assemblies），不是旧的 60-assembly mapping
- 如果 pocket_mate 人工审核未完成，模型仍可训练但 pocket_mate 指标不可靠

### Phase C：Step 3 几何验证接口

（与之前计划相同，但输入改为 Step 2 在新数据上的输出）

### Phase D：Step 4 pose graph + SolidWorks plan

（与之前计划相同）
    and independent_evidence_count >= 2
    and weak_single_interface_match is False
    and pose_graph_consistency is True
    and no_global_conflict is True
)
```

### Phase E：Step 5 exam protocol

目标：保证 SolidWorks 外部考试隔离。

最小新增：

1. `public_cad_dataset_audit/audit_exam_protocol.py`
2. `exam_protocol_report.md`

必须说明：

- 推理阶段输入只能是 STEP/geometry/config。
- 推理阶段不读取 `human_labels.json`。
- 只有判卷阶段读取 `human_labels.json`。
- 最终考试产物是 `solidworks_assembly_plan.json`。

## 8. 下一步建议

下一步不要直接训练网络，也不要继续 group discovery。

应先完成 Phase A：

1. 用 Assembly Dataset 构建 `fusion360_assembly_benchmark.jsonl`。
2. 从它派生 `fusion360_pair_relation_benchmark.jsonl`。
3. 明确 `same_assembly_non_edge` 负样本。
4. 生成 split/report。
5. 再用 Joint Dataset 补充 interface supervision 和 `pocket_mate_candidates`，但不要让 Joint Dataset 替代 assembly-level benchmark。

完成这一步后，Step 2 才有可靠训练目标。
