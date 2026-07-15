# Step 1-3 当前交付报告

日期：2026-07-08

## 总结

本轮完成了前三步的可运行闭环：

1. **Step 1：Fusion360 数据工程**
   - 扩展到本机可用的 60 个 Fusion360 assembly。
   - 成功转换 34 个可用 assembly graph。
   - 生成 pair-level relation benchmark。
   - 完成 Fusion360 标签到 SolidWorks 考试标签协议的映射。

2. **Step 2：Pair-level 神经网络 / JoinABLe 关系模型**
   - 新增轻量 pair-level relation head。
   - 训练只使用 Fusion360 split，不读取 SolidWorks 外部考试答案。
   - JoinABLe 继续作为接口候选定位工具；relation head 负责直接边和关系标签预测。

3. **Step 3：几何验证与 pose 求解**
   - 用 Fusion360 selected STEP 正边跑通几何关系识别、pose 求解、OCCT exact collision validation。
   - 5 个 Fusion360 smoke 样本全部完成流程。

当前还没有进入 Step 5 SolidWorks 外部考试。

## Step 1：Fusion360 数据工程

输出目录：

`public_cad_dataset_audit/outputs/fusion360_to_solidworks_exam_mapping_60`

核心产物：

- `fusion360_relation_benchmark.jsonl`
- `fusion360_train.jsonl`
- `fusion360_dev.jsonl`
- `fusion360_test.jsonl`
- `fusion360_pocket_mate_candidates.json`
- `fusion360_relation_mapping_summary.json`
- `solidworks_external_exam_manifest.json`
- `exam_readiness.json`

当前数据规模：

| 项目 | 数值 |
|---|---:|
| Fusion360 原始 assembly.json | 60 |
| 成功转换 assembly graph | 34 |
| part-pair 样本总数 | 10,072 |
| 正边 | 396 |
| closed-world 负边 | 9,676 |
| train/dev/test | 8,970 / 380 / 722 |

映射标签覆盖：

| 标签 | 样本数 |
|---|---:|
| `planar_mate` | 303 |
| `coaxial` | 186 |
| `clearance` | 4 |
| `pocket_mate` | 13 |
| `planar_align` | 0 |

注意：

- `pocket_mate` 当前是候选挖掘标签，仍需人工抽查。
- 为保证 train split 覆盖 `pocket_mate`，脚本把 `19051_26735260` 从 test 移到 train；仍保持 assembly-level split，没有拆分同一 assembly。
- `planar_align` 当前没有 Fusion360 映射样本，不能声称已学习。

## Step 2：Pair-level 关系模型

新增脚本：

`public_cad_dataset_audit/train_pair_relation_head.py`

输出目录：

`public_cad_dataset_audit/outputs/step123_pair_relation_head`

核心产物：

- `pair_relation_head.pkl`
- `pair_relation_model_metrics.json`
- `pair_relation_predictions_dev_test.json`
- `pair_relation_model_report.md`

模型边界：

- 使用 sklearn MLP，属于轻量 relation head。
- 不是最终大模型。
- 不使用 SolidWorks 标签训练。
- 输入为 Fusion360 结构化字段、contact/joint 证据、part name evidence。
- JoinABLe 仍是接口候选定位网络，不被替代。

主要指标：

| 指标 | dev | test |
|---|---:|---:|
| direct connection AUC | 1.0 | 1.0 |
| relation micro-F1 | 0.768 | 0.656 |

需要谨慎解释：

- direct AUC 高，主要因为 closed-world 负样本和正样本在 Fusion360 元数据上很可分。
- `pocket_mate` 在 train 中只有 1 条，在 test 中有 12 条，当前 test recall 为 0。
- 这说明 `pocket_mate` 数据还不够，不说明模型路线失败。

## Step 3：几何验证与 pose 求解

新增脚本：

`public_cad_dataset_audit/run_step3_fusion_pose_smoke.py`

输出目录：

`public_cad_dataset_audit/outputs/step123_pose_smoke`

核心产物：

- `step3_pose_smoke_summary.json`
- `step3_pose_smoke_report.md`
- `cases/*/known_group_output/assembly_relations.json`
- `cases/*/known_group_output/pose_validation.json`
- `cases/*/known_group_output/assembly.step`

Smoke 结果：

| 项目 | 数值 |
|---|---:|
| Fusion360 STEP 正边样本 | 5 |
| 流程成功 | 5 |
| pose valid | 3 |
| pose failed | 1 |
| pose uncertain | 1 |
| 与 Fusion360 映射标签有交集 | 3 |

解释：

- pose valid 只证明物理 pose 可行，不证明功能正确。
- pose failed/uncertain 暴露的是几何求解器能力边界，不能直接反推 Fusion360 标签错误。
- 当前 Step 3 已经能作为后续模型候选的保守验证层。

## 当前结论

前三步已经形成最小闭环：

```text
Fusion360 数据
  → 统一 pair-level relation benchmark
  → 轻量关系模型训练/预测
  → 几何关系识别 + pose/碰撞验证
```

但还不能说最终模型已经完成。主要短板：

1. Fusion360 使用量仍远小于官方完整数据集。
2. `pocket_mate` 样本太少，且仍需人工审核。
3. pair-level 模型目前是轻量结构化模型，不是完整 B-Rep/JoinABLe graph neural relation model。
4. Step 3 仍有 pose failed/uncertain，需要继续提升接口定位和 pose 搜索质量。

## 下一步最小建议

1. 抽查并清洗 `fusion360_pocket_mate_candidates.json`。
2. 扩大 Fusion360 assembly 转换规模，不停留在 60 个 assembly。
3. 用 JoinABLe joint dataset 训练或校准真正的 interface-level 模型输出。
4. 把 relation head 的输入从结构化字段升级为 JoinABLe/B-Rep graph embedding。
5. 再进入 Step 4 保守 accepted/review/rejected 决策层。

