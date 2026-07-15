# P1–P3 功能装配与保守交付报告

## 结论

P1、P2、P3 已完成并冻结。功能数据集、功能真值 mixed-pool benchmark、候选召回审计、保守四分类交付、DeepSeek explanation-only 门控和 P3 局部接口召回修复均已实际运行。

当前结果优先保证安全：

- 自动接受组：0
- 自动接受 false positive：0
- accepted precision：不可估计（没有自动接受样本，不能报告为 100%）
- 即时人工复核队列：72
- 延后复核候选：5,542
- 全部 review 候选：5,614
- rejected：386
- unresolved pool-local parts：42
- semantic reranking：关闭

这不是“成功自动分组”的结果，而是一次有效的保守性基线：系统没有错误自动接受，但复核候选仍多、所有零件仍 unresolved。

## P1：功能装配数据集

新增 `functional_dataset_v1`，仅包含三类功能模板，每类 3 个参数变体：

| Family | 工程含义 | Case 数 |
|---|---|---:|
| `cover_base` | 带矩形止口、重复孔阵列和双定位销的底座/盖板 | 3 |
| `shaft_hub_key` | 带轴孔、轴/毂键槽和平键的扭矩传递组件 | 3 |
| `bearing_housing` | 轴承座、轴承、轴和带止口端盖 | 3 |

数据集包含 33 个正样本零件 STEP、27 个受控负样本 STEP、9 个装配 STEP 和 9 份 `metadata.json`。69/69 个 STEP 文件均完成 OCCT 重新读取与 shape-valid 检查。

每个 case 都具备：

- `part_role`
- `interface_types`
- `functional_relation`
- 至少两个独立功能/几何证据
- `valid_groups`
- `interchangeable_parts`
- `easy_negative`
- `geometric_hard_negative`
- `semantic_hard_negative`

功能正样本中没有匿名 cone/ring/block/plate 随机堆叠。旧 primitive generator 仍可用于几何 smoke test，但已明确标记：

- `dataset_intended_use = geometry_smoke_only`
- `functional_positive_eligible = false`

视觉总览：[functional_contact_sheet.png](C:/Users/11049/Desktop/Model_match/sw/data/functional_dataset_v1/functional_contact_sheet.png)

## P2：功能 benchmark 与保守交付

建立 3 个 variant-disjoint mixed pool：

- train / calibration / test 各 1 个 pool
- 每个 pool 14 个匿名零件
- 每个 pool 含 3 个真实功能组和 3 类受控负样本
- truth basis 为 `functional_validity`
- `source_id_is_production_truth = false`

候选召回审计结果：

- truth interfaces：33
- generated：33，召回率 100%
- pruning 后保留：33，召回率 100%
- planar mate：12/12
- clearance：21/21
- missed true candidates：0
- pruned true candidates：0

保守流水线结果：

- geometry accepted：0
- geometry review：5,624
- geometry rejected：376
- pose valid：26
- pose failed：10
- pose uncertain / 未检查：5,588
- final accepted：0
- final review：5,614
- final rejected：386
- rejected reason coverage：100%

9 个受控功能负样本中：

- 自动接受：0
- 1 个 `geometric_hard_negative` 的精确二元候选被拒绝
- 其余 8 个负关系只出现在 review 候选的更大组合中，没有进入 accepted
- 3 个 `semantic_hard_negative` 全部未被自动接受

DeepSeek 未被调用。72 份即时 review semantic input 已使用结构化 role/interface/family/relation 字段生成，但 calibration gate 保持关闭，语义结果不能改变分组、排序或最终分数。

## P3：确定性候选召回修复

P3 未训练模型、未引入 RL、未扩大 beam、未启用 LLM reranking。

修改内容：

- 为小型零件增加有界局部平面接口提取；
- 保留全局大平面阈值，避免无界增加平面噪声；
- prescreen 显式记录物理证据类型和数量；
- 候选按独立证据数和粗几何分数确定性排序；
- 提高每个 part-pair 的保留上限，防止正确接口过早被剪枝；
- 将 `candidate_origin` 写入审计原因。

前后对比：

| 指标 | P2 baseline | P3 |
|---|---:|---:|
| generated typed-edge recall | 81.82% | 100.00% |
| post-pruning typed-edge recall | 78.79% | 100.00% |
| mean pair reduction | 31.14% | 31.14% |

P3 修复了候选阶段丢失真值接口的问题，没有牺牲 pair-level reduction。但它没有解决组级排序：9 个完整真组中 8 个处于 deferred review，1 个因 geometry score 低被拒绝，0 个进入即时 review frontier。

## 文件变更

主要新增代码：

- `sw/functional_dataset_generator.py`
- `sw/schemas/functional_metadata.schema.json`
- `sw/build_functional_pools.py`
- `sw/prepare_functional_semantic_inputs.py`
- `sw/audit_functional_negatives.py`
- `sw/freeze_p123_delivery.py`
- `sw/tests/test_functional_dataset.py`

主要修改代码：

- `sw/constraints.py`
- `sw/pool_index.py`
- `sw/match_scoring.py`
- `sw/configs/pool_pipeline.json`
- `sw/conservative_pipeline.py`
- `sw/sw_dataset_generator/templates/library.py`
- `sw/sw_dataset_generator/write_ground_truth.py`
- `sw/sw_dataset_generator/batch_generate.py`
- `sw/sw_dataset_generator/README.md`

主要新增数据与报告：

- `sw/data/functional_dataset_v1/`
- `sw/data/functional_mixed_pools_v1/`
- `sw/data/functional_results/`
- `P123_DELIVERY_FREEZE.json`

## 对要求的逐项回答

1. 修改文件：见“文件变更”。
2. 新增文件：见“文件变更”，完整产物哈希在 `P123_DELIVERY_FREEZE.json`。
3. 当前 functional dataset 是否仍有无意义堆叠：没有；旧 primitive 数据只允许作为 geometry smoke。
4. 当前 accepted precision：不可估计，因为 accepted group count 为 0。
5. 当前 false positive：0。
6. 当前 review：72 个即时复核，5,542 个 deferred，共 5,614 个。
7. 当前 unresolved parts：42 个 pool-local parts。
8. 成功转入 review 的错误：8 个受控负关系只存在于 review 组合中，包括全部 3 个 semantic hard negatives；另 1 个 geometric hard negative 被明确拒绝。
9. 仍被自动接受的错误：没有。
10. 下一步最小修改：只改 review 排序，不放宽 accepted gate。增加“完整功能拓扑代理”排序特征，例如中心件覆盖、接口角色多样性、关键小零件依附关系和完整 3/4 件结构；先要求 9 个真组进入 bounded review frontier，再考虑减少 deferred 数量。不得用该调整直接提升自动接受。

## 验证

- 单元测试：55/55 passed
- STEP roundtrip / shape validity：69/69 passed
- P1–P3 freeze checks：全部通过
- 冻结文件：[P123_DELIVERY_FREEZE.json](C:/Users/11049/Desktop/Model_match/P123_DELIVERY_FREEZE.json)
- 最终指标：[conservative_metrics.json](C:/Users/11049/Desktop/Model_match/sw/data/functional_results/conservative_metrics.json)
- 候选召回报告：[candidate_recall_audit.md](C:/Users/11049/Desktop/Model_match/sw/data/functional_results/candidate_recall_audit.md)
- hard-negative 审计：[functional_hard_negative_audit.md](C:/Users/11049/Desktop/Model_match/sw/data/functional_results/functional_hard_negative_audit.md)
