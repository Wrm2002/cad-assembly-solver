# SolidWorks / STEP 零件匹配系统严格技术路线对齐报告

日期：2026-07-07  
审查对象：`TECHNICAL_REVIEW_AND_MODIFICATIONS.md`、DeepSeek 所改代码、当前功能数据集、JoinABLe 复现实验、保守分组流水线与独立 holdout。

## 1. 结论

DeepSeek 对项目上限的宏观判断基本正确：几何、pose 和碰撞只能回答“能否装”，不能单独回答“是否属于同一功能装配”。但它给出的若干工程结论超过了现有证据：JoinABLe 不能按独立物理证据计数，单个 Qwen 正样本不能代表校准成功，降低门控阈值没有提高真组前沿召回，synthetic role/family 字段直接输入模型属于评估真值泄漏。

本轮已经将路线重新固定为：

1. 高召回候选生成；
2. 拓扑方向、接口极性和 pose 只验证物理可行性；
3. 多条件保守门控控制自动接受；
4. JoinABLe 仅进入 shadow recall / review corroboration；
5. Qwen/DeepSeek 仅进入 explanation-only 和独立校准；
6. 未经校准的模型输出不得改变分组、排序、接受或拒绝；
7. 不强行完整分组，保留 review 和 unresolved。

当前系统比 DeepSeek 修改后的版本更安全，pose 能力也有实质提升，但仍不具备工程投产条件。核心瓶颈已从 pose 转移到候选组排序和复核前沿召回。

## 2. DeepSeek 技术认知逐项审计

| DeepSeek 判断 | 审计结论 | 证据与处理 |
|---|---|---|
| 几何可行不等于功能正确 | 保留 | 与 hard negative 实验和项目最高原则一致。 |
| 匿名几何文本导致 DeepSeek 校准失败 | 基本正确 | 原结果 AUC=0.50、Brier=0.7322，输入缺少真实生产语义。 |
| JoinABLe 应替代手工规则 | 未证实 | 27 joints 上 pretrained top-1 优于规则，但 top-10 为 0.630，低于规则 0.741；union top-10 为 0.815。证据支持候选并集，不支持替代。 |
| JoinABLe 可计为一个独立证据 | 撤销 | JoinABLe 与手工规则都观察同一 B-Rep 几何，provider agreement 不是新增物理约束；softmax 0.85 未做域校准。 |
| 阈值 0.80→0.65、0.70→0.60 | 撤销 | 真组 review-frontier recall 仍为 22.22%，review 从 5,616 增至 5,990，rejected 从 384 降至 10，只增加人工负担。 |
| Qwen-VL 单个 cover_base 样本成功 | 只能算 smoke test | 声称的 live 文件、缓存、日志与图片均不存在，结果无法审计；输入还含 synthetic role/family 真值。 |
| 多模态模型可以直接补足功能语义 | 待证 | 零件 tray 丢失相对尺度与装配 pose，最多提供外形线索。必须使用无泄漏 holdout 做 AUC、Brier、FP 和 abstain 校准。 |
| AUC≥0.70 即可开启 reranking | 条件不足 | 还必须优于 geometry baseline、不降低 auto-accept precision、不增加 FP，并覆盖 accept/reject/abstain。通过后也只获得候选资格，不自动启用。 |
| 建立角色兼容知识库 | 有条件保留 | 只能使用真实 CAD metadata、BOM 或经过验证的角色推断；不能把 synthetic truth 当生产输入。 |
| 扩到 50+ case / 10–15 families | 暂缓 | 先完成外部机械工程师签核、拓扑变化和真实 CAD holdout；否则只是扩大同模板偏差。 |
| 当前方案与 SOTA 有明确替代关系 | 证据不足 | 现有实验不能支持“替代手工特征”或“达到 SOTA”的域内结论。 |

## 3. 对 DeepSeek 代码改动的实际核验

### 3.1 不可审计或未真正接入的部分

- 文档宣称新增的 `smoke_multimodal.py`、`test_qwen_live.py`、`debug_qwen.py`、`tmp_v2.py` 及三张 tray PNG 均不存在。
- 当前 functional pool 和 holdout 的 `pruned_candidates.json` 中没有 `joinable_interface_rank` 边；JoinABLe 未进入 `sw` 主路径。
- 文档引用的 `merge_candidate_providers.py` 位于另一个工具目录，不代表已集成到当前流水线。
- 原多模态路径读取 `filepath`，而当前 part index 使用 `source_file`，并会产生 `.step.step`；实际离线运行 6/6 为 `no_step_files_found`。
- 原 Qwen 输入直接包含 `functional_semantics` 中的 `part_role`、`part_name`、`assembly_family` 和 `functional_relation`，属于 synthetic evaluation truth 泄漏。
- 原 `off` 模式可能读取旧 live cache，使“关闭模型”仍受历史模型输出影响。

### 3.2 已修正的部分

- 恢复保守阈值：geometry 0.80、group consistency 0.70。
- `learned_evidence_enabled=false`，JoinABLe 仅为 `review_corroboration_only`。
- provider agreement 明确不增加 `independent_evidence_count`。
- 修复 STEP 路径解析和 `.step.step` 问题。
- 默认剔除 synthetic/holdout truth，只允许真实生产 CAD metadata 进入模型输入。
- `off` 模式在读取 cache 前立即 abstain。
- Qwen CLI 已显式区分 text / multimodal，且模型输出不影响 grouping。
- 建立 36 样本、三分割、先锁 hash 后评估的无泄漏多模态校准框架。
- 已移除技术报告中暴露的 API key 值；建议相关 key 轮换。

## 4. 本轮新增的几何基础修复

本轮继续排查 3/3 cover_base pose 失败，发现不是 beam width 不够，而是几何表示存在四个基础问题：

1. OCCT 平面法向未应用 face orientation，盖板底面和底座顶面可能都被记为同向；
2. 圆柱面未区分 convex 外圆与 concave 孔内壁，定位销可能被当成孔；
3. text STEP 记录覆盖了 OCCT 的面积和拓扑极性信息；
4. 重复等半径孔在 pose 搜索前被压成一个 radius bucket，双定位销无法分配到两个孔。

对应修复：

- 提取并保留 oriented planar normal；
- 提取 `surface_polarity=convex|concave|unknown`；
- 用 OCCT area/polarity 补全 text parser 的稳定实体顺序；
- known polarity 下，clearance 只允许 convex 小圆柱进入 concave 大圆柱；
- 不再用宽松 coaxial 假设覆盖已知的 convex-concave clearance；
- pose 模式保留重复圆柱面的独立 feature hypothesis；
- per-pair pruning 为 planar seating 保留有界类型配额；
- 仅当当前 pose 满足 clearance/planar/pocket 约束时，降低 AABB broad-phase 碰撞惩罚；最终仍由 OCCT exact collision 决定。

这些修改没有扩大 beam，仍使用原有最大 100 个完整 pose 候选。

## 5. 实验结果

### 5.1 路线对比

| 路线 | accepted | review | rejected | FP | frontier recall | group recall@2000 | true-group pose recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek 放宽阈值审计 | 0 | 5,990 | 10 | 0 | 22.22% | 88.89% | 66.67% |
| 恢复保守阈值、拓扑修复前 | 0 | 5,616 | 384 | 0 | 22.22% | 88.89% | 66.67% |
| 拓扑感知完整重跑 | 0 | 5,995 | 5 | 0 | 0.00% | 100.00% | 100.00% |

解释：

- 拓扑修复把 9 个功能真组 pose 从 6 个提升到 9 个，且公共体积均为 0，这是明确进步。
- 27 个强制 hard negative 中自动接受仍为 0；12 个 reject、15 个 review。
- 但 topology-aware 索引提高了大量几何候选的分数，使 review 增至 5,995、即时 review frontier 漏掉全部真组。
- 因此不能把 pose 100% 解读成整体系统成功；当前候选组排序退化，必须单独修复。
- accepted 仍为 0，所以 `auto_accept_precision=null`。不能把“0 false positive”表述为“precision=100%”。

### 5.2 独立 topology-varied holdout

- 真组：4；
- pose valid：2；
- pose recall：50%；
- cover_base 和 bearing_housing 通过；两个 shaft_hub_key 仍失败。

功能 benchmark 的 100% 证明修复解决了已知模板中的表示错误，但 holdout 50% 表明泛化尚未提升，下一步仍需分析 shaft/key/hub 的接口选择与微小干涉。

### 5.3 多模态校准

当前校准运行模式为 `off`：

- 36/36 abstain；
- 各 split AUC=0.50；
- auto-accept precision 不可估计；
- calibration gate=false；
- semantic reranking=false。

这是安全基线，不是模型效果结论。任何 live Qwen 校准都需要用户明确授权，因为会将匿名 CAD 渲染图发送到外部 API 并产生费用。

### 5.4 JoinABLe

当前可审计实验仅有 27 joints：

- pretrained top-1：0.444，rule top-1：0.259；
- pretrained top-5：0.630，rule top-5：0.519；
- pretrained top-10：0.630，rule top-10：0.741；
- rule10 ∪ pretrained10：0.815。

结论：最合理用途是 shadow candidate union / recall audit，不是替代规则，更不是独立自动接受证据。

## 6. 当前仍未完成的关键工作

### P0：候选组排序与 review frontier

这是当前最大瓶颈：

- typed edge recall=100%；
- group proposal recall=100%；
- group recall@2000=100%；
- 但 72 个即时 review 位的真组 recall=0%；
- 现有 completeness、central coverage、interface diversity 在大量组合上饱和为 1，不能区分真实组与稠密几何子图。

下一步应实现轻量、可审计的接口分配一致性，而不是再调总分或阈值：

1. 同一具体 face/cylinder 不能同时分配给多个互斥 mate；
2. 重复等径孔要先按 origin/axis 聚类为不同接口实例；
3. 计算是否存在覆盖全组的无冲突接口 assignment；
4. 将 feature-exclusivity、unassigned-part、overused-interface 仅用于 review/pose 排序；
5. 自动接受门控保持不变；
6. 在 functional calibration 和 topology-varied holdout 上同时验证，不允许只对 9 个模板调参。

### P1：holdout shaft_hub_key pose

两个失败组分别留下约 0.53 mm³ 和 11.70 mm³ 的 OCCT 公共体积。需要判断是：

- key/keyway face 选择错误；
- 轴向位置欠约束；
- chamfer/tolerance 被当作严重干涉；
- 生成器 ground-truth 自身存在接触建模问题。

不应通过扩大 beam 或放宽 collision 阈值直接掩盖。

### P1：真实工程有效性

- 9 个功能模板仍过少；
- topology-varied holdout 仍由同一项目代码生成；
- 尚无外部机械工程师签核；
- 尚无真实 BOM/CAD metadata 驱动的生产评估；
- `auto_accept_precision` 仍不可估计。

### P2：JoinABLe shadow 接入

将 JoinABLe 输出接入 functional pool 的候选并集，单独报告：

- pair/typed-edge recall 增量；
- candidate budget 增量；
- domain softmax reliability；
- 对 review frontier 的增益；
- 保证不改变 accepted gate。

### P2：Qwen live 校准

只有在获得外部 API 授权后执行。最低通过条件：test AUC≥0.70、Brier 优于 geometry baseline、三类 verdict 齐全、FP 不增加、auto-accept precision 不下降。通过后仍需单独审批才能启用 reranking。

## 7. 推荐的下一轮最小工作包

1. 实现 `interface_assignment_consistency`，只作用于 review/pose queue 排序；
2. 在 frozen functional benchmark 上比较 frontier recall、frontier precision、pose-valid yield；
3. 在 topology-varied holdout 上做无调参复核；
4. 保持 hard-negative auto accept=0；
5. 单独修复两个 holdout shaft_hub_key pose；
6. 获得机械工程师签核后再扩展 functional families；
7. 用户明确授权后才运行 Qwen live calibration。

下一轮验收建议：

- functional review-frontier recall 从 0 提升，同时不得增加 hard-negative auto accept；
- holdout frontier/pose 指标不得下降；
- rejected reason coverage 保持 100%；
- accepted 为空时继续报告 precision 不可估计，不制造成功率；
- 不改变 0.80/0.70 自动接受阈值；
- 不启用 JoinABLe 或 Qwen 自动裁决。

## 8. 工程状态

- 路线安全锁：通过；
- 主测试：71/71；
- 数据生成器测试：3/3；
- 功能真组 pose：9/9；
- topology-varied holdout pose：2/4；
- hard-negative 自动误接受：0/27；
- accepted：0；
- auto-accept precision：不可估计；
- semantic reranking：关闭；
- 外部机械签核：待完成；
- 生产就绪：否。

机器可验证状态见 `STRICT_ROUTE_ALIGNMENT_STATUS.json`。
