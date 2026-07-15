# Case1 / Case2 保守精密 Pose 交付

## 结论

- **case1：precision-valid，可接受。** 复合孔阵列候选替代了旧单平面假成功；横向接口误差从约 4.648 mm 降到 0.000048 mm，孔阵列 RMS 约为 0，OCCT common volume 为 0。
- **case2：exact-valid，但只能 review。** 当前得到 5 个 OCCT exact-valid 候选；所选候选的轴距约 0.0649 mm、轴夹角约 0.0451°、孔阵列 RMS 约为 0，但没有独立测得键槽插入深度，且接触审计/自由度闭锁仍不完整，因此没有自动接受。
- **没有启动服务器训练。** 本轮证据表明瓶颈是确定性的多接口约束、对称候选组合和插入深度验收，而不是网络容量。
- **最终合并回归：54/54 通过。** 覆盖 manifold solver、precision gate、compound pair exact audit、finalizer、保守分层、D0 生成、known-group 契约与 JoinABLe 适配。

## 最终图片

- case1：`final/case1_precision/precision_pose_render.png`
- case2：`final/case2/precision_pose_render.png`

详细机器可读结果见 `case12_final_summary.json`、两个 case 的 `exam_summary.json`。

## 实现内容

1. 新增无名称/case/role 泄漏的 precision pose gate，硬检查 exact collision、接触支持、多证据、轴距/轴角、平面 gap、孔阵列 RMS、插入深度与 common volume。
2. 把解析单接口、geometry-only compound、learned sidecar 改为独立候选预算；网络候选不能挤掉解析与复合候选。
3. 对小型离散对称积加入闭环一致性预排序；最终优化 hypothesis 数保持固定，不靠扩大 beam 硬救。
4. 给刚性 compound pair 候选增加隔离 OCCT 审计。pair-valid 只改变候选优先级，不等于组装接受。
5. 接通 repeated-hole multi-axis RANSAC 与 topology-backed prismatic slot proposal，并保留对称分支。
6. 修正 known-group 保守输出：`pose_status=valid` 但没有 precision-valid 时必须进入 review。
7. 基线考试 runner 与最终化工具按 precision tier 选择渲染结果，不再按优化 cost 或 OCCT-valid 单独选解。

## D0 / D3.5 / D7 审计

### D0

- 正式 functional dataset：9 cases；`cover_base`、`shaft_hub_key`、`bearing_housing` 各 3 个。
- `generic_geometry_stack_positive_count=0`。
- 每个 case 有明确 role、interface、functional mate 和三类 negative。
- `source_id_used_as_production_truth=false`。
- 旧 primitive generator 仍保留，但明确标记为 `geometry_smoke_only`、`functional_positive_eligible=false`，不能用于功能正样本证明。

限制：每个 family 目前只有 3 个近似参数变体；多数 hard negative 尚未形成精确、独立的候选组验收，因此 D0 解决了数据语义格式和明显无意义堆叠，但尚未充分证明功能判别泛化。

### D3.5

frozen 12-pool 当前报告的 pair+candidate-type recall 为 100%（68/68，pruning 后仍为 68/68）。这个数字不等价于正确 B-Rep entity、polarity、phase 或 reachable-pose recall，后续必须增加 entity-level 与 pose-level recall audit。

### D7

重新运行 frozen 12-pool 到 `sw/data/results_20260713_final`：

- 修改前：40 accepted，1 true positive，auto-accept precision 2.5%，false positive 39。
- 修改后：0 accepted，false positive 0，review 9,162（立即人工队列 288，deferred 8,874），rejected 506，unresolved parts 110。
- 当前 auto-accept precision **不可估计**，因为没有自动接受组；不能把它写成 100%。
- semantic reranking 仍为 false，DeepSeek 不影响分组、分数或 tier。
- D6 四项交付已写入 frozen 结果目录；由于这些 pool 没有真实 role/interface/family/relation 字段，`semantic_inputs.json` 与 `semantic_reviews.json` 为空，gate 明确关闭且没有 provider call。

## 成功转入 review 的错误

- frozen 12-pool 的旧 39 个错误自动接受全部不再 accepted。
- 这 39 个旧 false positive 全部被映射到 review：17 个进入 bounded 人工前沿，22 个进入 deferred review；`false_positive_audit.csv` 现已直接记录每个候选的 `post_gate_decision`、队列状态和原因。
- case1 的旧单平面“贴合但偏心”候选现在是 review；最终由多孔阵列+轴向接触+表面接触的复合候选替代。
- case2 的“无碰撞但分离”继续是 review；新得到的 exact-valid 候选也因插入深度未验证而保持 review。

## 仍未解决

- case2 键槽的真实 insertion depth / containment 没有独立实测。
- 当前 `unresolved_manifold_dofs` 是保守的 factor free-DoF 汇总，不是最终联合 Jacobian 的秩判定，可能高估剩余自由度。
- frozen 12-pool 的 recall audit 仍过粗；functional hard negatives 的 exact-candidate 覆盖不足。
- 没有统计意义上的 accepted precision：case1 是一个工程门通过样本，不足以估计总体精度。

## 论文与开源调研落点

- JoinABLe 保留为 B-Rep entity-pair / joint-axis 高召回旁路，不承担最终 Pose 正确性。
- CVPR 2024 Multi-Part Multi-Joint 只移植“part graph + joint/interface graph、多个接口联合约束”的结构思想，不移植其家具点云网络。
- TEASER++ 仅在开发集证明 RANSAC 被错误对应限制时，才作为解析 landmark 的可选 robust seed；不对整个表面做点云配准。
- BOP 的离散/连续 symmetry 表达适合移植到接口对称候选；当前先保留全部几何相位并用闭环与 exact gate筛选。
- FoundationPose、AutoMate/RL、LLM/DeepSeek prompt 都不能补上轴距、孔阵列相位和键槽 containment，因此没有进入实现。

## 下一步最小修改

1. 在 prismatic slot factor 中测量 opening plane、slot depth、key containment 和实际插入长度；只有测量值通过才解除 case2 review。
2. 用联合约束 Jacobian rank 替换自由度集合并集，准确判断多 factor 是否真正锁死 `tz/rz`。
3. 把 `planar + central axis + repeated-hole/key-slot` 显式表示为同一 pair bundle，而不是仅依赖一个刚性 compound seed。
4. 将候选召回审计扩展到 entity pair、polarity、phase 和 reachable pose；保持候选阶段高召回、接受阶段严格。
5. 给 D0 每类增加少量拓扑变化，并让每类 hard negative 形成独立 exact candidate 进入 accepted/review/rejected 验收。

在完成第 1～2 项前，不建议再训练网络、增加模型规模或扩大 beam。
