# 纯 B-Rep 学习与多零件全局 Pose Solver 三阶段审计报告

日期：2026-07-11

## 一、最终结论

本轮完成了截图要求的三层工程闭环，但必须区分“工程已经实现”和“所有考试已经做对”：

1. **第一阶段已完成**：从 Fusion 360 Joint 原始数据构建了纯 B-Rep、设计组隔离、无文件名/零件名/case/BOM/SolidWorks 答案泄漏的监督索引。
2. **第二阶段已完成工程化实现**：官方 JoinABLe 学习实体对，系统把 top-k 实体对提升为 `局部坐标系 + 轴/法向 + 自由度流形 + 初始相对 Pose + 置信度 + 对称等价类`。初始 Pose 只决定优化起点，不作为固定约束。
3. **第三阶段已完成工程化实现**：实现有界离散拓扑/候选搜索、所有零件 SE(3) 联合 `least_squares`、一致闭环、局部非对称 B-Rep 证据、连续接触/穿透残差以及 OCCT top-N 验证。
4. **最终考试结果并非 3/3**：case1 通过；case2 拓扑恢复正确但大件贴合仍有间隙，只能进入 review；case3 原始精确验证超时且代理旁证高风险，未通过。

因此，不能把本轮包装成“case1–3 全部解决”。真正的剩余瓶颈已从 pair 接口召回转移到：复杂凹腔/多接触条件下的高保真连续接触目标，以及大型原始 STEP 的可恢复精确验证。

## 二、第一阶段：纯 B-Rep Fusion 360 监督集

### 2.1 数据规模

- 原始 `joint_set_*.json`：19,156 个。
- 成功转换的 paired B-Rep record：18,002 个。
- joint supervision：29,560 条。
- split：train 14,231；dev 2,018；test 1,753。
- 设计组跨 split 重叠：0。
- 模型输入禁用字段命中：0。
- 转换异常：0。

输出位于：

- `D:\Model_match_public_data\fusion360_pure_brep_v1\fusion360_pure_brep_train.jsonl`
- `D:\Model_match_public_data\fusion360_pure_brep_v1\fusion360_pure_brep_dev.jsonl`
- `D:\Model_match_public_data\fusion360_pure_brep_v1\fusion360_pure_brep_test.jsonl`
- `reports/pure_brep_contract_audit.json`

模型张量入口只允许面/边类型、UV/曲线采样、拓扑邻接、面积/长度/半径、轴/法向等数值几何。存储路径只存在于 loader 定位层，不能进入 `model_input`。Fusion joint 类型只作为辅助监督生成自由度 target，不会被求解器解释成命名机械规则。

### 2.2 监督分布

- Rigid：16,243
- Revolute：8,184
- Cylindrical：2,613
- Slider：1,703
- Planar：477
- Ball：276
- PinSlot：64

### 2.3 学习基线

沿用官方 JoinABLe checkpoint 作为实体对学习器。现有完整官方测试审计为：

- Top-1：79.32%
- Top-5：87.56%
- Top-10：90.79%

STEP 重导入会改变边拓扑，项目已有 20 对 transfer audit；在 27 个 equivalent-evaluable joint 上，STEP equivalent Top-10 为 62.96%。这说明训练数据和网络本身已可用，但 STEP/OCCT 拓扑迁移仍是明显召回损失来源。

## 三、第二阶段：学习式 Pair Joint Hypothesis

### 3.1 输出契约

每个 top-k 候选包含：

- learned entity pair；
- joint origin / direction；
- 两侧局部方向坐标系；
- `free_dof_mask` 与 `constrained_dof_mask`；
- 流形上的初始相对 Pose；
- JoinABLe confidence/rank；
- polarity、phase 和 symmetry class；
- pair SDF 搜索与 pair OCCT 的审计信息。

同轴候选默认保留轴向平移和绕轴旋转；平面候选默认保留面内两个平移和绕法向旋转。只有两侧一环 B-Rep 都给出稳定局部非对称主方向时，才收紧相位自由度，同时保留 0°/180° 等价候选。

### 3.2 关键防泄漏边界

- JoinABLe/流形构建不读取 case ID、文件 token、零件语义名或人工答案。
- 不存在 `KeySlotFactor`、`FlangeFactor`、`pocket_mate` 等生产求解规则。
- Pair SDF 搜索 Pose 是初始化，不是 `T_ab` 硬约束。
- 对称/相位候选按实体对 round-robin 保留，避免一个实体对的多个 phase 占满 top-k。

该候选生命周期参考了 JoinABLe 的实体 link prediction 与搜索思想，并只借鉴 FoundationPose/MegaPose 的“生成—多样化—细化—评分—保留 top-k”流程；没有接入 RGB-D 网络。[JoinABLe 论文](https://damassets.autodesk.net/content/dam/autodesk/research/publications-assets/pdf/joinable-learning-bottom-up-assembly-of-parametric-cad-joints.pdf)、[FoundationPose 官方代码](https://github.com/NVlabs/FoundationPose)、[MegaPose 官方代码](https://github.com/megapose6d/megapose6d)。

## 四、第三阶段：多零件全局 Pose Solver

### 4.1 离散变量

- 连接拓扑；
- 每条边选择的 learned entity pair；
- polarity / phase / pair-search initial；
- 只有满足投影残差阈值的非树边才能成为闭环 factor。

大候选池不再完整展开笛卡尔积。当前实现使用：

- k-best confidence heap；
- 单边候选 sweep；
- 确定性混合角点；
- 预计算 SE(3) 距离的增量 farthest-first。

40×40×40 的候选组合不会再完整生成；120 个前沿组合的单元回归在 1 秒量级完成。

### 4.2 连续变量与残差

- 节点：每个零件的全局 `T_i ∈ SE(3)`；
- 优化：SciPy `least_squares`，固定一个 anchor；
- 约束：局部接口 frame 的投影 se(3) 残差；
- 轴流形：不惩罚 `tz/rz`；
- 平面流形：不惩罚 `tx/ty/rz`；
- 局部非对称证据可收紧 `rz`；
- 接触：连续最近表面间隙；
- 穿透：双向有向局部表面深度；
- 非连接 pair：只惩罚穿透，不强迫接触；
- 精确验证：OCCT Boolean Common，不参与梯度。

### 4.3 保守输出边界

全局 solver 始终输出 review-only 几何假设。无碰撞、低残差或闭环一致都不能独立证明语义正确；本轮也没有引入大模型语义裁判。

## 五、case1 / case2 / case3 最终考试

### 5.1 case1：通过

- 最终候选：35 个。
- OCCT 无碰撞：10 个。
- 选中的近接触候选：`topology_00_choice_000`。
- 原始 STL 最近距离：约 0.00146 mm。
- 有采样接触支持，OCCT 无公共体积碰撞。

结论：**物理装配通过**。由于当前任务没有独立人工真值矩阵，最终状态仍保留 audit/review 字段，但它已满足本轮可验证几何要求。

输出：`sw/exam_brep_manifold_20260710/case_1/global_pose_result_v12.json`

### 5.2 case2：部分通过，仍需 review

最好的强接触结果为：

- topology：`part_3` 为中心，连接 `part_0 / part_1 / part_2`；
- OCCT：四个原始 STEP 零公共体积碰撞；
- `part_2–part_3` 最近距离：约 0.090 mm，并出现采样接触；
- `part_0–part_3` 最近距离：约 1.195 mm；
- `part_1–part_3` 最近距离：约 1.933 mm；
- 两大件闭环最近距离：约 2.213 mm。

连接结构已经恢复到“中心件连接其余三件”，但两块大件尚未达到贴合要求。**不能判 case2 完全装配成功**。

输出：`sw/exam_brep_manifold_20260710/case_2/global_pose_result_v13_contact_refine.json`

### 5.3 case3：未通过 / exact uncertain

原模型节点规模导致完整 JoinABLe candidate graph 需要约 2 TB 分配。为参加考试，创建了只用于候选生成的有损几何代理：

- 笼体：57 个 solid 保留 14 个、234 faces、重要度覆盖约 61.83%；
- 风扇：3 个 solid 保留 1 个、414 faces、重要度覆盖约 49.88%；
- proxy JoinABLe candidate count：1,337,640；
- 流形/初值候选：416；
- 全局候选：39。

最终原始 STEP 验证：

- top-39 整批 OCCT：604 秒超时；
- top-1 单候选 OCCT：124 秒超时；
- 状态必须记为 `uncertain_timeout`，不能视作无碰撞；
- 原始高分辨率 STL 的非权威旁证将 39/39 标为 likely penetrating，但笼体是凹腔，该近似可能误报，因此只能作为风险证据。

结论：**case3 未通过**。代理证明 pipeline 能运行，但没有证明原始风扇正确插入笼体。

输出：

- `sw/exam_brep_manifold_20260710/case_3/proxy/*.audit.json`
- `sw/exam_brep_manifold_20260710/case_3/global_pose_proxy_v12.json`
- `sw/exam_brep_manifold_20260710/case_3/original_mesh_pose_audit.json`

## 六、测试、冻结与复查

- 新/核心回归：25/25 通过。
- JoinABLe pose/search/group 兼容回归：9/9 通过。
- 最终冻结：`reports/three_stage_brep_manifold_freeze.json`。
- 冻结检查：禁用生产 token 命中 0；纯 B-Rep contract 通过。

本轮实验暴露并修复了以下通用问题：

1. symmetry phase 占满实体 top-k；
2. polarity 未进入有限预算；
3. piecewise-constant contact fraction 无梯度；
4. pair-search Pose 没有接到 global initial；
5. 大笛卡尔积完整展开导致 600 秒超时；
6. 不一致非树 pair 被错误强迫成为 contact/cycle factor；
7. Conda NumPy DLL 与 `pysdf/rtree` 运行时不稳定，改为稳定的 KD-tree oriented signed-distance surrogate；
8. 大型凹腔 STEP 的同步 OCCT Boolean 不可恢复。

## 七、下一步最小路线

1. **不要再扩大 JoinABLe top-k 或全局组合数**。case2 已证明正确中心拓扑进入前沿，瓶颈是贴合目标。
2. case2 增加基于真实 B-Rep 面片的 contact-area / gap residual，而不是继续依赖稀疏表面点；重点让两块大件的实际接触面距离收敛到容差内。
3. case3 将 OCCT 改成单候选、单 pair、带超时并逐条落盘的 worker 队列；先用 BRepExtrema 距离和粗碰撞筛掉明显失败，再运行 Boolean Common。
4. case3 构建更高保真的凹腔代理：保留外壳、开口边界和内腔接触面，而不是只按体积选择 solid。该代理规则仍应基于 B-Rep 可观测几何，不能读取风扇/笼体名称。
5. 若后续训练辅助 head，优先学习 `free_dof_mask / polarity / local asymmetric direction confidence`；不训练机械标签分类器，不建立 case 专用解码表。

## 八、相关研究依据

- JoinABLe 将 B-Rep 实体连接建模为 link prediction，并通过搜索恢复 joint pose；它是本项目 pair-level 学习主干，但不是完整多零件全局 solver。[论文](https://damassets.autodesk.net/content/dam/autodesk/research/publications-assets/pdf/joinable-learning-bottom-up-assembly-of-parametric-cad-joints.pdf)、[项目页](https://www.karlddwillis.com/joinable/)
- AutoMate 强调 CAD mate 定义在 B-Rep 拓扑实体上，而不是只回归世界坐标位姿，支持本轮“实体监督 + 约束流形”的路线。[论文](https://arxiv.org/abs/2105.12238)
- FoundationPose/MegaPose 的价值限于多假设生成、细化和评分流程；其视觉网络输入与本项目纯 CAD 输入不一致，因此没有直接接入。[FoundationPose](https://github.com/NVlabs/FoundationPose)、[MegaPose](https://github.com/megapose6d/megapose6d)
