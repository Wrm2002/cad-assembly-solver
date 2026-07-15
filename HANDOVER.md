# CAD 装配 Pose 恢复项目接手文档

更新日期：2026-07-13

## 0. 本次 case1～5 泛化闭合交付

完整实现说明、审计结论和五个最终渲染：
[`sw/GENERALIZATION_WORK_REPORT_CASE1_5_20260713.md`](sw/GENERALIZATION_WORK_REPORT_CASE1_5_20260713.md)。

- case1、case3 沿用已经人工确认的正确结果；case5 保持 enclosure/bay 结果。
- case2 已修正并由用户确认：两法兰内侧面贴合，轴位于联合中心，键同时跨过
  接触面并与两侧轮毂键槽和轴键槽相位一致。所选 rank 为 `882/1760`；
  接触残差约 `1.78e-15 mm`、中心残差约 `5.33e-15 mm`、双侧插入约
  `45.0/45.0 mm`，OCCT common-volume 碰撞数为 0。
- case2 的实现是匿名几何组约束，不读取名称、case ID、颜色，也不写死最终平移。
  它要求两个可比支撑、两个同轴适配孔、真实 CLEARANCE、两套成对
  `topological_key_slot` witness、对称轴向范围和跨接键；任何一项缺失就 abstain。
- sidecar 必须有匹配的 STEP SHA-256，键槽 witness 必须有具体拓扑 ID，适配孔必须
  是凹面圆柱；`0.1x/1x/10x` 匿名尺度回归已通过。
- case4 的 DIMM 已由 repeated bounded edge-slot family 放入槽中；槽宽约
  `1.6673 mm`、功能体厚度约 `1.27 mm`、入槽边到槽底残差约
  `2.84e-14 mm`。但主板含 `104864` 个未覆盖 open-shell/orphan faces，
  因此状态必须保持 `review/uncertain`，不能宣称完整 collision-free。
- case2 组级规则仍为 `proposal_only/review_required/can_auto_accept=false`；
  正确考试结果不会被用来放宽 accepted gate。
- case3 仍因约 `192.715 mm³` 局部干涉进入 review；case4 所选 rank 的 exact
  validation 因预算跳过，邻近 exact ranks 另有 CPU—主板严重碰撞；case5 仅
  `1/2` connection closure。五张图用于关系核对，不能统称 exact-valid。
- frozen 12-pool 的既有保守基线仍是 false positive `39 -> 0`、accepted `0`，
  因而 accepted precision 不可估计；本轮没有用 case1～5 重新调该门槛。
- 下文旧的 case1/case2 失败描述是历史基线；若与本节冲突，以本节和工作报告为准。

## 1. 一句话状态

项目已能在已知真组上组合 JoinABLe/解析 B-Rep 候选、组级多接口约束、
刚体依赖传播和 OCCT 审计；case1～5 已生成当前关系渲染，但这仍不等于
mixed-pool 分组泛化已经解决。生产策略继续是高精度优先：证据不完整、几何覆盖不全
或仅有 proposal 的结果进入 review，而不是为了覆盖率自动接受。

下一个最小任务不是继续训练或扩大 beam，而是把这些通用接口族放进独立的
非考试 holdout：不同尺度、不同法兰/轮毂外形、无键/单键槽/错位键槽、阶梯轴，
以及 DIMM 槽宽和插入深度困难负例。

## 2. 不可违反的边界

### 2.1 输入与防泄漏

模型和生产求解器禁止读取：

- 文件名和零件名；
- case ID；
- BOM 文本；
- SolidWorks 答案；
- 用户对 case1～5 的人工解释；
- `flange`、`shaft`、`key` 等名称 token；
- source template / generator case ID 作为生产真值。

路径和 part ID 只能用于编排、日志和输出映射，不能进入神经网络特征或评分规则。

### 2.2 冻结考试

`sw/1`～`sw/5` 以及 `sw/exam_brep_manifold_20260710` 是 evaluation-only：

- 不进入训练；
- 不决定 loss；
- 不调阈值；
- 不做 early stopping；
- 不根据人工答案写 case-specific override。

### 2.3 物理与语义边界

- `collision_free` 不是装配正确；
- OCCT `valid` 不是共轴、贴合或插入正确；
- 单一平面、单孔或单轴证据不能自动接受；
- DeepSeek/多模态模型当前只能写 review 解释；
- 不引入 RL；
- 不扩大复杂多 Agent；
- 不为了覆盖率牺牲 false positive。

## 3. 当前系统结构

### 3.1 B-Rep 与 JoinABLe

入口：`sw/joinable_e2e.py`

职责：

1. 读取 STEP；
2. 使用 OCCT 提取 face/edge/adjacency 和解析几何；
3. 运行官方 JoinABLe checkpoint；
4. 输出跨零件实体对 top-k；
5. 将 plane/cylinder/circle/line 等实体提升为局部接口 frame 和约束流形。

JoinABLe 的职责是召回“哪些局部实体可能形成接口”，不是证明整个装配正确。

### 3.2 第一网络：CAD Pair Pose

代码：`sw/learned_joint/pose_learning.py::CADPairPoseModel`

输入：

- 两侧 JoinABLe entity/one-ring B-Rep embedding；
- 不包含名称或 case 信息。

输出：

- 8 个相对 Pose modes；
- mode logits；
- 6 维自由度概率；
- candidate-conditioned score。

当前结论：可以生成 top-k，但冻结 case1/2 的 network-only 消融均为 0 个严格有效 Pose。

### 3.3 第二网络：B-Rep Patch Interface / Contact

代码：`sw/learned_joint/pose_learning.py::CADPairPosePatchModel`

额外输入：

- 两侧局部 B-Rep surface patches，每点为 position + direction；
- 当前推理只支持有采样 patch 的 face candidate；edge candidate 会跳过。

额外输出：

- pose-conditioned interface score；
- gap / coverage / normal mismatch 预测。

当前结论：

- 训练指标有改善；
- 冻结 STEP 的旧图最初没有 patch，导致网络生成 0 条有效 proposal；
- 已增加 `sw/enrich_brep_graph_patches.py`，可从 STEP 重新采样面 patch；
- case1 生成 30 条 learned rows，case2 六个零件对共生成 178 条；
- 但这些 learned candidates 尚未在 case1/2 单独产生严格成功结果。

### 3.4 全局 Pose Solver

代码：

- `sw/learned_joint/manifold_solver.py`
- `sw/learned_joint/mesh_residuals.py`
- `sw/run_manifold_global_pose.py`

变量：每个非 anchor 零件的全局 `SE(3)` Pose。

离散部分：

- 连接拓扑；
- 每条 pair edge 的 candidate；
- polarity / phase / learned seed。

连续残差：

- manifold constrained DoF residual；
- learned full-Pose soft prior；
- sampled contact gap；
- sampled penetration；
- non-edge overlap。

当前核心缺陷：一个零件对通常只选择一个 factor，无法同时保留 planar、central axis 和 repeated-hole phase 三种独立证据。

### 3.5 Exact 与保守门

代码：

- `sw/validate_manifold_pose_exact.py`
- `sw/learned_joint/pose_acceptance.py`

OCCT exact gate：检查实体 Boolean common volume。

contact gate：要求每条 selected constraint edge 的归一化接触间隙不超过当前通用阈值。

已解决：

- 能把“OCCT 无碰撞但零件分离”的 case2 候选转入 review。

未解决：

- 不能拦截“面贴合但横向偏心”的 case1；
- 尚无中央轴距离和孔阵列 RMS 验收。

## 4. 数据与 checkpoint

### 4.1 当前主数据集

`D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1`

| split | samples |
|---|---:|
| train | 61,348 |
| dev | 9,345 |
| test | 7,847 |

特点：

- assembly_id 严格隔离；
- 多正例等价 Pose；
- B-Rep 实测困难负例；
- 不读取 SolidWorks case；
- 平均每个 supervision 约 2.2 个有效 Pose mode；
- 平均约 5 个 B-Rep measured hard negatives。

### 4.2 训练结果

`D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_train`

Pair Pose：

- checkpoint：`pose_proposal/best.pt`；
- best epoch：12；
- test loss：约 0.7201；
- test pose loss：约 0.2074；
- test rank-1：约 0.9792；
- test top-k metric：约 1.1243。

Patch Interface：

- checkpoint：`interface_score/best.pt`；
- best epoch：16；
- test loss：约 0.8840；
- test pose loss：约 0.0507；
- test score loss：约 0.3239；
- test contact loss：约 0.0002；
- test rank-1：约 0.8842；
- test top-k metric：约 0.1493。

这些指标来自不同任务头，不能直接横向比较，也不能替代 SolidWorks holdout。

### 4.3 第一网络历史 checkpoint

`D:\Model_match_public_data\joinable_pose_heads_v1_8k_train\best.pt`

- 生成时间：2026-07-11 16:37；
- 约 7.57 MB；
- best epoch：4；
- dev top-k error：约 3.484；
- 质量不高，未上传 GitHub。

用户已经取消上传旧版本。当前远端没有新 commit 或分支。

## 5. 冻结实验结果

汇总：`sw/exam_brep_manifold_20260710/case12_ablation_v2_summary.json`

| run | hypotheses | OCCT valid | precision/contact valid | 结论 |
|---|---:|---:|---:|---|
| case1 baseline | 4 | 1 | 仍未通过共轴门 | 面贴合但偏心 |
| case1 network-only | 4 | 0 | 0 | 网络单独失败 |
| case1 protected sidecar | 8 | 1 | 仍未通过共轴门 | 正确候选来自 baseline |
| case2 baseline | 24 | 0 | 0 | 失败 |
| case2 network-only | 24 | 0 | 0 | 失败 |
| case2 protected sidecar | 24 | 1 | 0 | 无碰撞但分离，review |

### 5.1 case1 定量根因

当前最佳 transform 中 moving flange：

- 横向平移：`sqrt(tx² + ty²) = 4.6478 mm`；
- `tz ≈ -0.00008 mm`；
- 面内转角：`3.6743°`；
- 轴倾角：`0.00024°`。

结论：轴线平行但不同轴。

为什么会发生：

1. 选中的 factor 为 `plane_coincidence`；
2. `free_dof_mask` 允许平面内 `tx/ty/rz`；
3. sampled contact optimizer 沿这些自由度滑动以降低穿透；
4. 中央 cylinder axis candidate 排名较后，没有与 planar factor 联合约束；
5. OCCT 不检查共轴。

### 5.2 case2 定量根因

sidecar 中唯一 OCCT-valid 候选：

- 至少一条 selected edge 的归一化 contact gap 约 `0.3475`；
- 当前 review threshold 为 `0.05`；
- 两法兰明显分离；
- key 未进入 keyway；
- 因此 `contact_supported_valid_count = 0`。

## 6. 本轮已经修改的关键文件

| 文件 | 修改内容 |
|---|---|
| `sw/enrich_brep_graph_patches.py` | 从 STEP/OCCT 生成 face patch points/directions |
| `sw/learned_joint/pose_head_adapter.py` | 不支持 patch 的 entity candidate 单独跳过，不再令整 pair 崩溃 |
| `sw/augment_manifold_frontier_with_learned_pose.py` | learned score 不再提升结构置信度 |
| `sw/learned_joint/report_adapter.py` | baseline 与 learned 使用独立预算；保留 strongest pair 两种 polarity |
| `sw/learned_joint/manifold_solver.py` | learned initial 纳入去重签名；避免 learned seed 被全部删除 |
| `sw/learned_joint/pose_acceptance.py` | OCCT valid 后增加 selected-edge contact gate |
| `sw/run_strong_contact_exam.py` | patch enrichment、solver runtime 分离、protected sidecar |
| `sw/run_baseline_pose_exam.py` | baseline/learned channel 消融入口 |
| `public_cad_dataset_audit/audit_pose_ablation_results.py` | 统一汇总 case 消融与接触门结果 |
| `sw/tests/test_manifold_report_adapter.py` | polarity 与 baseline protection 回归测试 |
| `sw/tests/test_pose_acceptance.py` | 分离但无碰撞必须 review 的测试 |

## 7. 本机环境

### 7.1 模型推理

`C:\Users\11049\Desktop\Model_match\.conda\python.exe`

- PyTorch 可读取当前 checkpoints；
- 用于 JoinABLe 和 learned heads；
- 不建议用于全局 SciPy least-squares。

### 7.2 数值求解

`C:\Users\11049\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

- NumPy/SciPy 测试稳定；
- 用于 `run_manifold_global_pose.py`；
- 当前 solver 改为 `tr_solver="lsmr"`。

### 7.3 OCCT

`C:\Users\11049\miniforge3\envs\cad311\python.exe`

- STEP/B-Rep/Boolean/STL；
- 不用于主 NumPy 优化。

### 7.4 已知原生崩溃

- `.conda` SciPy dense SVD/QR 曾以 `0xc06d007f` 退出；
- `cad311` 的部分 NumPy matrix multiply 也出现相同原生错误；
- 不要为了“统一环境”破坏当前三运行时隔离。

## 8. 最小复现命令

```powershell
$MODEL_PY = 'C:\Users\11049\Desktop\Model_match\.conda\python.exe'
$SOLVER_PY = 'C:\Users\11049\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$OCCT_PY = 'C:\Users\11049\miniforge3\envs\cad311\python.exe'
```

### 8.1 case1 baseline

```powershell
& $SOLVER_PY sw\run_baseline_pose_exam.py `
  sw\exam_brep_manifold_20260710\case_1\pair_frontier\pair_frontier_manifest.json `
  sw\exam_brep_manifold_20260710\case_1\handover_recheck\baseline `
  --solver-python $SOLVER_PY `
  --occt-python $OCCT_PY `
  --render-python $MODEL_PY
```

### 8.2 case1 protected sidecar

```powershell
& $MODEL_PY sw\run_strong_contact_exam.py `
  sw\exam_brep_manifold_20260710\case_1\pair_frontier\pair_frontier_manifest.json `
  D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_train\interface_score\best.pt `
  sw\exam_brep_manifold_20260710\case_1\handover_recheck\protected_sidecar `
  --runtime-python $MODEL_PY `
  --solver-python $SOLVER_PY `
  --occt-python $OCCT_PY `
  --candidate-limit 20 `
  --modes-per-candidate 6 `
  --learned-pose-prior-weight 0.35
```

### 8.3 审计与测试

```powershell
Get-Content sw\exam_brep_manifold_20260710\case12_ablation_v2_summary.json

& $SOLVER_PY -m unittest discover -s sw\tests -p 'test*manifold*.py'
& $SOLVER_PY -m unittest sw.tests.test_pose_acceptance
```

最近结果：10 个 manifold tests 与 3 个 pose acceptance tests 全部通过。

## 9. 下一步任务清单

### P0：Precision Pose Validator

新增通用模块，建议路径：

`sw/learned_joint/precision_pose_validator.py`

每个 pair 至少输出：

```json
{
  "axis_distance_mm": 0.0,
  "axis_angle_degrees": 0.0,
  "plane_gap_mm": 0.0,
  "hole_pattern_rms_mm": 0.0,
  "insertion_depth_mm": 0.0,
  "occt_common_volume_mm3": 0.0,
  "independent_evidence_count": 0,
  "precision_status": "valid | review | failed"
}
```

阈值必须在 Fusion/functional holdout 上校准，不能用 case1 人工调到刚好通过。

### P1：Compound Pair Constraint Bundle

修改 factor graph 表达：

- 一个 pair candidate 可以包含多个 factor；
- planar、axis、hole-pattern 是一个 bundle 的独立证据；
- bundle 联合选择、联合优化、联合审计；
- 不能只保留其中分数最高的一条。

### P2：接入 Multi-interface RANSAC

已有代码：`sw/augment_pair_frontier_with_multi_axis_ransac.py`

它可以从三组以上同半径圆柱接口恢复刚体 Pose，不依赖零件名称。接手者需要：

1. 在 pair frontier 构建阶段调用；
2. 保留 provenance 中的 support/residual/matches；
3. 让 compound candidate 在预算中优先于单平面弱证据；
4. 用 precision validator 验证，不直接自动接受。

### P3：重新评测

严格消融：

1. analytic baseline；
2. compound geometry only；
3. learned network only；
4. protected union；
5. 每组都输出 OCCT、contact、axis、plane、pattern 指标。

目标不是让渲染“看起来差不多”，而是明确证明误差进入工程容差。

## 10. 暂时不要做

- 不要再次扩大 Fusion 数据规模；
- 不要重训更大网络；
- 不要提高 learned prior weight 来硬救 case2；
- 不要用更大 beam width 暴力搜索；
- 不要写 flange/key/shaft 专用 factor；
- 不要用文件名 token；
- 不要把 case1 当前结果标为成功；
- 不要让网络候选覆盖解析基线；
- 不要上传整个工作区或 `D:\Model_match_public_data`。

## 11. Git 状态

- 当前分支：`master`；
- 当前远端：`origin https://github.com/Wrm2002/cad-assembly-solver.git`；
- 远端 HEAD：`24f3c07`；
- 用户已取消旧版本上传；
- 没有新 commit，没有 push，没有残留临时分支；
- 工作区非常脏，包含大量历史实验和未跟踪文件；
- 提交时必须逐文件 `git add`，禁止 `git add .`。

## 12. 接手后的第一小时建议

1. 阅读本文件和根目录 `README.md`；
2. 打开 `case12_ablation_v2_summary.json`；
3. 检查 case1 transform 的 `tx/ty`，确认 4.648 mm 偏心；
4. 阅读 `report_adapter.py` 的候选预算；
5. 阅读 `augment_pair_frontier_with_multi_axis_ransac.py`；
6. 设计 precision validator 的数据结构和 Fusion holdout 校准方式；
7. 在没有完成 precision gate 前，不继续训练。

只要下一位接手者牢记下面这句话，就不会重复当前误区：

> 无碰撞不是装配成功；面贴合也不是共轴；多接口装配必须由多个独立几何证据联合锁定。
