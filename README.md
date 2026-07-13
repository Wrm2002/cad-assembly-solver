# CAD 装配 Pose 恢复系统

## 2026-07-13 精密多接口更新

最新 case1/2 结果见
[`sw/exam_final_20260713/FINAL_REPORT.md`](sw/exam_final_20260713/FINAL_REPORT.md)：

- case1 已由 repeated-hole multi-axis compound candidate 恢复为
  `precision-valid`；轴距约 `0.000048 mm`、孔阵列 RMS 约为 0、
  OCCT common volume 为 0。
- case2 已得到 5 个 OCCT exact-valid 候选，但因键槽插入深度和联合
  自由度闭锁尚未独立验证，最终仍为 `review`，没有自动接受。
- known-group 输出现在额外要求 multi-evidence precision gate；旧的
  “connected + closure + collision-free” 不再足以进入 accepted。
- 以下第 3.3～5 节记录的是本次更新前的冻结基线与根因，保留用于对照。

## 1. 项目目标

当前主任务不是从混合零件池猜测哪些零件属于一组，而是：

> 输入一组已知属于同一机械装配体的 STEP / SolidWorks 零件，从 B-Rep 几何中恢复每个零件的全局 `SE(3)` Pose，并输出可回写 CAD 的 JSON。

系统必须同时满足：

- 不能把文件名、零件名、case ID、BOM、人工答案作为模型输入；
- 候选必须可解释、可审计、可复查；
- `collision_free` 只表示没有严重实体相交，不能代表装配正确；
- Pose 必须验证贴合、共轴、相位、插入深度、穿透和多零件闭环一致性；
- SolidWorks case1～5 是冻结考试，只能用于最终评测，不能用于训练、阈值选择或 early stopping。

混合池自动分组、DeepSeek 语义裁判、RL 和复杂多 Agent 当前均不是主线。

## 2. 当前技术路线

```text
STEP / SolidWorks parts
        │
        ▼
OCCT B-Rep graph
  faces / edges / adjacency / analytic geometry / sampled patches
        │
        ▼
JoinABLe top-k entity-pair proposal
        │
        ├───────────────┐
        ▼               ▼
analytic manifold      CAD Pair Pose network
plane / axis / DoF     top-k relative Pose modes
        │               │
        └──── protected union ────┐
                                  ▼
multi-part SE(3) factor graph + sampled contact residuals
                                  │
                                  ▼
OCCT exact collision validation
                                  │
                                  ▼
precision/contact gate
accepted | review | failed
```

第二套 B-Rep patch/interface 网络目前只能作为旁路候选。它不能覆盖解析基线、提高结构置信度或单独决定结果。

## 3. 2026-07-13 的真实状态

### 3.1 训练数据

当前等价 Pose + B-Rep 实测困难负例数据集：

`D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1`

| split | entity-pair samples |
|---|---:|
| train | 61,348 |
| dev | 9,345 |
| test | 7,847 |

数据只来自 Fusion360；SolidWorks 考试样本未进入训练。

### 3.2 已训练网络

训练产物目录：

`D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_train`

| 网络 | checkpoint | 参数量 | 当前定位 |
|---|---|---:|---|
| Pair Pose | `pose_proposal/best.pt` | 约 1.89M head 参数 | top-k Pose 旁路提案 |
| Patch Interface / Contact | `interface_score/best.pt` | 约 2.54M head 参数 | 候选评分与接触估计，尚未通过考试泛化验收 |
| frozen JoinABLe encoder | 官方 checkpoint | 约 1.34M | B-Rep 实体编码与候选召回 |

远端 GPU 训练已经完成。训练指标不能直接等价为 SolidWorks 装配成功率。

### 3.3 冻结 case1/2 消融

完整汇总：

`sw/exam_brep_manifold_20260710/case12_ablation_v2_summary.json`

| 运行方式 | case1 contact-supported valid | case2 contact-supported valid |
|---|---:|---:|
| 解析基线 | 1 | 0 |
| 网络单独 | 0 | 0 |
| 解析基线 + 网络旁路 | 1 | 0 |

注意：case1 的“contact-supported valid”仍未通过精密共轴验收，不能作为最终成功。

case1 当前最佳 Pose 的精确误差：

- 两轴夹角：约 `0.00024°`，基本平行；
- 横向轴线偏移：约 `4.648 mm`，明显不同轴；
- 轴向面间误差：约 `0.00008 mm`；
- 面内旋转：约 `3.674°`，孔阵列相位未严格一致。

因此当前 case1 的正确描述是：

> 法兰面贴合、轴线近似平行，但不同轴，孔阵列相位也不精确。

case2 中出现过一个 OCCT 无碰撞候选，但两个法兰分离、键未进入键槽；它已被接触门降级为 `review`，不算成功。

## 4. 当前已确认的根因

1. `plane_coincidence` 合法保留平面内 `x/y/rz` 自由度；单独使用时会发生横向漂移。
2. sampled contact residual 为了降低局部穿透，可能把两个贴合件沿平面滑开。
3. 当前小候选预算容易只保留平面候选，丢失中央圆柱轴或另一法向极性。
4. 当前全局图通常对一个零件对选择一个 factor，无法同时施加：
   - 平面贴合；
   - 中央共轴；
   - 重复孔阵列相位一致。
5. OCCT Boolean common 只验证相交体积，不验证共轴、接触闭合或功能装配。
6. 当前接触门能拦截“无碰撞但分离”，但还不能拦截“贴合但偏心”。

## 5. 下一步最小治本路线

当前不要继续扩大网络、数据规模或 beam width。按以下顺序推进：

1. **Precision Pose Validator**
   - 轴线距离；
   - 轴夹角；
   - 平面 gap；
   - 重复孔阵列 RMS；
   - 插入深度；
   - exact collision / common volume。
2. **同一零件对的多接口约束束**
   - 一个 pair 允许同时携带 planar、axis、hole-pattern 多个独立 factor；
   - 不再用一个单平面 factor 代表整个装配关系。
3. **接入通用 multi-interface RANSAC**
   - 使用重复等半径圆柱孔阵列求刚体 Pose；
   - 这是通用 B-Rep 几何，不依赖“法兰”名称或 case token。
4. **候选召回与预算修复**
   - 复合多证据候选优先；
   - 保留轴/法向两种极性；
   - 网络候选只增不减。
5. **重新考 case1/2**
   - case1 必须同时通过贴合、共轴与孔阵列相位；
   - case2 必须验证轴贯穿、法兰位置和键槽插入。

在以上几何工程通过前，不建议再次训练网络。

## 6. 目录地图

| 路径 | 作用 |
|---|---|
| `sw/joinable_e2e.py` | STEP pair 的 JoinABLe/B-Rep 候选入口 |
| `sw/learned_joint/pose_learning.py` | Pair Pose、patch interface/contact 网络与 loss |
| `sw/learned_joint/pose_head_adapter.py` | JoinABLe entity candidate 到网络 Pose 的推理适配 |
| `sw/learned_joint/manifold_solver.py` | 有界离散候选 + 连续多零件 SE(3) 优化 |
| `sw/learned_joint/mesh_residuals.py` | sampled contact / penetration 搜索代理 |
| `sw/learned_joint/pose_acceptance.py` | OCCT + 接触支持的保守门控 |
| `sw/learned_joint/report_adapter.py` | 解析基线和网络旁路的候选预算保护 |
| `sw/augment_pair_frontier_with_multi_axis_ransac.py` | 重复圆柱接口的通用复合 Pose 候选 |
| `sw/run_manifold_global_pose.py` | 全局 Pose 求解 CLI |
| `sw/validate_manifold_pose_exact.py` | OCCT 精确碰撞验证 |
| `sw/run_baseline_pose_exam.py` | 单独运行 baseline 或 learned channel 消融 |
| `sw/run_strong_contact_exam.py` | 解析基线 + 网络旁路完整考试链 |
| `public_cad_dataset_audit/train_joinable_pose_heads.py` | 两个轻量网络的训练入口 |
| `public_cad_dataset_audit/audit_pose_ablation_results.py` | 冻结 case 消融审计 |
| `HANDOVER.md` | 详细接手状态、环境、命令与任务列表 |

## 7. 本机运行环境

本机存在三个职责不同的 Python 环境：

```powershell
$MODEL_PY = 'C:\Users\11049\Desktop\Model_match\.conda\python.exe'
$SOLVER_PY = 'C:\Users\11049\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$OCCT_PY = 'C:\Users\11049\miniforge3\envs\cad311\python.exe'
```

- `$MODEL_PY`：加载 PyTorch / JoinABLe / learned heads；
- `$SOLVER_PY`：稳定的 NumPy/SciPy 全局求解；
- `$OCCT_PY`：STEP、B-Rep、Boolean common、STL 导出。

不要混用：

- `.conda` 中的 SciPy dense SVD/QR 曾出现 Windows 原生崩溃；
- `cad311` 中的 NumPy matrix multiply 也出现过原生 DLL 崩溃；
- 因此模型、数值求解和 OCCT 必须保持三运行时隔离。

## 8. 快速复现

### 8.1 case1 解析基线

```powershell
& $SOLVER_PY sw\run_baseline_pose_exam.py `
  sw\exam_brep_manifold_20260710\case_1\pair_frontier\pair_frontier_manifest.json `
  sw\exam_brep_manifold_20260710\case_1\recheck\baseline `
  --solver-python $SOLVER_PY `
  --occt-python $OCCT_PY `
  --render-python $MODEL_PY
```

### 8.2 case1 解析基线 + 网络旁路

```powershell
& $MODEL_PY sw\run_strong_contact_exam.py `
  sw\exam_brep_manifold_20260710\case_1\pair_frontier\pair_frontier_manifest.json `
  D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_train\interface_score\best.pt `
  sw\exam_brep_manifold_20260710\case_1\recheck\protected_sidecar `
  --runtime-python $MODEL_PY `
  --solver-python $SOLVER_PY `
  --occt-python $OCCT_PY `
  --candidate-limit 20 `
  --modes-per-candidate 6 `
  --learned-pose-prior-weight 0.35
```

### 8.3 测试

```powershell
& $SOLVER_PY -m unittest discover -s sw\tests -p 'test*manifold*.py'
& $SOLVER_PY -m unittest sw.tests.test_pose_acceptance
```

最近验证结果：10 个 manifold 测试与 3 个 pose acceptance 测试全部通过。

## 9. 输出语义

建议所有最终输出至少区分：

- `accepted`：精密几何、Pose、接触和冲突门均通过；
- `review`：无严重碰撞，但存在分离、偏心、相位不明、自由度未锁定或证据不足；
- `failed`：碰撞、穿透、约束残差超限或求解失败；
- `unresolved`：当前候选不足，不能可靠决定。

不得仅根据 `final_score`、`collision_free` 或网络 logit 自动接受。

## 10. Git 与数据边界

- GitHub 远端目前仍停留在 `24f3c07`；本轮没有上传新分支或 checkpoint。
- 工作区包含大量历史实验和未跟踪文件；提交前必须逐文件暂存，禁止 `git add .`。
- `D:\Model_match_public_data`、SolidWorks case、训练 checkpoint 和远端凭据不得误提交。
- 桌面压缩包只在用户明确要求时更新。

详细接手信息见 [HANDOVER.md](HANDOVER.md)。
