# Model_match 项目结构与唯一运行主线

本文件定义当前源码职责。整理原则是先统一入口与依赖方向，不移动仍被历史脚本引用的文件，也不删除实验取证数据。

## 1. 当前生产主线

已知输入零件属于同一装配体时，唯一主入口是：

```powershell
python sw/known_group_assembly.py <case_dir> --output-dir <output_dir>
```

主链依次执行：

1. STEP 解析与几何特征提取；
2. direct pair candidate 生成与保守选边；
3. 小规模装配初始 Pose；
4. 离散自由度扩展与组级 Pose 打分；
5. OCCT 精确碰撞验证；
6. `accepted / review / rejected / unresolved` 保守输出。

`sw/run_exam_v2.py` 只负责 case1–5 回归，不参与推理，不得向算法传递 case id、文件名 token 或人工答案。

## 2. Pose 与 JoinABLe 模块

### 2.1 唯一 Pose 核心

`sw/pose_search/` 是新增的通用 Pose 核心：

- `transforms.py`：双轴对齐、轴向旋转、轴向偏移、刚体矩阵与 placement 转换；
- `joinable_search.py`：JoinABLe-style top-k、offset/rotation/axis-sign、Nelder–Mead、SDF overlap/contact；
- `group_pose.py`：把多个 pair Pose 传播成 group Pose，并审计环一致性。

这里不包含 case 名称、零件名称或装配标签特判。所有输出仍需经过组级约束闭合和 OCCT exact collision。

### 2.2 JoinABLe 官方模型链

- `joinable_source/`：官方模型源码与 checkpoint，只读参考；
- `joinable_migration_audit/step_to_brep_graph_probe.py`：STEP → 审计型 B-Rep 图；
- `cad_assembly_agent/tools/joinable_interface_predictor/pretrained_joinable_predictor.py`：官方 checkpoint 的规范推理适配器；
- `sw/joinable_e2e.py`：当前唯一 E2E 入口，执行 B-Rep top-k 到 pair Pose proposal；
- `joinable_gpu_reproduction/`：官方数据/训练/迁移实验，不是生产入口。

兼容入口：

- `sw/joinable_inference.py`：转发到规范 E2E，不再维护独立图构造；
- `sw/joinable_pose_solver.py`：转发到 `pose_search`，供旧脚本调用；
- `sw/search_simplex.py`：保留旧类名，内部使用统一 SDF Pose 核心。

## 3. 其他源码职责

- `sw/features.py`、`constraints.py`：几何特征和候选约束；
- `sw/direct_assembly_graph.py`：已知组的直接装配边；
- `sw/small_assembly_solver.py`：初始多零件 Pose 搜索；
- `sw/placement_validation.py`：残差、AABB 与 OCCT 精确碰撞；
- `sw/global_optimizer/`：混池保守候选分层与组一致性；
- `sw/conservative_pipeline.py`：D3.5–D7 保守工程链；
- `public_cad_dataset_audit/`：Fusion360/真实 CAD 数据转换与 benchmark；
- `cad_assembly_agent/`：工具化组件；只有上述 JoinABLe predictor 被当前主线复用。

## 4. 历史/实验入口

以下文件保留用于对照，不作为当前最终求解入口：

- `sw/stop_plane_solver.py`；
- `sw/sequential_assembly.py`；
- `sw/compute_manifest.py` 与旧 BFS 流程；
- `joinable_step4_bundle_20260705/`、`joinable_step4_addon_20260705/`；
- `_audit_sw_origin/`。

历史脚本可以运行，但其结果不能覆盖 `known_group_assembly.py` 的保守审计结果。

## 5. 数据与生成物

以下属于数据、缓存或实验输出，不是源码：

- `sw/1`–`sw/5*`：考试输入与运行结果；
- `sw/data/`、`sw/synthetic_*`、`sw/baseline_*`；
- `joinable_gpu_reproduction/*.json|*.log|*.zip`；
- Word/PDF 报告、压缩包、崩溃转储；
- `.conda/`、`__pycache__/`、`.vscode/`。

这些内容默认保留审计证据，不在代码整理时擅自删除。

## 6. 验证命令

```powershell
$env:PYTHONPATH='sw;.'
python -m unittest discover -s sw/tests -v
python sw/run_exam_v2.py
```

JoinABLe pair Pose 示例：

```powershell
python sw/joinable_e2e.py part_a.step part_b.step `
  --output-dir pair_pose_output --top-k 20 --pose-top-k 5
```

若已有多个 pair E2E 报告，可把目录传给已知组入口：

```powershell
python sw/known_group_assembly.py <case_dir> `
  --joinable-pose-dir <pair_report_root>
```

只有 pair Pose 图全连通且环一致时才会形成 group proposal；否则保留原解析求解并记录 review 原因。
