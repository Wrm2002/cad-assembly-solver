# 遗留 STEP 自动装配工程概览

## 范围与当前状态

该工程是传统 CAD 几何流水线，不是机器学习或强化学习系统。输入为一个
目录中的 `.step/.stp` 零件，输出为 `assembly_manifest.json` 和
`assembly.step`。第一轮保持原算法及原 CLI 不变，只补环境、自检、批跑
和基线记录。

## 主调用链

```text
compute_manifest.py
  -> features.extract_features()
  -> constraints.match_features()
  -> coordinate_solver.solve_in_global_frame()
  -> coordinate_solver.placements_to_manifest()
  -> refinement.refine_placements()
  -> assembly_manifest.json

build_assembly.py
  -> 读取 assembly_manifest.json
  -> OCC 加载、变换并合并各 STEP
  -> assembly.step
  -> 重新读取输出作 I/O 验证
```

### `compute_manifest.py`

`generate_manifest(folder_path, decompose=False)` 扫描目录顶层非
`assembly*` 的 STEP 文件。标准模式将每个文件视为一个零件；分解模式
调用 `decompose_step.py` 拆多实体、用 `prescreen.py` 生成跨 parent
候选，再走同一特征、约束、求解和精调流程。

CLI 兼容入口：

```text
python compute_manifest.py <folder>
python compute_manifest.py <folder> --decompose
python compute_manifest.py <folder> --write-diagnostics
```

`--write-diagnostics` 是接管后增加的可选开关；它不改变候选生成、BFS 或
精调，只在 manifest 写出后调用 `diagnostics.py`。

### `features.py`

`extract_features(filepath, use_occt=True)` 先用 STEP 文本正则快速抽取
圆柱与回退 bbox，再用 OCCT 遍历所有面。返回结构包含：

- `filepath`
- `cylinders`: `radius/origin/axis`，OCCT 面还可能带 `area`
- `planes`: `position/normal/area`
- `cones`、`torii`、`spheres`
- `bbox`: `min/max`
- `occt_stats`: 各类曲面的计数和 OCCT 是否启用

圆柱采用文本结果和 OCCT 结果合并；bbox 优先使用文本点集，仅在缺失时
采用 OCCT bbox。这一优先级可能使文本中的非实体辅助点放大 bbox。

### `constraints.py`

`match_features(parts_features)` 先生成几何候选，再加入 pocket 候选并
去重。现有 match 类型为：

- `coaxial`
- `clearance`
- `planar_mate`
- `planar_align`
- `pocket_mate`

公共字段为 `type`、`parts`、`feat_a_idx`、`feat_b_idx`。类型相关字段有
`radius_match`、`gap`、`distance`，pocket 和圆柱还保留部分以下划线
开头的内部元数据。当前结构没有统一的 `score/confidence/reason`。

阈值是硬编码规则：圆柱半径差、最小圆柱半径、最小平面面积及平面面积
比例。去重按零件对、类型、半径桶或主轴法向桶只留一个“最佳”候选，
因此原始候选数量在进入 BFS 前已发生有损压缩。

### `coordinate_solver.py`

`solve_in_global_frame()` 将 match 建成无权邻接图，参考件按轴向边数量和
最大圆柱半径选择，并固定为 identity。随后执行单向 BFS：

1. 从当前件收集到同一未访问邻件的全部 match；
2. 轴向/间隙/pocket 候选提供旋转和位移；
3. 平面候选只在没有轴向候选时提供位移；
4. 第一次得到 placement 后立即把邻件标为 visited；
5. 不重新评估已放置件，也不回溯。

多约束松弛代码被注释禁用。未遍历到的零件最后也被写成 identity，
`placements_to_manifest()` 不保留 solved/unsolved 区别。这是当前诊断
中最危险的假成功来源。

### `refinement.py`

文件注册了十种策略，但 `refine_placements()` 当前只启用：

- `kabsch_bolt_pattern`
- `coaxial_flip`

碰撞检测没有在主精调入口执行，传给策略的 `collisions` 固定为 `None`。
策略异常被裸 `except Exception` 静默吞掉，因此精调失败不会反映到 CLI
退出码或 manifest。

### `build_assembly.py`

读取 manifest 的 `components`，对每个 `source` 加载 STEP，按
`rotate_sequence`（兼容旧旋转字段）和 `translate` 生成 OCCT 变换，
合并为 compound，并以 AP242 写出 `assembly.step`。`--use-parents`
会优先使用分解模式写入的 `parent_source`。写出后只验证 STEP 能被重新
读取，不验证配合残差、碰撞或装配正确性。

## 辅助模块

- `decompose_step.py`：拆分多实体 STEP，并缓存子实体结果。
- `prescreen.py`：按 bbox 相似度筛分解后的跨 parent 候选。
- `pocket.py`：由平面簇构造 pocket，并生成 pocket match。
- `analyze_geometry.py`：检测螺栓孔和孔键槽，主要供精调策略使用。

## 多零件失败的高风险位置

1. 规则候选没有置信度，弱边和强边在 BFS 中地位相同。
2. 去重发生在评分之前，可能提前丢掉后续全局一致解需要的候选。
3. 参考件和遍历顺序决定结果，第一条可用路径永久固定 placement。
4. “多 match 合并”不是联合优化，只各取首个轴向和平面候选。
5. 未求解件伪装成 identity，导出成功容易被误判为装配成功。
6. refinement 静默吞异常，且主流程不做碰撞检测。
7. `assembly.step` 可重新读取只证明文件有效，不证明几何装配正确。
8. 分解模式可把 3 个输入 parent 扩展到上百个 sub-part，组合规模骤增。
9. 源文件注释和输出字符串存在历史编码乱码，虽不阻塞执行，但降低日志
   可读性和后续正则解析可靠性。

## 建议修改顺序

在基线冻结后，按以下顺序做最小增量改造：

1. 先增加 diagnostics，并显式区分 reference、solved 和 unsolved；
2. 集中实现 match scoring，保留原规则候选生成；
3. 增加有审计日志的 pruning，量化每条边为什么删除；
4. 增加 placement validation，先有可信成功标准；
5. 对不超过 6 个零件增加 beam search / branch-and-bound，保留 BFS
   作为 fast baseline；
6. 最后建设 SolidWorks 程序化数据生成管线；
7. 只有在真实失败集和规则评分稳定后，再考虑学习型 match scorer。

第一轮不应训练端到端网络、不应引入 RL，也不应重写现有几何模块。

## 接管后新增的旁路能力

- `diagnostics.py`：输出 `assembly_diagnostics.json` 和
  `assembly_report.txt`，显式标出断图、孤立件、未求解件、identity
  歧义与异常变换。
- `match_scoring.py`：为原规则候选添加 `score`、`confidence` 和结构化
  `reason`；目前只用于诊断和独立 `scored_matches.json`，尚未进入 BFS
  或剪枝，因此不改变基线求解结果。
- `match_pruning.py`：按分数、零件对 top-k、最大邻居数和证据类型剪枝，
  输出完整 `kept_matches.json` / `removed_matches.json` 审计日志。
- `placement_validation.py`：计算约束残差、断图、identity/未求解状态与
  变换后 bbox 碰撞预检；bbox 不直接冒充精确穿透结论。
- `small_assembly_solver.py`：对 1–6 件执行多假设 beam search、分支上界
  和逐项 penalty；旧 BFS 仍是默认。
- `run_experiments.py` / `evaluate_results.py`：运行四种消融模式并按组
  汇总指标。
- `sw_dataset_generator/`：通过 SolidWorks COM 生成原生零件、装配、
  STEP、真值和困难负样本元数据。
