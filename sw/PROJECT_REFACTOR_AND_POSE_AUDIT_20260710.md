# 项目代码整理、JoinABLe 复现与 Pose 求解审计

日期：2026-07-10

## 一、结论

本轮完成了安全重构和一轮真实 STEP Pose 实验。直接装配边能力保持稳定：case1–5 全部通过；Pose 仍只有 case1 达到“约束全闭合 + OCCT 精确无碰撞”，case2–5 继续保守进入 review，没有把不确定结果伪装为成功。

JoinABLe 现在已经从三套互相矛盾的实现收敛为一条可审计链：

```text
STEP
  -> audited OCCT B-Rep graph
  -> released JoinABLe checkpoint
  -> all face/edge cross-body top-k
  -> two-sided joint-axis extraction
  -> rigid axis alignment
  -> offset / rotation / axis-sign search
  -> normalized SDF overlap/contact objective
  -> OCCT exact collision rerank
  -> optional bounded multi-pair group composition
  -> known-group closure/review gate
```

它的正确职责仍是接口和 Pose proposal 工具，不是装配正确性的单独裁判。

## 二、代码审计发现与修复

### 1. JoinABLe 推理图错误

旧 `sw/joinable_inference.py` 存在两个关键问题：

- 只在 face-face 子矩阵上取 top-k，丢失官方任务支持的 face-edge、edge-face、edge-edge；
- joint graph 的 body-B 节点没有加 body-A 节点数偏移，跨体边实际指回了 body-A 的前若干节点。

处理：旧文件改为兼容入口，统一调用已经审计的官方 checkpoint predictor 和 `sw/joinable_e2e.py`。

### 2. 旧 Simplex 物理单位失真

旧 `sw/search_simplex.py` 的 `_compute_offset_limit()` 没有真正给 `offset_limit` 赋值，随后搜索又把毫米偏移乘以 1500，可能产生十万毫米级无意义位移。

处理：重写为 `sw/pose_search/joinable_search.py` 的兼容包装。当前 offset 使用模型真实单位，并由两个零件沿关节轴的几何跨度限制。

### 3. Pose 只使用一侧轴

旧 E2E 从 top-k 中只选一侧实体轴，另一侧轴没有参与对齐，因此不等价于 JoinABLe 的 joint-axis alignment。

处理：每个候选必须同时提供 fixed/moving 两侧的 origin+direction；缺任一侧就跳过该 Pose seed。

### 4. “best”不是最低代价候选

旧 E2E 把 top-k 的第一个 Pose 直接写为 best，没有按搜索代价排序。

处理：所有 `(prediction, axis_sign)` 的结果统一排序，并分别保留 SDF 最优和 OCCT 精确无碰撞最优。

### 5. flip 不是合法 CAD 刚体

官方发布代码把 flip 实现为行列式 -1 的反射。它适合论文 mesh search 参数化，但不能直接写入 CAD 装配变换。

处理：生产实现把 flip 解释为关节轴方向符号歧义，通过对齐到反向轴生成行列式 +1 的刚体旋转；报告中显式记录此适配。

### 6. 验证后再次移动零件

旧 `known_group_assembly.py` 在约束闭合和 OCCT 碰撞验证后，又执行一次 cylinder axial stop，导致导出 Pose 与被验证 Pose 不一致。

处理：取消验证后的位姿修改。轴向 stop/depth 只能作为验证前的普通候选进入统一审计。

### 7. 精确验证预算失控

旧逻辑可能对所有小型候选做 OCCT boolean common；大型 STEP 的全局布尔检查也可能长时间阻塞。

处理：小型 case 的 exact frontier 上限为 180；大型 STEP 默认关闭全局 exact，维持 review，并保留闭合、SDF/AABB 和原因字段。

## 三、新增/重构模块

- `sw/pose_search/transforms.py`
  - 双轴对齐；
  - 轴向旋转与偏移；
  - proper-rigid axis sign；
  - 4x4 matrix 与项目 placement 双向转换。
- `sw/pose_search/joinable_search.py`
  - top-k joint axis seed；
  - Nelder–Mead offset/rotation；
  - 轴方向枚举；
  - JoinABLe default/smooth SDF objective；
  - 保留零偏移 interface seed，防止采样目标把接触面推入微小穿透。
- `sw/pose_search/group_pose.py`
  - pair Pose 到 group Pose 的图传播；
  - 环一致性审计；
  - 多个 pair 候选的有界组合；
  - 不一致和不连通结果进入 review。
- `sw/joinable_e2e.py`
  - 当前唯一 JoinABLe STEP pair E2E 入口；
  - 官方 checkpoint、all-entity top-k、双轴搜索、OCCT rerank。
- `PROJECT_STRUCTURE.md`
  - 修复历史乱码；
  - 明确唯一主入口、兼容层、数据和历史实验边界。

兼容层：

- `sw/joinable_inference.py`；
- `sw/joinable_pose_solver.py`；
- `sw/search_simplex.py`。

## 四、JoinABLe 复现程度

| 模块 | 当前状态 | 说明 |
|---|---|---|
| B-Rep face/edge 图 | 已接通 | OCCT 输出 face/edge 节点、邻接、checkpoint 最小特征 |
| 官方 checkpoint | 已复用 | 使用发布的 `last_run_*.ckpt` |
| all-entity top-k | 已复现 | 不再限制为 face-face |
| joint axis | 已复现并补全 | plane/cylinder/circle/ellipse/line 可提供轴或法向 |
| 双轴 alignment | 已复现 | moving axis origin/direction 映射到 fixed axis |
| offset/rotation | 已复现 | 物理单位有界 Nelder–Mead |
| flip | 工程适配复现 | 论文反射改为 proper-rigid 轴方向枚举 |
| overlap/contact cost | 已复现 | 双向体积/面积比例归一化，保留 default/smooth |
| top-k pose search | 已复现 | 每个预测枚举两种轴方向并统一排序 |
| OCCT exact gate | 项目增强 | 论文 SDF 后增加精确公共体积验证 |
| multi-part group pose | 项目增强，仍在发展 | pair 图传播、环审计、有界组合、轴向 relaxation |
| 完整训练数据增强/重新训练 | 本轮未做 | 当前没有必要，也不应为短期 Pose 问题盲目训练 |
| no-joint / pair membership | 官方模型不具备 | 不能用 JoinABLe softmax 判断两个零件是否应连接 |

发布 checkpoint 使用的输入是 `entity_types,length,face_reversed,edge_reversed`，因此缺失 UV grid 不阻塞当前 checkpoint 推理；若以后复现论文其它输入消融或重新训练，仍需补 UV/curve sampling、convexity 和 dihedral 的严格跨核对齐。

## 五、真实 STEP 实验

### 1. case1 法兰 pair

- joint entity candidate 总数：1600；
- top-k 中同时出现 plane、cylinder 和 circle edge；
- SDF 连续最优把 moving flange 推进约 0.035 mm；
- OCCT 检出约 387 mm³ 公共体积；
- 保留的轴对齐零偏移 seed 无精确碰撞，最终被选为 exact-collision-free pair Pose。

这证明 SDF 目标适合搜索，但不能取代 exact CAD gate。

### 2. case2 三条 pair 边

- shaft–flange A：1265 个 entity-pair，找到 exact-collision-free Pose；
- shaft–flange B：1265 个 entity-pair，找到 exact-collision-free Pose；
- shaft–key：414 个 entity-pair，找到 exact-collision-free Pose。

但 top-1 pair Pose 合成 group 后：

- 两个法兰之间公共体积约 131,117 mm³；
- 法兰还分别与 key 发生实体相交；
- group Pose 因此被正确拒绝，没有自动接受。

继续保留多个 pair 无碰撞候选后：

- 共读取 19 个 pair Pose candidates；
- 有界形成 48 个 group hypotheses；
- 其中 16 个约束全闭合，但没有一个 group exact-collision-free；
- 加入 shared-axis relaxation 后总候选 902；
- relaxed candidates 中 65 个全闭合、2 个精确无碰撞，但仍没有候选同时满足两者。

结论：case2 的剩余问题是 group-level coupled optimization，不是再提高单边 JoinABLe top-k。

### 3. case1–5 回归

| case | 直接边 | Pose | 主要原因 |
|---|---|---|---|
| case1 | 正确 | valid | 全闭合且 OCCT 精确无碰撞 |
| case2 | 正确 | uncertain | 全闭合候选仍有组内碰撞；无碰撞候选少一条 clearance 闭合 |
| case3 | 正确 | uncertain | closure 1/1，但大型 STEP exact 默认停用 |
| case4 | 正确 | uncertain | planar 已贴合，但弱 pocket 候选未闭合，当前保守门仍要求 pocket |
| case5 | 正确 | uncertain | 两条边均有 planar 接触；高置信 pocket/插入关系未闭合 |

直接边总计 5/5 case 通过，未出现新增 false edge。Pose 仍为 1/5 valid，未通过的 4 组全部进入 review。

## 六、测试审计

- `sw/tests`：96/96；
- `public_cad_dataset_audit/tests`：13/13；
- `joinable_migration_audit/tests`：4/4；
- 合计：113/113。

修复了一条旧测试语义错误：测试目标是“小零件 planar interface 召回”，但输入法向平行，正确关系应为 `planar_align`，旧断言却只接受 `planar_mate`。现按接口召回检查两种合法 planar polarity。

## 七、当前仍未解决的问题

1. pair SDF optimum 不等于 group optimum；
2. case2 需要同时优化两个法兰的轴向位置、方向和 key 位置；
3. case3 需要更高效的 per-solid/per-pair exact broad phase，不能直接对百万实体 STEP 反复做全局 boolean；
4. case4/5 的 pocket detector 会把局部三平面结构当成候选，需区分“强插槽”与“弱角点/盒角”；
5. 当前 contact objective 仍是采样比例/解析残差，不是精确 CAD contact area；
6. `known_group_assembly.py` 仍偏大，后续可继续拆分 candidate generation、group objective 和 report serialization，但本轮没有为追求目录整齐而冒险大搬家。

## 八、下一步最小路线

1. 实现真正的 multi-body continuous optimizer：以每个 satellite 的轴向 offset/rotation 为变量，联合最小化 selected-edge residual、non-edge overlap 和 group collision proxy；
2. 在内循环使用 mesh BVH/SDF union，在最终少量候选上调用 OCCT exact；
3. 为 pocket evidence 增加强度分层：只有尺寸、方向、壁面、插入深度都成立时才成为 mandatory closure；弱 pocket 只作 review evidence；
4. 给 case3 增加可超时的分实体精确碰撞 worker；
5. 在非 case1–5 的 Fusion360/functional holdout 上复查 pair Pose 与 group Pose，防止只优化考试样本。

本轮没有使用文件名 token、case id 或人工标签改变任何算法判定。
