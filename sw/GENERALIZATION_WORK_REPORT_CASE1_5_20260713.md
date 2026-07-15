# case1～5 装配 Pose 泛化修复工作报告

日期：2026-07-13  
范围：已知真组 Pose recovery、保守验证、case1～5 冻结考试渲染  
原则：成功率优先；不读取名称/case ID/颜色；不以碰撞自由代替装配正确；不因考试答案增加自动接受权限。

## 1. 结论

case1、case3 保持已确认结果；case4 已从“内存条平放”修正为边缘入槽；
case5 保持 enclosure/bay 放置；case2 已修正为两法兰面贴合、轴位于联合中心、
键跨过接触面并同时进入轴键槽和两侧轮毂键槽。用户已对最新 case2 渲染确认正确。

case2 的修复没有保存人工答案的位移，也没有按 case 或零件名称分支。系统从匿名
B-Rep 关系中识别一个组级约束环：

```text
支撑 A ── 复合共轴 + 端面 + 键槽相位 ── 支撑 B
   │                                           │
   └──── 同轴适配孔 ── CLEARANCE 轴 ─────────┘
                              │
                         刚体键跨过接触面
```

只有整个环闭合，才生成“轴在两支撑联合中心、双侧均有有效插入”的候选；
缺失任一关键证据时 abstain。即使闭合成功，该候选仍是
`proposal_only/review_required/can_auto_accept=false`，不会直接进入自动接受。

五组当前状态必须与渲染关系分开看：

| Case | 保守状态 | checked/rank | 当前验证结论 |
|---|---|---:|---|
| 1 | `accepted / valid` | `220 / 8` | `1/1` closure，OCCT exact 0 collision |
| 2 | `review / valid` | `1760 / 882` | `3/3` closure，OCCT exact 0 collision；proposal-only |
| 3 | `review / uncertain` | `55 / 1` | `1/1` closure；2 个 solid pair、约 `192.715 mm³` 局部干涉 |
| 4 | `review / uncertain` | `176 / 155` | `2/2` closure；所选 rank exact 因预算跳过且主板覆盖不全 |
| 5 | `review / uncertain` | `126 / 28` | 仅 `1/2` closure；exact 因闭合不完整跳过 |

在这五个已知真组的窄评测范围内，自动接受为 `1`、review 为 `4`、已知错误自动
接受为 `0`，所以表面上 auto-accept precision 为 `1/1`；样本量太小且不是 mixed-pool，
不能把这个数当作项目级 accepted precision 或泛化证明。

## 2. 技术路线

```text
STEP / SolidWorks
  -> OCCT 匿名 B-Rep 特征与拓扑 sidecar（含 SHA 校验）
  -> JoinABLe + 解析接口候选的 protected union
  -> 按接口族生成有界 Pose：轴向复合、平面足迹、边缘槽、包络/机箱 bay
  -> 多零件因子图与刚体依赖传播
  -> 组级闭合：轴向/径向/相位/插入深度/跨接口零件分别验收
  -> OCCT exact collision + 几何覆盖审计
  -> accepted | review | rejected | unresolved
```

网络只负责候选召回/排序旁路；本轮关键修复在几何闭合与审计层，没有训练新网络，
没有引入 RL、复杂多 Agent 或 DeepSeek 裁判。DeepSeek 仍只能用于 review 解释。

## 3. case2：从错误的二元端面止挡改为组级装配闭合

### 3.1 之前为什么会反复错

单独求解“法兰—法兰”和“法兰—轴”会产生互相不兼容的局部正确解：

- 轴端面贴某个法兰端面，可能共轴但轴不在中间；
- 把轴移到中间，若不重算法兰 compound evidence，法兰面又可能分开；
- 只对轴做相位约束，不能证明键同时穿过两个轮毂的内侧键槽；
- 某个旧 proposal 的证据若只按 connection ID 复用，会错误“救活”新的位姿。

因此这不是再调一个 top-1 分数，而是把 pairwise pose 提升为一个小型组级因子环。

### 3.2 当前通用触发条件

`sw/known_group_assembly.py` 的轴向组闭合只使用几何/拓扑字段，并要求：

1. 两个主圆柱半径可比，且从同一接触面向相反方向延伸；
2. 小径轴通过 selected `CLEARANCE` 边连接到其中一个支撑；
3. 两个支撑都存在与轴近似同轴、半径适配、表面极性为凹面的内孔；
4. 支撑—支撑和支撑—轴的 compound history 都有成对
   `topological_key_slot` witness，且 witness ID 来自与 STEP SHA-256 匹配的 sidecar；
5. 支撑端面当前位姿确实贴合，相位仍位于有效 orbit；
6. 轴的真实终端面区间以支撑联合中心为中点，且在两侧都有足够、平衡的插入重叠；
7. 与轴刚性连接的非轴向小零件必须跨过支撑接触面；
8. evidence 绑定精确的 `candidate_id + proposal_id`，每次 closure 都在当前位姿重算。
9. 长度、中心和跨面门槛以零件尺度归一化；`0.1x/1x/10x` 匿名夹具走同一逻辑。

没有读取 `case2`、文件名中的 `flange/shaft/key`、零件 label 或渲染颜色；
也没有写入本例最终 `-55 mm` 平移。候选位移来自当前几何的支撑接触面与轴终端面中点之差。

### 3.3 数值审计

数据目录：`generalization_work/case2_compound_v8/`

| 指标 | 当前值 |
|---|---:|
| selected origin | `axial_group_symmetric_centering` |
| selected pose rank | `882 / 1760` |
| pose status | `valid` |
| 两法兰端面距离残差 | `1.776e-15 mm` |
| 轴中心残差 | `5.329e-15 mm` |
| 支撑联合中心残差 | `8.882e-15 mm` |
| 轴双侧插入重叠 | `45.0 / 45.0 mm` |
| overlap balance | `1.0` |
| 键槽 phase residual | `0 deg` |
| 键是否跨过支撑接触面 | `true` |
| OCCT solid collision count | `0` |

保守输出仍为 `accepted=0, review=1`：正确的考试位姿没有被用作放宽自动接受门槛的理由。

## 4. case4：边缘槽接口与不完整几何覆盖

case4 的错误根因是把 DIMM 当作普通平面足迹零件，得到“共面但未入槽”的候选。
新增的 edge-slot provider 从重复、等间距、带镜像侧壁的内部槽族召回候选，并把薄功能体
沿槽法向插入。当前审计值：

| 指标 | 当前值 |
|---|---:|
| 检出的槽族 | 6 个等间距槽 |
| 槽距 | `7.5438 mm` |
| 槽宽 | `1.6673 mm` |
| DIMM 功能体厚度 | `1.27 mm` |
| 理论单侧间隙 | `0.1987 mm` |
| 入槽边到槽底残差 | `2.84e-14 mm` |
| 输出状态 | `review / uncertain` |

主板 STEP 同时包含实体和大量 open-shell/orphan faces；当前有 `104864` 个 face、
`391` 个 shell 未被 solid boolean 覆盖。因此碰撞验证现在显式输出
`coverage_audit.complete=false`、`collision_result=uncertain`，不再把未覆盖区域误报成
完整的 `collision_free=true`。渲染关系正确不等于 exact validation 已完整通过。

## 5. 修改与新增文件

本轮核心代码：

- `contracts.py`：把 compound、组级轴向居中、edge-slot、enclosure 证据纳入连接契约；
- `known_group_assembly.py`：有界接口族候选、刚体依赖传播、组级轴向闭合、精确 evidence 绑定、保守输出传播；
- `placement_validation.py`：实体/open-shell 覆盖审计，partial/uncertain 与已检测碰撞并存；
- `pose_search/axial_compound_interface.py`：共轴、端面、极性、相位 orbit 与非对称 witness；
- `pose_search/edge_slot_interface.py`：重复内部槽族、薄体插入和 review-only 证据；
- `pose_search/enclosure_bay.py`、`planar_footprint.py`、`obb_insertion.py`：其他通用接口族；
- `render_assembly_manifest_occt.py`：关系视图、透明上下文和多视角审计渲染。

关键测试：

- `tests/test_axial_group_centering.py`；
- `tests/test_axial_compound_interface.py`；
- `tests/test_rigid_dependent_propagation.py`；
- `tests/test_edge_slot_interface.py`；
- `tests/test_placement_validation.py`；
- `tests/test_known_group_contract.py`。

当前组合回归：`72 tests passed`。覆盖的反例包括 stale proposal、轴向端面止挡但无插入、
缺少 CLEARANCE、缺少第二支撑孔、外凸圆柱冒充内孔、无/错 SHA sidecar、无 ID witness、
键未跨接触面、孤立边界面伪槽、长度不兼容槽和 open-shell 未覆盖；匿名
`0.1x` 与 `10x` 尺度夹具都能生成同类候选。

## 6. 泛化边界与下一步最小验证

本轮代码审计和不透明 ID 试跑证明：新组约束不依赖 case/name/color，也没有
`-55 mm` 或某个固定旋转；它能在匿名合成反例上 abstain。仓库中的 legacy
exam/debug/report 脚本仍会出现 case 名，不能把结论扩大成“整个仓库没有 case 字符串”。

当前仍有一个明确证据边界：跨面 dependent 的检查只能证明其与轴刚性相连并跨过
支撑接触面，还没有用 exact slot-frame/截面体积证明它确实进入三个键槽。由于该路径
始终 review-only，这个缺口不会直接制造 accepted false positive，但必须在独立 holdout
和局部 OCCT clearance 中补齐。

它还没有证明所有轴—轮毂—键装配的统计泛化。下一步不应再加 case-specific rule，而应建立
一个独立小 holdout：

1. 同一功能关系的不同尺度、不同外轮廓、不同孔阵列；
2. 单轮毂、无键、只有一侧键槽、键未跨面、错相位、阶梯轴和带头轴困难负例；
3. 支撑长度明显不对称但工程上允许轴向偏置的 abstain 样本；
4. 不同 DIMM/edge-card 厚度、槽宽、槽深和有/无底部止挡；
5. 统计 proposal recall、review rate、false positive，而不是只看五个考试图。

在 holdout 通过前，轴向组居中和 edge-slot 均继续 review-only；`final_score` 只排序，
不能单独决定 accepted。

## 7. case1～5 最终渲染

### Case 1

![case1](generalization_work/render_gallery/case1_current.png)

### Case 2

![case2](generalization_work/render_gallery/case2_current.png)

### Case 3

![case3](generalization_work/render_gallery/case3_current.png)

### Case 4

![case4](generalization_work/render_gallery/case4_current.png)

### Case 5

![case5](generalization_work/render_gallery/case5_current_full.png)

简化聚焦视图另见
[`case5_current.png`](generalization_work/render_gallery/case5_current.png)。
