# JoinABLe 迁移审计报告

**明确结论：部分适合迁移，但需要先补齐数据结构**

JoinABLe 适合被评估为“已知零件对之间的 B-Rep interface / joint candidate
predictor”，可以替换当前全表面 plane/cylinder 暴力枚举的候选生成层。它不能
直接解决 mixed-pool 分组、功能语义判断、多零件全局一致性或最终物理验证，也
不能未经适配直接读取当前 SolidWorks/STEP。

审计基于官方仓库 commit
`5a3b9933e91269d19b6cc733b244a879eb4bfc30`、官方 j1.0.0 Joint Data
和当前项目 3 个真实 STEP 的实际代码输出。没有训练模型、调用 DeepSeek 或修改
主 solver。

## 结论表

| 问题 | 结论 | 证据 | 对当前项目的意义 |
|---|---|---|---|
| 1. 输入是什么 | 一对 B-Rep body 的预处理 graph；pose search 还需要原始 joint JSON 和 OBJ | `JointGraphDataset.load_graph()` 同时读取 `body_one/body_two.json`；README 要求 JSON/OBJ | 输入粒度是 part pair，不是 mixed pool |
| 2. 输出是什么 | 所有跨 body entity pair 的 joint 分数；再生成 axis、offset、rotation、flip 和 4×4 transform | `make_joint_graph()` 构造笛卡尔积；`SearchSimplex.search()` 返回 pose 参数 | 可给现有 solver 提供 top-K 接口及初始位姿 |
| 3. B-Rep graph 如何构建 | 每个 body 一个 NetworkX node-link graph，face/edge 作为节点，B-Rep adjacency 作为 link | 官方 body JSON 实测含 `nodes/links/properties` | 可由 OCCT 构造同类异构图 |
| 4. graph node 是什么 | face 和 edge 都是 node；face 在前，edge 索引加 `face_count` | `offset_joint_index()`；实测 graph 77 nodes = 23 faces + 54 edges | 不能只提取 plane/cylinder faces |
| 5. graph edge 是什么 | body graph 是 face-edge 拓扑邻接；joint graph 是两 body 全节点笛卡尔积 | 实测 link `source=face,target=edge`；`make_joint_graph()` | 模型把暴力 surface pair 转成学习排序，但内部仍评分跨图 entity pairs |
| 6. face feature | 10×10 points、normals、trimming mask；surface type、area、reversed、radius | `face_grid_feature_map`、`face_entity_feature_map` | 当前最小 STEP graph 还不满足预训练输入 |
| 7. edge feature | 10 点 points/tangents；curve type、length、reversed、convexity、dihedral angle、radius | `edge_grid_feature_map`、`edge_entity_feature_map` | 需要增加 OCCT curve sampling 与邻面特征 |
| 8. joint axis 如何表示 | 从选中的 analytic face/edge 派生无限直线 origin+direction | `joint/joint_axis.py` | plane、cylinder、circle 等可与当前轴线规则对齐 |
| 9. pose search | 先对齐预测轴，再用 Nelder–Mead 搜 offset/rotation，并枚举 flip；目标基于 SDF overlap/contact | `search_simplex.py`、`joint_environment.py` | 可作 pose seed；不能替代 OCCT exact collision |
| 10. 依赖哪些 Fusion 字段 | `body_one/body_two`、`joints`、两侧 `geometry_or_origin`、entity type/index、transform、holes、body graph/OBJ/properties | 20 个 j1.0.0 样本实测 | 官方 Joint Data 足以形成监督和评估闭环 |
| 11. 是否用 designer-selected entity | 是。用户选中的 BRepFace/BRepEdge 是 weak supervision joint label，另有 equivalent/ambiguous/hole augmentation | README 与 `get_label_matrix()` | 标签是设计操作记录，不是匿名几何猜测 |
| 12. 能解决哪一层 | 已知 pair 的 joint/interface entity top-K 和 joint axis 候选 | 官方任务定义为 pair-of-parts joint | 最适合替换候选召回层 |
| 13. 不能解决哪一层 | mixed-pool grouping、功能语义、BOM、全局多零件一致性、精确物理接受 | 训练样本本身已给定两个 body | 仍需 pair graph、保守 gate 和 OCCT validator |
| 14. 能否直接 mixed-pool grouping | 不能 | 模型 API 是 body pair，label matrix 是两个图的笛卡尔积 | 必须在外层先枚举/检索 part pair，再构连接图 |
| 15. 能否直接处理当前 SolidWorks/STEP | 不能直接；可做 adapter | 官方输入是 Fusion 预处理 JSON；当前 3 STEP 只完成 coarse graph | 需 feature parity、单位/归一化和 id round-trip |
| 16. 与 STEP/OCCT 差异 | STEP 有几何拓扑但无 designer joint/contact/pose/hierarchy；OCCT id 也非永久 CAD id | `schema_gap_report.json` 与 3/3 STEP probe | 可用于推理特征，不能自动制造监督真值 |

## 官方图结构实测

一个抽查 body graph：

- `properties.face_count=23`
- `properties.edge_count=54`
- `nodes=77`
- `links=108`
- face node 含 surface、area、centroid、normal、UV sampled points/normals/mask
- edge node 含 curve、length、radius、convexity、dihedral、sampled points/tangents
- graph 为 undirected node-link 格式，link 连接 face node 与 edge node

标签矩阵大小为 `(#face + #edge)_A × (#face + #edge)_B`。标签包含
Non-joint、Joint、Ambiguous、JointEquivalent、AmbiguousEquivalent、Hole 和
HoleEquivalent；默认训练最终可映射为 joint/non-joint link prediction。

## 20 个官方 Joint Data 样本实测

| 项目 | 数量 |
|---|---:|
| 转换成功 | 20 / 20 |
| 转换失败 | 0 |
| 有两侧 interface entity id | 20 |
| 有 axis origin/direction | 20 |
| 有 transform | 20 |
| 有对应 joint 的 contact face label | 18 |
| 有 holes label | 18 |

Joint type 分布：

- RevoluteJointType：11
- RigidJointType：4
- CylindricalJointType：2
- PlanarJointType：2
- SliderJointType：1

Joint-set JSON 没有独立的显式 `assembly_id` 字段。本适配器使用两个 body 名称
的公共前缀作为可审计推断 id，并把 `source_explicit_assembly_id` 写入不可用字段；
不会把推断值伪装成官方 assembly 主键。

## 当前 STEP/OCCT 实测

| 文件 | face | edge | face-edge adjacency | 状态 |
|---|---:|---:|---:|---|
| `sw/1/flange_part_a.step` | 13 | 27 | 45 | success |
| `sw/1/flange_part_b.step` | 13 | 27 | 45 | success |
| `sw/2/shaft_with_keyway.step` | 8 | 15 | 29 | success |

每个节点已有 1-based OCCT topology index、类型、测度、质心、方向和几何签名。
同一 worker 内可通过 `IndexedMap.FindKey(index)` 精确反查 shape。序列化后只在
输入 SHA256、OCCT build 和导入设置不变时可按同序重建；STEP 重导出、healing
或布尔修改后不保证 id 不变。

## 迁移可行性表

| 模块 | 可迁移性 | 风险 | 建议 |
|---|---:|---|---|
| Common joint-interface schema | 高 | occurrence/body 语义混淆 | 保留 source id semantics 和 unavailable fields |
| Fusion Joint Data parser | 高 | 显式 assembly id 缺失 | body 公共前缀只作推断 id |
| Designer joint entity label | 高 | weak supervision 含设计偏好 | 用于 top-K recall，不当作功能装配真值 |
| Fusion contact/hole label | 中高 | contact 可能不完备 | 分开质量标记，不能等同 joint |
| STEP face/edge topology graph | 高 | topology id 非永久稳定 | hash + indexed id + signature + 每次重验 |
| JoinABLe exact feature parity | 中 | UV trim sampling、convexity、dihedral 尚缺 | 先补 adapter 并做逐字段对照测试 |
| 官方 pretrained checkpoint 复用 | 中低 | Python 3.7/PyTorch 1.8/PyG 1.7.2 老环境与 domain gap | 隔离环境，先在官方 test split 复现 |
| JoinABLe pose search | 中 | 使用 mesh/SDF 近似和旧依赖 | 只作 seed，最终交给现有 OCCT validator |
| mixed-pool grouping | 低，非直接能力 | O(n²) pair 推理与 false positive | 外层检索、top-K、保守图门控 |
| 功能/语义装配判断 | 不可迁移 | 模型无 BOM/role/功能标签 | 不宣称可解决；交给真实数据和人工复核 |

## 环境与复现风险

官方环境固定在 Python 3.7、PyTorch 1.8、torch-geometric 1.7.2、
PyTorch Lightning 1.3.7，并依赖 `igl`、`pysdf` 和 `trimesh`。仓库只有一个
2022 年提交。当前任务没有把这些旧依赖装进主项目环境，也没有运行训练；这是
有意的隔离，不是遗漏。预训练复现应在单独环境完成。

本机 `cad_asm` 环境的 NumPy `linalg.inv` 曾触发 Windows 原生运行时异常
`0xc06d007f`。适配器已改为纯 Python 刚体逆变换，20/20 转换通过。这个异常与
JoinABLe 算法无关，但说明迁移时必须保持 native worker 隔离。

## 能用的部分

1. 官方 Joint Data 的 face/edge 监督、部分 contact、holes、transform。
2. B-Rep heterogeneous graph 表达。
3. analytic entity → joint axis 的规则。
4. 已知 part pair 的跨图 entity-pair top-K 排序任务定义。
5. 预测 axis 后的 pose seed 思路。
6. pretrained checkpoints，可在独立旧环境完成复现后再判断是否复用。

## 当前不能用或不能声称的部分

1. 不能直接从 mixed pool 输出真实装配分组。
2. 不能从 collision-free 推导功能正确。
3. 不能从孤立 STEP 恢复 designer joint、mate type、assembly pose 或 hierarchy。
4. 当前 coarse STEP graph 不能直接喂给 pretrained JoinABLe。
5. 尚未在当前真实 STEP 上得到模型预测；本任务没有伪造该结果。
6. 不应把 JoinABLe pose search 直接替换 exact OCCT collision validation。

## 最小决策

允许进入 **adapter feature-parity 开发与规则 top-K baseline**，暂不允许接入主
solver。只有官方 test 复现通过、STEP graph 特征逐字段一致、当前数据 top-K
召回可审计后，才考虑以独立服务形式提供候选，不直接修改主 solver。

## 证据索引

- 官方代码：`vendor/JoinABLe/`
- 官方数据解析：`fusion_joint_schema_sample.json`
- 20 样本转换：`conversion_summary.json`
- STEP 实测：`step_brep_graph_probe_report.json`
- Schema gap：`schema_gap_report.json`
- 官方项目：https://github.com/AutodeskAILab/JoinABLe
- 官方数据：https://github.com/AutodeskAILab/Fusion360GalleryDataset
