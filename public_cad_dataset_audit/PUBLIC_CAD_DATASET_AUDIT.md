# 公开 CAD 装配数据集审计报告

审计日期：2026-07-03

## 最终判断

1. **主数据源：Autodesk Fusion 360 Gallery Assembly Dataset。**
   它显式提供 occurrence、body、joint、contact 和 B-Rep/OBJ 几何，能够构造
   occurrence-body 级 assembly graph、正负零件对和接口监督。原始 contact
   不能视为完备真值，joint 应作为更高置信关系，contact 必须保留来源与质量标签。
2. **辅助数据源：AutoMate。**
   它适合大规模 mate 类型预测和正零件对任务，但 parquet 元数据没有直接的
   B-Rep face/edge id；若要生成同装配负样本和区分重复零件 occurrence，还需
   下载并解析 `assemblies.zip`。
3. **候选增强源：Linkify。**
   代码和分块数据地址已经公开，并补算了 contact face、面积、体积及局部 PLY。
   但完整下载与解压峰值约 304 GB，本机 D 盘审计时仅剩约 94.55 GB，未做本地
   全量验证；仓库也未发现明确 LICENSE 文件，因此当前不作为立即投入的主源。

## Fusion 360 实测

从官方首个发布分卷中抽取 10 个带完整 JSON/SMT/STEP/OBJ 的装配，逐条解析并
转换，没有读取用户 SolidWorks 文件。

| 指标 | 实测值 |
|---|---:|
| 装配 | 10 |
| 可见 occurrence-body 零件实例 | 184 |
| joint | 13 |
| contact | 410 |
| joint/contact 原始关系 | 423 |
| 成功映射到 part pair | 423 / 423 |
| 可按拓扑索引寻址的接口实体 | 846 |
| 在实际 indexed OBJ 中验证的接口实体 | 846 / 846 |
| 成功转换统一 graph | 10 / 10 |

映射成立的边界是：Fusion 原生 SMT 与 indexed OBJ 保留了 JSON 所用的
body/face/edge 索引语义；中性 STEP 重新导入后不保证 face 编号仍相同。

官方公布的全量规模为 8,251 个装配、154,468 个零件；本报告没有把 10 个样本
的经验比例外推为全量质量统计。Fusion 数据采用 Autodesk 自定义的非商业研究
许可，不能重新分发完整数据集。

## AutoMate 全量元数据实测

本次读取了完整的 `assemblies.parquet`、`parts.parquet` 和 `mates.parquet`，
没有下载大体积 STEP/Parasolid 几何压缩包。

| 指标 | 实测值 |
|---|---:|
| assembly rows | 255,211 |
| part rows | 451,967 |
| mate rows | 1,292,016 |
| 精确去重的非同 ID 正零件对 | 566,064 |
| 声明可用 STEP 的零件 | 447,442 |
| 可读 Parasolid 标记的零件 | 451,966 |
| mate 类型数 | 8 |

Mate 类型为 BALL、CYLINDRICAL、FASTENED、PARALLEL、PIN_SLOT、PLANAR、
REVOLUTE、SLIDER。元数据足以构造 mate 类型和正零件对监督；不能直接构造
face/edge 接口监督。同一 part id 的 mate 有 168,508 行，这可能代表同一零件
定义的两个 occurrence，不能简单删除或当作自环错误。AutoMate 发布许可为 CC0。

## 统一 assembly graph 与零件对

统一格式见 `assembly_graph_schema.md`。节点采用 occurrence-body 实例，避免把
一个 body 定义在多个 occurrence 中的实例错误合并。正边聚合同一 pair 上的
joint/contact；负边定义为“同一装配内没有观测到 joint/contact 的 pair”。

10 个 graph 共生成：

| 样本 | 数量 |
|---|---:|
| positive part pair | 171 |
| negative part pair | 1,840 |
| 合计 | 2,011 |

负边是 closed-world 训练标签，不代表两个零件在所有工程场景下绝对不能装配。
数据划分必须按 `assembly_id` 完成，不能随机拆散 pair，否则会造成装配泄漏。

## 失败与不可用字段

- Fusion 原始 contact 的完备性和正确性没有保证。
- Fusion JSON 的拓扑编号不能直接假设等同于 STEP 重新导入后的 face 编号。
- AutoMate parquet 没有直接 mate face/edge id。
- AutoMate 几何归档未下载，因此本次没有本地打开其 STEP/Parasolid。
- 缺少 `assemblies.zip` 时，AutoMate parquet 不能可靠生成同装配负边或解析
  重复 part definition 的 occurrence 身份。
- Linkify 未做本地全量记录质量验证，且未发现明确的仓库/数据许可证。

以上内容同时以机器可读的 `failure_reasons` 和 `unavailable_fields` 写入各 JSON。

## 产物

- `outputs/fusion360_audit_report.json`
- `outputs/automate_audit_report.json`
- `outputs/linkify_audit_report.json`
- `outputs/fusion360_assembly_graphs/*.json`
- `outputs/fusion360_assembly_graphs/conversion_manifest.json`
- `outputs/pair_dataset_manifest.json`
- `outputs/public_dataset_decision_report.json`

原始公开数据放在 `D:\Model_match_public_data`，第三方源码快照放在本目录
`vendor/`；两者均未接入或修改旧的 `sw/` 流程。

## 验证状态

- 8 个本项目 Python 文件均通过语法编译。
- 2 个基于 Autodesk 官方 `belt_clamp` 示例的回归测试通过。
- 10 个目标 Fusion 装配全部转换成功，0 个被拒绝。
- 17 个输出 JSON 均可重新解析；本次新增输出均含
  `failure_reasons` 和 `unavailable_fields`。
- 未训练模型，未生成合成 CAD，未处理用户 SolidWorks 文件。

## 最小下一步

先把 Fusion 的 joint 边与 contact 边分层，使用 joint 建立高置信主任务，再对
contact 做抽样人工审计或引入经过许可确认的 Linkify 子集。不要在验证 contact
质量之前直接训练，也不要把同装配 non-edge 解释为绝对机械负样本。

## 官方来源

- Fusion 360 Gallery Dataset:
  https://github.com/AutodeskAILab/Fusion360GalleryDataset
- AutoMate 项目：
  https://degravity.github.io/automate/
- AutoMate 数据：
  https://zenodo.org/records/7776208
- Linkify：
  https://github.com/ajignasu/linkify
