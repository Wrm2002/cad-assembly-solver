# Common Joint–Interface Schema

Version: `1.0.0`

这个格式是 Fusion Joint Data、JoinABLe B-Rep graph 与 STEP/OCCT 之间的独立
交换层。它不表示“整套装配已经正确”，只表示一个有来源记录的零件对、关系标签
及其局部接口。

## 顶层

```json
{
  "schema_version": "1.0.0",
  "sample_id": "joint_set_00001:joint:0",
  "source_dataset": "fusion360_joinable_joint",
  "assembly_id": "source assembly or joint-set id",
  "part_a": {},
  "part_b": {},
  "relation": {},
  "interface_a": {},
  "interface_b": {},
  "contacts": [],
  "holes": [],
  "metadata": {},
  "failure_reasons": [],
  "unavailable_fields": []
}
```

所有输出都必须保留 `failure_reasons` 和 `unavailable_fields`。成功转换时也必须
存在，可以为空数组。

## Part

```json
{
  "part_id": "source body or occurrence-body id",
  "geometry_path": "body.obj | body.step | body.smt | body.x_t",
  "brep_graph_path": "body.json",
  "source_body_role": "body_one",
  "geometry_format": "obj | step | smt | parasolid",
  "source_occurrence_id": null,
  "source_geometry_sha256": null
}
```

`part_id` 必须标明是 body definition 还是 occurrence-body instance。重复零件的
不同 occurrence 不能在不知情的情况下合并。

## Relation

```json
{
  "has_joint": true,
  "label_semantics": "designer_selected_joint",
  "joint_type": "RigidJointType",
  "compatibility_label": "positive",
  "axis_origin": [0, 0, 0],
  "axis_direction": [0, 0, 1],
  "axis_origin_b": [0, 0, 0],
  "axis_direction_b": [0, 0, 1],
  "transform_a_to_world": [[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]],
  "transform_b_to_world": [[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]],
  "transform_a_to_b": [[1, 0, 0, 0], [0, 1, 0, 0],
                       [0, 0, 1, 0], [0, 0, 0, 1]],
  "offset": null,
  "angle": null,
  "is_flipped": null
}
```

- 正样本：`has_joint=true`, `compatibility_label=positive`。
- 负样本：`has_joint=false`, `compatibility_label=negative`，接口和轴允许为空。
- 未标注 pair：`compatibility_label=unknown`，不能自动当作负样本。
- `has_joint=false` 只表示定义好的数据集负标签，不证明在所有场景下机械不兼容。

## Interface

```json
{
  "entity_type": "BRepFace",
  "entity_ids": [
    {
      "source_entity_type": "BRepFace",
      "source_entity_index": 12,
      "joinable_node_index": 12,
      "occt_entity_type": "face",
      "occt_topology_index": null,
      "geometry_signature": null
    }
  ],
  "surface_or_curve_type": "CylinderSurfaceType",
  "semantic_role": "designer_selected_joint_origin",
  "axis_origin": [0, 0, 0],
  "axis_direction": [0, 0, 1],
  "brep_graph_path": "body.json",
  "source_node_features": {},
  "failure_reasons": [],
  "unavailable_fields": []
}
```

该层支持：

- interface localization：用 `entity_ids` 指向 face/edge；
- joint/mate type prediction：用 `relation.joint_type`；
- pair compatibility：用 `relation.compatibility_label`；
- 从 JoinABLe node 反查 Fusion face/edge；
- 从 STEP adapter 反查同一次 OCCT import 的 indexed shape；
- 用 `geometry_signature` 辅助检查重导入后的 topology id 漂移。

## Contacts 与 holes

```json
{
  "contacts": [
    {
      "part_a_entity_ids": [{"source_entity_index": 3}],
      "part_b_entity_ids": [{"source_entity_index": 8}],
      "contact_area": null,
      "label_quality": "source_recorded"
    }
  ],
  "holes": [
    {
      "part_id": "body_a",
      "face_ids": [5],
      "axis_origin": [0, 0, 0],
      "axis_direction": [0, 0, 1]
    }
  ]
}
```

JoinABLe Joint Data 中 contact 或 holes 为空不算转换失败，但必须进入
`unavailable_fields`，不能凭几何猜测后伪装成原始标签。

## Metadata 与可审计性

```json
{
  "unit": "cm",
  "source_file": "joint_set_00001.json",
  "source_sha256": "...",
  "source_joint_index": 0,
  "conversion_status": "success | partial | failed",
  "failure_reason": null,
  "failure_reasons": [],
  "unavailable_fields": []
}
```

## STEP/OCCT 预留约定

STEP 文件本身通常没有 designer-selected joint、contact、mate type 或装配层级。
因此 STEP adapter 只能生成 part B-Rep graph，并预留：

- `occt_topology_index`：同一次导入中的 1-based indexed shape id；
- `source_geometry_sha256`：固定输入版本；
- `geometry_signature`：类型、测度、质心等组成的辅助签名；
- `id_stability_scope=same_file_same_importer_build`；
- `reverse_lookup_scope=in_process_indexed_map`。

任何跨 CAD 导出、healing、布尔运算后的稳定 face id 都不在本 schema 的保证范围。
