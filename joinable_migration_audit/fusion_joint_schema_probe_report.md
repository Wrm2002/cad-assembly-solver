# Fusion Joint Schema Probe Report

- 状态：`success`
- joint-set 文件：20
- 解析 joint samples：20 / 20

## 字段实测

| 字段 | 出现样本数 |
|---|---:|
| `assembly_id` | 20 |
| `body_one` | 20 |
| `body_two` | 20 |
| `brep_entity_a` | 20 |
| `brep_entity_b` | 20 |
| `contact_face_pair` | 18 |
| `contacts` | 20 |
| `geometry_or_origin_one` | 20 |
| `geometry_or_origin_two` | 20 |
| `hole_or_cylindrical_feature` | 18 |
| `holes` | 18 |
| `inferred_assembly_id` | 20 |
| `joint_axis_line` | 20 |
| `joint_type` | 20 |
| `joints` | 20 |
| `transform` | 20 |

## Joint type

| 类型 | 数量 |
|---|---:|
| `CylindricalJointType` | 2 |
| `PlanarJointType` | 2 |
| `RevoluteJointType` | 11 |
| `RigidJointType` | 4 |
| `SliderJointType` | 1 |

## 失败与不可用

- 不可用：`contacts`
- 不可用：`holes`
- 不可用：`source_explicit_assembly_id`

复现命令：

```powershell
python fusion_joint_schema_probe.py --data_root D:\path\to\j1.0.0 --limit 20
```
