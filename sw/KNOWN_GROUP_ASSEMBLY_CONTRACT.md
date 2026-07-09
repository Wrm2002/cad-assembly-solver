# 已知同组零件装配关系识别契约

## 任务边界

输入为 1～5 个 STEP 零件。调用方保证这些零件属于同一个装配体，但不提供直接接触拓扑、接口、关系标签或位姿。

系统只回答：

1. 哪些零件之间存在直接装配连接；
2. 每条连接使用哪些几何接口；
3. 接口属于哪类约束；
4. 零件的相对位姿和全局位姿；
5. 候选位姿是否满足约束闭合且通过 OCCT 实体碰撞检查。

系统不再执行 mixed-pool 分组、来源判断、功能语义裁判或强制完整分区。

## 第一版关系标签

- `coaxial`：两个圆柱接口轴线重合；
- `clearance`：凸圆柱插入半径更大的凹圆柱；
- `planar_mate`：两个反向平面贴合；
- `planar_align`：两个同向平面共面；
- `pocket_mate`：插入件与槽或口袋配合。

一条直接连接可以同时包含多个约束。例如法兰连接通常由 `coaxial` 和 `planar_mate` 共同支持。`primary_relation_type` 是主标签，`supporting_relation_types` 保留所有在最终位姿中实际闭合的标签。

## 主输出

`known_group_output/assembly_relations.json` 使用 `schemas/known_group_assembly_result.schema.json`，关键字段为：

- `input_assumption=all_parts_belong_to_one_assembly`；
- `direct_connections`：选中的直接零件连接；
- `assembly_relations`：带接口索引的五类几何约束；
- `relative_transform_a_to_b`：A局部坐标到B局部坐标的4×4刚体变换；
- `components`：所有零件在统一装配坐标系中的位姿；
- `assembly_connected`：关系图是否覆盖全部输入零件；
- `pose_status=valid|failed|uncertain`；
- `collision_validation`：逐候选OCCT布尔交碰撞审计。

## 有效位姿判定

`pose_status=valid` 同时要求：

1. 直接装配图连接全部输入零件；
2. 每条骨架连接至少有一个约束闭合；
3. 若同一连接同时存在轴向与平面贴合证据，两类证据必须同时闭合；
4. OCCT Boolean Common 未发现超过容差的实体穿透；
5. OCCT检查完整完成。

仅同轴但轴向分离、仅无碰撞但没有接口闭合、或碰撞检查不可用，均不得标记为有效。

## JoinABLe边界

JoinABLe作为学习型接口定位证据接入：

- 保留Top-K B-Rep面/边候选；
- 按轴向/平面接口族对解析几何候选提供有界加分；
- 不凭JoinABLe单独生成物理关系；
- 未提供缓存报告时，显式记录`not_provided`，解析几何回退仍可运行。

## 命令

```powershell
python known_group_assembly.py <case_dir> [--joinable-report report.json] [--beam-width 20]
```

除主输出外还生成候选审计、位姿审计、装配manifest和装配STEP。
