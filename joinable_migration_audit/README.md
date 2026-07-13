# JoinABLe Migration Audit

该目录与主 solver 隔离，用于验证 JoinABLe 能否作为 B-Rep interface/joint
candidate provider。没有训练模型、调用 DeepSeek、生成 toy CAD 或修改主 solver。

## 已验证数据

- 官方 JoinABLe 仓库 commit：
  `5a3b9933e91269d19b6cc733b244a879eb4bfc30`
- Fusion Joint Data j1.0.0：
  `D:\Model_match_public_data\fusion360_joint\j1.0.0.7z`
- 20 样本子集：
  `D:\Model_match_public_data\fusion360_joint\sample20\j1.0.0\joint`
- 当前项目 3 个 STEP：`sw/1` 与 `sw/2`，只读。

## 复现

```powershell
$py = 'C:\Users\11049\miniforge3\envs\cad_asm\python.exe'
$data = 'D:\Model_match_public_data\fusion360_joint\sample20\j1.0.0\joint'

& $py fusion_joint_schema_probe.py `
  --data_root $data --limit 20

& $py convert_fusion_joint_to_common_schema.py `
  --data_root $data --out_dir converted_joint_samples --limit 20

& $py step_to_brep_graph_probe.py `
  '..\sw\1\flange_part_a.step' `
  '..\sw\1\flange_part_b.step' `
  '..\sw\2\shaft_with_keyway.step' `
  --limit 3

& $py schema_gap_report.py
```

## 主要结果

- 官方 joint sample 转换：20/20；
- 当前 STEP graph：3/3；
- 主结论：`部分适合迁移，但需要先补齐数据结构`。

详见 `JOINABLE_MIGRATION_REPORT.md` 和 `MIGRATION_PLAN.md`。
