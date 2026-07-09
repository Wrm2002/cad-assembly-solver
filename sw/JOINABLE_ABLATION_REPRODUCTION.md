# JoinABLe harder-holdout 一键消融

## 运行命令

在 `sw/` 下运行：

```powershell
C:\Users\11049\miniforge3\envs\cad_asm\python.exe `
  run_joinable_holdout_ablation.py `
  --output-root data\joinable_holdout_ablation_repro_v2 `
  --device cuda
```

命令一次完成：

1. 校验并准备 functional CAD holdout 的 JoinABLe 图；
2. 分别执行 CUDA 和 CPU pair ranking；
3. 在隔离工作区重建 analytic-only 与 analytic+JoinABLe 两臂索引；
4. 两臂分别运行 candidate recall、D4 tiering、D5 pose/碰撞验证；
5. 在 2600-domain source-design-disjoint test split 上评估 exact entity ranking；
6. 生成真实接口补回、false-candidate 去向和专家结论。

输出目录中的 `run_manifest.json` 固定输入 STEP、真值、配置、checkpoint 和评估报告 SHA256。脚本拒绝写入非空输出目录，避免历史结果混入。

## 主要输出

- `audit/strict_audit.json`：机器可读总审计；
- `audit/EXPERT_REVIEW.md`：识别专家复查；
- `audit/real_joint_rescue_cases.csv`：27-joint 逐项结果；
- `audit/false_candidate_routes.csv`：新增 edge/group 的全链路去向；
- `results/*/conservative_metrics.json`：两臂保守指标；
- `cache/domain_holdout_union_test_report.json`：221 个 exact-evaluable pair 的大 holdout 结果。

## 边界

- 不训练模型；
- JoinABLe 不写入 geometry score；
- JoinABLe 不增加独立物理证据；
- JoinABLe 不具有自动接受权限；
- functional CAD holdout 尚未通过机械工程师签核，相关功能真值必须标记为 provisional；
- 2600-domain test source-design-disjoint，但曾用于历史域适配实验的评估，不属于全新盲测。
