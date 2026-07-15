# 图片任务实施报告

## 当前状态

六项任务中的代码、数据和自动验证部分已经完成。唯一未完成项是外部机械工程师对新 CAD holdout 的 12 项签字确认；系统不会伪造该签字，因此整体状态为：

`pending_external_mechanical_engineer_signoff`

## 1. 九个真组 evaluation-only pose audit

已新增并运行 `sw/audit_true_group_pose.py`。真值只用于选择审计对象和计算指标，不向 pose solver 提供真值 placement 或语义。

- 真组：9
- group proposal recall：100%
- pose valid：6
- pose failed：3
- `true_group_pose_recall`：66.67%

三个 `cover_base` 全部失败；`shaft_hub_key` 和 `bearing_housing` 全部成功。失败主要来自提案内部选择了错误约束边，而不是候选组缺失。

报告：

- `sw/data/functional_results/true_group_pose_audit.json`
- `sw/data/functional_results/true_group_pose_audit.csv`
- `sw/data/functional_results/true_group_pose_audit.md`

## 2. Bounded review 结构化重排

review 排名现已加入：

- `group_completeness_score`
- `central_part_coverage`
- `interface_diversity_score`
- 候选间 part-set novelty

这些字段只影响人工 review 顺序，`affects_auto_accept=false`。自动接受阈值没有放宽；相反，无语义校准时二元组现在必须进入 review。

结果：

- 原即时 frontier 真组命中：0/9
- 新即时 frontier 真组命中：2/9
- `review_frontier_recall`：22.22%
- `review_frontier_precision`：2.78%

结果仍然较弱，但改进已被单独计量，没有被包装成自动分组成功。

## 3. 全部 27 个 hard negatives 精确经过 D4/D5

已新增 `sw/audit_forced_hard_negatives.py`。

- 精确 hard-negative candidates：27/27
- production candidate edge 可用：27/27
- D4 未拒绝并实际进入 D5：9
- D4 已拒绝、因此 D5 不适用：18

首次审计发现 5 个错误自动接受：

- 3 个 geometric hard negatives
- 2 个 semantic hard negatives

增加二元组保守门后重新运行：

- accepted false positives：0
- review：9
- rejected：18

前后审计分别保存在：

- `sw/data/functional_results_hard_negative_baseline/`
- `sw/data/functional_results/forced_hard_negative_audit.json`

## 4. 新增组级指标

已新增 `sw/functional_group_metrics.py`，输出：

- `group_proposal_recall`
- `group_recall@24/50/100/500/2000`
- `review_frontier_recall`
- `review_frontier_precision`
- `true_group_pose_recall`

当前主要结果：

| 指标 | 结果 |
|---|---:|
| group proposal recall | 100% |
| group_recall@24 | 22.22% |
| group_recall@500 | 22.22% |
| group_recall@2000 | 88.89% |
| review frontier recall | 22.22% |
| true-group pose recall | 66.67% |

报告位于 `sw/data/functional_results/functional_group_metrics.*`。

## 5. 独立 CAD holdout 与人工确认包

新增了禁止用于调参的 topology-varied holdout：

1. 圆形止口仪表盖；
2. 带法兰、平键和可选轴向限位件的轴毂组件；
3. 带保持环的法兰 cartridge bearing housing。

包含：

- 3 个正装配；
- 9 个负样本；
- 24 个有效 STEP 文件；
- assembled / exploded / negative-pair PNG；
- SHA-256 holdout lock；
- 12 行机械工程审核表；
- 自动签字校验脚本。

位置：

- `sw/data/functional_cad_holdout_v1/holdout_contact_sheet.png`
- `sw/data/functional_cad_holdout_v1/ENGINEERING_REVIEW_INSTRUCTIONS.md`
- `sw/data/functional_cad_holdout_v1/ENGINEERING_REVIEW_FORM.csv`

待完成：由具备机械工程背景的真实审核人填写并签署全部 12 行，然后运行：

```powershell
conda run --no-capture-output -n cad_asm python sw/validate_holdout_engineering_review.py
```

只有 `engineering_signoff.json` 中 `gate_passed=true` 才算第 5 项完整完成。

### 锁定 holdout 首次基线

holdout 在生成 SHA-256 lock 后，以当前规则完成了一次“不调参”运行：

| 暂定指标 | 结果 |
|---|---:|
| candidate interface recall | 100% |
| post-pruning interface recall | 100% |
| group proposal recall | 100% |
| review frontier recall | 25% |
| true-group pose recall | 50% |
| hard-negative auto accepts | 0 |

这些结果只用于测量泛化，后续不得使用同一 holdout 调整规则。由于机械工程签字尚未完成，真值和指标仍标记为 provisional。

基线报告：

- `sw/data/functional_cad_holdout_results_v1/locked_holdout_baseline.json`
- `sw/data/functional_cad_holdout_results_v1/locked_holdout_baseline.md`

## 6. DeepSeek 状态

- provider calls：0
- semantic reranking：关闭
- semantic application mode：explanation-only
- holdout DeepSeek：关闭

## 验证

- 单元测试：56/56 passed
- holdout STEP readback/shape validity：24/24 passed
- 27 个 exact hard negatives：false positive 5 → 0
- 自动实施检查：全部通过
- 外部机械工程签字检查：待完成

机器可验证状态：

- `IMAGE_TASK_DELIVERY_STATUS.json`
