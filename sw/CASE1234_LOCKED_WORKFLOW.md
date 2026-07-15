# Case1–4 已确认工作流锁

唯一回归入口：

```powershell
.\.conda\python.exe sw\run_case1234_locked_visual_workflow.py --verify-lock
```

需要实际重跑时：

```powershell
.\.conda\python.exe sw\run_case1234_locked_visual_workflow.py --run --cases 1,2,3,4 --visual-mode live
```

锁定流程为：

```text
known_group_assembly 几何候选生成
→ 受保护且来源多样的 Pose 候选前沿
→ Qwen 多视图角色/区域/方向/动作判断
→ 不泄漏几何分数的视觉语义 Top-K
→ 视觉 Top-K 后的 OCCT 精确门控
→ Accepted / Review / Rejected / Unresolved
→ 与人工确认 Case1–4 Pose 基准做回归比较
```

硬性规则：

- 不允许 geometry-only fallback；
- 视觉 API `abstain` 或缓存缺失不能算成功；
- 视觉模型不能看到几何分数；
- 视觉语义不能自动 Accepted；
- 运行结果若偏离人工确认 Pose，标记 `regression_failed`；
- 金标准只用于 Case1–4 回归比较，不参与新 Case 的候选生成或评分。

配置与基准指纹：

- `sw/configs/case1234_visual_pose_workflow.v1.json`

最新锁审计：

- `sw/generalization_work/case1234_visual_pose_locked_v1/workflow_lock_audit.json`

历史目录 `case1234_new_route_v1` 中 Case2/3 的几何旧结果是错误实验，不能作为当前工作流输出；这里只保留其中经人工确认的 Case1/4 Pose 作为回归基准。Case2/3 基准来自 `case23_visual_route_v1` 的视觉重排结果。
