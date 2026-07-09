# JoinABLe Phase 1：核心候选提供器

## 职责

JoinABLe 当前接入候选召回入口，而不是最终接受门：

1. 对 mixed-pool 中的零件对生成 B-Rep 图；
2. 使用官方预训练 checkpoint 对每个零件对的接口实体排序；
3. 每个零件保留有界的 learned candidate 邻居；
4. 将解析几何预筛漏掉的邻居送入详细几何匹配；
5. 只有详细几何产生物理证据后，候选才能继续进入 D4/D5。

官方 checkpoint 没有训练 `no joint` 类，因此 JoinABLe 排名不能直接证明两个零件应当连接。

## 命令

```powershell
cd sw
python prepare_joinable_pool_graphs.py data/functional_mixed_pools_v1

python ../cad_assembly_agent/tools/joinable_interface_predictor/batch_mixed_pool_inference.py `
  --mixed-pool-root data/functional_mixed_pools_v1 `
  --output data/functional_mixed_pools_v1/joinable_pair_rankings.json `
  --device cpu `
  --top-k 20

python pool_index.py data/functional_mixed_pools_v1/functional_pool_001/parts `
  --output-dir data/functional_mixed_pools_v1/functional_pool_001/index_joinable `
  --joinable-report data/functional_mixed_pools_v1/joinable_pair_rankings.json
```

批量正式推理时可把 `--device cpu` 改为 `--device cuda`。编码、契约测试和小规模冒烟测试不需要租 GPU。

## 安全边界

- learned frontier 与 analytic frontier 做并集，不会挤掉解析几何候选；
- JoinABLe 不增加 `independent_evidence_count`；
- JoinABLe softmax 不写入 `geometry_score`；
- JoinABLe 不能自动接受；
- 所有候选仍需通过几何分数、多证据、pose、精确碰撞和组一致性门。

## 2026-07-07 小基准结果

- 42/42 个零件图提取成功；
- 273/273 个零件对完成 CPU 推理，模型前向总计约 5.04 秒；
- JoinABLe 救回 6 个解析预筛遗漏对，但这 6 个都不属于真实功能组；
- 详细匹配增加 15 个 generated candidates，剪枝后只增加 3 个；
- retained true-mate pair 增量为 0，因为该小基准的 analytic baseline 原本已是 100% true-pair recall。

这说明接入链路有效，但当前小基准不能证明召回收益。下一步应在存在明确 analytic prescreen miss 的 harder holdout 上做相同消融。

本机 CUDA 在批量完成后的同步阶段出现 `illegal instruction`，当前 273 对 CPU 推理足够快，因此暂不租 GPU。等 harder holdout 和一键消融命令冻结后，再修复本机 CUDA 或租用显卡做大批量推理。
