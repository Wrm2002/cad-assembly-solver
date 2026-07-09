# JoinABLe harder-holdout 最终识别审计

## 结论

JoinABLe 已被证实是有价值的核心候选提供器，但不能替代 analytic ranker，也不能成为自动接受裁判。正确结构是 analytic 与 JoinABLe 取并集，再交给保守几何、pose、碰撞和 group consistency 门控。

## 1. harder holdout 是否补回 analytic 漏召回

采用三层证据：

1. 27-joint real STEP transfer：analytic Top-10 为 20/27，JoinABLe 为 17/27，并集为 22/27。JoinABLe 独有补回 2 个，但 analytic 独有命中 5 个，说明两者互补而非可替代。
2. 2600-domain source-design-disjoint test：221 个 exact-evaluable pair 中，analytic Top-10 为 85/221（38.46%），JoinABLe 为 124/221（56.11%），并集为 145/221（65.61%）。JoinABLe 独有补回 60 个，analytic 独有命中 21 个；配对 exact-binomial p=1.69e-5。
3. functional CAD mixed-pool holdout：当前 analytic true-pair recall 已为 100%，因此不能用它证明补回；这里只用于安全审计。

限制：27-joint sample 与官方预训练数据的独立性没有完全证明；2600-domain test 做到了 source-design-disjoint，但历史域适配实验曾用它做 evaluation，因此也不是全新盲测。上述结果证明当前数据上的互补性，不等于未知外部域的最终泛化结论。

## 2. 新增 false candidates 是否被挡住

在 locked functional CAD holdout 上：

- 新增 generated edges：5；
- pruning 删除：4；
- pruning 后保留：1；
- 保留项是 bearing-like ring 与 oversize clocking pin 之间的弱局部 planar mate，geometry score=0.622842；
- 该 edge 引发 7 个新 group ID 和 2 个既有 group 决策迁移；
- JoinABLe arm 中受影响 group 的最终去向：accepted=0、review=6、rejected=3；
- false auto accepts=0；
- immediate review frontier 净增加 2。

结论是安全门控有效，但不是零成本：错误候选没有进入 accepted，却增加了人工复核工作量。

## 3. 一键消融

新增 `run_joinable_holdout_ablation.py`，从图生成、CPU/GPU ranking、两臂索引、D4/D5 到最终审计一次完成。脚本为每次运行创建隔离工作区并冻结输入 SHA256，不修改 locked truth。

## 4. 是否租卡

当前不租：64 pair 的模型前向 CPU 1.17 秒，CUDA 1.76 秒；两臂完整 pipeline 各约 90.6 秒，瓶颈是 pose/碰撞。当前推理逐 pair 执行，GPU kernel launch 和数据搬运开销占主导。先做真正的 pair batching，再依据端到端 profiling 决定租卡。

## 5. 是否启动训练或域适配

当前不启动。历史 300-sample 四种子域适配 mean test Top-10 为 -6.62 个百分点，0/4 种子不退化；扩展 2600 版本 mean test Top-10 为 -0.34 个百分点，仅 2/4 不退化。promotion gate 未通过。

下一次讨论训练前，必须先获得全新、工程师签核、未参与调参的真实装配 holdout，并证明预训练并集在该集合上仍无法满足召回/工作量目标。

## 复查结论

- 没有训练；
- 没有让 JoinABLe softmax 进入自动接受；
- 没有用 source_id 作为生产真值；
- 没有以候选数量代替召回；
- accepted precision 仍不可估计，因为两臂 accepted 均为 0；
- functional holdout 的工程师签核仍是外部未完成项，不能被本次算法审计替代。
