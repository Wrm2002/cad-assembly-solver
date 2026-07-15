# JoinABLe 官方模型复现与本机 CUDA 评估

日期：2026-07-04

## 结论

第一步已经完成：官方 JoinABLe 模型、官方 checkpoint 和官方预处理测试集已在隔离环境中成功复现。

第二步也完成了短时 CUDA 训练测速。RTX 4070 Laptop 8 GB 对 JoinABLe 默认
`batch_size=2` 的模型计算并不慢，当前没有必要仅为了算力马上租 AutoDL。真正的本机风险是
32 GB 内存下读取旧版 16 GB 单体训练 pickle，以及 Windows/Python 的长任务稳定性。

如果只是继续接口预测验证、少量微调和适配 SolidWorks，先用本机。如果要立即完整训练
100 epochs、重复 5 个随机种子或做多组超参数实验，建议租用至少 64 GB 系统内存的实例；
GPU 12 GB 已可用，常见的 24 GB RTX 3090/4090 会更从容。

## 第一步：官方 checkpoint 复现

- 官方代码：`joinable_migration_audit/vendor/JoinABLe`
- checkpoint：`pretrained/paper/last_run_0.ckpt`
- checkpoint 状态：epoch 100，global step 159000
- 权重：26/26 个张量严格匹配，未缺失、未放宽 `strict`
- 官方模型配置：hidden=384，GATv2，输入特征为
  `entity_types,length,face_reversed,edge_reversed`
- 官方 checkpoint 和 pickle 均未修改
- 旧 PyG 1.7 `Data` 仅在内存中转换到 PyG 2.6 数据结构

隔离环境位于 `D:\Model_match_envs\joinable_gpu`：

- Python 3.10.20
- PyTorch 2.5.1 + CUDA 12.4
- PyG 2.6.1
- RTX 4070 Laptop，8 GB VRAM，compute capability 8.9

不能直接复用官方 CUDA 10.2 环境，因为 CUDA 10.2 早于 RTX 4070/Ada。当前适配保留了
官方网络结构和权重，只替换运行时与旧 PyG 对象存储结构。

## 完整官方测试集结果

官方 `test.pickle` 含 1,926 对；按官方 `max_node_count=950` 过滤 69 对，实际评价 1,857 对。

| 指标 | 结果 |
|---|---:|
| Top-1 accuracy | 79.32% |
| Top-5 recall | 87.56% |
| Top-10 recall | 90.79% |
| 数据转换失败 | 0 |
| 缺少正标签样本 | 0 |

Top-1 与官方 README 所述约 79.53% 只差 0.21 个百分点，说明迁移后的模型行为与发布结果
基本对齐。差异可能来自五个发布随机种子的口径或运行时浮点实现。

另取 10 对同时运行 CPU/CUDA：

- 10/10 的 Top-10 排序完全一致；
- 最大绝对 logit 差为 0.001655；
- 该差异属于不同设备浮点内核的正常范围。

注意：这里评价的是“已知两个零件时的 B-Rep joint entity 排名”，不是 mixed-pool 自动分组，
也不能独立证明最终装配 pose 无碰撞或功能语义正确。

## 第二步：RTX 4070 Laptop CUDA 训练测速

为避免把约 16 GB 的旧 `train.pickle` 整体压入 32 GB 内存，测速使用官方测试缓存中沿全体
顺序均匀抽取的 32 对真实图张量，并使用训练口径标签执行真实 forward、loss、backward 和
Adam step。

配置：

- FP32
- batch size 2
- 3 steps warmup + 10 steps measured
- 官方 checkpoint 架构与 MLE/symmetric loss

结果：

| 指标 | 结果 |
|---|---:|
| 中位 step 时间 | 19.75 ms |
| 吞吐 | 99.83 samples/s |
| 峰值 allocated VRAM | 285.58 MiB |
| 峰值 reserved VRAM | 526 MiB |
| 估算单 epoch | 2.21 min |
| 估算 100 epochs 纯训练计算 | 3.68 h |

3.68 小时不含 16 GB pickle 读取、完整验证、checkpoint 保存和异常重启。保守预计本机单次
100-epoch 端到端运行约 5–7 小时；五随机种子约 25–35 小时。

AMP FP16 没有作为参考结果：官方代码会显式创建 FP32 临时张量，直接 autocast 会产生类型
冲突。要启用 AMP 需修改模型实现，不能宣称是未经改动的官方复现。

## 稳定性风险

本轮发现两类与算力不同的问题：

1. 长序列、变尺寸 CUDA 推理曾出现一次 `illegal instruction`，对应样本单独运行正常，
   更像运行时/驱动的间歇问题。
2. 反复读取 1.9 GB 旧 pickle 时出现 Python `0xc0000005` 访问冲突和反序列化内部状态异常；
   同一时段 Windows 事件日志中微信也有多次 `0xc0000005`。

因此不能把本机描述为“稳定完成长训练已经验证”。GPU 温度、功耗和显存当时都正常，
问题不是过热，也不是 8 GB 显存耗尽。旧版单体 pickle 和本机应用/运行时稳定性是主要风险。

## 是否租 AutoDL

当前建议：

- **现在不必仅为了 GPU 速度租卡。** 4070 Laptop 对默认 JoinABLe batch=2 足够快。
- **若今天就要完整重训或跑五随机种子，建议租。** 重点选择系统内存至少 64 GB，
  再选择 12–24 GB 显存 GPU；RTX 3090/4090 24 GB 是稳妥配置。
- **更合理的本地下一步**是先把官方 16 GB 单体 pickle 只读迁移成现代、分片、可流式加载
  的数据，不改变标签和图内容。完成后再做一个真实完整 epoch 基准，届时才能给出更精确的
  本机/AutoDL 成本比较。

## 新增文件

- `joinable_compat.py`：旧 checkpoint/PyG 数据只读兼容层
- `run_official_inference.py`：官方测试集 Top-1/5/10 复现
- `benchmark_cuda_training.py`：受控 CUDA forward/backward 测速
- `official_inference_report.json`：CPU/CUDA 一致性报告
- `official_inference_full_report.json`：完整测试集报告
- `cuda_training_benchmark.json`：本机训练吞吐报告
- `README.md`：复现实验命令和边界

## 下一步边界

JoinABLe 输出应作为 interface 候选和几何证据，不能直接成为最终 accepted group。它需要
接入现有保守门控：pose valid、无碰撞、至少两个独立证据、group consistency、无全局冲突；
否则进入 review，而不是为了覆盖率强行接受。
