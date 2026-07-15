# JoinABLe STEP 域迁移与 AutoDL 接力报告

日期：2026-07-04

## 结论

第一阶段已经完成，当前确实到达了需要云 GPU 的节点，但不能把本地微调
checkpoint 宣称为有效成果。

- 官方 JoinABLe checkpoint 已在现代 PyTorch/CUDA 环境严格加载，并复现官方
  预处理测试集：Top-1 79.32%、Top-5 87.56%、Top-10 90.79%。
- 已建立 STEP -> B-Rep 图 -> JoinABLe interface ranking 的完整接口，模型只做
  候选提供者，保持 shadow/review-only，不能直接自动接受装配。
- 同一 Fusion 零件导出 STEP 后，预训练模型在严格可映射接口上的 Top-10
  equivalent recall 为 62.96%；源 Fusion 图为 81.48%。主要损失来自 STEP
  重建后的 edge 拓扑变化。
- 规则候选 Top-10 为 74.07%；规则 Top-10 与模型 Top-10 的并集在不超过
  20 个候选预算下达到 81.48%。因此当前最可靠方案是混合候选召回，不是让
  神经网络单独裁决。
- 300 对紧凑 STEP 子集已完成无泄漏 train/validation/test 划分。570/570 个
  STEP 图提取成功，训练、验证和测试 source design 无交叉。
- 四个随机种子的低学习率微调全部把 validation exact Top-10 从 48.39%
  提高到 58.06%，增益均为 9.68 个百分点。
- 四个随机种子的 independent test exact Top-10 全部退化，平均从 61.76%
  降到 57.35%，平均下降 4.41 个百分点。

因此：域适配存在稳定的验证集信号，但小样本泛化失败。当前 adapted
checkpoint 不得进入生产候选链；下一轮应扩大紧凑 STEP 配对数据，在稳定的
Linux 云 GPU 上进行多随机种子实验。

## 四随机种子结果

| Seed | Best epoch | Validation Top-10 | Validation delta | Test Top-10 | Test delta |
|---:|---:|---:|---:|---:|---:|
| 42 | 10 | 58.06% | +9.68 pp | 58.82% | -2.94 pp |
| 7 | 12 | 58.06% | +9.68 pp | 55.88% | -5.88 pp |
| 17 | 5 | 58.06% | +9.68 pp | 58.82% | -2.94 pp |
| 73 | 11 | 58.06% | +9.68 pp | 55.88% | -5.88 pp |

模型选择只使用 validation exact Top-10/Top-5/Top-1。test 只做一次独立
评价，没有按 test 挑 checkpoint。

机器可读汇总：
`joinable_gpu_reproduction/domain_finetune_multiseed_summary.json`

## 本机崩溃诊断

这里有三类不同问题，不能混为一个原因。

1. PowerShell 的 `An empty pipe element is not allowed` 是命令语法错误：
   `foreach { ... } | ...` 在当前 Windows PowerShell 解析路径下不合法。它与
   GPU 或硬件加速无关，改为先接收结果再管道输出即可。
2. 部分 `Codex process is not available` / UI 灰屏属于 Codex/VS Code
   前端或进程通信中断。VS Code `argv.json` 已关闭硬件加速，因此不能把每次
   UI 故障都归因于 GPU。
3. 本次持续 CUDA 微调发生了真实的 NVIDIA 驱动故障：PyTorch 报
   `CUDA error: an illegal instruction was encountered`，同一秒 Windows
   System 日志记录多个 `nvlddmkm` Event ID 13 和 153。故障发生时显存约
   2 GB、温度约 74–75°C，不是显存耗尽。驱动重置可能连带影响桌面图形应用，
   因而是 UI 不稳定的一个可信诱因。

为避免再次触发，后续本机只运行 CPU 数据审计和短测试，不再进行持续 CUDA
训练。

## 为什么现在需要 AutoDL

当前 130 个可训练样本在 CPU 上即可完成实验，没必要为这个规模租卡。真正的
租卡触发条件是下一步：

1. 将紧凑配对集扩大到至少 2,000 train / 300 validation / 300 test；
2. 保持 source-design 隔离；
3. 运行至少 4 个随机种子；
4. 检查独立 test Top-5/Top-10 是否不再退化；
5. 只有多种子 test 提升稳定时才允许替换官方 checkpoint。

本机 CUDA 驱动不适合持续训练，所以这一步应转到 Linux 云 GPU。

## 推荐实例

- 首选：RTX 3090 24 GB 或同级 24 GB NVIDIA GPU。
- 不必追求 4090；当前模型很小，稳定性和价格比极限算力更重要。
- 系统内存：至少 32 GB；如计划处理旧版大 pickle，选 64 GB。但推荐继续使用
  当前紧凑图数据路线，避免 16 GB legacy pickle。
- 数据盘：至少 50 GB；若上传完整 2.77 GB 原始压缩包并扩大数据，建议
  100 GB。
- 镜像：Linux、Python 3.10、PyTorch 2.5.1、CUDA 12.4（或兼容的现代
  PyTorch/CUDA 组合）。

## 接力包用法

桌面接力包：
`JoinABLe_AutoDL_handoff_linux_20260704.zip`

- 大小：197.00 MiB；
- ZIP entries：2356；
- ZIP 内部路径：全部使用 Linux 兼容的 `/`；
- SHA-256：
  `6db431d39ec1eafaca5a0e2b5b47e6c59e96603c76f75b440dae46e2440f198a`。

接力包内包含：

- `joinable_gpu_reproduction/`：训练、评价、聚合与审计代码；
- `cad_assembly_agent/`：STEP 图到模型输入的适配器；
- `joinable_migration_audit/vendor/JoinABLe/`：官方源码和 checkpoint；
- `data/domain_adapt_300/`：当前 300 对 smoke-test 数据。

云端安装附加依赖：

```bash
python -m pip install -r joinable_gpu_reproduction/requirements_autodl.txt
```

执行：

```bash
chmod +x joinable_gpu_reproduction/run_autodl_training.sh
DATA_ROOT="$PWD/data/domain_adapt_300" \
  bash joinable_gpu_reproduction/run_autodl_training.sh
```

训练脚本支持 `--data-root`，会把 manifest 中保存的 Windows 绝对路径安全地
重定位到 Linux 数据目录，不需要手工修改每条样本。

该路径重定位已做端到端 CPU smoke test：130 train、33 validation、37 test
全部成功加载，failure count 为 0；云端启动脚本也已通过 Git Bash
`bash -n` 语法检查。

当前 300 对数据只用于云环境 smoke test。完整实验前仍需生成扩大后的紧凑
STEP 图数据，并将 `DATA_ROOT` 指向扩大数据集。

## 安全边界

- JoinABLe 只输出 face/edge interface 候选。
- adapted checkpoint 当前不启用。
- 模型输出不能改变 accepted/rejected。
- 所有候选仍需经过 pose、碰撞、multi-evidence 和 group-consistency gate。
- 不引入 RL，不扩大多 Agent，不让 DeepSeek 做分组裁判。
- 任何 test 退化的 checkpoint 都不晋级。

## 本轮主要新增/修改

新增：

- `pretrained_joinable_predictor.py`
- `merge_candidate_providers.py`
- `audit_official_step_transfer.py`
- `prepare_domain_adaptation_subset.py`
- `build_domain_adaptation_manifest.py`
- `finetune_step_domain.py` 的配套实验与评价产物
- `aggregate_multiseed_results.py`
- `run_autodl_training.sh`
- `requirements_autodl.txt`
- `task_heartbeat.ps1`

修改：

- `step_to_brep_graph_probe.py`：补齐官方 checkpoint 所需 B-Rep 特征和拓扑审计；
- `finetune_step_domain.py`：exact designer entity 选择指标、CPU/CUDA、
  full/post-only、云端数据路径重定位；
- `README.md` 和 `ME.md`：记录 shadow-only 与五分钟桌面心跳约定。

## 下一步需要用户提供

租好实例后提供以下任一种连接信息即可：

1. AutoDL SSH 主机、端口、用户名、密码；或
2. SSH 主机、端口、用户名和私钥路径。

拿到连接后可以由 Codex 完成上传、环境核验、smoke test、扩大数据实验和日志
回收。不要在聊天中公开长期有效的密钥；优先使用临时实例密码或限定用途的
SSH key。
