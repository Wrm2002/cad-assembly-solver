# 治本路线可行性与数据准入审计

## 结论

项目具备训练真正 B-Rep Pair Pose / 接口评分网络的主要正样本、相对 Pose、局部接口面片和 assembly 隔离切分；但当前完整模型的 holdout 分数不足，不能用于 SolidWorks 考试或宣称功能语义正确。

## 已具备的硬条件

- Fusion B-Rep 合同：18002 个记录，29560 条 joint supervision，split overlap 为 0。
- 完整 Pair Pose 训练集：train/dev/test = 22917/3489/2901；三份 embedding 非零率分别为 1.000/1.000/1.000。
- 自由度标签不是全零：训练集中有 7 类 DOF mask。
- 局部强接触基线：rank-1 = 0.969，但它只评测真实接触对抗局部扰动。

## 不能误读的结果

- 完整 Pair Pose + 接口评分模型的 assembly-holdout rank-1 仅为 0.597。这是目前唯一应作为完整模型基线使用的数字。
- 强接触集的高分不能替代真实多模式 Pose、同装配干扰接口、跨 CAD 域与多零件闭合测试。

## 尚未补齐的硬条件

1. **Pose 等价类**：同一接口的对称旋转、可滑移区间和多种有效装配状态尚未组成一个正例集合；单一 Pose 标签会把其他有效解误作负例。
2. **困难负例**：需要由真实 occurrence Pose 自动构造近接触滑移、翻转、穿透、错误插入深度，以及同装配的错接口负例；不能仅用随机局部噪声。
3. **接口闭合 target**：现有 gap/coverage/normal target 没有记录完整插入长度、包络比例、重复孔阵列一致性等可由 B-Rep 自动量测的 target。
4. **外部考试标签**：当前没有隔离的 SolidWorks 真实装配 Pose 真值，因此不能量化跨 CAD 域泛化，更不能据 case1–5 调参。
5. **GPU 训练环境**：默认 OCCT 环境为 CPU-only；预检会单独探测 GPU 训练环境，二者不得互相覆盖。

## 允许的下一步

- 先从 Fusion occurrence 与 B-Rep 自动生成等价 Pose 集和困难负例；不读取 SolidWorks 考试目录。
- 训练 Pair Pose top-k 与 candidate-conditioned Interface Scorer，并在 Fusion assembly holdout 上分别报告 top-k Pose 误差、真实接触排序、错误自动接受率。
- 只有完整训练基线显著超过当前 0.597 rank-1，且外部真值 benchmark 不回退，才接入多零件因子图。
- SolidWorks case1–5 只作为冻结考试；若无真值，结果只能做人工可视化 review。
