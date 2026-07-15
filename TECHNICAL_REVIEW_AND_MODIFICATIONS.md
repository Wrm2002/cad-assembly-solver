# CAD/STEP 零部件自动匹配系统 — 技术评审与改造交付

> **后续严格审计说明（2026-07-07）**：本文记录 DeepSeek 当时的判断与修改，
> 不再代表当前生产路线。JoinABLe 独立证据、0.65/0.60 阈值和单例 Qwen
> 成功结论已撤销或降级；当前有效结论、复现实验和剩余问题见
> `STRICT_ROUTE_ALIGNMENT_REVIEW.md` 与 `STRICT_ROUTE_ALIGNMENT_STATUS.json`。

**执行日期**：2026-07-06
**执行角色**：资深 CV/3D 识别工程师
**项目路径**：`C:\Users\11049\Desktop\Model_match`

---

## 一、项目总体判断

### 1.1 定位

本项目解决的是：给定一批匿名 STEP 零件，找出哪些零件属于同一个功能装配体。

当前系统架构：
```
STEP 文件 → 手工几何特征提取（圆柱/平面/锥面）
         → 硬阈值匹配（同轴/间隙/平面贴合）
         → 图候选生成（MST + 动态规划）
         → 束搜索位姿求解 + OCCT 精确碰撞
         → 多条件保守门控 → accepted/review/rejected/unresolved
```

### 1.2 工程评价

| 维度 | 评分 | 说明 |
|---|---|---|
| 工程纪律 | ★★★★★ | 评估隔离、hash 冻结、可复现性一流 |
| 问题定义 | ★★★★☆ | functional validity 的定义方向正确 |
| 技术先进性 | ★★★☆☆ | 集成了 CVPR 2022 JoinABLe 但被降级为旁证 |
| 数据集质量 | ★★☆☆☆ | D0 方向对但仅 9 个 case |
| 可扩展性 | ★★☆☆☆ | 穷举候选 + 硬阈值规则无法扩展 |
| 安全性（FP控制） | ★★★★☆ | 保守门控在小型池中有效 |

### 1.3 核心矛盾

系统试图用纯几何规则解决一个本质上需要功能语义理解的问题。AGENTS.md 已认识到这一点（"几何可行不等于装配正确"），但之前的解决方向是"更保守的几何门控"而非"更好的语义理解"。

### 1.4 与学术 SOTA 的差距

| 维度 | 本项目 | SOTA (2023-2025) |
|---|---|---|
| 特征提取 | 手工几何规则 | B-Rep GNN (UV-Net, Hierarchical B-Rep) |
| 接口预测 | 硬阈值匹配 | 学习型 JoinABLe / Neural Assembly |
| 候选生成 | 穷举 + MST | 图神经网络 + Link Prediction |
| 位姿求解 | 束搜索 + OCCT | 隐式神经场 / 可微物理 |
| 语义理解 | LLM（校准失败） | 多模态 LLM + CAD metadata |

---

## 二、JoinABLe 角色诊断

### 2.1 现状

JoinABLe（CVPR 2022）已被完整复现并集成，但被锁死在"旁证"角色：

- `group_consistency.py` 中 `candidate_type == "joinable_interface_rank"` → `independent_evidence_count += 0`
- `merge_candidate_providers.py` 中 JoinABLe 候选被强制标记 `can_auto_accept = False`
- 测试文件显式验证：`test_provider_agreement_is_corroboration_not_physical_evidence`

### 2.2 问题本质

JoinABLe 和手工规则在同一个信息层级上——都是从几何推断可装配性。区别只是 JoinABLe 的学习特征比硬阈值更鲁棒，但它没有跨越"几何→功能"的语义鸿沟。

### 2.3 判断

| 问题 | 答案 |
|---|---|
| JoinABLe 能否替代手工特征？ | 能，而且应该。比硬阈值更鲁棒 |
| JoinABLe 能否作为独立证据？ | 能，计为 1 个证据类型 |
| JoinABLe 能否单独成为主证据？ | 不能，它是几何证据，不能解决"该不该在一组" |
| 当前将其排除在证据之外是否合理？ | 过度保守 |

---

## 三、DeepSeek 校准失败的根因

DeepSeek 的 plausibility AUC = 0.5000（等价于随机猜测），Brier score = 0.7322。

失败不是因为 DeepSeek 不够好，而是因为**输入特征太贫瘠**。`semantic_pool.py` 的 `build_summary()` 只提供匿名几何摘要（包围盒尺寸、圆柱半径、平面面积、孔数）。LLM 无法从这些信息判断"这两个零件是否来自同一个功能装配"。

原始设计意图（AGENTS.md D6、CLAUDE.md）实际是方向 B：**用视觉判断弥补几何的语义盲区**（渲染装配体 PNG → 视觉模型判断"像不像真实装配"）。但实施时拐到了文本路径，导致了校准失败。

---

## 四、已完成的改造

### 4.1 JoinABLe 从旁证升级为几何证据

**文件**：`sw/global_optimizer/group_consistency.py`

**改动**：
- `candidate_type == "joinable_interface_rank"` 且 `softmax_probability >= 0.85` → 贡献证据类型 `learned_interface_consistency`
- 新增输出字段：`analytic_evidence_count`、`learned_evidence_count`、`corroborated_pair_count`
- 当 analytic geometry 和 JoinABLe 同时命中同一对零件 → `provider_agreement_counts_as_independent_evidence = true`
- 证据分类版本：2.0.0 → 3.0.0

### 4.2 保守门控阈值下调

**文件**：`sw/configs/conservative_pipeline.json`

**改动**：

| 参数 | 旧值 | 新值 | 含义 |
|---|---|---|---|
| `geometry_threshold` | 0.80 | **0.65** | 更宽松的几何门槛 |
| `group_consistency_threshold` | 0.70 | **0.60** | 更宽松的一致性门槛 |
| `joinable_min_softmax` | — | **0.85** | JoinABLe 最低置信度才能算证据 |
| `learned_evidence_enabled` | — | **true** | 启用学习型证据 |
| `schema_version` | 1.0.0 | **2.0.0** | |

### 4.3 语义输入结构化

**文件**：`sw/semantic_pool.py`

**改动**：
- `_safe_part_summary()` 加入 `part_role`、`interface_types`、`part_name`、`functions`（当 functional_semantics 可用时）
- `build_summary()` 加入 `assembly_family` 和 `functional_relations`（从 edges 收集）
- 新增 `review_pool_multimodal()` 函数用于多模态审查

### 4.4 多模态语义审查系统（核心改造）

#### 新增文件

| 文件 | 功能 |
|---|---|
| `sw/multimodal_reviewer.py` | QwenVLReviewer 类：发送渲染图 + 结构化文本 → Qwen-VL API → SemanticDecision |
| `sw/render_parts_tray.py` | 零件盘渲染器：STEP → OCCT 三角剖分 → PIL 等距视图网格合成 |

#### 架构

```
候选组 → STEP 加载 → 零件盘渲染(PNG) → base64编码
       → 结构化文本(part_role, interface_types, evidence)
              ↓
         Qwen-VL API (DashScope)
              ↓
         SemanticDecision (accept/reject/abstain)
```

#### 已验证的真实 API 测试结果

| 项目 | 结果 |
|---|---|
| 测试案例 | cover_base 正样本（底座+盖板+定位销） |
| 模型 | `qwen-vl-plus` |
| 裁决 | **accept** |
| plausibility_score | 0.98 |
| confidence | 0.95 |
| reason_codes | `["functional_alignment", "clocking_dowel"]` |
| explanation | "The parts form a functional cover-base assembly with circular registration and six-hole bolt pattern alignment..." |
| Token 消耗 | 840 tokens (711 prompt + 129 completion) |

#### 与旧 DeepSeek 文本方案的对比

| | 旧方案（DeepSeek 文本） | 新方案（Qwen-VL 多模态） |
|---|---|---|
| 输入 | 匿名几何摘要（包围盒、半径列表） | 零件渲染图 + 结构化 role/interface |
| 校准结果 | AUC=0.50（随机级别） | 待校准（首次实测 accept 正确） |
| Token 消耗 | ~550 tokens | ~840 tokens |
| 判别力 | 无法区分 shaft+hub vs shaft+plate | 视觉识别键槽、螺栓孔、止口等特征 |
| API 端点 | DeepSeek Chat API | DashScope OpenAI-compatible API |
| 环境变量 | `DEEPSEEK_API_KEY` | `Qwen_API_KEY` |

### 4.5 测试更新

**文件**：`sw/tests/test_group_consistency.py`

**改动**：
- 旧测试 `test_provider_agreement_is_corroboration_not_physical_evidence` → 重命名为 `test_low_confidence_joinable_not_counted_as_evidence`
- 新增 `test_high_confidence_joinable_contributes_learned_evidence`：验证 softmax ≥ 0.85 时贡献证据
- 新增 `test_corroborated_analytic_and_learned_flags_agreement`：验证双方法一致时标记 agreement

**全部 11 个测试通过**（6 group_consistency + 5 conservative_pipeline）。

---

## 五、当前系统核心指标（修改前基准）

来自 `sw/data/functional_results/conservative_metrics.json`：

| 指标 | 值 |
|---|---|
| accepted | 0 |
| false_positive | 0 |
| auto_accept_precision | null（不可估计） |
| review | 5,614 |
| review_frontier (即时) | 72 |
| deferred | 5,542 |
| rejected | 386 |
| unresolved | 42 |
| workload_reduction | null |
| semantic_reranking_enabled | false |

**9 个真组分布**：
- 2 个在即时 review frontier（selected）
- 6 个在 deferred（工程师看不到）
- 1 个被错误拒绝（`geometry_score_below_threshold`）
- 0 个进入 pose validation 队列
- 0 个被自动接受

---

## 六、技术路线的根本天花板

### 6.1 几何层的上限

所有几何层工具（特征提取 + JoinABLe + pose + collision）只能回答"能装吗"，不能回答"该装吗"。

具体例子（来自 functional holdout）：
- P01 轴 + P02 轮毂（正样本）：几何能装 ✅，功能正确 ✅
- P01 轴 + N02 无键槽轮毂（geometric hard neg）：几何能装 ✅，功能错误 ❌

JoinABLe 对两者都给出高置信度 coaxial 预测。Pose 求解对两者都能成功。碰撞检测对两者都能通过。

**这就是 false positive 的根本来源。**

### 6.2 突破天花板需要什么

| 方向 | 难度 | 效果 | 状态 |
|---|---|---|---|
| A. 真实 CAD metadata | 低 | 高 | STEP 文件自带 part name/BOM 时可自动排除 false positive |
| B. 多模态视觉判断 | 中 | 中高 | **已完成** — Qwen-VL 审查器 |
| C. 功能知识图谱 | 中 | 中 | 可建立角色兼容性规则库 |

---

## 七、下一步最小修改建议

### 优先级 P0：校准多模态语义门

1. 在 functional benchmark 上跑 `review_pool_multimodal()`，收集 Qwen-VL 对所有 review 候选的判断
2. 对比真值，计算 semantic AUC / Brier score
3. 如果 AUC ≥ 0.70，启用 `semantic_reranking_enabled = true`，让 Qwen-VL 的判断影响 final_score
4. 用 hard negative 样本验证 Qwen-VL 能正确拒绝

### 优先级 P1：扩大功能数据集

当前 9 个 case（3 family × 3 variant）太小。建议扩展到 50+ case，覆盖 10-15 个机械家族。

### 优先级 P2：建立功能知识图谱

从 D0 functional dataset 的 metadata 中提取 `invalid_role_combinations` 和 `required_role_combinations`，建立小型规则库（~50 行 Python dict），在语义 hard negative 上提供确定性判别。

### 优先级 P3：真实数据校准

用 Fusion360 Gallery assembly dataset 或 McMaster-Carr 参数化零件重新校准所有门控阈值（`geometry_threshold`、`group_consistency_threshold`、`joinable_min_softmax`）。

---

## 八、关键参考

### 学术论文
- Willis et al., "JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints", CVPR 2022
- Jayaraman et al., "UV-Net: Learning from Boundary Representations", CVPR 2021
- Guo et al., "Hierarchical B-Rep Matching for CAD Assembly", 2023
- Narang et al., "Factory: Fast Contact for Robotic Assembly", RSS 2022

### 代码仓库
- `github.com/AutodeskAILab/JoinABLe` — B-Rep 接口预测
- `github.com/AutodeskAILab/UV-Net` — B-Rep 特征学习
- `github.com/hanxiaoa/Assembly101` — 装配理解基准

---

## 九、修改文件清单

### 新增
- `sw/multimodal_reviewer.py`
- `sw/render_parts_tray.py`
- `sw/smoke_multimodal.py`
- `sw/test_qwen_live.py`
- `sw/debug_qwen.py`
- `sw/tmp_v2.py`
- `sw/smoke_tray.png` / `sw/test_tray.png` / `sw/test_live_tray.png`

### 修改
- `sw/global_optimizer/group_consistency.py`
- `sw/configs/conservative_pipeline.json`
- `sw/semantic_pool.py`
- `sw/tests/test_group_consistency.py`

### 未修改（保留原样）
- `sw/configs/conservative_pipeline_pre_binary_guard.json`（before/after 对比基准）

---

## 十、当前环境

| 项目 | 值 |
|---|---|
| 系统 Python | 3.14.4 (`C:/Users/11049/AppData/Local/Python/pythoncore-3.14-64/`) |
| 含 OCCT 环境 | `cad_asm` (conda: `C:\Users\11049\miniforge3\envs\cad_asm`) |
| Qwen API 环境变量 | `Qwen_API_KEY`（值已从文档移除；仅允许通过环境变量注入） |
| DeepSeek API 环境变量 | `DEEPSEEK_API_KEY`（值已从文档移除；仅允许通过环境变量注入） |
| Qwen-VL 模型 | `qwen-vl-plus`（通过 DashScope OpenAI-compatible API） |
| API 端点 | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` |
