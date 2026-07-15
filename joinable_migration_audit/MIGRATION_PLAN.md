# JoinABLe 可执行迁移路线

原则：JoinABLe 始终是独立 candidate provider；主 solver、OCCT validation 和
保守 accepted/review/rejected gate 在所有阶段保持独立。任何阶段不达门槛都不
进入下一阶段。

## Phase 0：复现 JoinABLe 原始流程

### 动作

1. 新建隔离环境 `joinable_legacy`，严格使用官方 CPU 依赖：
   Python 3.7、PyTorch 1.8、PyG 1.7.2、Lightning 1.3.7。
2. 下载官方 raw j1.0.0（本机已完成）和 preprocessed j1.0.0 514MB。
3. 不训练，先加载 `pretrained/paper/last_run_0.ckpt`。
4. 在官方 test split 跑固定 20 个样本，保存 raw logits、top-K entity pairs、
   axis 和 pose-search 输出。
5. 将官方 README 的 79.53% 视为论文参考，不把本地未复现值写入结果。

### 输出

- `phase0_environment_lock.yml`
- `phase0_official_inference.json`
- `phase0_reproduction_report.md`

### Gate

- 官方 checkpoint 可加载；
- 同一输入重复运行的 top-K 一致；
- 至少 20 个官方样本有非空预测；
- entity index 可反查 body graph；
- 不修改主项目 Python 环境。

当前状态：**未执行 pretrained inference**。原因是本任务禁止训练且优先做迁移
审计；旧运行环境应单独建立，避免再次污染/击穿主环境。

## Phase 1：数据 schema 打通

### 已完成

- Fusion Joint Data → common schema：20/20；
- 当前 STEP → coarse B-Rep graph：3/3；
- designer entity、axis、contact、holes、transform 的字段审计；
- Fusion/STEP schema gap。

### 待补

1. OCCT face 10×10 UV points/normals/trimming mask；
2. OCCT edge 10-point points/tangents；
3. edge convexity、dihedral angle；
4. JoinABLe entity enum 与 normalization 完全一致；
5. 官方 body graph 与同一 STEP adapter graph 的逐节点数值对照；
6. topology id round-trip 测试。

### Gate

- 官方同一 body 的 face/edge 数量和 adjacency 一致率 100%；
- surface/curve enum 映射覆盖率 ≥99%；
- sampled feature 无 NaN/Inf；
- selected joint entity round-trip ≥99.9%；
- 所有缺失字段都显式记录。

## Phase 2：规则 baseline

### 动作

1. 不训练模型。
2. 在 common schema 上为已知 part pair 枚举可产生 axis 的 face/edge。
3. 用类型兼容、半径、轴线、尺寸和 adjacency 形成可解释分数。
4. 输出 top-1/top-5/top-10 interface pair。
5. 用官方 designer-selected entity 及 axis-equivalent labels 评估。
6. 按 joint type、face-face/face-edge/edge-edge 分层报告。

### 指标

- exact selected-entity pair top-K recall；
- axis-equivalent top-K recall；
- candidate reduction ratio；
- unsupported entity rate；
- 每个拒绝/跳过原因覆盖率。

### Gate

- top-10 axis-equivalent recall ≥95%；
- 候选量比全表面笛卡尔积减少 ≥80%；
- 不因预筛选删除无法解释的真值；
- evaluation split 按 source design，而不是随机 joint 行。

## Phase 3：JoinABLe-style 模型训练或复用

### 动作

1. 首先复用 pretrained checkpoint，不先重训。
2. 比较 official graph 与 OCCT graph 的 embedding 输入分布。
3. 若 domain gap 明显，再设计 Fusion/AutoMate 训练集；AutoMate 没有直接 face
   id，不能伪装成 interface localization 标签。
4. 模型只输出 top-K face/edge/interface candidate 和置信度，不输出最终分组。

### Gate

- official holdout 指标复现；
- STEP adapter 上无 schema/runtime failure；
- 相比 Phase 2，提高 top-K recall 或在相同 recall 下进一步降候选；
- 真实样本人工检查无明显 topology id 错位。

## Phase 4：迁移到当前 SolidWorks / STEP

### 接口

```text
STEP
  -> isolated OCCT graph adapter
  -> rule/JoinABLe candidate provider
  -> top-K interface + axis seeds
  -> existing pose solver
  -> exact OCCT collision/residual validator
  -> accepted/review/rejected
```

### 动作

1. 以 JSON 文件或独立进程通信，不 import 到主 solver。
2. 每个 candidate 携带 source SHA256、face/edge id、signature 和模型版本。
3. 先 shadow mode：只记录候选，不改变主系统结果。
4. 对真实人工标注 pair 评估 top-K recall 与 false-positive workload。

### Gate

- 真实 pair top-10 recall 达到 Phase 2 预设门槛；
- candidate provider 崩溃不会带崩主 solver；
- exact validator 仍是物理接受必要条件；
- 无法映射的 entity 自动进入 review。

## Phase 5：混合零件池分组

### 动作

1. 先用尺寸/检索索引减少 part-pair 数，不对全池做无界 O(n²) 深推理。
2. 对剩余 pair 调 candidate provider，构建带 top-K interface 证据的连接图。
3. connected components 只产生候选区域，不直接当 final groups。
4. 用 group consistency、central-part structure、全局冲突和 3D collision 分层。
5. 高置信 accepted；冲突或单接口证据进入 review；剩余 unresolved。

### Gate

- auto-accept precision ≥90%；
- false positive 不高于当前保守基线；
- 所有 accepted edge 至少有两个独立证据；
- mixed-pool 结果继续输出 accepted/review/rejected/unresolved；
- JoinABLe 分数绝不单独决定 accepted。

## 当前立即可做的最小下一步

只执行 Phase 1 的 feature parity，不接主 solver、不训练：

1. 给 `step_to_brep_graph_probe.py` 增加 UV/curve sampling；
2. 用已下载 Fusion body 的 STEP 作为 cross-kernel 对照；
3. 建立 selected entity round-trip 单元测试；
4. 再实现 Phase 2 规则 top-K baseline。
