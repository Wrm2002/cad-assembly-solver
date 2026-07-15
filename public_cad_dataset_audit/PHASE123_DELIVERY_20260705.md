# CAD装配项目第1～3步交付报告

总状态：已完成

## 第1步：冻结JoinABLe双零件Tool

- 官方checkpoint及运行源码已用SHA-256冻结。
- 官方测试集：1857对。
- Top-1/Top-5/Top-10：79.32%/87.56%/90.79%。
- 已重新运行一对真实STEP图的CPU推理，Top-1节点对为[2, 3]，冻结校验通过。
- 工具保持shadow mode，不能直接接受装配组。

## 第2步：真实Fusion装配级数据

- 可用装配：10/10。
- occurrence-body实例：184。
- 直接正边：171，其中designer joint边13，contact-only边158。
- 关系映射：423/423。
- 接口实体映射：846/846。
- 10个装配的STEP均可用，正/负pair全集分区完整。
- cover_base、shaft_hub_key、bearing_housing已建立功能标注契约，但尚未伪造为已验证CAD样本。

## 第3步：真实mixed-pool

- 源装配：10，池：6。
- 匿名STEP实例：49，真实来源组：16。
- 直接joint/contact正对：38。
- 同组但无直接边（不误标为负）：23。
- 跨来源装配负对：122。
- 几何相似负例候选：58，均明确标记为尚未经过pose/collision验证。
- train/validation/test之间assembly_id零重叠，pool_input源身份泄漏为0。
- 数据位置：D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705

## 边界与下一步

- The current Fusion subset has no audited part_role, assembly_family, or functional_relation labels.
- Contact-only edges are observations, not guaranteed designer intent.
- Cross-assembly negatives are provenance negatives, not proof of universal mechanical incompatibility.
- The 58 geometry-similarity negatives are candidates for later pose/collision audit; none is claimed as a verified hard negative.
- The functional family catalog is an annotation contract only; it is not counted as human-validated CAD benchmark data.
- No mixed-pool JoinABLe batch inference was run in Steps 1-3; that is the Step-4/AutoDL trigger.

第4步开始才需要打开AutoDL：对mixed-pool候选零件对批量运行JoinABLe，并做候选召回审计。
