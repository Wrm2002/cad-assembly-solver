# Case3 / Case4 / Case5 泛化考试

## 结论

用户对泛化性的担心成立。根据 2026-07-13 用户提供的真实工程装配图进行人工复核，case3、case4、case5 的最终渲染位姿全部错误，工程 Pose 通过率为 **0/3**。此前把“零件关系边命中”表述成 case3/case5 正确是不严谨的，现予以撤回。

- 原始 case3/4/5 没有一组达到 `precision-valid` 或 `accepted`。
- 三组全部进入 `review`，因此本轮没有 false auto-accept。
- pass-1 关系边按旧标签合计为 TP=4、FP=1、FN=1；这个数字只衡量“哪些零件可能有关联”，不能证明接口、方向、插入深度或最终位姿正确，不能再作为 case 通过依据。
- case4 出现实质性错误：系统预测 `memory module <-> CPU`，漏掉应有的 `PCBA <-> CPU`。
- case3 虽然关系边和代数 closure 看似成立，但风扇没有恢复到风扇笼中的真实槽位、朝向和止挡位置，因此 Pose 错误。
- case5 虽然旧关系标签认为两条边命中，但电源模块没有恢复到机箱仓位，耳板也没有贴合机箱侧壁；closure=0/2，Pose 错误。PSU 含大量 open shell，common-volume 也不能作为完备碰撞证明。

这说明当前系统的“保守验收”比“Pose 泛化”更可靠：它成功避免了错误自动接受，但还不能在大型、多 solid、局部接口占比很小的真实 CAD 上恢复完整 Pose。

## 冻结规则

- `beam_width=20`
- 三个 case 不做单独阈值或候选预算调整
- 原始 `sw/3`、`sw/4`、`sw/5` 为正式输入
- `sw/4_lightweight`、`sw/5_lightweight` 仅作执行 smoke，不计入泛化通过率
- 旧 `assembly.step` 和 identity placement 不作为 Pose 真值

## Raw 结果

| Case | 候选 | 边 TP/FP/FN | Closure | Exact | 系统 tier | 工程 Pose 人审 | 耗时 |
|---|---:|---:|---:|---|---|---|---:|
| case3 | 55 | 1/0/0 | 1/1 | large-step budget 未完成 | review | **错误** | 53.7 s |
| case4 | 67 | 1/1/1 | 0/2 | closure short-circuit | review | **错误** | 165.7 s |
| case5 | 65 | 2/0/0 | 0/2 | closure short-circuit；open-shell 风险 | review | **错误** | 148.5 s |

这里的 `review` 只说明保守门控没有误自动接受，不代表求解结果接近正确答案。

## Proxy smoke

case4/5 lightweight 的旧关系标签命中，且 OCCT worker 返回 success；但是两组 closure 都是 0/2，因此仍为 review。代理几何是人为保留考试接口且接近装配坐标的简化模型，只能证明代码路径可执行，不能用于声称 raw CAD 泛化成功或工程 Pose 正确。

## 当前路线暴露出的缺口

1. case1/2 使用过的 all-pairs JoinABLe、multi-axis、prismatic、isolated pair OCCT 和 precision manifest 尚没有通用批处理入口；raw case3/4/5 不能自动得到同样完整的 compound sidecar。
2. 多 solid 大部件产生大量无关平面/角点，真实小接口被淹没。case4 的错误第二条边就是直接证据。
3. 当前 compound proposer 强于重复圆柱阵列和简单 key-slot，但不覆盖 fan cage rail、DIMM socket、CPU socket、PSU bay 等局部拓扑。
4. 大型凹腔和 open-shell 不能依赖单次全局 common-volume；需要 per-solid broad phase、局部 ROI 和带超时的 exact worker。
5. 当前没有可信 pass-2 face/edge Pose GT，因此不能用最低 cost 或旧 assembly coordinates 证明姿态正确。

## 下一步最小改动

1. 先补一个通用、固定参数的 pair-frontier augmentation runner，使 case3/4/5 真正走与 case1/2 相同的候选路线。
2. 在 B-Rep graph 上先做局部接口 ROI / rarity-weighted retrieval，抑制 PCBA/机箱内部重复平面。
3. 新增通用的 `planar seating + bounded pocket walls + insertion direction + depth` compound factor，而不是按零件名称写规则。
4. exact validation 改为 per-solid BVH broad phase，再仅对疑似接触 solid pair 调 OCCT。
5. 在评测副本上施加确定性的独立刚体扰动，消除原始共同坐标系或 identity pose 带来的假成功。

## 本轮额外修复

- 修复 `known_group_assembly.py` 对 `../<part>` 的硬编码假设；独立考试输出目录现在能正确引用原始 STEP。
- 新增 `export_assembly_manifest_stl.py`，按 manifest 导出变换后的逐零件 STL，用于四视图审查。
