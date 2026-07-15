# Fusion / JoinABLe 与 STEP / OCCT Schema Gap

状态：`success`

| 字段 | Fusion / JoinABLe 是否有 | 当前 STEP 是否有 | 缺失影响 | 修复建议 |
|---|---|---|---|---|
| face nodes | True | True | none for coarse topology | retain OCCT IndexedMap face id |
| edge nodes | True | True | none for coarse topology | retain OCCT IndexedMap edge id |
| stable persistent face/edge id | source index stable inside released record | conditional: same file/importer only | predictions cannot safely survive STEP rewrite/healing | hash input, store indexed id and geometry signature; revalidate after every import |
| surface/curve type | True | True | minor enum/domain mapping required | map OCCT GeomAbs types to JoinABLe vocabulary |
| face-edge adjacency | True | True | none | preserve heterogeneous incidence links |
| face 10x10 point/normal/trim grid | True | False | pretrained JoinABLe feature contract cannot be reproduced | sample OCCT parametric faces with trimming mask |
| edge point/tangent grid | True | False | pretrained edge encoder input missing | sample OCCT curves at normalized parameters |
| edge convexity/dihedral angle | True | False | local interface descriptor domain gap | estimate from adjacent face normals |
| designer-selected joint entity label | True | False | no supervised truth on user STEP | prediction only; obtain labels from assembly mates or manual annotation for evaluation |
| contact label | True | False | cannot train/evaluate contact localization | requires assembled pose and exact proximity/contact query |
| hole labels | True | False | JoinABLe hole augmentation unavailable | rule-estimate cylindrical loops, mark as inferred |
| joint/mate type label | True | False | cannot evaluate mate type on standalone STEP | obtain from source CAD assembly or manual ground truth |
| assembly transform | True | False | no ground-truth pose reconstruction target | requires assembly-level source, not isolated parts |
| assembly hierarchy | joint set is a body pair, source has context | False | cannot infer mixed-pool grouping truth | ingest assembly tree/BOM or create separate labels |

## 直接回答

- Fusion designer-selected joint 能映射到 B-Rep face/edge。
- 当前 STEP graph 有 face、edge、surface/curve type 和 adjacency。
- OCCT id 仅在同文件、同版本、同导入设置下条件稳定；不是 CAD persistent id。
- 单独 STEP 不含 contact、joint、mate、assembly transform 或 hierarchy 真值。
- 因此 JoinABLe 可替代候选接口生成层，不能直接完成 mixed-pool 分组。

## 推理时 domain gap

- Fusion graph comes from native Fusion B-Rep; STEP is a neutral exchange import with possible topology splitting/healing.
- JoinABLe uses sampled face/edge grids, convexity and dihedral features not yet emitted by the minimal OCCT probe.
- Units and normalization must be reproduced exactly before pretrained inference.
- Entity ordering and persistent identity differ across Fusion and OCCT.
- Standalone STEP has no designer joint, mate, contact, pose, hierarchy or functional-intent labels.

## 可规则估计

- surface/curve type
- area/length/radius
- local axis candidates
- face-edge adjacency
- approximate hole candidates
- edge convexity and dihedral angle

## 无法从孤立 STEP 获得

- designer-selected joint entities
- ground-truth mate type
- ground-truth assembly transform
- assembly hierarchy and BOM
- true contact pairs before pose is known
- functional compatibility and source grouping
