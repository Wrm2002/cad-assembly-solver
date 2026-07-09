# Mixed-pool Functional Assembly Grouping V2

## Outcome

The eight requested components are implemented and tested. The route is now a conservative, family-aware proposal and review system. It does not force a full partition, does not use semantic models for hard decisions, and does not allow JoinABLe confidence to become physical evidence.

## Implemented route

1. Unified `PairEdge` records aggregate analytic and JoinABLe providers while separating provider agreement from physical evidence.
2. Geometry-only multi-label center/role estimation ignores embedded synthetic semantic truth.
3. Family slot templates cover `cover_base`, `shaft_hub_key`, and `bearing_housing`, including optional axial/bearing retainers.
4. Bounded expansion enumerates center + slots rather than arbitrary subsets.
5. Subset/superset links are recorded before pose; demotion occurs only when the superset is also pose-valid.
6. Proposal clustering is family-isolated and only compresses review presentation.
7. Failure diagnosis separates candidate recall, proposal generation, ranking, pose, and semantic/calibration failures.
8. The locked topology-varied functional holdout is evaluated separately from the ordinary functional benchmark.

## Proposal and review results

| Benchmark | Proposals | Proposal recall | Review frontier | Frontier recall | Frontier precision |
|---|---:|---:|---:|---:|---:|
| Previous ordinary baseline | 9,668 | 100% | 72 | 22.22% | 2.78% |
| V2 ordinary functional | 957 | 100.00% | 60 | 100.00% | 15.00% |
| V2 locked harder holdout | 135 | 100.00% | 40 | 100.00% | 10.00% |

The ordinary proposal count fell by 90.10%, while review-frontier true-group recall rose from 2/9 to 9/9. The locked harder holdout generated and surfaced 4/4 provisional true groups.

The D0 functional dataset audit passed: 9 cases, three cases per family, zero invalid cases, zero generic cone/ring/block/plate stack positives, all three required hard-negative types present, and no production decision treats source ID as truth.

## Exact pose and false-positive audit

| Benchmark | Checked | Pose valid | Failed | Uncertain | Final accepted | Review | Rejected | False auto accepts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary functional | 60 | 58 | 2 | 0 | 0 | 58 | 2 | 0 |
| Locked harder holdout | 40 | 23 | 16 | 1 | 0 | 24 | 16 | 0 |

On the ordinary benchmark, 49 false groups were pose-valid; on the harder holdout, 21 false groups were pose-valid. This empirically confirms that pose success and collision freedom are not evidence of functional membership.

An exploratory gate would have auto-accepted 2 harder-holdout groups; all 2 were false. Those groups are now transferred to review and the role/template calibration gate is closed. Final false auto accepts are zero. Accepted precision is therefore not estimable rather than being reported as 100%.

## Ablation conclusions

- Queue size 5 recalled 75.00%; queue size 10 recalled 100.00%. Ten per pool is the smallest tested budget with 4/4 harder-holdout recall.
- Analytic-only and analytic+JoinABLe union both generated 4/4 true proposals, but each surfaced only 3/4 at its current ablation frontier. JoinABLe remains a core recall provider, but this holdout shows it does not solve group ranking by itself.
- Pre-pose subset demotion was tested and rejected: it reduced ordinary frontier recall from 9/9 to 6/9. The implementation now waits for pose confirmation.
- Qwen/DeepSeek reranking remained disabled in every experiment.

## External-method review

The [JoinABLe paper](https://arxiv.org/abs/2111.12772) and [official implementation](https://github.com/AutodeskAILab/JoinABLe) define a pair-of-parts B-Rep entity/link-prediction and joint-pose task. That supports using JoinABLe as a first-class `PairEdge` provider but does not support treating it as a complete mixed-pool grouping judge. The [Fusion 360 Gallery Assembly Dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) provides joint and assembly graph supervision, but functional validity still requires the project-specific signed holdout.

## Gate status and remaining blocker

- Semantic reranking: disabled.
- Role/template auto-accept calibration: failed and closed.
- Mechanical-engineer signoff for the locked holdout: pending.
- GPU training: not started because proposal recall is already saturated on both tested benchmarks; current errors are semantic/functional ambiguity and pose-valid false combinations.

The next minimal step is not larger search or model training. It is signed review of the locked holdout plus discriminative functional-interface features that separate the 49/21 pose-valid false groups from true groups. Until that calibration passes, the correct engineering output is high-recall review with zero automatic false acceptance.
