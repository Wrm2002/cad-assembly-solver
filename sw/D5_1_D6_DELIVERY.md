# D5.1-D6 Delivery: Collision Backtracking and Calibrated Agent

Date: 2026-07-02

## D5.1 complete-pose backtracking

The initial solver retained only its highest-scoring complete pose. Exact OCCT
collision validation could reject that pose, but could not recover a lower
ranked collision-free pose.

The revised solver:

- branches over up to three connected next-part choices;
- retains the complete beam instead of only its top state;
- checks up to 100 complete poses in score order;
- rejects candidates with excessive selected-constraint residual first;
- uses OCCT Boolean Common volume to choose the first physically valid pose;
- records every checked rank and rejection reason;
- runs each CAD validation in an isolated worker.

Increasing beam width from 100 to 300 and checking 200 poses did not solve a
representative six-part group, so the production configuration remains
bounded at 100/100.

Evaluation-only true-group pose audit:

| Parts per group | Groups | Accepted | Recall |
|---:|---:|---:|---:|
| 2 | 10 | 10 | 100% |
| 3 | 10 | 10 | 100% |
| 4 | 4 | 4 | 100% |
| 5 | 4 | 4 | 100% |
| 6 | 2 | 0 | 0% |
| **All** | **30** | **28** | **93.33%** |

This improves the initial audit from 73.33% to 93.33%. It does not solve the
six-part search boundary.

### Expanded four/five-part audit

The four observations for each of sizes 4 and 5 in the mixed-pool audit were
too small for a persuasive rate estimate. A separate frozen audit therefore
ran every available source case for these sizes, with one isolated process per
case and no pool-group selection:

| Parts | Generator template | Cases | Accepted | Rate | Wilson 95% CI | Worker failures |
|---:|---|---:|---:|---:|---:|---:|
| 4 | `cover_base` | 100 | 100 | 100% | 96.30%-100% | 0 |
| 5 | `small_gearbox` | 100 | 100 | 100% | 96.30%-100% | 0 |

Runtime and search diagnostics:

| Parts | Mean time/case | Maximum time | Mean expanded states | Mean exact poses checked |
|---:|---:|---:|---:|---:|
| 4 | 1.087 s | 1.419 s | 914.11 | 5.47 |
| 5 | 3.754 s | 9.164 s | 2814.06 | 27.00 |

The complete machine-readable report is
`pose_audit_4_5/pose_validation_audit.json`. This establishes strong evidence
for known-group pose reconstruction within these two parameterized generator
families. It does **not** establish the same rate for unknown mixed-pool
grouping, different mechanical families, or real external CAD.

Frozen 12-pool result after D5.1:

- controller failures: 0;
- converged pools: 12/12;
- exact-group macro F1: 4.17%;
- co-membership pair precision: 38.74%;
- co-membership pair recall: 31.90%;
- co-membership pair F1: 33.43%.

The earlier D5 pair F1 was 30.61%. The gain is real but small because a
collision-free pose proves physical feasibility, not source-assembly
membership. Some wrong cross-case groups are also physically assemblable.

## D6 DeepSeek semantic reviewer

Implementation:

- provider-neutral review interface with a DeepSeek adapter;
- API key read only from `DEEPSEEK_API_KEY`;
- no key, GT, local path, or STEP body enters requests or cache;
- strict `SemanticDecision` contract and generated JSON Schema;
- JSON parsing, schema validation, proposal-ID validation, bounded retry;
- explicit `accept`, `reject`, and `abstain`;
- deterministic request cache and replay;
- empty, invalid, mismatched, or unavailable provider output becomes abstain;
- semantic utility perturbation is bounded by the configured geometry
  ambiguity band and can never bypass pose validation.

The connectivity smoke used `deepseek-v4-flash`, returned schema-valid JSON,
and consumed 383 prompt plus 162 completion tokens.

## Calibration gate

DeepSeek was calibrated only on pools 001-003. Pools 004-012 remained holdout.

Calibration result:

- reviewed candidates: 18;
- exact true groups: 2;
- negatives: 16;
- geometry-score AUC: 0.8125;
- DeepSeek plausibility AUC: 0.5000;
- DeepSeek Brier score: 0.7322;
- verdicts observed: `accept`, `abstain`;
- semantic reranking enabled: **false**.

The model assigned high plausibility to many anonymous, geometrically feasible
but provenance-wrong groups. The calibration gate therefore prohibited
semantic scores from changing holdout assignments. No DeepSeek calls were
made on the nine holdout pools.

## Agent controller

The bounded controller records:

1. deterministic geometry grouping;
2. optional ambiguity review;
3. calibration-gate decision;
4. reliable pose search and exact collision validation;
5. bounded rejection/reassignment;
6. final structured assignment and event trace.

Geometry failures are non-overridable. With the failed semantic calibration
gate, the Agent automatically used geometry-only mode on all nine holdout
pools. Holdout co-membership pair F1 was 32.67%, exactly the corresponding
geometry behavior; no unearned semantic improvement is claimed.

## Commands

```powershell
python semantic_review.py --smoke
python calibrate_semantic_review.py mixed_pools_v1 `
  --pools pool_001 pool_002 pool_003 --mode live
python run_agent_benchmark.py mixed_pools_v1

python audit_dataset_pose_validation.py `
  synthetic_dataset_600 pose_audit_4_5 `
  --group-size 4 5 --resume

python run_agent_pipeline.py `
  --input <parts_folder> `
  --output <results_folder> `
  --semantic off
```

`--semantic deepseek` is available, but the current calibration file prevents
its scores from affecting grouping. Improving semantic input requires real
functional metadata or richer labelled mechanical families; repeated prompting
on the same anonymous primitives is not justified by the calibration evidence.

## Remaining research boundary

- Six-part pose search needs a stronger joint constraint formulation or
  collision-aware optimization inside the search, not a larger beam alone.
- Geometry cannot distinguish provenance when cross-case parts are genuinely
  interchangeable.
- The current anonymous synthetic summaries do not contain enough functional
  semantics for DeepSeek to outperform geometry.
- Future semantic experiments must add meaningful part roles, interface types,
  or real labelled CAD metadata and must pass calibration before holdout use.
