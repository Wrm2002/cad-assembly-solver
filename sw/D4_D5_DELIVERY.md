# D4-D5 Geometry Baseline Delivery

Date: 2026-07-02

> This is the frozen initial D4-D5 baseline. D5.1 and D6 results supersede the
> pose audit below; see `D5_1_D6_DELIVERY.md`.

## Scope and leakage boundary

This stage implements deterministic global grouping and reliable pose
validation. The inference path derives the part list from `parts/`; it does not
require or inspect group labels. If `pool_gt.json` exists, metrics are attached
after inference. The same command runs on an unlabeled folder.

Ground-truth group validation is a separate diagnostic command. Its output is
stored under `oracle_validation/`, marked `diagnostic_only`, and never enters
candidate selection or retry decisions.

## D4 global grouping

`global_grouping.py`:

- enumerates connected candidate subsets of 2-6 parts;
- scores MST support, internal edge quality, and graph density;
- uses exact bitmask dynamic programming for a mutually exclusive partition;
- permits singletons instead of forcing every part into an assembly;
- records every unselected proposal and conflict reason.

Frozen 12-pool result:

- truth-group coverage among proposals: 100% (30/30);
- exact-group macro F1: 4.17%;
- co-membership pair precision: 36.65%;
- co-membership pair recall: 33.81%;
- co-membership pair F1: 33.38%.

Interpretation: candidate generation does not lose the answers, but the mixed
pools contain many same-family parts with nearly interchangeable geometric
evidence. A local edge score is therefore insufficient to identify source
assembly membership. This negative result is retained rather than tuned
against test labels.

## D5 pose solving and validation

`geometry_pipeline.py`:

- regenerates detailed matches for each selected group;
- preserves alternative parallel-face hypotheses for pose search;
- runs the reliable beam/branch-bound solver;
- validates connectivity, solved parts, and constraint residual;
- confirms solid penetration by OCCT Boolean Common volume;
- builds and reopens an assembly STEP;
- rejects failed groups and reruns global assignment with a bounded budget;
- emits schema-valid `ValidationResult` and `AgentEvent` records.

Stability:

- every CAD group validation runs in a child process;
- native access violations become explicit failed validations;
- a failed worker cannot terminate the pool controller;
- cached validation is accepted only when a fingerprint of configuration and
  all relevant solver sources matches;
- retry exhaustion may output only previously validated groups and singletons.

Frozen 12-pool result:

- controller failures: 0;
- converged pools: 12/12;
- exact-group macro F1: 4.17%;
- co-membership pair precision: 38.00%;
- co-membership pair recall: 28.27%;
- co-membership pair F1: 30.61%.

Validation increases precision slightly but decreases recall, so it does not
improve final co-membership F1 on this hard pool. This is reported as a
baseline limitation, not a semantic-model claim.

## Pose false-negative audit

The evaluation-only audit checks whether the pose solver accepts groups when
their membership is supplied from GT:

| Parts per group | Groups | Accepted | Recall |
|---:|---:|---:|---:|
| 2 | 10 | 10 | 100% |
| 3 | 10 | 10 | 100% |
| 4 | 4 | 2 | 50% |
| 5 | 4 | 0 | 0% |
| 6 | 2 | 0 | 0% |
| **All** | **30** | **22** | **73.33%** |

Preserving parallel-face hypotheses and correcting collision-penalty scale
raised this audit from 33.33% to 73.33%. Remaining 4-6 part failures are
confirmed penetration outcomes from the current search result. Increasing
beam width alone did not resolve a representative five-part failure.

## Commands

Unlabeled inference:

```powershell
python run_agent_pipeline.py `
  --input <parts_folder> `
  --output <results_folder> `
  --semantic off
```

Frozen evaluation:

```powershell
python run_global_grouping.py mixed_pools_v1
python run_geometry_benchmark.py mixed_pools_v1
python audit_true_group_validation.py mixed_pools_v1
```

Tests:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s sw_dataset_generator/tests -v
python -m compileall -q .
```

Final check: 24 core tests and 2 generator tests pass. The unlabeled six-part
smoke run is stored in `geometry_demo_pool_001/` and completed without GT or an
API key.

## Gate for D6

The frozen geometry-only baseline is now complete. The next stage is the
optional provider-neutral semantic reviewer and bounded Agent controller.
DeepSeek must receive only structured candidate summaries, return
schema-validated JSON, support abstention/cache/replay, and remain unable to
override geometric failure.
