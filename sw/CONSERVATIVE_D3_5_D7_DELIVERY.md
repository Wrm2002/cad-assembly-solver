# D3.5-D7 Conservative Delivery

Date: 2026-07-02

> Historical snapshot. The P0 semantics repair on 2026-07-06 supersedes the
> review/rejected counts and workload-reduction interpretation in this file.
> Use `../P0_CONSERVATIVE_FREEZE.json` and the regenerated
> `data/results/` artifacts for the current state. In particular, candidates
> outside the bounded operator frontier now remain deferred review rather than
> being counted as rejected.

## Outcome

The system now behaves as a precision-first candidate recommendation tool. It
does not force a complete partition, does not treat a collision-free pose as
source-assembly truth, and does not permit DeepSeek to change grouping.

On the frozen 12-pool benchmark:

| Metric | Legacy final assignment | Conservative output |
|---|---:|---:|
| Automatically accepted groups | 40 | 0 |
| Exact true groups among accepts | 1 | 0 |
| False automatic accepts | 39 | 0 |
| Auto-accept precision | 2.5% | Not estimable (zero accepts) |
| Human-review groups | 0 | 288 |
| Review rate over proposals | 0% | 2.98% |
| Unresolved pool-local parts | 6 | 110 |
| Candidate workload reduction | Not reported | 97.02% |
| Rejected-reason coverage | Not reported | 100% |

Zero accepts must not be reported as 100% precision. The initial 90% precision
target remains statistically unverified.

Of the 39 legacy false positives, 24 are now routed to review and 15 are
rejected by the bounded review frontier. None remains automatically accepted.

## D3.5 — candidate recall audit

The audit emits:

- `missed_true_candidates.csv`;
- `pruned_true_candidates.csv`;
- `candidate_recall_by_type.json`;
- `candidate_recall_by_group_size.json`;
- `candidate_recall_audit.md`.

The current `mixed_pools_v1` index has 68 truth interfaces. At the
part-pair-plus-interface-type level, generation and post-pruning recall are
both 68/68 (100%). The earlier 79.4%/63.2% figures came from an older index
state and must not be presented as current frozen-v1 results.

Ground truth is read only by the audit. It is not exposed to candidate ranking
or production decisions.

## D4 — conservative geometry tiers

`global_optimizer/group_consistency.py` records:

- evidence and independent-evidence counts;
- interface coverage;
- weak single-interface matches;
- central-part topology;
- larger-group blocking;
- near-tied overlapping conflicts;
- a group-consistency score and explicit reasons.

The 9,668 frozen proposals are split into:

- 0 geometry candidates safe enough to proceed toward automatic acceptance;
- 9,168 geometry review candidates before review-frontier bounding;
- 500 direct geometry rejects.

After pose failure handling and the score-ranked, group-size-diverse review
frontier, the final operator queue contains 288 groups and the final reject
set contains 9,380 groups.

The frontier is label-independent. Truth is used only afterward for
evaluation. It retains 6/30 exact truth groups, so truth-group review recall is
20%; this low recall is an explicit known limitation.

## D5 — physical pose only

Complete-pose backtracking and OCCT Boolean Common collision checks are
retained. Outputs now distinguish `valid`, `failed`, and `uncertain`.

- bounded pose checks: 144;
- physically valid: 143;
- physically failed: 1;
- not checked/search-uncertain: 9,024.

Each checked record contains checked rank count, selected rank, per-rank
rejection reasons, residual, collision status, common volume, worker status,
and final pose status. A valid pose never directly creates an accepted group.
Groups of six parts are always routed to review.

## D6 — DeepSeek explanation-only study

The semantic gate now requires:

- AUC at least 0.70;
- Brier score better than the geometry baseline;
- `accept`, `reject`, and `abstain` verdicts;
- no holdout auto-accept precision decrease;
- no holdout false-positive increase.

The default application mode is `explanation_only`; utility overrides are not
applied.

A blinded study was generated from the same frozen pools:

- 30 exact source groups;
- 24 pose-valid provenance-wrong hard negatives;
- 54 DeepSeek reviews;
- 54 human-review STL files and 54 screenshots.

DeepSeek results against hidden source truth:

- AUC: 0.2833;
- Brier score: 0.3598;
- geometry Brier score: 0.4227;
- verdicts observed: accept, reject, abstain.

The slightly lower Brier score is insufficient because the ranking AUC is far
below 0.70 and human semantic labels/holdout safety are absent. Semantic
reranking remains disabled.

The desktop review package is:

`C:\Users\11049\Desktop\CAD语义人工复核_20260702_完成版`

The user should inspect `01_待人工判断`, fill
`人工语义标注表.csv`, and only afterward open the DeepSeek and source-truth
folders. Every STL was independently reopened with OCCT and passed shape
validity checks.

## D7 outputs

`data/results/` contains:

- candidate recall CSV/JSON/Markdown audit;
- accepted/review/rejected geometry candidates;
- pose-valid/failed/uncertain candidates;
- final accepted/review/rejected groups;
- unresolved parts;
- false-positive audit;
- conservative metrics and before/after comparison;
- candidate score audit;
- structured semantic inputs/reviews and gate report;
- assembly report.

`run_conservative_delivery.py` is the one-command entry point. DeepSeek
defaults to cache-only in that entry point, and a new empty human-pack path
must be supplied explicitly.

## Files added

- `candidate_recall_audit.py`
- `global_optimizer/__init__.py`
- `global_optimizer/group_consistency.py`
- `conservative_pipeline.py`
- `semantic_explanation_batch.py`
- `build_human_semantic_review_pack.py`
- `run_conservative_delivery.py`
- `freeze_conservative_delivery.py`
- `configs/conservative_pipeline.json`
- four new unit-test modules

## Files modified

- `geometry_pipeline.py`
- `agent_controller.py`
- `semantic_pool.py`
- `calibrate_semantic_review.py`
- `configs/pool_pipeline.json`
- `configs/semantic_calibration.json`
- `tests/test_agent_controller.py`

## Verification

- core tests: 42/42 passed;
- generator tests: 3/3 passed;
- complete Python compilation passed;
- 54/54 STL independent readbacks passed;
- 54/54 PNG files opened and verified;
- no SolidWorks generation task remains active.

## Next minimum step

The next action is not prompt tuning. First complete the human CSV. Then
compare human functional judgments, hidden source truth, and DeepSeek
decisions. If humans also cannot separate the provenance-wrong groups from
images, the required missing evidence is BOM/CAD metadata or functional role
data rather than a larger search beam or a more assertive language-model
prompt.
