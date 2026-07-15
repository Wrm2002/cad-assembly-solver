# P0 Conservative Semantics Repair

Date: 2026-07-06

## Outcome

P0 is complete.  The two conservative pipelines now preserve uncertainty as
review, isolate evaluation truth from production artifacts, require explicit
collision success for automatic acceptance, and never choose a score-greedy
winner between overlapping auto-accept candidates.

Provider agreement is recorded as corroboration only.  JoinABLe and an
analytic rule reading the same STEP geometry no longer count as two
independent physical facts.

## Decision semantics

- `accepted`: passed every geometry, evidence, consistency, pose, collision,
  conflict, and group-size gate.
- `review / selected`: review-required and selected for the immediate bounded
  operator queue.
- `review / deferred`: still unresolved and review-required, but outside the
  immediate operator queue.
- `rejected`: hard geometry failure or a completed, definitive pose failure.
- Review-frontier capacity is never rejection evidence.
- Overlapping otherwise-acceptable groups all move to review; there is no
  greedy winner.

## Evaluation isolation

Production candidate and final-decision JSON files contain no
`evaluation_is_true_group`, `truth_group_id`, `source_assembly_id`, or
evaluation-only queue marker.

Evaluation data is stored separately:

- `public_cad_dataset_audit/outputs/phase5_group_search/group_proposal_truth_audit.json`
- `public_cad_dataset_audit/outputs/phase6_candidate_tiers/geometry_tiering_truth_audit.json`
- `public_cad_dataset_audit/outputs/phase7_pose_validation/pose_validation_truth_audit*.json`
- `public_cad_dataset_audit/data/results/final_decision_truth_audit.json`
- `sw/data/results/final_decision_truth_audit.json`

## Regenerated benchmark results

### Legacy frozen 12-pool benchmark

- Candidates: 9,668
- Accepted: 0
- Review-required: 9,167
- Immediate review frontier: 288
- Deferred review: 8,879
- Confirmed rejected: 501
- Unresolved pool-local parts: 110
- Automatic false positives: 0
- Accepted precision: not estimable
- Workload reduction: not established
- Review-frontier compression: 97.02%

The historical legacy assignment had 40 automatic accepts, of which 39 were
source-truth false positives.  P0 does not reinterpret the synthetic
source-group label as functional truth.

### Real Fusion mixed-pool 6-pool benchmark

- Parts: 49
- Candidates: 1,191
- Accepted: 0
- Review-required: 1,190
- Immediate review frontier: 11
- Deferred review: 1,179
- Confirmed rejected: 1
- Unresolved parts: 49
- Automatic false positives: 0
- Accepted precision: not estimable
- Workload reduction: not established
- Review-frontier compression: 99.08%

Fifteen of sixteen source-provenance groups remain review-required; two are in
the immediate review frontier.  Source provenance is still not functional
validity truth.

## Verification

- `sw/tests`: 51/51 passed.
- `public_cad_dataset_audit/tests`: 13/13 passed.
- `sw/sw_dataset_generator/tests`: 3/3 passed.
- P0 delivery invariants: 26/26 passed.
- Semantic reranking remains disabled and no provider call was made.
- Frozen hashes and validation details:
  `P0_CONSERVATIVE_FREEZE.json`.

## Files changed

- `sw/global_optimizer/group_consistency.py`
- `sw/conservative_pipeline.py`
- `sw/tests/test_group_consistency.py`
- `sw/tests/test_conservative_pipeline.py`
- `sw/CONSERVATIVE_D3_5_D7_DELIVERY.md`
- `sw/D7_CONSERVATIVE_FREEZE.json`
- `public_cad_dataset_audit/search_real_mixed_pool_groups.py`
- `public_cad_dataset_audit/tier_real_group_candidates.py`
- `public_cad_dataset_audit/run_real_pose_validation.py`
- `public_cad_dataset_audit/evaluate_conservative_real_benchmark.py`
- `public_cad_dataset_audit/tests/test_conservative_real_pipeline.py`

## Files added

- `public_cad_dataset_audit/freeze_p0_delivery.py`
- `P0_CONSERVATIVE_DELIVERY.md`
- `P0_CONSERVATIVE_FREEZE.json`

## Boundary

P0 does not repair D0.  The current synthetic functional dataset still
contains primitive geometric stacking and cannot validate functional grouping
or semantic correctness.  D0 remains the next stage.
