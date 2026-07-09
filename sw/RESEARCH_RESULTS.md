# Small-Assembly Reconstruction Results

## Implemented scope

1. `match_scoring.py` produces an auditable score, confidence class, and
   type-specific reason fields. `calibrate_scoring.py` performs threshold
   sweeps and Platt probability calibration against `gt.json`.
2. `match_pruning.py` writes every retained and removed edge, including
   `low_score`, `exceeded_top_k_pair`, `exceeded_max_neighbors`,
   `weak_planar_only`, and `duplicate` reasons.
3. `placement_validation.py` checks constraint residuals, transformed bounding
   boxes, graph connectivity, unsolved/identity placements, and transform
   plausibility. Bounding-box overlap is reported as a conservative warning
   and is not mislabeled as exact solid penetration.
4. The legacy BFS remains the default. `small_assembly_solver.py` adds
   multi-hypothesis beam search, a branch upper bound, penalty breakdowns, and
   best-partial-solution reporting for one to six parts.
5. `run_experiments.py` runs all four ablations and records the requested
   per-case and per-group metrics.
6. `sw_dataset_generator/` creates native parts/assemblies, part STEP files,
   ground-truth assembly STEP, randomized parameters, hard-negative metadata,
   and `gt.json` through the SolidWorks API.

## Preliminary controlled evaluation

Dataset: one real SolidWorks-generated case for each group size 1 through 6.
This is a pipeline validation set, not a statistically sufficient benchmark.

| Mode | Validation success | STEP success | Mean match precision* | Mean match recall* | Mean max residual |
|---|---:|---:|---:|---:|---:|
| Original BFS | 6/6 | 6/6 | 0.284 | 0.713 | 0.0000 |
| BFS + scoring | 6/6 | 6/6 | 0.284 | 0.713 | 0.0000 |
| BFS + scoring + pruning | 4/6 | 6/6 | 0.363 | 0.713 | 90.9817 |
| Reliable solver | 6/6 | 6/6 | 0.363 | 0.713 | 0.0000 |

`*` Precision/recall means exclude the one-part case.

The controlled ablation shows the intended distinction: pruning increases
precision but can make first-hit BFS choose an inconsistent pose. The reliable
solver retains the pruned graph's precision while testing alternative poses
and restoring all six validation successes.

Full case metrics are in
`synthetic_experiments_final/experiment_cases.csv`; grouped results are in
`synthetic_experiments_final/experiment_group_summary.csv`.

## Score calibration

On the six-case smoke set, the best raw classification threshold is 0.90.
Platt scaling maps raw score `s` to:

`p(true mate) = sigmoid(4.632048 * s - 5.013661)`

This reduces in-sample Brier score from 0.4425 to 0.1447. The calibrated
probability threshold with best F1 is 0.30. The topology-safe pruning default
remains 0.50 on raw score because classification-optimal thresholding can
disconnect the candidate graph. These values must be re-estimated on the full
synthetic set and checked on a separate real external set.

Reports:

- `synthetic_calibration/calibration_report.json`
- `synthetic_calibration/threshold_sweep.csv`
- `synthetic_calibration/calibrated_threshold_sweep.csv`
- `synthetic_calibration/scored_candidates.csv`

## Formal synthetic dataset

The configured target is 100 cases for each group size (600 total). Generation
uses periodic SolidWorks-session recycling, per-case checkpoints, resume, and
continue-on-error support:

```powershell
python sw_dataset_generator/batch_generate.py `
  --group-size 1 2 3 4 5 6 --num-cases 100 `
  --output-root synthetic_dataset_600 --seed 20260702 `
  --resume --continue-on-error --session-batch-size 20
```

Synthetic data is for controlled testing, hard-negative construction, and
candidate scorer/search research. It is not evidence of universal
generalization to arbitrary real CAD. Real SolidWorks-exported data remains an
external, manually checked test set.
