# Small STEP Assembly Reconstruction

This repository preserves the legacy geometry/BFS baseline and adds auditable
scoring, pruning, validation, reliable search, evaluation, and SolidWorks API
synthetic-data generation.

## Environment

```powershell
conda activate cad_asm
python check_env.py .
```

## Legacy-compatible baseline

```powershell
python compute_manifest.py <case>
python build_assembly.py <case>
```

## Scoring, pruning, reliable search, and validation

```powershell
python match_scoring.py <case>
python match_pruning.py <case> --min-score 0.5 --max-neighbors 4
python compute_manifest.py <case> --solver reliable --beam-width 20 --write-diagnostics
python placement_validation.py <case> --matches <case>/kept_matches.json
```

The default solver remains BFS. `--solver reliable` supports 1–6 parts and
uses scoring, auditable pruning, beam search, a branch upper bound, multiple
pose hypotheses, and explicit penalty details.

## Experiments

```powershell
python calibrate_scoring.py <dataset_root> --output-dir <calibration_dir>
python run_experiments.py <dataset_root>
python evaluate_results.py <dataset_root> --summary <summary.csv>
```

The four standard modes are:

1. `baseline_1_bfs`
2. `baseline_2_bfs_scoring`
3. `baseline_3_bfs_scoring_pruning`
4. `proposed_reliable`

## SolidWorks synthetic data

```powershell
python sw_dataset_generator/batch_generate.py `
  --group-size 1 2 3 4 5 6 --num-cases 100 `
  --output-root synthetic_dataset
```

Interrupted batches can continue without regenerating complete cases:

```powershell
python sw_dataset_generator/batch_generate.py `
  --group-size 1 2 3 4 5 6 --num-cases 100 `
  --output-root synthetic_dataset --resume --continue-on-error
```

Generated data is programmatic synthetic CAD for controlled testing,
hard-negative construction, and match-scorer experiments. Real STEP data
remains an external test set; no universal real-CAD generalization is claimed.

## Tests

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s sw_dataset_generator/tests -v
```

## D1-D3 pool pipeline

Generate versioned public schemas and freeze the geometry API:

```powershell
python generate_schemas.py
python freeze_baseline.py
```

Audit and build anonymous mixed pools:

```powershell
python sw_dataset_generator/audit_dataset.py synthetic_dataset_600 `
  --expected-per-group 100 --verify-step `
  --geometry-samples-per-group 3 `
  --output synthetic_dataset_600/audit_report.json

python build_mixed_pools.py synthetic_dataset_600 `
  --output-root mixed_pools_v1 --num-pools 12 --seed 20260702
```

Index and prescreen all frozen pools:

```powershell
python index_mixed_pools.py mixed_pools_v1
```

All prescreen thresholds are versioned in `configs/pool_pipeline.json`.
Rejected pairs and edges retain explicit audit reasons.

## D4-D5 geometry-only pool pipeline

Run a folder containing 2-12 STEP parts. Labels and API keys are not required:

```powershell
python run_agent_pipeline.py `
  --input <parts_folder> `
  --output <results_folder> `
  --semantic off
```

The command writes the part index, candidate/rejection audit, global group
proposals, validated assignment, per-group manifests and assembly STEP files,
exact OCCT penetration checks, retry events, and `run_summary.json`.

Frozen-pool evaluation:

```powershell
python run_global_grouping.py mixed_pools_v1
python run_geometry_benchmark.py mixed_pools_v1
python audit_true_group_validation.py mixed_pools_v1
```

`audit_true_group_validation.py` is evaluation-only: it measures pose-solver
false negatives and is prohibited from feeding group selection. CAD validation
runs one group per child process so an OCCT native fault is recorded as a
rejection rather than terminating the controller.

See `D4_D5_DELIVERY.md` for measurements and limitations. DeepSeek semantic
review is deliberately gated after this frozen no-LLM baseline.

## D5.1-D6 calibrated Agent

The later collision-backtracking and semantic-Agent results are in
`D5_1_D6_DELIVERY.md`.

```powershell
python semantic_review.py --smoke
python calibrate_semantic_review.py mixed_pools_v1 `
  --pools pool_001 pool_002 pool_003 --mode live
python run_agent_benchmark.py mixed_pools_v1

python audit_dataset_pose_validation.py `
  synthetic_dataset_600 pose_audit_4_5 `
  --group-size 4 5 --resume
```

Semantic reviews are schema-validated, cached, and calibration-gated. The
current anonymous synthetic summaries did not beat geometry on calibration, so
the frozen gate disables semantic influence and holdout API calls. Geometry-only
operation remains the default and requires no API key.

The expanded known-group pose audit runs 100 four-part and 100 five-part
cases with per-case process isolation and Wilson confidence intervals. See
`D5_1_D6_DELIVERY.md`; these rates apply to the tested generator families, not
to arbitrary real CAD or unknown pool grouping.
## 当前主入口：已知同组零件的装配关系识别

当输入的 1～5 个 STEP 零件已知属于同一装配体时，使用：

```powershell
python known_group_assembly.py .\1
```

可选接入已缓存的 JoinABLe 两两接口排序：

```powershell
python known_group_assembly.py .\1 --joinable-report path\to\pair_report.json
```

输出位于 `<case>/known_group_output/`：

- `assembly_relations.json`：主交付，包含直接连接、五类约束标签、接口、相对位姿和验证状态；
- `candidate_relations.json`：全部候选与 JoinABLe 审计；
- `pose_validation.json`：逐位姿约束闭合和 OCCT 碰撞结果；
- `assembly_manifest.json`：各零件全局位姿；
- `assembly.step`：重建装配体。

该入口明确假设所有输入零件属于同一装配体，不执行 mixed-pool 分组、来源判断或语义裁判。
