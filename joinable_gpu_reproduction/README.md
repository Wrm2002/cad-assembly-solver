# JoinABLe modern CUDA reproduction

This directory runs the released JoinABLe checkpoint and official preprocessed
Fusion 360 joint graphs without changing either artifact.

The release targeted Python 3.7, PyTorch 1.8, PyG 1.7.2, and CUDA 10.2.  That
CUDA stack predates the RTX 4070 Laptop GPU, so the reproducible local route is
a small compatibility adapter on PyTorch 2.5.1 + CUDA 12.4.  The adapter:

1. strictly loads all released checkpoint tensors;
2. migrates legacy PyG `Data` objects in memory;
3. preserves the release's model architecture and input feature set;
4. never rewrites the checkpoint or official pickle files.

Environment:

```powershell
D:\Model_match_envs\joinable_gpu\python.exe -m pip install `
  torch==2.5.1 torchvision==0.20.1 torch-geometric==2.6.1 `
  numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 `
  pytorch-lightning==1.9.5 torchmetrics==0.11.4 setuptools==80.9.0
```

Run the official test subset:

```powershell
D:\Model_match_envs\joinable_gpu\python.exe `
  joinable_gpu_reproduction\run_official_inference.py --limit 200
```

Run the bounded CUDA training benchmark:

```powershell
D:\Model_match_envs\joinable_gpu\python.exe `
  joinable_gpu_reproduction\benchmark_cuda_training.py
```

These experiments rank B-Rep joint entities for a known pair of parts. They do
not prove mixed-pool grouping or semantic assembly correctness.

## STEP transfer and bounded domain adaptation

The released checkpoint only requests
`entity_types,length,face_reversed,edge_reversed`.  The enhanced OCCT probe
emits these fields plus source hashes and topology signatures.  Run the
pretrained model on two extracted graphs with:

```powershell
D:\Model_match_envs\joinable_gpu\python.exe `
  ..\cad_assembly_agent\tools\joinable_interface_predictor\pretrained_joinable_predictor.py `
  --part-a-graph part_a.brep_graph.json `
  --part-b-graph part_b.brep_graph.json `
  --output prediction.json
```

`audit_official_step_transfer.py` compares source Fusion graphs and the same
parts re-imported from STEP.  Truth entities are matched geometrically; an
out-of-tolerance nearest entity is recorded for diagnosis but is never counted
as mapped truth.

The bounded adaptation pipeline deliberately avoids the 16 GB legacy pickle:

1. `prepare_domain_adaptation_subset.py` extracts only joint JSON, body JSON,
   and STEP from the official archive.
2. `extract_domain_step_graphs.py` runs every OCCT import in an isolated worker.
3. `build_domain_adaptation_manifest.py` maps labels and removes source-design
   leakage between train, validation, and test.
4. `finetune_step_domain.py` performs a small checkpoint fine-tune and selects
   epochs only on validation Top-10/Top-5/Top-1.

When the subset is moved from Windows to Linux, pass `--data-root` to resolve
the graph paths relative to the manifest's recorded `subset_root`; do not edit
individual sample paths. `run_autodl_training.sh` uses this mechanism and runs
multiple seeds. `aggregate_multiseed_results.py` never selects a checkpoint on
the test split.

The deterministic and pretrained candidate lists can be combined with
`merge_candidate_providers.py`.  The merger is shadow-only: every result still
requires pose, collision, multi-evidence, and group-consistency validation.
