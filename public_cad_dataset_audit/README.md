# Public CAD Dataset Audit

This directory is isolated from the legacy SolidWorks/STEP pipeline. It audits
public assembly datasets only. It does not generate synthetic CAD, train a
neural network, or read the user's private SolidWorks cases.

Generated public-data downloads are kept under `data/` and third-party source
snapshots under `vendor/`; neither is an input to the legacy `sw/` project.

The empirical decision report is
`outputs/public_dataset_decision_report.json`; the human-readable findings are
in `PUBLIC_CAD_DATASET_AUDIT.md`.

## Frozen Step 1-3 delivery

The conservative Step 1-3 delivery is documented in
`PHASE123_DELIVERY_20260705.md` and
`outputs/phase123_delivery_report.json`.

- The official JoinABLe pair-interface tool is frozen under
  `cad_assembly_agent/tools/joinable_interface_predictor/`.
- Strict Fusion graph audit outputs are under
  `outputs/phase123_real_assembly_dataset/`.
- Anonymized real mixed-pool STEP files are stored outside the repository at
  `D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705`.

Reproducible commands:

```powershell
python public_cad_dataset_audit\audit_assembly_graph_quality.py `
  public_cad_dataset_audit\outputs\fusion360_assembly_graphs `
  --output-dir public_cad_dataset_audit\outputs\phase123_real_assembly_dataset

python public_cad_dataset_audit\build_real_mixed_pools.py `
  public_cad_dataset_audit\outputs\phase123_real_assembly_dataset\assembly_dataset_manifest.json `
  --output-root D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705

python public_cad_dataset_audit\validate_real_mixed_pools.py `
  D:\Model_match_public_data\fusion360_mixed_pools_real_v1_20260705 `
  --output public_cad_dataset_audit\outputs\phase123_real_assembly_dataset\mixed_pool_validation_report.json
```

The mixed-pool builder intentionally refuses to overwrite a non-empty output
directory.  Cross-assembly negatives are provenance negatives only; verified
geometric and semantic hard negatives remain unavailable until later physical
and functional review.
