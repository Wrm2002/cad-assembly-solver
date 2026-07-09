# D1–D3 Delivery Record

## Scope boundary

This delivery implements interface freezing, dataset/pool auditing, mixed-pool
construction, part indexing, and recall-oriented candidate prescreening. It
does not implement D4 global grouping, LLM semantics, or Agent decisions.

## D1 — Frozen interfaces

Artifacts:

- `contracts.py`: Pydantic source of truth for five public contracts.
- `schemas/*.schema.json`: generated JSON Schema documents.
- `pipeline_api.py`: stable façade for part extraction, pool indexing, and
  known-group solving.
- `configs/pool_pipeline.json`: versioned thresholds and search limits.
- `INTERFACE_CONTRACTS.md`: units, coordinates, IDs, and uncertainty rules.
- `baseline_freeze.json`: hashes, environment, versions, and known limits.

Conventions:

- length: millimetres;
- angle: degrees;
- feature coordinates: local STEP part frame;
- part ID: complete pool-local filename;
- uncertain hole/pattern detections: explicitly `heuristic`, never asserted as
  measured functional semantics.

Compatibility:

- existing BFS and reliable CLIs are unchanged;
- `pipeline_api.solve_known_group` successfully ran the one-part smoke case;
- generated PartFeature/CandidateEdge documents pass Pydantic validation.

## D2 — Dataset and mixed pools

Generation:

- 600/600 cases generated;
- 100 cases for each group size 1–6;
- zero final generation failures.

Supervision:

- STEP checks run in isolated child processes so an OCCT native crash is
  recorded per case instead of killing the full audit.
- GT geometry sampling compares transformed part bbox union and volume against
  the exported ground-truth assembly.

Detected issue:

- SolidWorks `AddComponent5` insertion coordinates place the generated
  primitive bbox center, while the original `gt.json` treated the insertion
  point as the local-origin translation.
- A two-part sample showed 17.1621 mm bbox error despite matching volume.
- The repair derives:
  `true_translation = insertion_point - local_part_bbox_center`.
- A copied smoke case passed afterward with 0 bbox error and approximately
  `2.4e-16` volume relative error.
- Bulk repair is isolated per case, idempotent, checkpointed, and retains both
  old/new transforms plus the original insertion point.

Mixed pools:

- 12 deterministic pools;
- 5–12 anonymous STEP parts per pool;
- 2–3 true groups plus distractors;
- filenames are deterministic hashes with no group label;
- `pool_gt.json` retains group membership, mates, placements, distractors, and
  provenance.

## D3 — Index and prescreen

Part index includes:

- authoritative OCCT bbox;
- volume and center of mass;
- OCCT inertia axes with explicit coordinate-axis fallback;
- planar and cylindrical faces;
- heuristic cylindrical-interface and equal-radius pattern candidates;
- source SHA-256 and extraction metadata.

Prescreen:

- all thresholds are read from `configs/pool_pipeline.json`;
- bbox scale is context, not sufficient evidence;
- cylinder, planar-area, and heuristic pattern evidence are recorded;
- top coarse neighbors are bounded and every rejection is logged;
- accepted pairs are passed through the existing detailed matcher, scorer, and
  auditable pruner.

Frozen 12-pool benchmark:

- prescreen true-pair recall: 100%;
- 12-part pools reduce 66 all-pairs comparisons to 47–49;
- 12-part pair reduction: 25.8%–28.8%;
- detailed typed-edge recall: approximately 79.4% overall;
- pruned typed-edge recall: approximately 63.2% overall.

The typed-edge gap is not hidden or attributed to prescreening. It identifies
legacy detailed-match/GT-semantic inconsistency for later correction.

## Commands

```powershell
python generate_schemas.py
python freeze_baseline.py

python sw_dataset_generator/audit_dataset.py synthetic_dataset_600 `
  --expected-per-group 100 --verify-step `
  --geometry-samples-per-group 3 `
  --output synthetic_dataset_600/audit_report.json

python build_mixed_pools.py synthetic_dataset_600 `
  --output-root mixed_pools_v1 --num-pools 12 --seed 20260702

python index_mixed_pools.py mixed_pools_v1
```

## Acceptance status

- D1 contracts and compatibility: implemented and tested.
- D2 generation: complete, 600/600.
- D2 GT transform repair: complete, including isolated retry of five transient
  OCCT worker crashes.
- D2 final audit: success; 600 valid cases, 100 per group, zero invalid cases,
  all STEP files reopened, and 18/18 sampled geometry checks passed.
- D2 mixed pools: rebuilt from repaired GT.
- D3 index/prescreen: implemented and benchmarked.
- D4 and later work: deliberately not started.

## Post-freeze addendum

This file is the historical D1-D3 checkpoint. Subsequent local-frame planar
matching corrections raised both generated and pruned typed-edge recall to
100% on the 12 frozen pools; mean pair reduction is approximately 11.5%.
D4-D5 are now implemented and measured in `D4_D5_DELIVERY.md`.
