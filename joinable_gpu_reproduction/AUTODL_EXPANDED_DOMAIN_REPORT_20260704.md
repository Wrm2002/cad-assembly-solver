# AutoDL Expanded STEP-Domain Experiment

Date: 2026-07-04

## Decision

The expanded STEP-domain experiment completed successfully, but the adapted
checkpoint is **not promoted**.

- Validation exact Top-10 improved consistently across all four seeds.
- Independent test exact Top-10 improved for only two of four seeds.
- Mean test Top-10 changed by -0.34 percentage points.
- Mean test Top-5 changed by -1.47 percentage points.
- `safe_to_promote_adapted_checkpoint=false`.

The larger dataset substantially reduced the small-set test degradation, but
did not establish a stable held-out gain.

## Dataset

Requested source joint sets:

- train: 2000;
- validation: 300;
- test: 300.

Extracted and audited:

- unique STEP bodies: 4044;
- OCCT graphs: 4044/4044 successful;
- graph extraction failures: 0;
- canonical cached re-audit: 4044/4044 successful;
- exact entity-side mapping rate: 84.6100%;
- external failures: 0.

After source-design leakage filtering:

- training-usable train: 695;
- training-usable validation: 151;
- training-usable test: 221;
- evaluation loader validation: 164;
- evaluation loader test: 236;
- train/validation/test source-design overlap: 0/0/0.

The large reduction from 2000 requested train rows to 695 usable rows is a
known data-splitting inefficiency, not a model failure. Test and validation
designs are given priority, and 998 selected train rows are excluded because
their source designs also occur in held-out splits.

## Environment

- AutoDL RTX 5090 D, 32 GB;
- Ubuntu 22.04;
- driver 580.105.08;
- Python 3.10.20;
- PyTorch 2.7.0+cu128;
- torchvision 0.22.0+cu128;
- torch-geometric 2.6.1;
- compute capability 12.0.

Three independent CUDA matrix/backward/shutdown probes passed. The official
checkpoint also passed a complete one-epoch expanded-data smoke test before
the formal run.

## Formal four-seed result

Official checkpoint baseline:

- validation exact Top-10: 47.02%;
- test exact Top-10: 55.66%.

| Seed | Best epoch | Validation Top-10 | Validation delta | Test Top-10 | Test delta |
|---:|---:|---:|---:|---:|---:|
| 7 | 9 | 54.97% | +7.95 pp | 54.30% | -1.36 pp |
| 17 | 6 | 54.97% | +7.95 pp | 57.01% | +1.36 pp |
| 42 | 7 | 55.63% | +8.61 pp | 56.56% | +0.90 pp |
| 73 | 9 | 55.63% | +8.61 pp | 53.39% | -2.26 pp |

Mean deltas:

- validation Top-1: -2.15 pp;
- validation Top-5: +4.97 pp;
- validation Top-10: +8.28 pp;
- validation Top-20: +0.50 pp;
- test Top-1: -0.68 pp;
- test Top-5: -1.47 pp;
- test Top-10: -0.34 pp;
- test Top-20: +0.34 pp.

Only two of four seeds had non-degraded test Top-10. This fails the promotion
gate even though validation improved in four of four seeds.

## Runtime and stability

- formal run wall-clock monitoring interval: about 5 minutes 23 seconds;
- completed epochs per seed: 14, 11, 12, 14;
- training failures: 0;
- GPU samples: 65 at five-second intervals;
- maximum temperature: 49 C;
- peak used memory: 1089 MiB;
- peak power: 134.3 W.

No NVIDIA driver reset, illegal instruction, OOM, or non-finite loss occurred.

## Artifact integrity

Expanded cloud payload:

- file:
  `C:\Users\11049\Desktop\JoinABLe_domain_adapt_2600_linux_20260704.zip`;
- SHA-256:
  `9dfcd0a2c8acb24b5edbab00d213b1c6aa3969ad097fe9ac08e3cc5f5b2655c0`;
- entries: 4048;
- CRC: passed.

Downloaded formal result bundle:

- file:
  `joinable_gpu_reproduction/autodl_2600_results_20260704.zip`;
- SHA-256:
  `9085a671c2fe49f3d14c34686f39b3ece17ea91046e993b6faf403dbdaac9d14`;
- entries: 28;
- CRC: passed;
- remote and local hashes match.

## Important fixes made during the run

1. The base image had Miniconda but no activated Python or framework.
2. A Python 3.10 / PyTorch 2.7 / CUDA 12.8 environment was created on the
   data disk.
3. `matplotlib` and `trimesh` were added to the permanent AutoDL requirement
   list because the released JoinABLe module imports them eagerly.
4. The cloud launcher now discovers the data-disk Python interpreter.
5. The SSH client pins the server host key, retries banner failures, avoids
   password command-line arguments, and limits SFTP prefetch concurrency.
6. The OCCT extractor supports disjoint shards and was canonically re-audited.
7. The manifest builder now defaults output relative to `--subset-root`;
   this prevents a new manifest from overwriting the 300-pair manifest.
8. The accidentally overwritten 300-pair manifest was restored byte-for-byte
   from the validated handoff bundle, and both dataset roots were verified.

## Next minimum experiment

Do not increase epochs or beam width.

First change data splitting to group by source-design/component before
sampling. The current post-selection leakage filter discards 998 train rows.
Build a split with roughly 2000 genuinely usable train samples while keeping
all source-design overlap at zero, then repeat the same four-seed protocol.

Promotion still requires:

- positive mean independent-test Top-10;
- non-degraded test Top-10 in at least three of four seeds;
- no test Top-5 regression;
- no new failures;
- no mixed-pool or semantic correctness claims from interface ranking alone.

Until then, keep the official checkpoint and the deterministic rule provider
as a review-only candidate union.
