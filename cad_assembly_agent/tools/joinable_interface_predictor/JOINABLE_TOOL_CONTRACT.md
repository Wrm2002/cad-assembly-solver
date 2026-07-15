# Frozen JoinABLe Pair-Interface Tool Contract

Version: `1.0.0-frozen`

## Purpose

The tool ranks B-Rep entity pairs for **one already selected pair of parts**.
It reproduces the released JoinABLe checkpoint through a modern PyTorch/PyG
compatibility adapter.

## Inputs

- Two audited OCCT B-Rep graph JSON files.
- Each graph must report successful extraction and the minimal released-model
  feature set.
- Combined node count must not exceed 950.
- The released checkpoint is
  `joinable_migration_audit/vendor/JoinABLe/pretrained/paper/last_run_0.ckpt`.

## Outputs

The predictor always emits JSON.  Successful output contains ranked interface
candidates, entity identifiers, model scores, normalized probabilities,
mapping audit information, `failure_reasons`, and `unavailable_fields`.
Failure output contains an empty candidate list and an explicit reason.

## Frozen evidence

The official filtered test split contains 1,857 evaluated part pairs:

- Top-1 accuracy: 79.32%
- Top-5 recall: 87.56%
- Top-10 recall: 90.79%
- samples with no positive label: 0

The checkpoint is loaded with strict weight matching.  The checkpoint, adapter,
predictor, vendored model sources, and full evaluation report are locked by
SHA-256 in `frozen_tool_manifest.json`.

## Safety boundary

This tool does **not** decide whether two parts belong to the same real
assembly.  It does not prove a collision-free pose or functional correctness.
Its output remains shadow evidence until candidate recall, group consistency,
pose, collision, and conservative acceptance gates have all passed.

The official checkpoint is the frozen baseline.  Experimental STEP-adapted
checkpoints are not promoted because the independent test result did not
improve consistently.
