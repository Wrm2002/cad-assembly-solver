#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/model_match
DATA=/workspace/data/fusion360_equiv_v1
OUT=/workspace/output/equivalent_pose_brep_v1_train
PY=/root/miniconda3/bin/python

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/sw${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$ROOT/public_cad_dataset_audit/train_joinable_pose_heads.py" "$DATA" "$OUT/pose_proposal" \
  --device cuda --epochs 30 --batch-size 512 --negatives 8

"$PY" "$ROOT/public_cad_dataset_audit/train_joinable_pose_heads.py" "$DATA" "$OUT/interface_score" \
  --device cuda --epochs 30 --batch-size 512 --negatives 8 --patch-geometry --contact-target
