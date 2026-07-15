#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Model_match
export PYTHON=/root/autodl-tmp/envs/joinable/bin/python
export DATA_ROOT="${ROOT}/data/domain_adapt_2600"
export MANIFEST="${DATA_ROOT}/domain_adaptation_manifest.json"
export CHECKPOINT="${ROOT}/joinable_migration_audit/vendor/JoinABLe/pretrained/paper/last_run_0.ckpt"
export OUTPUT_ROOT="${ROOT}/joinable_gpu_reproduction/autodl_2600_runs"
export SEEDS="7 17 42 73"
export EPOCHS=20
export PATIENCE=5
export LEARNING_RATE=3e-6
export BATCH_SIZE=1

mkdir -p "${OUTPUT_ROOT}"
main_pid=$$
(
  while kill -0 "${main_pid}" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,temperature.gpu,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep 5
  done
) > "${OUTPUT_ROOT}/gpu_monitor.csv" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

cd "${ROOT}"
bash joinable_gpu_reproduction/run_autodl_training.sh
