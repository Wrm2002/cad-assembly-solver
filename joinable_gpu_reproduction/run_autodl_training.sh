#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/autodl-tmp/envs/joinable/bin/python ]]; then
    PYTHON=/root/autodl-tmp/envs/joinable/bin/python
  elif command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  else
    echo "No Python interpreter found. Set PYTHON explicitly." >&2
    exit 127
  fi
fi
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/domain_adapt}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/domain_adaptation_manifest.json}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/joinable_migration_audit/vendor/JoinABLe/pretrained/paper/last_run_0.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/joinable_gpu_reproduction/autodl_runs}"
SEEDS="${SEEDS:-7 17 42 73}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
BATCH_SIZE="${BATCH_SIZE:-1}"

test -f "${MANIFEST}"
test -f "${CHECKPOINT}"
nvidia-smi
"${PYTHON}" - <<'PY'
import torch
import torch_geometric
print(
    {
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
)
assert torch.cuda.is_available(), "CUDA is required for this launcher"
PY

mkdir -p "${OUTPUT_ROOT}"
REPORTS=()
for SEED in ${SEEDS}; do
  RUN_DIR="${OUTPUT_ROOT}/seed_${SEED}"
  "${PYTHON}" "${ROOT}/joinable_gpu_reproduction/finetune_step_domain.py" \
    --manifest "${MANIFEST}" \
    --data-root "${DATA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --patience "${PATIENCE}" \
    --seed "${SEED}" \
    --device cuda \
    --train-scope full \
    --output-dir "${RUN_DIR}" \
    2>&1 | tee "${OUTPUT_ROOT}/seed_${SEED}.log"
  REPORTS+=("${RUN_DIR}/finetune_report.json")
done

"${PYTHON}" \
  "${ROOT}/joinable_gpu_reproduction/aggregate_multiseed_results.py" \
  "${REPORTS[@]}" \
  --output "${OUTPUT_ROOT}/multiseed_summary.json"
