#!/usr/bin/env bash
set -euo pipefail

WORK=/root/autodl-tmp/model_match_cloud
ENV_ROOT=/root/autodl-tmp/envs/joinable
mkdir -p "${WORK}" /root/autodl-tmp/envs

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  /root/miniconda3/bin/conda create \
    -p "${ENV_ROOT}" python=3.10 pip -y
fi

"${ENV_ROOT}/bin/python" -m pip install --upgrade pip
"${ENV_ROOT}/bin/python" -m pip install \
  torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

"${ENV_ROOT}/bin/python" - <<'PY'
import torch

print(
    {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "capability": (
            torch.cuda.get_device_capability(0)
            if torch.cuda.is_available()
            else None
        ),
    }
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
