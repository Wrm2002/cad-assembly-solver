"""Exercise CUDA initialization, forward/backward work, and clean shutdown."""

from __future__ import annotations

import json
import platform

import torch


def main() -> int:
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    left = torch.randn(2048, 2048, device=device, requires_grad=True)
    right = torch.randn(2048, 2048, device=device, requires_grad=True)
    output = (left @ right).square().mean()
    output.backward()
    torch.cuda.synchronize()
    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "loss": float(output.detach().cpu()),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 2**20, 2
        ),
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
