# CUDA / NVIDIA driver repair — 2026-07-07

## Diagnosis

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (compute capability 8.9)
- Previous driver: 596.49 Game Ready WHQL
- Python environment: PyTorch 2.5.1 + CUDA 12.4, PyG 2.6.1
- Failure: one JoinABLe CUDA batch ended with `CUDA error: an illegal instruction was encountered`.
- Windows System log contained 35 `nvlddmkm` errors and 11 corrected WHEA events in the preceding 30 days.
- The original failure could not be reproduced in three immediate retry batches, but the historical driver errors justified a clean driver replacement.

## Repair

- Exported the previous `oem47.inf` NVIDIA package to:
  `C:\Users\11049\Desktop\NVIDIA_Driver_Repair_20260707\backup`
- Downloaded the official NVIDIA Studio Driver 610.47 installer.
- Installer SHA256: `59AC4A1659664AAD0A6FC525E5DF99B3FA76887BDE663F9E36E0E7EBB5DBA937`
- Authenticode signature: valid; signer `NVIDIA Corporation`.
- Installed with clean-install and no-automatic-reboot options.
- Installed package: `oem10.inf` / `nvamsi.inf`, driver `32.0.16.1047`.

## Post-install validation

- Device Manager/PnP status: Started / OK
- `nvidia-smi`: 610.47, CUDA UMD 13.3
- Basic CUDA matrix multiplication and softmax: passed
- JoinABLe full mixed-pool CUDA inference: 273/273 pairs passed in three consecutive runs
- New post-install `nvlddmkm` events: 0
- New post-install WHEA events: 0

## Remaining action

Windows reports pending file rename operations. Restart Windows once before the next long GPU batch. If WHEA corrected-machine-check events recur after restart, reset Armoury Crate CPU/GPU tuning to stock and investigate platform stability; a display-driver reinstall cannot correct CPU parity errors.
