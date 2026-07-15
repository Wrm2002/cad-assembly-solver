$ErrorActionPreference = "Stop"

$python = "D:\Model_match_envs\joinable_gpu\python.exe"
$manifest = "D:\Model_match_public_data\fusion360_joint\domain_adapt_300\domain_adaptation_manifest.json"

& $python "joinable_gpu_reproduction\finetune_step_domain.py" `
  --manifest $manifest `
  --epochs 12 `
  --batch-size 1 `
  --learning-rate 3e-6 `
  --patience 4 `
  --device cpu `
  --torch-threads 4 `
  --train-scope full `
  --output-dir "joinable_gpu_reproduction\domain_finetune_cpu_low_lr"
if ($LASTEXITCODE -ne 0) {
  throw "low_lr experiment failed with exit code $LASTEXITCODE"
}

& $python "joinable_gpu_reproduction\finetune_step_domain.py" `
  --manifest $manifest `
  --epochs 12 `
  --batch-size 1 `
  --learning-rate 1e-5 `
  --patience 4 `
  --device cpu `
  --torch-threads 4 `
  --train-scope post_only `
  --output-dir "joinable_gpu_reproduction\domain_finetune_cpu_post_only"
if ($LASTEXITCODE -ne 0) {
  throw "post_only experiment failed with exit code $LASTEXITCODE"
}
