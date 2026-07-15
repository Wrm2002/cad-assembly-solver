$ErrorActionPreference = "Stop"

$python = "D:\Model_match_envs\joinable_gpu\python.exe"
$manifest = "D:\Model_match_public_data\fusion360_joint\domain_adapt_300\domain_adaptation_manifest.json"
$seeds = @(7, 17, 73)

foreach ($seed in $seeds) {
    $output = "joinable_gpu_reproduction\domain_finetune_cpu_seed_$seed"
    & $python "joinable_gpu_reproduction\finetune_step_domain.py" `
      --manifest $manifest `
      --epochs 12 `
      --batch-size 1 `
      --learning-rate 3e-6 `
      --patience 4 `
      --seed $seed `
      --device cpu `
      --torch-threads 4 `
      --train-scope full `
      --output-dir $output
    if ($LASTEXITCODE -ne 0) {
        throw "seed $seed failed with exit code $LASTEXITCODE"
    }
}
