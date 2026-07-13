# Failure-isolated GPU construction.  CUDA illegal access fails one chunk and
# stops the pipeline rather than silently converting remaining samples to skips.
param(
    [string]$Pure = 'D:\Model_match_public_data\fusion360_pure_brep_v1',
    [string]$Manifest = 'D:\Model_match_public_data\fusion360_pose_equivalence_v1\pose_equivalence_manifest.jsonl',
    [string]$ChunkRoot = 'D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_chunks',
    [string]$Output = 'D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1',
    # A few unusually large B-Rep graphs can trigger an upstream PyG/CUDA
    # illegal-access bug after prolonged graph processing.  Fifty records
    # keeps every worker well below the observed failure window; completed
    # larger chunks remain valid and are reused by the ordered merger.
    [int]$ChunkSize = 50
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\11049\Desktop\Model_match'
$python = 'D:\Model_match_envs\joinable_gpu\python.exe'
New-Item -ItemType Directory -Force -Path $ChunkRoot | Out-Null
foreach ($split in @('train','dev','test')) {
    $count = (Get-Content "$Pure\fusion360_pure_brep_$split.jsonl" | Measure-Object -Line).Lines
    $start = 0
    while ($start -lt $count) {
        # Resume from the actual recorded end, not from the current chunk-size
        # grid.  This permits safe continuation after an earlier 200-record
        # run is followed by finer 50-record chunks.
        $completed = Get-ChildItem -Path $ChunkRoot -Directory -Filter "$split`_*" | ForEach-Object {
            $audit = Join-Path $_.FullName 'dataset_audit.json'
            if (Test-Path $audit) {
                $row = (Get-Content $audit -Raw | ConvertFrom-Json).splits.$split.source_record_range
                [PSCustomObject]@{ Start = [int]$row.start; End = [int]$row.end_exclusive }
            }
        } | Where-Object { $_.Start -eq $start } | Select-Object -First 1
        if ($null -ne $completed) {
            $start = $completed.End
            continue
        }
        $chunk = Join-Path $ChunkRoot ("{0}_{1:D6}" -f $split, $start)
        if (Test-Path $chunk) { Remove-Item -LiteralPath $chunk -Recurse -Force }
        & $python "$root\public_cad_dataset_audit\build_equivalent_pose_brep_hard_negative_dataset.py" $Pure $chunk `
            --device cuda --equivalence-manifest $Manifest --splits $split --record-start $start --limit-records $ChunkSize --progress-every 50
        if ($LASTEXITCODE -ne 0) {
            # A small number of graphs trigger an upstream PyG CUDA kernel
            # fault.  Re-run the *same source range* on CPU; this is a device
            # fallback, not a sample skip.  The per-chunk audit records which
            # device created the tensors.
            Write-Host "gpu_chunk_failed_cpu_fallback split=$split start=$start" -ForegroundColor Yellow
            Remove-Item -LiteralPath $chunk -Recurse -Force -ErrorAction SilentlyContinue
            & $python "$root\public_cad_dataset_audit\build_equivalent_pose_brep_hard_negative_dataset.py" $Pure $chunk `
                --device cpu --equivalence-manifest $Manifest --splits $split --record-start $start --limit-records $ChunkSize --progress-every 50
            if ($LASTEXITCODE -ne 0) {
                # A Windows Python import occasionally races immediately after
                # a faulted CUDA worker exits.  One clean-process retry is
                # deterministic enough to distinguish that transient from a
                # genuinely unreadable B-Rep record.
                Start-Sleep -Seconds 3
                Remove-Item -LiteralPath $chunk -Recurse -Force -ErrorAction SilentlyContinue
                & $python "$root\public_cad_dataset_audit\build_equivalent_pose_brep_hard_negative_dataset.py" $Pure $chunk `
                    --device cpu --equivalence-manifest $Manifest --splits $split --record-start $start --limit-records $ChunkSize --progress-every 50
                if ($LASTEXITCODE -ne 0) { throw "chunk_failed_on_gpu_and_cpu_$split`_$start" }
            }
        }
        # The loop's next pass reads the just-written range from its audit.
    }
}
& $python "$root\public_cad_dataset_audit\merge_equivalent_pose_brep_chunks.py" $ChunkRoot $Output
if ($LASTEXITCODE -ne 0) { throw 'chunk_merge_failed' }
