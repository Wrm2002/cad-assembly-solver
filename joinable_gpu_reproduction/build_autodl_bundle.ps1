param(
    [string]$ProjectRoot = "C:\Users\11049\Desktop\Model_match",
    [string]$DataRoot = "D:\Model_match_public_data\fusion360_joint\domain_adapt_300",
    [string]$StageRoot = "D:\Model_match_autodl_bundle_20260704",
    [string]$ZipPath = "C:\Users\11049\Desktop\JoinABLe_AutoDL_handoff_20260704.zip"
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $StageRoot) {
    throw "StageRoot already exists: $StageRoot"
}
if (Test-Path -LiteralPath $ZipPath) {
    throw "ZipPath already exists: $ZipPath"
}

New-Item -ItemType Directory -Path $StageRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $ProjectRoot "joinable_gpu_reproduction") `
    -Destination $StageRoot -Recurse
Copy-Item -LiteralPath (Join-Path $ProjectRoot "cad_assembly_agent") `
    -Destination $StageRoot -Recurse

$vendorParent = Join-Path $StageRoot "joinable_migration_audit\vendor"
New-Item -ItemType Directory -Path $vendorParent -Force | Out-Null
Copy-Item -LiteralPath (
    Join-Path $ProjectRoot "joinable_migration_audit\vendor\JoinABLe"
) -Destination $vendorParent -Recurse

$dataParent = Join-Path $StageRoot "data"
New-Item -ItemType Directory -Path $dataParent | Out-Null
Copy-Item -LiteralPath $DataRoot `
    -Destination (Join-Path $dataParent "domain_adapt_300") -Recurse

Compress-Archive -Path (Join-Path $StageRoot "*") `
    -DestinationPath $ZipPath -CompressionLevel Optimal

$hash = Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256
$files = Get-ChildItem -LiteralPath $StageRoot -Recurse -File
$bytes = ($files | Measure-Object Length -Sum).Sum
[pscustomobject]@{
    stage_root = $StageRoot
    zip_path = $ZipPath
    file_count = $files.Count
    uncompressed_bytes = $bytes
    zip_bytes = (Get-Item -LiteralPath $ZipPath).Length
    sha256 = $hash.Hash.ToLowerInvariant()
    completed_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content `
    -LiteralPath (Join-Path $StageRoot "bundle_manifest.json") `
    -Encoding UTF8
