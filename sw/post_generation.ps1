param(
    [Parameter(Mandatory = $true)]
    [int]$GenerationProcessId
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$conda = Join-Path $HOME "miniforge3\condabin\conda.bat"
$statusPath = Join-Path $root "post_generation_status.json"

function Write-Status([string]$stage, [string]$status, [string]$detail = "") {
    @{
        stage = $stage
        status = $status
        detail = $detail
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

function Run-Conda([string[]]$arguments, [string]$stdoutName, [string]$stderrName) {
    $process = Start-Process `
        -FilePath $conda `
        -ArgumentList $arguments `
        -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $root $stdoutName) `
        -RedirectStandardError (Join-Path $root $stderrName) `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    return $process.ExitCode
}

try {
    Write-Status "generation" "waiting"
    Wait-Process -Id $GenerationProcessId -ErrorAction SilentlyContinue

    Write-Status "repair_resume" "running"
    $repair = Run-Conda @(
        "run", "--no-capture-output", "-n", "cad_asm", "python",
        "sw_dataset_generator\batch_generate.py",
        "--group-size", "1", "2", "3", "4", "5", "6",
        "--num-cases", "100",
        "--output-root", "synthetic_dataset_600",
        "--seed", "20260702",
        "--resume", "--continue-on-error",
        "--session-batch-size", "5"
    ) "synthetic_600_repair.stdout.log" "synthetic_600_repair.stderr.log"
    if ($repair -ne 0) { throw "repair resume exited with $repair" }

    Write-Status "audit" "running"
    $audit = Run-Conda @(
        "run", "--no-capture-output", "-n", "cad_asm", "python",
        "sw_dataset_generator\audit_dataset.py",
        "synthetic_dataset_600",
        "--expected-per-group", "100",
        "--output", "synthetic_dataset_600\audit_report.json"
    ) "synthetic_600_audit.stdout.log" "synthetic_600_audit.stderr.log"
    if ($audit -ne 0) { throw "dataset audit exited with $audit" }

    Write-Status "calibration" "running"
    $calibration = Run-Conda @(
        "run", "--no-capture-output", "-n", "cad_asm", "python",
        "calibrate_scoring.py", "synthetic_dataset_600",
        "--output-dir", "synthetic_calibration_600"
    ) "synthetic_calibration_600.stdout.log" "synthetic_calibration_600.stderr.log"
    if ($calibration -ne 0) { throw "calibration exited with $calibration" }

    Write-Status "experiments" "running"
    $experiments = Run-Conda @(
        "run", "--no-capture-output", "-n", "cad_asm", "python",
        "run_experiments.py", "synthetic_dataset_600",
        "--output-dir", "synthetic_experiments_60",
        "--max-cases-per-group", "10"
    ) "synthetic_experiments_60.stdout.log" "synthetic_experiments_60.stderr.log"
    if ($experiments -ne 0) { throw "experiments exited with $experiments" }

    Write-Status "complete" "success"
}
catch {
    Write-Status "failed" "error" $_.Exception.Message
    throw
}
