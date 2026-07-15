# Reproducible GPU pipeline for the multi-positive / measured-hard-negative study.
# It intentionally uses Fusion360 only.  SolidWorks cases and screenshots are
# never read by this script and remain final evaluation / review material.

$ErrorActionPreference = 'Stop'
$root = 'C:\Users\11049\Desktop\Model_match'
$python = 'D:\Model_match_envs\joinable_gpu\python.exe'
$pure = 'D:\Model_match_public_data\fusion360_pure_brep_v1'
$equivalenceManifest = 'D:\Model_match_public_data\fusion360_pose_equivalence_v1\pose_equivalence_manifest.jsonl'
$dataset = 'D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1'
$runRoot = 'D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_train'

if (-not (Test-Path "$dataset\dataset_audit.json")) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\public_cad_dataset_audit\build_equivalent_pose_brep_chunks.ps1" `
        -Pure $pure -Manifest $equivalenceManifest -Output $dataset
    if ($LASTEXITCODE -ne 0) { throw 'dataset_build_failed' }
} else {
    Write-Host 'reusing_completed_audited_dataset'
}

& $python "$root\public_cad_dataset_audit\train_joinable_pose_heads.py" $dataset "$runRoot\pose_proposal" `
    --device cpu --epochs 30 --batch-size 256 --negatives 8
if ($LASTEXITCODE -ne 0) { throw 'pose_proposal_training_failed' }

& $python "$root\public_cad_dataset_audit\train_joinable_pose_heads.py" $dataset "$runRoot\interface_score" `
    --device cpu --epochs 30 --batch-size 128 --negatives 8 --patch-geometry --contact-target
if ($LASTEXITCODE -ne 0) { throw 'interface_score_training_failed' }

# Frozen evaluation only: these cases are intentionally not visible to either
# dataset construction or training.  The script produces JSON and renderings
# for case1/2 under their own new output directories.
& $python "$root\public_cad_dataset_audit\run_frozen_case12_exam.py" $runRoot `
    --runtime-python $python --occt-python 'C:\Users\11049\miniforge3\envs\cad311\python.exe'
if ($LASTEXITCODE -ne 0) { throw 'frozen_case12_exam_failed' }
