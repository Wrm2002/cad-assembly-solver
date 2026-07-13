# Continue from a completed Pair Pose checkpoint after an optional mesh/SDF
# import interrupted the Interface Score training stage.
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\11049\Desktop\Model_match'
$python = 'D:\Model_match_envs\joinable_gpu\python.exe'
$occtPython = 'C:\Users\11049\miniforge3\envs\cad311\python.exe'
$dataset = 'D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1'
$runRoot = 'D:\Model_match_public_data\fusion360_equivalent_pose_brep_hard_negative_v1_train'

if (-not (Test-Path "$runRoot\pose_proposal\best.pt")) { throw 'completed_pair_pose_checkpoint_missing' }

& $python "$root\public_cad_dataset_audit\train_joinable_pose_heads.py" $dataset "$runRoot\interface_score" `
    --device cpu --epochs 30 --batch-size 128 --negatives 8 --patch-geometry --contact-target
if ($LASTEXITCODE -ne 0) { throw 'interface_score_training_failed' }

& $python "$root\public_cad_dataset_audit\run_frozen_case12_exam.py" $runRoot `
    --runtime-python $python --occt-python $occtPython
if ($LASTEXITCODE -ne 0) { throw 'frozen_case12_exam_failed' }
