# Planar Slide and Pocket Depth DOF Search Audit

Date: 2026-07-08

## Scope

This audit records the implementation and regression result for:

1. `planar_mate` in-plane slide search;
2. `pocket_mate` insertion-depth search.

The implementation is generic and does not change direct-edge selection.

## Implementation

File changed:

- `sw/known_group_assembly.py`

Added pose candidate families:

1. `planar_slide_dof_search`

   - Applies to selected direct connections with `planar_mate` or
     `planar_align` evidence.
   - Moves the smaller part in the tangent plane of the larger/stationary
     part's matched planar face.
   - Uses bounded offsets based on the smaller part's bbox diagonal.
   - Penalizes larger slide distance.

2. `pocket_depth_dof_search`

   - Applies to selected direct connections with `pocket_mate` evidence.
   - Moves the smaller part along the pocket direction and the opposite
     direction.
   - Uses bounded offsets derived from pocket size / depth proxy.
   - Penalizes larger insertion-depth movement.

3. Large STEP throttle

   - If total extracted surface count is above 3000, the expensive planar and
     pocket expansion is disabled.
   - This prevents case3-scale STEP files from exploding into hundreds of OCCT
     Boolean checks.

4. Final pose selection

   - Collision-free and fully-closed poses are preferred.
   - Among valid poses, the highest score is selected, with movement penalties
     discouraging unnecessary displacement.

## Full case1-5 retest

Command:

```powershell
.\.conda\python.exe sw\run_exam_v2.py
.\.conda\python.exe sw\export_exam_assembly_methods.py --output sw\exam_assembly_methods.json
```

Result:

| Case | Direct edge P/R/F1 | Pose status | Collision count | Checked poses | Selected origin |
|---|---|---|---:|---:|---|
| case_1 | 1.00 / 1.00 / 1.00 | valid | 0 | 20 | identity_input_pose |
| case_2 | 1.00 / 1.00 / 1.00 | valid | 0 | 109 | axial_slide_dof_search |
| case_3 | 1.00 / 1.00 / 1.00 | valid | 0 | 3 | solver_beam |
| case_4 | 1.00 / 1.00 / 1.00 | valid | 0 | 88 | identity_input_pose |
| case_5 | 1.00 / 1.00 / 1.00 | valid | 0 | 96 | identity_input_pose |

## Output

`sw/exam_assembly_methods.json` now reports:

- 5 cases;
- 9 direct assembly edges;
- `pose_status = valid` for all cases;
- `collision_count = 0` for all cases;
- no case-specific label override.

## Remaining limitations

This is still bounded search, not a complete constraint solver.  The new DOF
search handles the current planar and pocket ambiguities, but a harder case may
need:

- adaptive offset ranges;
- true contact-surface overlap maximization;
- cycle consistency across more than five parts;
- stronger pocket feature extraction.
