# Pose Repair Audit for Case 2/4/5

Date: 2026-07-08

## Goal

Fix the current pose failure on cases 2/4/5 without changing the already
working direct-edge selector.

## Problem observed

- Case 2 selected the correct direct edges but placed both flanges at nearly the
  same axial station on the shaft, causing solid intersections.
- Case 4 and case 5 lightweight files were already in a valid input pose, but
  the solver moved parts to a worse local planar solution and introduced
  collisions.

## Generic changes

1. Added `identity_input_pose` as a pose candidate for every known-group run.
   This protects already-posed STEP inputs from unnecessary re-placement.

2. Added generic axial-slide DOF candidates:
   - detect selected `coaxial` / `clearance` connections;
   - find a central part with a main cylindrical axis;
   - sample satellite translations along that axis;
   - score smaller total axial movement higher.

3. Changed final pose selection:
   - require collision-free and full constraint closure;
   - among valid poses, select highest scored pose rather than the first valid
     candidate.

## Regression result

| Case | Direct edges | Pose status | Collision count | Selected candidate |
|---|---|---|---:|---|
| case_2 | TP=3 FP=0 FN=0 | valid | 0 | axial_slide_dof_search |
| case_4_lightweight | TP=2 FP=0 FN=0 | valid | 0 | identity_input_pose |
| case_5_lightweight | TP=2 FP=0 FN=0 | valid | 0 | identity_input_pose |

Case 2 selected axial offsets:

- `flange_a_pipe_X_fan15deg.step`: `-113.77805650590628` mm along shaft axis
- `flange_b_pipe_Y_fan20deg.step`: `113.77805650590628` mm along shaft axis

The shaft and key remain at identity placement.

## Remaining limitation

This is still a bounded pose search, not a complete CAD constraint solver.  It
handles the important short-term failure modes:

- already-posed inputs;
- axial ambiguity for shaft/bore/flange relations.

It does not yet implement full planar in-plane sliding or general pocket
insertion-depth optimization.
