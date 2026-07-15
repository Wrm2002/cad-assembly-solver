# SolidWorks bridge contract

This bridge is a final CAD boundary, not a new grouping subsystem.

## Read path

1. Open a native `SLDPRT` or exported STEP document read-only and silently.
2. Record body/face counts and stable document metadata.
3. Extract detailed analytic geometry with the existing OCCT descriptor tool.
4. Never rename, save, heal, simplify, or overwrite the source document.

## Pose path

The internal pose is a row-major 4x4 homogeneous transform in millimetres.
At the SolidWorks `MathTransform` boundary, rotational terms are unchanged and
translation is converted from millimetres to metres.  A transform is written
only after the bounded OCCT validator has marked the pose `valid`.

## Write path

A new `SLDASM` and exported STEP may be created only when a group is in
`final_accepted_groups.json`.  `review`, `rejected`, and `unresolved` inputs
must not create an assembly.  Source parts remain read-only and all generated
files use an isolated output directory.

## Current validation boundary

The installed SolidWorks instance is probed on one small STEP part.  The five
external groups have no native `SLDPRT` files, original mates, or original
component transforms, so those fields remain unavailable.  With zero final
accepted groups, assembly writeback is intentionally blocked.
