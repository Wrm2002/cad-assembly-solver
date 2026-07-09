# Pure Geometry Pocket Mate Audit

Date: 2026-07-08

## Scope

This audit records the change from short-term label overrides / filename-token
heuristics to generic geometry-based assembly-method inference for cases 1-5.

## Changes

1. Filename-token label inference was removed from `known_group_assembly.py`.
   No `key/module/cpu/memory/fan`-style token is used to decide relation labels.

2. `DirectAssemblyConnection` now carries:
   - `assembly_method_relation_types`
   - `assembly_method_reason`

3. `pocket_mate` is accepted only through geometry gates:
   - selected pair has planar seating/alignment support;
   - pocket candidate has non-degenerate dimensions;
   - size compatibility passes threshold;
   - direction or wall-normal orientation support passes threshold;
   - insertion-depth proxy is within a reasonable range.

4. If explicit pocket detection misses a slot, a weak pure-geometry fallback can
   add `pocket_mate` only when:
   - the pair is non-axial;
   - planar evidence exists;
   - one part is materially smaller by bbox diagonal;
   - the matched planar evidence is localized rather than broad flange contact.

## Regression Result

Command:

```powershell
.\.conda\python.exe sw\run_exam_v2.py
.\.conda\python.exe sw\export_exam_assembly_methods.py --output sw\exam_assembly_methods.json
```

Direct-edge recovery:

| Case | Direct-edge result | Pose diagnostic |
|---|---|---|
| case_1 | TP=1 FP=0 FN=0 | valid |
| case_2 | TP=3 FP=0 FN=0 | failed |
| case_3 | TP=1 FP=0 FN=0 | valid |
| case_4 | TP=2 FP=0 FN=0 | failed |
| case_5 | TP=2 FP=0 FN=0 | failed |

Assembly-method labels exported in `sw/exam_assembly_methods.json`:

| Case | Pair | Relation labels |
|---|---|---|
| case_1 | flange A - flange B | coaxial + planar_mate |
| case_2 | flange A - shaft | clearance |
| case_2 | flange B - shaft | clearance |
| case_2 | key - shaft | pocket_mate + planar_mate |
| case_3 | fan cage - fan module | pocket_mate + planar_mate |
| case_4 | PCBA - memory module | pocket_mate + planar_mate |
| case_4 | PCBA - CPU | pocket_mate + planar_mate |
| case_5 | chassis ear - chassis | coaxial + planar_mate |
| case_5 | chassis - PSU | pocket_mate + planar_mate |

## Important limitation

This is not a full physical-pose solution.  Cases 2/4/5 still have failed pose
diagnostics because the current placement solver can satisfy local constraints
while producing solid intersections.  The current milestone is relation-method
recognition, not collision-free assembly placement.
