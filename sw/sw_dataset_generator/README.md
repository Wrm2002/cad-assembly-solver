# SolidWorks API synthetic dataset generator

> Legacy geometry-smoke generator only. Its primitive cylinder/ring/box/plate
> cases are not function-grounded positives. D0 functional data is generated
> by `../functional_dataset_generator.py`.

This generator creates programmatic synthetic CAD for controlled testing,
hard-negative construction, and later match-scorer training. It is not claimed
to solve arbitrary real-CAD generalization.

```powershell
python sw_dataset_generator/batch_generate.py --group-size 1 2 3 4 5 6 --num-cases 1 --dry-run
python sw_dataset_generator/batch_generate.py --group-size 1 --num-cases 1
```

Each case contains native SolidWorks documents, part STEP files,
`assembly_gt.step`, `gt.json`, and the exact randomized specification. The
current first implementation uses cylinders, annular bores, and boxes, with
near-radius and symmetry hard negatives. More detailed bolt patterns,
chamfers, keyways, and tolerances can be added without changing the output
schema.
