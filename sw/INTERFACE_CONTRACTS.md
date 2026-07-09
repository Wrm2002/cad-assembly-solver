# D1 Public Interface Contracts

- API version: `1.0.0`
- Schema version: `1.0.0`
- Length unit: millimetre (`mm`)
- Angle unit: degree
- Coordinate frame: each STEP file's local part frame
- Part ID: stable pool-local identifier; it carries no functional meaning

Public Python entry points are in `pipeline_api.py`:

- `extract_part_feature(step_path, part_id=None)`
- `index_part_pool(parts_dir, output_dir, config_path=None)`
- `solve_known_group(case_dir, solver="reliable", ...)`

Later stages must consume the JSON documents defined in `schemas/`, not private
variables from the legacy geometry modules.

Geometry uncertainty is explicit:

- measured: directly obtained from OCCT geometry;
- heuristic: inferred by a documented geometric heuristic;
- unavailable: not computed and never fabricated.

The prescreen is recall-oriented. Rejection at this stage means no configured
coarse evidence was found; it is not a semantic claim that two parts cannot
belong to the same real machine.
