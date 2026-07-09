# Agent-based CAD Matching Technical Roadmap

Start date: 2026-07-02  
Target v1 completion: 2026-10-11

## Positioning

The project has two nested problems:

1. Within-group reconstruction: infer mates and placements for a known group of
   one to six parts.
2. Part-pool discovery: partition an unordered pool of 5–30 parts into
   plausible assembly groups, then solve each group.

Geometry remains the source of feasibility and final correctness. Semantic
models may rerank ambiguous candidates but cannot override geometric failure.
The Agent orchestrates tools, retries, diagnostics, and reporting; it is not a
replacement for the geometry solver.

## Milestones

| ID | Dates | Focus | Main deliverables | Exit criteria |
|---|---|---|---|---|
| M0 | Jul 2–5 | Freeze the reproducible within-group baseline | Audited 600-case dataset; scoring/pruning/search/validation code; environment and incident records | 100 valid cases for every group size; zero missing native/STEP/GT files; all tests pass |
| M1 | Jul 6–19 | Correct and diversify synthetic mechanics | Strict GT schema; shaft-hole, flange, cover/chassis families; pool generator with distractors | At least 2 independent structure families per relevant group size; GT placements reproduce the exported assembly |
| M2 | Jul 20–Aug 2 | Build part index and coarse retrieval | Unified `part_features.json`; volume, center of mass, principal axes, holes and simplified flange patterns; pair prescreen | A 30-part pool can be indexed and reduced to a bounded candidate set without losing more than 5% of true edges |
| M3 | Aug 3–16 | Geometry-only global grouping baseline | Candidate graph; conflict model; set-packing/CP-SAT or equivalent optimizer; `final_groups.json` | Every part appears in at most one incompatible group; group precision/recall and rejection reasons are reported |
| M4 | Aug 17–30 | Integrate reliable pose solving and validation | Group proposal → beam/BnB pose search → residual/collision validation; fallback to next grouping | Complete groups require connected graph, solved placements, bounded residual, and no confirmed severe penetration |
| M5 | Aug 31–Sep 13 | Add optional semantic verification | Provider-neutral semantic interface; DeepSeek adapter; JSON schema validation; cache/replay; abstention | Pipeline works offline without LLM; semantic module is called only for ambiguous candidates; ablation shows measured effect |
| M6 | Sep 14–27 | Agent workflow control | State-machine planner, executor, retry policy, trace log, Markdown report | Agent can diagnose no-candidate, over-pruning, collision, and grouping-conflict failures and perform bounded retries |
| M7 | Sep 28–Oct 11 | External evaluation and v1 handoff | Manually labelled real CAD external set; complete ablations; technical report and one-command demo | Geometry-only, semantic-assisted, and Agent-assisted modes are compared on frozen data; limitations are explicit |

## Stage details

### M0 — Current baseline

- Complete the running SolidWorks generation only.
- Do not automatically launch experiments before the technical route is
  confirmed.
- Audit every case for:
  - native parts and native assembly;
  - part STEP and assembly ground-truth STEP;
  - `gt.json` and generated parameters;
  - expected part count and non-empty files.
- Preserve BFS as the fast baseline.
- Preserve the reliable beam/branch-and-bound solver as the group solver.

### M1 — Synthetic-data quality

- Replace the single incremental family with several actual mechanical
  families.
- Make the GT feature semantics correspond to real generated geometry.
- Add controlled negatives:
  - near-radius cylinders;
  - similar planar faces;
  - bolt-pattern distractors;
  - rotational symmetry;
  - unrelated parts;
  - geometrically feasible but semantically wrong pairs.
- Add mixed pools containing multiple assemblies and distractors.
- Introduce `pool_gt.json` with group membership, true mates, placements, and
  distractor labels.

### M2–M4 — Deterministic core

- Use coarse retrieval before detailed feature matching.
- Keep every removed candidate auditable.
- Separate three scores:
  - geometric compatibility;
  - pose-validation quality;
  - optional semantic plausibility.
- Do not use a fixed semantic weight without calibration.
- Solve pool grouping and within-group placement as separate optimization
  levels.
- Require geometry validation after every complete group proposal.

### M5 — Semantic module

- DeepSeek is an optional reviewer, not the geometry solver.
- Send only structured summaries of already-generated candidates.
- Record provider, model, prompt version, request, response, latency, and cost.
- Enforce JSON schema and cache all results for deterministic replay.
- Allow `abstain/review` instead of forcing a binary decision.
- Compare:
  - geometry only;
  - geometry plus semantic reranking;
  - geometry plus a supervised structured-feature scorer.

### M6 — Agent layer

The first Agent is a bounded workflow controller:

1. inspect the current state;
2. choose a registered tool;
3. execute it with validated parameters;
4. inspect structured diagnostics;
5. retry within an explicit budget or stop with a reason.

It must not invent mates or placements directly. Every action and state
transition is written to a trace file.

### M7 — Evaluation

Required metrics:

- group precision, recall, F1, and exact-group accuracy;
- mate precision, recall, and false-positive count;
- all-parts-solved rate and unsolved count;
- connected-component count;
- pose residual;
- collision and severe-penetration count;
- assembly STEP export success;
- runtime, API calls, retry count, and expanded states;
- calibration/Brier score for probabilistic scorers.

Required ablations:

1. original BFS;
2. BFS + scoring;
3. BFS + scoring + pruning;
4. reliable geometry solver;
5. geometry-only global grouping;
6. optional semantic reranking;
7. Agent-controlled retries.

## Decision gates

- Gate A: Do not train a scorer until GT consistency is audited.
- Gate B: Do not add LLM semantics until geometry-only global grouping exists.
- Gate C: Do not Agentize an unreliable tool; every tool needs structured
  inputs, outputs, errors, and tests first.
- Gate D: Do not claim semantic benefit without a frozen no-LLM ablation.
- Gate E: Do not consider RL until deterministic search traces and a clear
  remaining bottleneck exist.

## Immediate next actions

1. Finish the remaining SolidWorks cases with six-case session recycling.
2. Stop after dataset generation as requested.
3. Audit dataset completeness.
4. Review the GT/geometry consistency of a stratified sample before running
   experiments.
5. Freeze the v0 interfaces and define `part_features`, `candidate`,
   `group_proposal`, `validation`, and `agent_trace` schemas.
6. Begin M1 with one shaft-hole family, one flange family, and one
   chassis/cover family.

## Scope deliberately excluded from v1

- reinforcement learning;
- end-to-end neural assembly prediction;
- LLM-generated geometry or placements;
- universal CAD generalization claims;
- silent retries without trace records;
- replacing the existing geometry stack with a greenfield rewrite.
