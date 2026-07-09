# One-Week Agent-based CAD Matching MVP

Sprint: 2026-07-02 to 2026-07-08

## MVP objective

Given an unordered pool of 5–12 STEP parts, produce:

- structured part features;
- auditable geometric mate candidates;
- geometry-only assembly group proposals;
- reliable within-group placements and validation;
- optional semantic review for ambiguous top-ranked candidates;
- an Agent trace and Markdown report;
- reproducible geometry-only and semantic-assisted comparison results.

The MVP is an engineering prototype, not a universal CAD solution.

## Daily schedule

| Day | Date | Work | Deliverable and acceptance |
|---|---|---|---|
| D1 | Jul 2 | Finish current dataset generation; freeze interfaces | Dataset generation stops cleanly; define schemas for part, candidate, group proposal, validation, and Agent trace |
| D2 | Jul 3 | Dataset audit and mixed-pool builder | Audited source cases; generate mixed pools containing 2–3 true groups plus distractors; write `pool_gt.json` |
| D3 | Jul 4 | Part indexing and coarse retrieval | Produce `part_features.json`; reduce all-pairs candidates using bbox, volume, cylinders, planes, and hole-pattern summaries |
| D4 | Jul 5 | Geometry-only global grouping | Build candidate graph and deterministic conflict-aware group assignment; output accepted and rejected group proposals |
| D5 | Jul 6 | Pose solve and validation integration | Run reliable beam/BnB solver for every proposed group; reject groups with unsolved parts, excessive residual, or confirmed invalidity |
| D6 | Jul 7 | Optional semantic verifier and Agent controller | Add provider-neutral semantic interface, cached JSON responses, abstention, bounded retry policy, and full Agent trace |
| D7 | Jul 8 | Evaluation, demo, and handoff | Run frozen-pool ablations; output metrics, Markdown report, one-command demo, tests, and limitations |

## Strict scope for one week

### Included

- reuse the existing geometry and reliable-solver code;
- support shaft/hole, flange-like, and chassis/cover-style evidence already
  represented by the current feature stack;
- pools of 5–12 parts;
- deterministic geometry-only baseline;
- one simple global assignment method;
- optional semantic reranking only for ambiguous candidates;
- state-machine Agent with registered tools and bounded retries;
- complete intermediate JSON and Markdown report.

### Deferred

- robust support for arbitrary 30-part pools;
- many independent mechanical template families;
- learned neural scorer;
- reinforcement learning;
- exact functional understanding from geometry alone;
- large manually labelled real-CAD test set;
- production service/UI;
- claims of universal generalization.

## Implementation order

1. Define schemas before adding modules.
2. Build geometry-only pool grouping.
3. Connect existing reliable pose solver and validation.
4. Freeze a no-LLM baseline.
5. Add semantic review behind a feature flag.
6. Add Agent orchestration around stable tools.
7. Run ablations and write limitations.

## Required outputs

```text
results/
├─ part_features.json
├─ geometry_candidates.json
├─ pruned_candidates.json
├─ group_proposals.json
├─ group_validations.json
├─ semantic_reviews.json
├─ final_groups.json
├─ agent_trace.jsonl
├─ candidate_scores.csv
└─ assembly_report.md
```

## One-command demo

```powershell
python run_agent_pipeline.py `
  --input <parts_folder> `
  --output <results_folder> `
  --semantic off
```

Semantic-assisted mode must remain optional:

```powershell
python run_agent_pipeline.py `
  --input <parts_folder> `
  --output <results_folder> `
  --semantic deepseek
```

## Acceptance gates

- Geometry-only mode runs without an API key.
- Every accepted/rejected edge and group has a structured reason.
- A part cannot be assigned to conflicting groups.
- Every accepted group passes within-group placement validation.
- Agent retries are bounded and recorded.
- LLM output cannot override a geometric failure.
- Tests and one frozen demo pool run from a single command.

## Contingency

If SolidWorks generation consumes more than D1, stop at the valid checkpoint
and use the completed cases. Dataset volume must not block development of the
pool-level MVP.
