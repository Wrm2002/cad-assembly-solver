# Conservative Assembly Report

## 1. Input parts
110 parts across 12 frozen pools.

## 2. Candidate recall audit
See `candidate_recall_audit.md` and the two CSV exception files.

## 3. Geometry candidates
9668 total proposals.

## 4. Final tiers
Accepted=0, review=9162, rejected=506.

## 5. Pose validation
Valid=138, failed=6, uncertain=9024.

## 6. Semantic gate
Disabled; DeepSeek is explanation-only.

## 7. Auto-accepted groups
0 groups; every record contains its evidence and gate reasons.

## 8. Review groups
9162 review-required groups: 288 selected for the immediate operator frontier and 8874 deferred. Deferred review is not rejection.

## 9. Rejected groups
506 groups; reason coverage 100.00%.

## 10. Unresolved parts
110 pool-local parts.

## 11. False-positive risk
Legacy baseline: available; before false positives: 39; after: 0. Auto-accept precision is None.

## 12. Next step
Reduce the immediate review frontier with deterministic local interface ranking while preserving measured truth-candidate recall. Keep semantic reranking disabled.
