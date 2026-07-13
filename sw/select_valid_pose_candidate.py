"""Select the least-displaced exact-valid Pose for a reviewable handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    valid = [
        row for row in source.get("hypotheses") or []
        if (row.get("exact_validation") or {}).get("status") == "valid"
    ]
    if not valid:
        raise SystemExit("no_exact_valid_pose_candidate")
    selected = min(
        valid,
        key=lambda row: abs(float((row.get("source_evidence") or {}).get(
            "pair_two_manifold_offset_mm", 0.0
        ))),
    )
    result = {
        "schema_version": "known_group_physical_pose_result.v1",
        "physical_pose_status": "valid",
        "semantic_status": "not_auto_claimed",
        "accepted": False,
        "review_required": True,
        "selection_policy": "exact_OCCT_valid_then_minimum_absolute_unresolved_axis_offset",
        "feature_policy": source.get("feature_policy"),
        "candidate": selected,
        "acceptance_boundary": (
            "This proves a collision-free, geometrically supported known-group Pose. "
            "It does not claim provenance or functional semantic correctness beyond available CAD geometry."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_hypothesis": selected.get("hypothesis_id"),
        "physical_pose_status": "valid",
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
