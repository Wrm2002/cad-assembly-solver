"""Add learned CAD Pair Pose proposals as soft-prior manifold candidates."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-12.0, min(12.0, value))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_result", type=Path)
    parser.add_argument("learned_frontier", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-entity-limit", type=int, default=2)
    args = parser.parse_args()
    base, learned = _read(args.base_result), _read(args.learned_frontier)
    rows = list((base.get("joint_hypotheses") or {}).get("rows") or [])
    by_entities: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_entities.setdefault((str(row.get("entity_a")), str(row.get("entity_b"))), []).append(row)
    additions = []
    for entity_row in learned.get("pose_hypotheses") or []:
        key = (str((entity_row.get("entity_a") or {}).get("entity_id")), str((entity_row.get("entity_b") or {}).get("entity_id")))
        templates = by_entities.get(key) or []
        if not templates:
            continue
        template = templates[0]
        for proposal in (entity_row.get("learned_pose_hypotheses") or [])[: max(1, args.per_entity_limit)]:
            candidate = copy.deepcopy(template)
            candidate["initial_pose_b_in_a"] = proposal["relative_transform"]
            score = float(proposal["combined_logit"])
            # Sidecar candidates must never inflate the structural confidence
            # of their JoinABLe/analytic template.  The learned score is kept
            # separately as a soft pose prior and for audit; promoting it to
            # pair confidence previously allowed an uncalibrated head to
            # displace the successful analytic baseline.
            candidate["confidence"] = float(template.get("confidence", 0.0))
            provenance = dict(candidate.get("provenance") or {})
            provenance.update({
                "learned_pose_initial": True,
                "learned_pose_score": score,
                "learned_pose_mode_index": int(proposal["mode_index"]),
                "learned_pose_probability": _sigmoid(score),
                "initial_pose_is_constraint": False,
                "source": "cad_pair_pose_head.v2",
                "selection_channel": "learned_sidecar",
            })
            candidate["provenance"] = provenance
            candidate["rank"] = int(entity_row.get("joinable_entity_rank", candidate.get("rank", 0)))
            additions.append(candidate)
    result = copy.deepcopy(base)
    result.setdefault("joint_hypotheses", {}).setdefault("rows", []).extend(additions)
    result["joint_hypotheses"]["learned_pose_augmentation"] = {
        "source": "cad_pair_pose_head.v2",
        "added_rows": len(additions),
        "soft_prior_only": True,
        "case_specific_override": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"added_rows": len(additions), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
