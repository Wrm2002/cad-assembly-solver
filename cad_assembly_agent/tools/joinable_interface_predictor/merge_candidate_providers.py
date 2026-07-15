"""Merge rule and pretrained interface candidates without auto-accepting them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def entity_key(entity: dict[str, Any]) -> tuple[str, int]:
    return str(entity["entity_type"]), int(entity["topology_index"])


def candidate_key(
    candidate: dict[str, Any],
) -> tuple[tuple[str, int], tuple[str, int]]:
    return (
        entity_key(candidate["part_a_entity"]),
        entity_key(candidate["part_b_entity"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-k", type=int, default=10)
    parser.add_argument("--pretrained-k", type=int, default=10)
    args = parser.parse_args()

    rule = read_json(args.rule)
    pretrained = read_json(args.pretrained)
    merged: dict[
        tuple[tuple[str, int], tuple[str, int]], dict[str, Any]
    ] = {}
    failures = []
    for provider, payload, limit in (
        ("rule", rule, max(0, args.rule_k)),
        ("pretrained_joinable", pretrained, max(0, args.pretrained_k)),
    ):
        if payload.get("failure_reasons"):
            failures.extend(
                f"{provider}:{reason}"
                for reason in payload["failure_reasons"]
            )
        for candidate in payload.get("candidates", [])[:limit]:
            key = candidate_key(candidate)
            if key not in merged:
                merged[key] = {
                    "candidate_id": (
                        f"{key[0][0]}_{key[0][1]:06d}__"
                        f"{key[1][0]}_{key[1][1]:06d}"
                    ),
                    "part_a_entity": candidate["part_a_entity"],
                    "part_b_entity": candidate["part_b_entity"],
                    "provider_evidence": {},
                }
            merged[key]["provider_evidence"][provider] = {
                "rank": int(candidate["rank"]),
                "score": candidate.get("score", candidate.get("logit")),
                "softmax_probability": candidate.get(
                    "softmax_probability"
                ),
                "score_evidence": candidate.get("score_evidence"),
            }

    candidates = list(merged.values())
    for candidate in candidates:
        ranks = [
            int(evidence["rank"])
            for evidence in candidate["provider_evidence"].values()
        ]
        provider_count = len(candidate["provider_evidence"])
        candidate.update(
            {
                "independent_provider_count": provider_count,
                "has_rule_support": (
                    "rule" in candidate["provider_evidence"]
                ),
                "has_pretrained_support": (
                    "pretrained_joinable"
                    in candidate["provider_evidence"]
                ),
                "best_provider_rank": min(ranks),
                "rank_sum": sum(ranks),
                "review_required": True,
                "can_auto_accept": False,
                "review_reason": (
                    "Interface candidate evidence must pass pose, collision, "
                    "multi-evidence, and group-consistency gates."
                ),
            }
        )
    candidates.sort(
        key=lambda candidate: (
            -candidate["independent_provider_count"],
            candidate["best_provider_rank"],
            candidate["rank_sum"],
            candidate["candidate_id"],
        )
    )
    for rank, candidate in enumerate(candidates, 1):
        candidate["merged_rank"] = rank

    output = {
        "schema_version": "1.0.0",
        "predictor": "conservative_rule_pretrained_union",
        "source_rule_prediction": str(args.rule.resolve()),
        "source_pretrained_prediction": str(args.pretrained.resolve()),
        "rule_k": args.rule_k,
        "pretrained_k": args.pretrained_k,
        "candidate_budget_upper_bound": args.rule_k + args.pretrained_k,
        "candidate_count_after_deduplication": len(candidates),
        "dual_provider_candidate_count": sum(
            candidate["independent_provider_count"] == 2
            for candidate in candidates
        ),
        "candidates": candidates,
        "gate_policy": {
            "shadow_mode": True,
            "provider_score_can_auto_accept": False,
            "requires_pose_validation": True,
            "requires_exact_collision_validation": True,
            "requires_group_consistency": True,
            "single_provider_candidates_default_to_review": True,
        },
        "failure_reasons": failures,
        "unavailable_fields": [
            "functional_assembly_validity",
            "final_pose_without_external_solver",
        ],
    }
    write_json(args.output, output)
    print(
        f"merged {len(candidates)} candidates; "
        f"dual={output['dual_provider_candidate_count']}"
    )
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
