"""Use cached JoinABLe rankings to expand detailed candidate recall.

The released model ranks interface entities for a supplied part pair and has
no trained "no joint" class.  Consequently, this provider may nominate pairs
for detailed analytic matching, but it never creates physical evidence or an
automatic acceptance decision by itself.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pair(part_a: str, part_b: str) -> tuple[str, str]:
    return tuple(sorted((str(part_a), str(part_b))))


def load_pair_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("pairs"), list):
        raise ValueError("JoinABLe report must contain a list field named 'pairs'")
    return payload


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    features = row.get("pair_features") or {}
    uniform_lift = float(features.get("top_1_uniform_lift", 0.0) or 0.0)
    margin = float(features.get("top_2_logit_margin", 0.0) or 0.0)
    entropy = float(features.get("normalized_entropy", 1.0) or 1.0)
    return (uniform_lift, margin, -entropy, str(row.get("pair_id", "")))


def select_joinable_pairs(
    report: dict[str, Any],
    *,
    pool_id: str,
    part_ids: list[str],
    maximum_neighbors_per_part: int,
    minimum_uniform_lift: float = 1.0,
    audit_top_candidates: int = 3,
) -> tuple[set[tuple[str, str]], dict[str, Any]]:
    """Select a bounded, deterministic JoinABLe recall frontier."""

    valid_parts = {str(part) for part in part_ids}
    eligible: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    for row in report.get("pairs", []):
        if str(row.get("pool_id")) != str(pool_id):
            continue
        pair = _pair(row.get("part_a", ""), row.get("part_b", ""))
        features = row.get("pair_features") or {}
        lift = float(features.get("top_1_uniform_lift", 0.0) or 0.0)
        reason = None
        if set(pair) - valid_parts or pair[0] == pair[1]:
            reason = "pair_not_in_current_pool"
        elif row.get("status") != "success":
            reason = f"provider_status:{row.get('status', 'missing')}"
        elif not row.get("candidates"):
            reason = "no_ranked_interface_candidates"
        elif lift < minimum_uniform_lift:
            reason = "below_minimum_uniform_lift"
        if reason:
            ignored.append({"parts": list(pair), "reason": reason})
            continue
        eligible.append(row)

    ranked_by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        ranked_by_part[str(row["part_a"])].append(row)
        ranked_by_part[str(row["part_b"])].append(row)

    selected_pairs: set[tuple[str, str]] = set()
    selected_rows: dict[tuple[str, str], dict[str, Any]] = {}
    limit = max(0, int(maximum_neighbors_per_part))
    for part_id in sorted(valid_parts):
        rows = sorted(
            ranked_by_part.get(part_id, []),
            key=_ranking_key,
            reverse=True,
        )[:limit]
        for row in rows:
            pair = _pair(row["part_a"], row["part_b"])
            selected_pairs.add(pair)
            selected_rows[pair] = row

    selected_audit = []
    for pair in sorted(selected_pairs):
        row = selected_rows[pair]
        candidates = row.get("candidates") or []
        selected_audit.append(
            {
                "parts": list(pair),
                "pair_id": row.get("pair_id"),
                "pair_features": row.get("pair_features") or {},
                "ranked_interface_candidate_count": len(candidates),
                "top_interface_candidates": candidates[:audit_top_candidates],
                "selection_role": "detailed_analytic_candidate_recall",
                "creates_physical_evidence": False,
                "can_auto_accept": False,
            }
        )

    audit = {
        "schema_version": "1.0.0",
        "provider": "pretrained_joinable",
        "pool_id": str(pool_id),
        "model_boundary": {
            "interface_ranking_available": True,
            "trained_no_joint_class": False,
            "pair_connection_decision_available": False,
            "provider_can_expand_candidate_recall": True,
            "provider_can_auto_accept": False,
        },
        "maximum_neighbors_per_part": limit,
        "minimum_uniform_lift": float(minimum_uniform_lift),
        "eligible_pair_count": len(eligible),
        "selected_pair_count": len(selected_pairs),
        "selected_pairs": selected_audit,
        "ignored_rows": ignored,
    }
    return selected_pairs, audit
