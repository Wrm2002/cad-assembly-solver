"""Unified, auditable pair-edge records for mixed-pool grouping.

The record deliberately separates physical analytic evidence from learned
provider corroboration.  A JoinABLe score can improve recall and ordering, but
it never increases the number of independent mechanical constraints.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable


ANALYTIC_TYPES = {
    "clearance",
    "coaxial",
    "pocket_mate",
    "planar_mate",
    "planar_align",
}
LEARNED_TYPES = {"joinable_interface_rank"}


def canonical_pair(parts: Iterable[str]) -> tuple[str, str]:
    pair = tuple(sorted(str(part) for part in parts))
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("pair edge requires two distinct parts")
    return pair


def pair_edge_id(parts: Iterable[str]) -> str:
    pair = canonical_pair(parts)
    digest = hashlib.sha256("|".join(pair).encode("utf-8")).hexdigest()[:16]
    return f"PE_{digest}"


def _joinable_fields(edge: dict[str, Any]) -> tuple[int | None, float | None]:
    provider = edge.get("provider_evidence", {}).get(
        "pretrained_joinable", {}
    )
    rank = provider.get("rank", edge.get("joinable_rank"))
    probability = provider.get(
        "softmax_probability", edge.get("joinable_score")
    )
    return (
        int(rank) if rank is not None else None,
        float(probability) if probability is not None else None,
    )


def _physical_evidence(edge: dict[str, Any]) -> set[str]:
    kind = str(edge.get("candidate_type", ""))
    reason = edge.get("audit_reason", {})
    evidence: set[str] = set()
    if kind in {"clearance", "coaxial", "pocket_mate"}:
        if max(
            float(reason.get("gap_quality", 0.0) or 0.0),
            float(reason.get("radius_quality", 0.0) or 0.0),
            float(reason.get("radius_similarity", 0.0) or 0.0),
        ) >= 0.7:
            evidence.add("radius_fit")
        if float(reason.get("axis_dot_abs", 0.0) or 0.0) >= 0.9:
            evidence.add("axis_alignment")
        if float(reason.get("area_reliability", 0.0) or 0.0) >= 0.75:
            evidence.add("axial_engagement")
    if kind in {"planar_mate", "planar_align"}:
        evidence.add("planar_seating")
        if float(reason.get("area_reliability", 0.0) or 0.0) >= 0.75:
            evidence.add("planar_extent_match")
    return evidence


def build_pair_edges(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate candidate-interface rows into one provider-aware pair edge."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[canonical_pair(candidate["parts"])].append(candidate)

    records = []
    for parts, rows in sorted(grouped.items()):
        analytic = [
            row for row in rows
            if str(row.get("candidate_type")) in ANALYTIC_TYPES
        ]
        learned = [
            row for row in rows
            if str(row.get("candidate_type")) in LEARNED_TYPES
        ]
        rank_probability = [_joinable_fields(row) for row in learned]
        ranks = [rank for rank, _ in rank_probability if rank is not None]
        probabilities = [
            probability for _, probability in rank_probability
            if probability is not None
        ]
        physical = sorted(
            {
                item
                for row in analytic
                for item in _physical_evidence(row)
            }
        )
        providers = []
        if analytic:
            providers.append("analytic_geometry")
        if learned:
            providers.append("joinable_interface_ranker")
        best_analytic = max(
            (float(row.get("geometry_score", 0.0)) for row in analytic),
            default=0.0,
        )
        best_learned = max(probabilities, default=0.0)
        records.append(
            {
                "schema_version": "2.0.0",
                "pair_edge_id": pair_edge_id(parts),
                "parts": list(parts),
                "providers": providers,
                "provider_count": len(providers),
                "analytic_candidate_ids": sorted(
                    str(row["candidate_id"]) for row in analytic
                ),
                "learned_candidate_ids": sorted(
                    str(row["candidate_id"]) for row in learned
                ),
                "candidate_ids": sorted(
                    str(row["candidate_id"]) for row in rows
                ),
                "relation_types": sorted(
                    {str(row.get("candidate_type", "unknown")) for row in rows}
                ),
                "best_analytic_geometry_score": round(best_analytic, 8),
                "best_joinable_probability": round(best_learned, 8),
                "best_joinable_rank": min(ranks) if ranks else None,
                "physical_evidence": physical,
                "independent_physical_evidence_count": len(physical),
                "provider_agreement_present": bool(analytic and learned),
                "provider_agreement_counts_as_independent_evidence": False,
                "learned_only": bool(learned and not analytic),
                "critical_learned_only": False,
                "audit_trace": [
                    f"analytic_rows={len(analytic)}",
                    f"learned_rows={len(learned)}",
                    f"physical_evidence={','.join(physical) or 'none'}",
                ],
            }
        )
    return records


def index_pair_edges(
    pair_edges: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {canonical_pair(row["parts"]): row for row in pair_edges}

