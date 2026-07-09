"""Lightweight, auditable group-consistency checks.

The module does not infer provenance and does not choose a complete partition.
It measures whether a proposal has independent geometric support and whether
near-tied overlapping alternatives make automatic acceptance unsafe.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


CYLINDRICAL_TYPES = {"clearance", "coaxial", "pocket_mate"}
PLANAR_TYPES = {"planar_mate", "planar_align"}
LEARNED_TYPES = {"joinable_interface_rank"}
EVIDENCE_TAXONOMY_VERSION = "3.1.0"


def _pair(parts: Iterable[str]) -> tuple[str, str]:
    return tuple(sorted(str(part) for part in parts))


def _edge_evidence(edge: dict[str, Any]) -> list[str]:
    """Return independent physical constraints from analytic geometry.

    Learned interface predictors observe the same geometry and therefore do
    not become an independent physical constraint merely because their
    softmax is high.  They are tracked separately as review corroboration.
    """
    kind = str(edge.get("candidate_type", ""))
    reason = edge.get("audit_reason", {})
    evidence = []
    if kind in PLANAR_TYPES:
        evidence.append("planar_seating")
        if float(reason.get("area_reliability", 0.0) or 0.0) >= 0.75:
            evidence.append("planar_extent_match")
    if kind in CYLINDRICAL_TYPES:
        if (
            float(reason.get("gap_quality", 0.0) or 0.0) >= 0.7
            or float(reason.get("radius_quality", 0.0) or 0.0) >= 0.7
            or float(reason.get("radius_similarity", 0.0) or 0.0) >= 0.7
        ):
            evidence.append("radius_fit")
        if float(reason.get("axis_dot_abs", 0.0) or 0.0) >= 0.9:
            evidence.append("axis_alignment")
        if float(reason.get("area_reliability", 0.0) or 0.0) >= 0.75:
            evidence.append("axial_engagement")
    return evidence


def _joinable_softmax(edge: dict[str, Any]) -> float:
    if str(edge.get("candidate_type", "")) not in LEARNED_TYPES:
        return 0.0
    return float(
        edge.get("provider_evidence", {})
        .get("pretrained_joinable", {})
        .get("softmax_probability", 0.0)
        or 0.0
    )


def _edge_provider(edge: dict[str, Any]) -> str | None:
    kind = str(edge.get("candidate_type", ""))
    if kind == "joinable_interface_rank":
        return "joinable_interface_ranker"
    if kind in PLANAR_TYPES | CYLINDRICAL_TYPES:
        return "analytic_geometry"
    return None


def assess_group_consistency(
    proposal: dict[str, Any],
    candidate_edges: list[dict[str, Any]],
    all_proposals: list[dict[str, Any]],
    *,
    conflict_score_margin: float = 0.04,
    learned_evidence_enabled: bool = False,
    joinable_min_softmax: float = 0.85,
) -> dict[str, Any]:
    parts = {str(part) for part in proposal["parts"]}
    edge_by_id = {
        str(edge["candidate_id"]): edge for edge in candidate_edges
    }
    internal = [
        edge_by_id[edge_id]
        for edge_id in proposal.get("candidate_edges", [])
        if edge_id in edge_by_id
    ]
    evidence_instances = [
        evidence
        for edge in internal
        for evidence in _edge_evidence(edge)
    ]
    independent_evidence = sorted(set(evidence_instances))
    analytic_evidence = list(independent_evidence)
    qualified_learned_edges = [
        edge
        for edge in internal
        if learned_evidence_enabled
        and _joinable_softmax(edge) >= joinable_min_softmax
    ]
    learned_pairs = {
        _pair(edge["parts"]) for edge in qualified_learned_edges
    }
    learned_evidence = (
        ["learned_interface_consistency"] if learned_pairs else []
    )
    learned_probabilities = [
        _joinable_softmax(edge) for edge in qualified_learned_edges
    ]
    corroborating_providers = sorted(
        {
            provider
            for edge in internal
            if (provider := _edge_provider(edge)) is not None
        }
    )
    # Provider agreement: when both analytic geometry and a learned predictor
    # signal the same pair, the agreement itself strengthens the evidence.
    analytic_pairs: set[tuple[str, str]] = set()
    for edge in internal:
        pair = _pair(edge["parts"])
        provider = _edge_provider(edge)
        if provider == "analytic_geometry":
            analytic_pairs.add(pair)
    corroborated_pairs = analytic_pairs & learned_pairs
    provider_agreement_present = bool(corroborated_pairs)
    pair_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    degree: dict[str, set[str]] = defaultdict(set)
    for edge in internal:
        pair = _pair(edge["parts"])
        pair_types[pair].add(str(edge["candidate_type"]))
        degree[pair[0]].add(pair[1])
        degree[pair[1]].add(pair[0])

    covered = {part for part in parts if degree.get(part)}
    interface_coverage = len(covered) / len(parts) if parts else 0.0
    maximum_degree = max((len(value) for value in degree.values()), default=0)
    has_central = len(parts) <= 2 or maximum_degree >= 2
    required_tree_pairs = max(len(parts) - 1, 1)
    supported_pair_count = len(pair_types)
    pair_coverage = min(supported_pair_count / required_tree_pairs, 1.0)
    group_completeness = (
        0.55 * interface_coverage
        + 0.30 * pair_coverage
        + 0.15 * float(bool(proposal.get("connected", False)))
    )
    central_part_coverage = min(
        maximum_degree / required_tree_pairs, 1.0
    )
    interface_families = set()
    for kinds in pair_types.values():
        if kinds & PLANAR_TYPES:
            interface_families.add("planar")
        if kinds & CYLINDRICAL_TYPES:
            interface_families.add("cylindrical")
        interface_families.update(
            kind
            for kind in kinds
            if kind not in PLANAR_TYPES | CYLINDRICAL_TYPES | LEARNED_TYPES
        )
    evidence_instances = {
        (_pair(edge["parts"]), evidence)
        for edge in internal
        for evidence in _edge_evidence(edge)
    }
    family_diversity = min(len(interface_families) / 2.0, 1.0)
    evidence_type_diversity = min(
        len(independent_evidence) / 3.0, 1.0
    )
    interface_instance_coverage = min(
        len(evidence_instances) / max(2 * required_tree_pairs, 1),
        1.0,
    )
    interface_diversity = (
        0.30 * family_diversity
        + 0.30 * evidence_type_diversity
        + 0.40 * interface_instance_coverage
    )
    only_one_pair = len(pair_types) <= 1
    only_planar = bool(pair_types) and all(
        kinds <= PLANAR_TYPES for kinds in pair_types.values()
    )
    weak_single = (
        not pair_types
        or (only_one_pair and only_planar)
        or len(independent_evidence) < 2
    )

    raw_score = proposal.get("geometry_score")
    if raw_score is None:
        raw_score = proposal.get("candidate_priority_score", 0.0)
    score = float(raw_score)
    competing = []
    larger = []
    for alternative in all_proposals:
        if alternative.get("group_id") == proposal.get("group_id"):
            continue
        raw_alt_parts = alternative.get("parts", [])
        if not isinstance(raw_alt_parts, list) or not all(
            isinstance(part, str) for part in raw_alt_parts
        ):
            continue
        alt_parts = set(raw_alt_parts)
        overlap = parts & alt_parts
        if not overlap:
            continue
        raw_alt_score = alternative.get("geometry_score")
        if raw_alt_score is None:
            raw_alt_score = alternative.get(
                "candidate_priority_score", 0.0
            )
        alt_score = float(raw_alt_score)
        if parts < alt_parts and alt_score >= score - conflict_score_margin:
            larger.append(str(alternative["group_id"]))
        high_overlap = len(overlap) >= max(1, min(len(parts), len(alt_parts)) - 1)
        if high_overlap and alt_score >= score - conflict_score_margin:
            competing.append(str(alternative["group_id"]))
    blocks_larger = bool(larger)
    has_global_conflict = bool(competing)

    evidence_quality = min(len(independent_evidence) / 3.0, 1.0)
    central_score = 1.0 if has_central else 0.0
    conflict_score = 0.0 if has_global_conflict else 1.0
    consistency = (
        0.35 * score
        + 0.25 * evidence_quality
        + 0.20 * interface_coverage
        + 0.10 * central_score
        + 0.10 * conflict_score
        - (0.20 if weak_single else 0.0)
    )
    consistency = min(1.0, max(0.0, consistency))
    review_required = (
        weak_single
        or not has_central
        or blocks_larger
        or has_global_conflict
    )
    reasons = []
    if weak_single:
        reasons.append("only weak or non-independent interface support")
    if not has_central:
        reasons.append("no central-part topology")
    if blocks_larger:
        reasons.append("candidate may block a near-tied larger group")
    if has_global_conflict:
        reasons.append("near-tied overlapping alternatives exist")
    if not reasons:
        reasons.append("independent evidence with no obvious global conflict")
    return {
        "evidence_taxonomy_version": EVIDENCE_TAXONOMY_VERSION,
        "group_consistency_score": round(consistency, 8),
        "evidence_count": len(evidence_instances),
        "independent_evidence_count": len(independent_evidence),
        "independent_evidence": independent_evidence,
        "analytic_evidence_count": len(analytic_evidence),
        "analytic_evidence": analytic_evidence,
        "learned_evidence_count": len(learned_evidence),
        "learned_evidence": learned_evidence,
        "learned_evidence_enabled": learned_evidence_enabled,
        "joinable_min_softmax": joinable_min_softmax,
        "learned_corroboration_score": round(
            max(learned_probabilities, default=0.0), 8
        ),
        "corroborating_provider_count": len(corroborating_providers),
        "corroborating_providers": corroborating_providers,
        "corroborated_pair_count": len(corroborated_pairs),
        "provider_agreement_present": provider_agreement_present,
        "provider_agreement_counts_as_independent_evidence": False,
        "interface_coverage": round(interface_coverage, 8),
        "supported_pair_count": supported_pair_count,
        "pair_coverage": round(pair_coverage, 8),
        "group_completeness_score": round(group_completeness, 8),
        "central_part_coverage": round(central_part_coverage, 8),
        "interface_family_count": len(interface_families),
        "interface_families": sorted(interface_families),
        "interface_instance_count": len(evidence_instances),
        "interface_diversity_score": round(interface_diversity, 8),
        "weak_single_interface_match": weak_single,
        "has_multi_evidence_support": len(independent_evidence) >= 2,
        "has_central_part_structure": has_central,
        "blocks_larger_better_group": blocks_larger,
        "has_global_conflict": has_global_conflict,
        "competing_group_ids": sorted(competing)[:20],
        "larger_group_ids": sorted(larger)[:20],
        "review_required": review_required,
        "reason": "; ".join(reasons),
    }
