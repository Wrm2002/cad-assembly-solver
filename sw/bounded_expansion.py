"""Family-aware bounded proposal expansion for mixed CAD part pools."""

from __future__ import annotations

import hashlib
import itertools
from statistics import mean
from typing import Any, Iterable

from contracts import GroupProposal
from family_templates import FamilyTemplate, TEMPLATES
from pair_edge import canonical_pair, index_pair_edges
from role_estimator import select_center_seeds


def _group_id(parts: Iterable[str]) -> str:
    ordered = tuple(sorted(parts))
    digest = hashlib.sha256("|".join(ordered).encode("utf-8")).hexdigest()[:12]
    return f"G_{digest}"


def _slot_candidates(
    template: FamilyTemplate,
    role_table: dict[str, dict[str, Any]],
    *,
    minimum_role_score: float,
) -> dict[str, list[str]]:
    slots: dict[str, list[str]] = {}
    for slot in template.slots:
        scored = [
            (
                float(row["role_scores"].get(slot.role, 0.0)),
                -float(row["learned_only_graph_risk"]),
                part_id,
            )
            for part_id, row in role_table.items()
            if float(row["role_scores"].get(slot.role, 0.0))
            >= minimum_role_score
        ]
        scored.sort(reverse=True)
        # Candidate caps bound branching.  All candidates tied at the boundary
        # are retained, preventing arbitrary ID ordering from losing recall.
        kept = scored[: slot.candidate_cap]
        if kept:
            boundary = kept[-1][0]
            kept.extend(
                row for row in scored[slot.candidate_cap :]
                if abs(row[0] - boundary) <= 1e-9
            )
        slots[slot.role] = [part_id for _, _, part_id in kept]
    return slots


def _role_selections(
    template: FamilyTemplate,
    candidates: dict[str, list[str]],
) -> list[tuple[str, tuple[str, ...]]]:
    selections = []
    for slot in template.slots:
        available = candidates.get(slot.role, [])
        values: list[tuple[str, ...]] = []
        for count in range(slot.minimum, slot.maximum + 1):
            values.extend(itertools.combinations(available, count))
        selections.append((slot.role, tuple(values)))
    return selections


def _assignment_products(
    selections: list[tuple[str, tuple[tuple[str, ...], ...]]],
) -> Iterable[dict[str, tuple[str, ...]]]:
    if any(not values for _, values in selections):
        return []
    roles = [role for role, _ in selections]
    products = itertools.product(*(values for _, values in selections))
    return (
        {role: tuple(value) for role, value in zip(roles, values)}
        for values in products
    )


def _relation_audit(
    template: FamilyTemplate,
    assignment: dict[str, tuple[str, ...]],
    pair_index: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    audits = []
    missing = []
    for relation in template.relations:
        left_role, right_role = relation.roles
        relation_pairs = [
            (left, right)
            for left in assignment[left_role]
            for right in assignment[right_role]
        ]
        for left, right in relation_pairs:
            pair = canonical_pair((left, right))
            edge = pair_index.get(pair)
            relation_types = set(edge["relation_types"]) if edge else set()
            type_supported = bool(
                relation_types & set(relation.accepted_relation_types)
            )
            evidence_count = (
                int(edge["independent_physical_evidence_count"])
                if edge else 0
            )
            # Missing analytic type remains proposal-level uncertainty when a
            # learned provider recalled the pair.  It is never physical proof.
            learned_bridge = bool(edge and edge["learned_only"])
            supported = type_supported and evidence_count >= min(
                relation.minimum_physical_evidence, 1
            )
            key = f"{left_role}:{left}|{right_role}:{right}"
            if not supported:
                missing.append(key)
            audits.append(
                {
                    "roles": [left_role, right_role],
                    "parts": [left, right],
                    "pair_edge_id": edge["pair_edge_id"] if edge else None,
                    "type_supported": type_supported,
                    "physical_evidence_count": evidence_count,
                    "minimum_physical_evidence": relation.minimum_physical_evidence,
                    "learned_only_bridge": learned_bridge,
                    "supported": supported,
                }
            )
    return audits, missing


def generate_bounded_proposals(
    part_features: list[dict[str, Any]],
    pair_edges: list[dict[str, Any]],
    role_table: dict[str, dict[str, Any]],
    *,
    templates: tuple[FamilyTemplate, ...] = TEMPLATES,
    minimum_role_score: float = 0.42,
    maximum_centers: int = 8,
    maximum_missing_relations: int = 1,
    maximum_proposals: int = 500,
) -> tuple[list[GroupProposal], dict[str, Any]]:
    """Generate complete-slot proposals without enumerating arbitrary subsets."""

    pair_index = index_pair_edges(pair_edges)
    edge_by_id = {row["pair_edge_id"]: row for row in pair_edges}
    centers = select_center_seeds(role_table, maximum=maximum_centers)
    feature_parts = {str(row["part_id"]) for row in part_features}
    proposals_by_parts: dict[tuple[str, ...], GroupProposal] = {}
    family_stats = {}

    for template in templates:
        candidates = _slot_candidates(
            template, role_table, minimum_role_score=minimum_role_score
        )
        selections = _role_selections(template, candidates)
        enumerated = distinct = relation_pruned = 0
        for assignment in _assignment_products(selections):
            enumerated += 1
            flat = tuple(
                part for values in assignment.values() for part in values
            )
            if len(flat) != len(set(flat)):
                continue
            if not set(flat) <= feature_parts:
                continue
            distinct += 1
            center_parts = sorted(
                {
                    part
                    for role in template.center_roles
                    for part in assignment.get(role, ())
                    if part in centers
                }
            )
            if not center_parts:
                continue
            relation_audit, missing_relations = _relation_audit(
                template, assignment, pair_index
            )
            if len(missing_relations) > maximum_missing_relations:
                relation_pruned += 1
                continue
            present_edges = [
                edge_by_id[row["pair_edge_id"]]
                for row in relation_audit
                if row["pair_edge_id"] in edge_by_id
            ]
            analytic_scores = [
                float(edge["best_analytic_geometry_score"])
                for edge in present_edges
                if float(edge["best_analytic_geometry_score"]) > 0.0
            ]
            role_scores = [
                float(role_table[part]["role_scores"][role])
                for role, parts in assignment.items()
                for part in parts
            ]
            relation_coverage = (
                (len(relation_audit) - len(missing_relations))
                / len(relation_audit)
                if relation_audit else 0.0
            )
            complete = not missing_relations
            completeness_score = (
                0.65 + 0.35 * relation_coverage
            )
            geometry_score = min(
                1.0,
                0.55 * (mean(analytic_scores) if analytic_scores else 0.0)
                + 0.25 * (mean(role_scores) if role_scores else 0.0)
                + 0.20 * completeness_score,
            )
            physical_evidence = {
                (tuple(edge["parts"]), evidence)
                for edge in present_edges
                for evidence in edge["physical_evidence"]
            }
            original_candidate_ids = sorted(
                {
                    candidate_id
                    for edge in present_edges
                    for candidate_id in edge["candidate_ids"]
                }
            )
            pair_edge_ids = sorted(
                {edge["pair_edge_id"] for edge in present_edges}
            )
            ordered_parts = tuple(sorted(flat))
            proposal = GroupProposal(
                group_id=_group_id(ordered_parts),
                parts=list(ordered_parts),
                candidate_edges=original_candidate_ids,
                pair_edge_ids=pair_edge_ids,
                geometry_score=round(geometry_score, 8),
                connected=relation_coverage > 0.0,
                status="candidate",
                reasons=[
                    f"family={template.family}",
                    f"relation_coverage={relation_coverage:.6f}",
                    f"mean_role_score={mean(role_scores):.6f}",
                ],
                assembly_family=template.family,
                center_part_ids=center_parts,
                role_assignment={
                    role: (parts[0] if len(parts) == 1 else list(parts))
                    for role, parts in assignment.items()
                },
                slot_coverage={
                    role: len(parts) for role, parts in assignment.items()
                },
                completeness_status=(
                    "family_complete" if complete
                    else "complete_slots_relation_uncertain"
                ),
                completeness_score=round(completeness_score, 8),
                relation_coverage=round(relation_coverage, 8),
                independent_evidence_count=len(physical_evidence),
                missing_required_slots=[],
                missing_required_relations=missing_relations,
                ranking_features={
                    "mean_role_score": round(mean(role_scores), 8),
                    "mean_analytic_score": round(
                        mean(analytic_scores) if analytic_scores else 0.0, 8
                    ),
                    "relation_coverage": round(relation_coverage, 8),
                    "complete_family_structure": float(complete),
                    "learned_only_critical_edge": float(
                        any(row["learned_only_bridge"] for row in relation_audit)
                    ),
                },
                audit_trace=[
                    "generated_by=family_aware_bounded_expansion",
                    "evaluation_semantics_used=false",
                    f"relation_audit={relation_audit}",
                ],
            )
            current = proposals_by_parts.get(ordered_parts)
            if current is None or (
                proposal.completeness_score,
                proposal.geometry_score,
                proposal.assembly_family,
            ) > (
                current.completeness_score,
                current.geometry_score,
                current.assembly_family,
            ):
                proposals_by_parts[ordered_parts] = proposal
        family_stats[template.family] = {
            "slot_candidate_counts": {
                role: len(values) for role, values in candidates.items()
            },
            "raw_assignment_count": enumerated,
            "distinct_assignment_count": distinct,
            "relation_pruned_count": relation_pruned,
        }

    proposals = sorted(
        proposals_by_parts.values(),
        key=lambda row: (
            row.completeness_status == "family_complete",
            row.completeness_score,
            row.geometry_score,
            row.independent_evidence_count,
            row.group_id,
        ),
        reverse=True,
    )[:maximum_proposals]
    audit = {
        "schema_version": "1.0.0",
        "algorithm": "family_aware_bounded_expansion",
        "center_seeds": centers,
        "family_stats": family_stats,
        "proposal_count": len(proposals),
        "maximum_proposals": maximum_proposals,
        "minimum_role_score": minimum_role_score,
        "maximum_missing_relations": maximum_missing_relations,
        "evaluation_semantics_used": False,
    }
    return proposals, audit

