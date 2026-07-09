"""可解释的组级功能接口闭环特征，不读取评估真值。"""

from __future__ import annotations

from statistics import mean
from typing import Any

from family_templates import TEMPLATE_BY_FAMILY
from pair_edge import canonical_pair, index_pair_edges


def _role_parts(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def compute_functional_interface_features(
    proposal: dict[str, Any],
    pair_edges: list[dict[str, Any]],
    role_table: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    family = str(proposal.get("assembly_family", "unknown"))
    template = TEMPLATE_BY_FAMILY.get(family)
    if template is None:
        return {
            "functional_feature_version": "1.0.0",
            "family_supported": False,
            "functional_closure_score": 0.0,
            "review_required": True,
            "reason": "unsupported family",
        }
    assignment = {
        role: _role_parts(value)
        for role, value in proposal.get("role_assignment", {}).items()
    }
    edge_index = index_pair_edges(pair_edges)
    relation_rows = []
    for relation in template.relations:
        left_role, right_role = relation.roles
        for left in assignment.get(left_role, []):
            for right in assignment.get(right_role, []):
                edge = edge_index.get(canonical_pair((left, right)))
                relation_types = set(edge["relation_types"]) if edge else set()
                type_supported = bool(
                    relation_types & set(relation.accepted_relation_types)
                )
                evidence_count = int(
                    edge["independent_physical_evidence_count"]
                ) if edge else 0
                strong = (
                    type_supported
                    and evidence_count >= relation.minimum_physical_evidence
                    and not bool(edge and edge["learned_only"])
                )
                relation_rows.append(
                    {
                        "roles": [left_role, right_role],
                        "parts": [left, right],
                        "required_for_complete": relation.required_for_complete,
                        "type_supported": type_supported,
                        "evidence_count": evidence_count,
                        "minimum_evidence": relation.minimum_physical_evidence,
                        "strong_support": strong,
                        "learned_only": bool(edge and edge["learned_only"]),
                    }
                )
    required_rows = [
        row for row in relation_rows if row["required_for_complete"]
    ]
    optional_rows = [
        row for row in relation_rows if not row["required_for_complete"]
    ]
    required_coverage = (
        sum(row["type_supported"] for row in required_rows) / len(required_rows)
        if required_rows else 0.0
    )
    strong_coverage = (
        sum(row["strong_support"] for row in required_rows) / len(required_rows)
        if required_rows else 0.0
    )
    optional_coverage = (
        sum(row["strong_support"] for row in optional_rows) / len(optional_rows)
        if optional_rows else 1.0
    )
    assigned_confidences = []
    assigned_margins = []
    for role, parts in assignment.items():
        for part in parts:
            scores = role_table.get(part, {}).get("role_scores", {})
            assigned = float(scores.get(role, 0.0))
            alternatives = [
                float(score) for candidate_role, score in scores.items()
                if candidate_role != role
            ]
            assigned_confidences.append(assigned)
            assigned_margins.append(
                assigned - max(alternatives, default=0.0)
            )
    mean_confidence = mean(assigned_confidences) if assigned_confidences else 0.0
    minimum_confidence = min(assigned_confidences, default=0.0)
    mean_margin = mean(assigned_margins) if assigned_margins else -1.0

    role_pairs = {
        tuple(row["roles"]): row["strong_support"] for row in relation_rows
    }
    key_three_way = (
        role_pairs.get(("shaft", "hub"), False)
        and role_pairs.get(("shaft", "key"), False)
        and role_pairs.get(("hub", "key"), False)
    ) if family == "shaft_hub_key" else None
    bearing_double_fit = (
        role_pairs.get(("housing", "bearing"), False)
        and role_pairs.get(("bearing", "shaft"), False)
    ) if family == "bearing_housing" else None
    axial_retention = (
        role_pairs.get(("housing", "end_cover"), False)
        and (
            not assignment.get("bearing_retainer")
            or role_pairs.get(("housing", "bearing_retainer"), False)
        )
    ) if family == "bearing_housing" else None
    locator_rows = [
        row for row in relation_rows
        if "locating_pin" in row["roles"]
    ]
    dual_sided_locator = (
        bool(locator_rows)
        and all(row["strong_support"] for row in locator_rows)
    ) if family == "cover_base" else None

    closure_flags = [
        flag for flag in (
            key_three_way, bearing_double_fit, axial_retention,
            dual_sided_locator,
        ) if flag is not None
    ]
    topology_closure = (
        sum(bool(flag) for flag in closure_flags) / len(closure_flags)
        if closure_flags else 0.0
    )
    confidence_component = max(0.0, min(mean_confidence, 1.0))
    margin_component = max(0.0, min((mean_margin + 0.25) / 0.5, 1.0))
    score = (
        0.35 * strong_coverage
        + 0.25 * topology_closure
        + 0.20 * confidence_component
        + 0.10 * margin_component
        + 0.10 * optional_coverage
    )
    critical_learned_only = any(
        row["required_for_complete"] and row["learned_only"]
        for row in relation_rows
    )
    return {
        "functional_feature_version": "1.0.0",
        "family_supported": True,
        "required_relation_coverage": round(required_coverage, 8),
        "strong_required_relation_coverage": round(strong_coverage, 8),
        "optional_relation_coverage": round(optional_coverage, 8),
        "mean_assigned_role_confidence": round(mean_confidence, 8),
        "minimum_assigned_role_confidence": round(minimum_confidence, 8),
        "mean_assigned_role_margin": round(mean_margin, 8),
        "key_three_way_closure": key_three_way,
        "bearing_inner_outer_double_fit": bearing_double_fit,
        "axial_retention_chain": axial_retention,
        "dual_sided_locator_closure": dual_sided_locator,
        "topology_closure_score": round(topology_closure, 8),
        "critical_learned_only_relation": critical_learned_only,
        "functional_closure_score": round(max(0.0, min(score, 1.0)), 8),
        "review_required": (
            strong_coverage < 1.0
            or topology_closure < 1.0
            or critical_learned_only
            or minimum_confidence < 0.35
        ),
        "relation_audit": relation_rows,
        "evaluation_semantics_used": False,
    }

