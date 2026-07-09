"""Select direct assembly edges when all input parts are known to be related.

The selector never asks whether parts belong to the same assembly.  It finds a
globally connected, maximum-evidence skeleton and conservatively retains
additional strongly supported direct edges.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from typing import Any

from constraints import CLEARANCE, COAXIAL, PLANAR_ALIGN, PLANAR_MATE, POCKET_MATE


TYPE_PRIORITY = {
    POCKET_MATE: 5,
    CLEARANCE: 4,
    COAXIAL: 3,
    PLANAR_MATE: 2,
    PLANAR_ALIGN: 1,
}
STRONG_TYPES = {POCKET_MATE, CLEARANCE, COAXIAL}


def canonical_pair(parts: list[str] | tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(str(part) for part in parts))


def stable_id(prefix: str, *values: str) -> str:
    digest = hashlib.sha1("\0".join(values).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _connected(parts: list[str], edges: tuple[tuple[str, str], ...] | list[tuple[str, str]]) -> bool:
    if len(parts) <= 1:
        return True
    adjacency = {part: set() for part in parts}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = {parts[0]}
    frontier = [parts[0]]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return len(visited) == len(parts)


def _confidence(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def build_pair_candidates(
    matches: list[dict[str, Any]],
    joinable_by_pair: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    joinable_by_pair = joinable_by_pair or {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for match in matches:
        grouped[canonical_pair(match["parts"])].append(match)
    candidates = []
    for pair, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                float(row.get("score", 0.0)),
                TYPE_PRIORITY.get(str(row.get("type")), 0),
            ),
            reverse=True,
        )
        relation_types = []
        for row in ranked:
            if row["type"] not in relation_types:
                relation_types.append(row["type"])
        best = float(ranked[0].get("score", 0.0))
        diversity = min(0.10, 0.04 * max(0, len(relation_types) - 1))
        joinable = joinable_by_pair.get(pair)
        learned_bonus = 0.04 if joinable else 0.0
        score = min(1.0, best + diversity + learned_bonus)
        providers = ["analytic_geometry"]
        if joinable:
            providers.append("joinable")
        candidates.append({
            "connection_id": stable_id("C", *pair),
            "parts": list(pair),
            "score": round(score, 6),
            "confidence": _confidence(score),
            "relation_types": relation_types,
            "primary_relation_type": ranked[0]["type"],
            "matches": ranked,
            "providers": providers,
            "joinable": joinable,
        })
    return candidates


def select_direct_connections(
    parts: list[str],
    pair_candidates: list[dict[str, Any]],
    *,
    conservative: bool = False,
    part_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Globally select a connected skeleton plus bounded supported edges.
    
    When conservative=True, only the minimum spanning tree is kept.
    No additional supported edges are added. This is appropriate for
    known-group assemblies where all parts are known to belong together
    and false-positive edges between satellite parts must be avoided.
    
    part_weights: optional dict mapping part name → weight (e.g. file size).
    Used as a tiebreaker to prefer central hubs with higher weight.
    """
    parts = sorted(str(part) for part in parts)
    if len(parts) == 1:
        return {"connected": True, "selected": [], "unresolved_parts": []}
    by_pair = {
        canonical_pair(row["parts"]): row for row in pair_candidates
    }
    pair_keys = sorted(by_pair)
    best_tree = None
    best_objective = None
    for combination in itertools.combinations(pair_keys, len(parts) - 1):
        if not _connected(parts, combination):
            continue
        rows = [by_pair[pair] for pair in combination]
        # Weighted type score: preference for strong constraint types (coaxial,
        # clearance, pocket_mate) over weak planar-only evidence.
        weighted_types = sum(
            sum(TYPE_PRIORITY.get(t, 0) for t in row["relation_types"])
            for row in rows
        )
        # Centrality: prefer star topology — max degree in the tree.
        degree = defaultdict(int)
        for a, b in combination:
            degree[a] += 1
            degree[b] += 1
        max_degree = max(degree.values()) if degree else 0
        # Hub weight: when degrees tie, prefer the higher-weight part as hub.
        # Weight is typically file size — the largest part is usually the
        # structural center (chassis, baseplate, main housing).
        hub_weight = 0.0
        if part_weights:
            hubs = [p for p, d in degree.items() if d == max_degree]
            hub_weight = max((part_weights.get(p, 0.0) for p in hubs), default=0.0)
        objective = (
            round(round(sum(float(row["score"]) for row in rows), 2), 9),
            weighted_types,
            max_degree,
            round(hub_weight, 1),
            tuple(combination),
        )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_tree = set(combination)
    if best_tree is None:
        # Preserve the strongest reachable forest for an auditable partial result.
        parent = {part: part for part in parts}

        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        forest = set()
        for pair in sorted(pair_keys, key=lambda p: by_pair[p]["score"], reverse=True):
            a, b = pair
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                forest.add(pair)
        best_tree = forest

    selected = []
    for pair in sorted(best_tree):
        row = dict(by_pair[pair])
        row["selection_role"] = "connected_skeleton"
        selected.append(row)

    # Retain non-tree edges only with strong non-planar evidence, multiple
    # independent relation types, or direct learned-interface support.
    # In conservative mode, skip all additional edges — the spanning tree
    # already connects all parts and extra edges risk false positives.
    if not conservative:
        for pair in pair_keys:
            if pair in best_tree:
                continue
            row = by_pair[pair]
            types = set(row["relation_types"])
            keep = (
                (float(row["score"]) >= 0.78 and bool(types & STRONG_TYPES))
                or (float(row["score"]) >= 0.80 and len(types) >= 2)
                or (float(row["score"]) >= 0.82 and row.get("joinable") is not None)
            )
            if keep:
                item = dict(row)
                item["selection_role"] = "additional_supported_edge"
                selected.append(item)

    selected_pairs = [canonical_pair(row["parts"]) for row in selected]
    connected = _connected(parts, selected_pairs)
    touched = {part for pair in selected_pairs for part in pair}
    return {
        "connected": connected,
        "selected": selected,
        "unresolved_parts": sorted(set(parts) - touched),
        "candidate_pair_count": len(pair_candidates),
        "selected_pair_count": len(selected),
        "selection_method": "exhaustive_maximum_evidence_spanning_skeleton_plus_bounded_support",
    }
