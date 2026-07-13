"""Pose graph builder: converts JoinABLe pairwise results to a pose graph.

Key insight: JoinABLe gives pairwise relative poses, but for multi-part
assembly we need to select which edges to activate and handle multiple
candidates per edge.

Strategy:
  1. Build full candidate graph from all pairwise results
  2. Compute Maximum Spanning Tree (MST) for connectivity
  3. Optionally add extra edges (cycle closures) for redundancy
  4. Output clean edge list for the global optimizer
"""

from __future__ import annotations

from typing import Any


def select_mst_edges(
    part_ids: list[str],
    candidates: list[dict[str, Any]],
    extra_edges: int = 0,
) -> list[dict[str, Any]]:
    """Select edges via Maximum Spanning Tree, optionally adding cycles.

    Args:
        part_ids:   Ordered list of part IDs.
        candidates: List of candidate edges, each with:
                      'src': part_id
                      'dst': part_id
                      'T_rel': 4×4 numpy array (measured relative transform)
                      'score': float (higher = better)
        extra_edges: How many additional edges (beyond MST) to include
                     for cycle redundancy.  0 = tree only.

    Returns:
        Selected edge list (n-1 + extra_edges items).
    """
    import networkx as nx

    n = len(part_ids)
    if n <= 1:
        return []

    # Build directed graph with scores as weights
    G = nx.Graph()
    for pid in part_ids:
        G.add_node(pid)

    for c in candidates:
        s, d = c["src"], c["dst"]
        w = c.get("score", c.get("weight", 1.0))
        if G.has_edge(s, d):
            # Keep the candidate with higher score
            if w > G[s][d].get("weight", 0):
                G[s][d]["weight"] = w
                G[s][d]["candidate"] = c
        else:
            G.add_edge(s, d, weight=w, candidate=c)

    # Check connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        raise ValueError(
            f"Part graph is not fully connected. Components: "
            f"{[sorted(c) for c in components]}. "
            f"Missing edges between: {_missing_edges(components, candidates)}"
        )

    # Maximum spanning tree (negate weights for min spanning tree → max)
    neg_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        neg_G.add_edge(u, v, weight=-data["weight"], candidate=data["candidate"])
    mst = nx.minimum_spanning_tree(neg_G)

    selected = [mst[u][v]["candidate"] for u, v in mst.edges()]

    # Add extra edges (highest-score remaining edges that add a cycle)
    if extra_edges > 0:
        remaining = []
        for u, v, data in G.edges(data=True):
            if not mst.has_edge(u, v):
                remaining.append((data["weight"], data["candidate"]))
        remaining.sort(key=lambda x: -x[0])  # descending by score
        for _, c in remaining[:extra_edges]:
            selected.append(c)

    return selected


def _missing_edges(components, candidates):
    """Give a hint about which parts are disconnected."""
    parts_by_comp = [set(c) for c in components]
    missing = []
    for i in range(len(parts_by_comp)):
        for j in range(i + 1, len(parts_by_comp)):
            missing.append(f"comp{i}({sorted(parts_by_comp[i])}) <-> comp{j}({sorted(parts_by_comp[j])})")
    return missing


def build_graph_from_pairwise_results(
    part_ids: list[str],
    pairwise_results: dict[tuple[str, str], dict[str, Any]],
    primary_rank: int = 1,
    fallback_ranks: tuple[int, ...] = (2, 3),
) -> list[dict[str, Any]]:
    """Build candidate edge list from a dict of pairwise JoinABLe results.

    Args:
        part_ids:          All part IDs in the assembly.
        pairwise_results:  Dict keyed by (part_a, part_b) tuple, value is the
                           JoinABLe E2E result dict (with 'pose_search'.'results').
        primary_rank:      Use this rank (1-based) as the primary candidate.
        fallback_ranks:    If primary has no pose, try these ranks.

    Returns:
        List of candidate edge dicts.
    """
    candidates = []
    for (a, b), r in pairwise_results.items():
        if a not in part_ids or b not in part_ids:
            continue

        ps = r.get("pose_search", {})
        ps_results = ps.get("results", [])

        for rank in [primary_rank] + list(fallback_ranks):
            # rank is 1-based, results list is 0-based
            idx = rank - 1
            if idx >= len(ps_results):
                continue
            pr = ps_results[idx]
            if pr.get("exact_collision", {}).get("status") != "success":
                continue

            # Build 4×4 transform from placement dict
            import numpy as np
            from scipy.spatial.transform import Rotation

            T = np.eye(4)
            placement = pr.get("placement_part_b_in_part_a_frame", {})
            for rs in reversed(placement.get("rotate_sequence", [])):
                aa = rs["axis_angle"]
                R = Rotation.from_rotvec(
                    np.array(aa[:3]) * aa[3] * np.pi / 180
                ).as_matrix()
                T[:3, :3] = R @ T[:3, :3]
            T[:3, 3] = np.array(placement.get("translate", [0, 0, 0]))

            # Score from prediction (higher score → better)
            score = pr.get("prediction_score", pr.get("evaluation", {}).get("cost", 1.0))
            # Negate cost if it looks like a cost (lower is better)
            if isinstance(score, (int, float)) and score < 0:
                score = abs(score)

            candidates.append({
                "src": a,
                "dst": b,
                "T_rel": T,
                "score": float(score),
                "rank": rank,
                "collision_free": True,
            })
            break  # Stop trying fallback ranks once we found a collision-free one

    return candidates
