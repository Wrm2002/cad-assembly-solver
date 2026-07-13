"""Candidate enumerator: try different edge-candidate combinations
until a collision-free global assembly is found.

Key insight: JoinABLe's simplex search gravitates to the nearest shaft end.
By trying candidates from different axis seeds, we can find placements
on both ends of the shaft without hard-coding geometry.
"""

from __future__ import annotations

import itertools
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def enumerate_candidate_combinations(
    part_ids: list[str],
    pairwise_results: dict[tuple[str, str], dict[str, Any]],
    max_combos: int = 27,
) -> list[list[dict[str, Any]]]:
    """Enumerate candidate edge combinations for global optimization.

    For each pair (src, dst), collect all collision-free pose results
    (from different axis seeds / prediction ranks).  Then enumerate
    the cross-product of choices.

    Args:
        part_ids:          All part IDs.
        pairwise_results:  Dict (src, dst) → JoinABLe E2E result JSON.
        max_combos:        Maximum number of combos to enumerate (safety limit).

    Returns:
        List of edge-lists, each being one candidate combination.
    """
    # Collect candidates per edge
    edge_candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (a, b), r in pairwise_results.items():
        if a not in part_ids or b not in part_ids:
            continue
        ps = r.get("pose_search", {})
        ps_results = ps.get("results", [])
        candidates = []
        seen_positions = set()  # de-dup by discretized axial position

        for pr in ps_results:
            if pr.get("exact_collision", {}).get("status") != "success":
                continue

            T = _placement_to_4x4(pr.get("placement_part_b_in_part_a_frame", {}))
            score = _extract_score(pr)

            # De-dup: discretize position to 5mm grid to avoid near-duplicates
            pos_bin = tuple(np.round(T[:3, 3] / 5.0).astype(int))
            if pos_bin in seen_positions:
                continue
            seen_positions.add(pos_bin)

            candidates.append({
                "src": a,
                "dst": b,
                "T_rel": T,
                "score": float(score),
                "rank": pr.get("prediction_rank", 0),
                "translate": T[:3, 3].tolist(),
            })

        # Sort by score descending
        candidates.sort(key=lambda c: -c["score"])
        edge_candidates[(a, b)] = candidates

    # Safety: limit candidates per edge to avoid combinatorial explosion
    max_per_edge = max(1, int(max_combos ** (1.0 / max(1, len(edge_candidates)))))
    for key in edge_candidates:
        edge_candidates[key] = edge_candidates[key][:max_per_edge]

    # Enumerate all combinations
    keys = list(edge_candidates.keys())
    candidate_lists = [edge_candidates[k] for k in keys]
    combos = []
    for combo in itertools.product(*candidate_lists):
        combos.append(list(combo))
        if len(combos) >= max_combos:
            break

    return combos


def _placement_to_4x4(placement: dict) -> np.ndarray:
    """Convert placement dict to 4×4 homogeneous matrix."""
    T = np.eye(4)
    for rs in reversed(placement.get("rotate_sequence", [])):
        aa = rs["axis_angle"]
        R = Rotation.from_rotvec(
            np.array(aa[:3]) * aa[3] * np.pi / 180
        ).as_matrix()
        T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = np.array(placement.get("translate", [0, 0, 0]))
    return T


def _extract_score(pr: dict) -> float:
    """Extract a positive score from a pose result (higher = better)."""
    score = pr.get("prediction_score", 0)
    if score is None:
        score = pr.get("evaluation", {}).get("contact", 0.0) * 10.0
    if isinstance(score, (int, float)) and score < 0:
        score = abs(score)
    return float(score)
