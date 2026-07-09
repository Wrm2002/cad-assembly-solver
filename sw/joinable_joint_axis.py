"""
joinable_joint_axis.py — Extract joint axes from JoinABLe predictions.

Two capabilities:
  1. Score boost: JoinABLe predictions boost existing analytic constraint scores
  2. Learned coaxial: high-confidence cylinder+cylinder predictions are converted
     to clearance constraints with precise axis origin+direction parameters
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(v: list[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in v))
    if length < 1e-12:
        return [0.0, 0.0, 1.0]
    return [x / length for x in v]


def load_brep_graphs(case_dir: Path) -> dict[str, dict[str, Any]]:
    graph_dir = case_dir / "_brep_graphs311"
    if not graph_dir.is_dir():
        return {}
    graphs = {}
    for graph_file in sorted(graph_dir.glob("*_graph.json")):
        stem = graph_file.stem.replace("_graph", "")
        graphs[stem] = _read_json(graph_file)
    return graphs


def load_joinable_predictions(case_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    joinable_dir = case_dir / "_joinable"
    if not joinable_dir.is_dir():
        return {}
    predictions = {}
    for pred_file in sorted(joinable_dir.glob("pred_*.json")):
        pred = _read_json(pred_file)
        part_a = Path(pred.get("part_a", "")).stem
        part_b = Path(pred.get("part_b", "")).stem
        pair = tuple(sorted((part_a, part_b)))
        candidates = pred.get("candidates", [])
        if pair in predictions:
            predictions[pair].extend(candidates)
        else:
            predictions[pair] = list(candidates)
    return predictions


# ── Entity → geometry mapping ─────────────────────────────────────

def _find_cylinder_by_topology(
    topology_index: int,
    brep_nodes: list[dict[str, Any]],
    part_features: dict[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    """Match a JoinABLe entity to a features.py cylinder by centroid proximity.
    Returns (cylinder_dict, index_in_list) or (None, -1)."""
    brep_node = None
    for node in brep_nodes:
        if node.get("occt_topology_index") == topology_index:
            brep_node = node
            break
    if brep_node is None:
        return None, -1
    centroid = brep_node.get("centroid")
    if not centroid or len(centroid) != 3:
        return None, -1
    cylinders = part_features.get("cylinders", [])
    if not cylinders:
        return None, -1
    best_cyl, best_idx, best_dist = None, -1, float("inf")
    for i, cyl in enumerate(cylinders):
        origin = cyl.get("origin", [0, 0, 0])
        dist = sum((centroid[i] - origin[i]) ** 2 for i in range(3))
        if dist < best_dist:
            best_dist = dist
            best_cyl = cyl
            best_idx = i
    return best_cyl, best_idx


def extract_joint_axis(
    entity: dict[str, Any],
    brep_nodes: list[dict[str, Any]],
    part_features: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    """Extract (origin, direction) from a JoinABLe entity prediction."""
    topo = entity.get("topology_index")
    etype = entity.get("entity_type")
    geo = entity.get("geometry_type")
    if topo is None:
        return None

    if etype == "face" and geo == "cylinder":
        cyl, _ = _find_cylinder_by_topology(topo, brep_nodes, part_features)
        if cyl:
            return (list(cyl.get("origin", [0, 0, 0])), _norm(list(cyl.get("axis", [0, 0, 1]))))
    return None


# ── Learned coaxial constraint generation ─────────────────────────

def derive_learned_coaxial(
    case_dir: Path,
    parts_features: dict[str, dict[str, Any]],
    top_k: int = 3,
    min_score: float = 2.0,
) -> list[dict[str, Any]]:
    """Generate clearance constraints from high-confidence JoinABLe coaxial predictions."""
    brep_graphs = load_brep_graphs(case_dir)
    predictions = load_joinable_predictions(case_dir)
    if not predictions or not brep_graphs:
        return []

    stem_to_key = {Path(k).stem: k for k in parts_features}
    constraints = []

    for (stem_a, stem_b), candidates in predictions.items():
        key_a = stem_to_key.get(stem_a)
        key_b = stem_to_key.get(stem_b)
        if not key_a or not key_b:
            continue

        nodes_a = brep_graphs.get(stem_a, {}).get("nodes", [])
        nodes_b = brep_graphs.get(stem_b, {}).get("nodes", [])
        feats_a = parts_features[key_a]
        feats_b = parts_features[key_b]

        for c in candidates[:top_k]:
            if c.get("joint_family_candidate") != "coaxial_or_cylindrical":
                continue
            score = float(c.get("score", 0))
            if score < min_score:
                continue

            ea = c.get("part_a_entity", {})
            eb = c.get("part_b_entity", {})
            if ea.get("geometry_type") != "cylinder" or eb.get("geometry_type") != "cylinder":
                continue

            axis_a = extract_joint_axis(ea, nodes_a, feats_a)
            axis_b = extract_joint_axis(eb, nodes_b, feats_b)
            if not axis_a or not axis_b:
                continue

            cyl_a, idx_a = _find_cylinder_by_topology(ea.get("topology_index", 0), nodes_a, feats_a)
            cyl_b, idx_b = _find_cylinder_by_topology(eb.get("topology_index", 0), nodes_b, feats_b)
            if cyl_a is None or cyl_b is None:
                continue
            ra = cyl_a.get("radius", 0)
            rb = cyl_b.get("radius", 0)

            # shaft = smaller radius, bore = larger
            if ra <= rb:
                shaft_key, bore_key = key_a, key_b
                shaft_feat_idx, bore_feat_idx = idx_a, idx_b
                gap = rb - ra
                radius = ra
            else:
                shaft_key, bore_key = key_b, key_a
                shaft_feat_idx, bore_feat_idx = idx_b, idx_a
                gap = ra - rb
                radius = rb

            constraints.append({
                "type": "clearance",
                "parts": (shaft_key, bore_key),
                "feat_a_idx": shaft_feat_idx,
                "feat_b_idx": bore_feat_idx,
                "gap": gap,
                "_sort_key": -score,
                "_radius_a": radius,
                "_joinable_rank": c.get("rank", 999),
                "_joinable_score": score,
                "_learned_axis_origin": axis_a[0] if shaft_key == key_a else axis_b[0],
                "_learned_axis_direction": axis_a[1] if shaft_key == key_a else axis_b[1],
                "candidate_origin": "joinable_learned_joint_axis",
            })

    return constraints


# ── Main entry point ──────────────────────────────────────────────

def inject_joinable_constraints(
    case_dir: Path,
    parts_features: dict[str, dict[str, Any]],
    raw_matches: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Enhance constraints with JoinABLe evidence: score boost + learned coaxial."""
    predictions = load_joinable_predictions(case_dir)

    # ── Score boost lookup ──
    boost_map: dict[tuple, float] = {}
    if predictions:
        for (sa, sb), cands in predictions.items():
            pair = tuple(sorted((sa, sb)))
            for c in cands[:top_k]:
                fam = c.get("joint_family_candidate", "")
                sc = float(c.get("score", 0))
                grp = "coaxial" if fam == "coaxial_or_cylindrical" else ("planar" if fam == "planar" else "other")
                key = (pair, grp)
                if key not in boost_map or sc > boost_map[key]:
                    boost_map[key] = sc

    # ── Apply score boosts ──
    enhanced = []
    pairs_with_coaxial: set[tuple[str, str]] = set()
    for m in raw_matches:
        m = dict(m)
        pa, pb = Path(m["parts"][0]).stem, Path(m["parts"][1]).stem
        pair = tuple(sorted((pa, pb)))
        mt = m["type"]
        grp = "coaxial" if mt in ("coaxial", "clearance") else ("planar" if mt in ("planar_mate", "planar_align") else "other")
        if grp == "coaxial":
            pairs_with_coaxial.add(pair)

        boost = boost_map.get((pair, grp), 0)
        if boost > 0:
            cur = float(m.get("score", m.get("_sort_key", 0)))
            lb = min(0.10, 0.02 * boost)
            m["score"] = round(cur + lb, 6) if isinstance(m.get("score"), (int, float)) else lb
            m["_joinable_boost"] = round(lb, 6)
            m["_joinable_max_score"] = round(boost, 3)
            m["candidate_origin"] = m.get("candidate_origin", "analytic") + "+joinable"
        enhanced.append(m)

    # ── Inject learned coaxial (only for pairs with existing coaxial evidence) ──
    learned = derive_learned_coaxial(case_dir, parts_features, top_k=3, min_score=2.0)
    for lc in learned:
        pair = tuple(sorted(Path(p).stem for p in lc["parts"]))
        if pair in pairs_with_coaxial:
            enhanced.append(lc)

    return enhanced
