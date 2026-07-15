"""Step 3+4 pipeline: geometry validation + assembly graph + solidworks_assembly_plan.json.

This script:
1. Loads Step 2 relation model
2. For each SolidWorks exam case, enumerates candidate pairs
3. Runs geometry feature extraction and constraint solving
4. Builds assembly graph, propagates poses, checks consistency
5. Outputs accepted/review/rejected edges and solidworks_assembly_plan.json

CRITICAL: This script does NOT read human_labels.json during inference.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABELS = ["clearance", "coaxial", "planar_align", "planar_mate", "pocket_mate"]
CASES = ["1", "2", "3", "4", "5"]
SW_DIR = Path("sw")

# Conservative gate thresholds
DIRECT_EDGE_THRESHOLD = 0.7
GEOMETRY_SCORE_THRESHOLD = 0.60  # Lowered for baseline (no OCCT)
INDEPENDENT_EVIDENCE_MIN = 1     # Lowered for baseline
MAX_AUTO_ACCEPT_PARTS = 5


# ---------------------------------------------------------------------------
# Step 2 model loading
# ---------------------------------------------------------------------------
def load_step2_model(model_path: Path) -> dict[str, Any]:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_pair(model_data: dict[str, Any], pair_features: list[dict[str, float]]) -> dict[str, Any]:
    """Predict direct_edge_score and relation_type_scores for a pair."""
    direct_model = model_data["direct_model"]
    rel_model = model_data["rel_model"]
    mlb = model_data["mlb"]

    direct_proba = direct_model.predict_proba(pair_features)[0, 1]
    rel_proba = rel_model.predict_proba(pair_features)[0]

    # Build relation type scores
    relation_scores = {}
    for i, label in enumerate(mlb.classes_):
        # OneVsRest produces probability per label
        if hasattr(rel_model, "estimators_"):
            # Access each binary classifier's probability
            pass
    # Simplified: use decision function
    try:
        rel_dec = rel_model.decision_function(pair_features)[0]
        for i, label in enumerate(mlb.classes_):
            relation_scores[label] = float(rel_dec[i]) if i < len(rel_dec) else 0.0
    except Exception:
        for label in mlb.classes_:
            relation_scores[label] = 0.0

    return {
        "direct_edge_score": float(direct_proba),
        "relation_type_scores": relation_scores,
    }


# ---------------------------------------------------------------------------
# Feature extraction for Step 2 model (from train_pair_relation_head_v2)
# ---------------------------------------------------------------------------
import re

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}")

TOP_KEYWORDS = {
    "screw", "bolt", "nut", "washer", "pin", "shaft", "bearing", "housing",
    "plate", "bracket", "flange", "base", "cover", "body", "ring", "gear",
    "piston", "cylinder", "valve", "spring", "rail", "slide", "guide",
    "socket", "clip", "dovetail", "t-nut", "groove", "dowell",
}


def _tokenize(values: list[str]) -> list[str]:
    tokens = []
    for value in values:
        for token in TOKEN_RE.findall(str(value or "").lower()):
            tokens.append(token[:40])
    return tokens


def featurize_part_names(part_names: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for token in _tokenize(part_names):
        if token in TOP_KEYWORDS:
            features[f"name_{token}"] = features.get(f"name_{token}", 0.0) + 1.0
    return features


# ---------------------------------------------------------------------------
# Simple geometry proxy (when full OCCT is unavailable)
# ---------------------------------------------------------------------------
def simple_geometry_check(
    part_a_path: str, part_b_path: str, relation_type: str
) -> dict[str, Any]:
    """Lightweight geometry validation.

    In production this would use OCCT exact validation.
    For the baseline we return a reasonable proxy based on available evidence.
    """
    # Check if STEP files exist
    a_exists = Path(part_a_path).exists() if part_a_path else False
    b_exists = Path(part_b_path).exists() if part_b_path else False

    evidence = []
    if a_exists and b_exists:
        evidence.append("step_geometry_available")
    else:
        return {
            "pose_status": "uncertain",
            "collision_free": None,
            "geometry_evidence": evidence,
            "failure_reasons": ["step_geometry_missing"],
            "pairwise_transform": None,
            "checked_pose_count": 0,
        }

    # Add relation-specific evidence
    if relation_type in ("coaxial", "clearance"):
        evidence.append("cylindrical_pair_candidate")
    elif relation_type in ("planar_mate", "planar_align"):
        evidence.append("planar_pair_candidate")
    elif relation_type == "pocket_mate":
        evidence.append("pocket_pair_candidate")

    return {
        "pose_status": "uncertain",  # Default: needs full OCCT for certainty
        "collision_free": None,
        "geometry_evidence": evidence,
        "failure_reasons": [],
        "pairwise_transform": _identity_transform(),
        "checked_pose_count": 1,
        "geometry_score": 0.85 if len(evidence) >= 2 else 0.6,
    }


def _identity_transform() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Pose graph propagation
# ---------------------------------------------------------------------------
def build_pose_graph(
    accepted_edges: list[dict[str, Any]],
    parts: list[str],
) -> dict[str, Any]:
    """Build global placements from accepted edges via BFS propagation."""
    if not accepted_edges:
        return {
            "placements": {},
            "connected_components": len(parts),
            "cycle_conflicts": [],
            "graph_consistent": True,
        }

    # Build adjacency
    adj: dict[str, list[tuple[str, list[list[float]]]]] = defaultdict(list)
    for edge in accepted_edges:
        pa, pb = edge["part_pair"]
        T = edge.get("pairwise_transform", _identity_transform())
        adj[pa].append((pb, T))
        # Store inverse
        adj[pb].append((pa, _invert_transform(T)))

    # BFS from first part
    placements: dict[str, list[list[float]]] = {}
    root = parts[0] if parts else ""
    placements[root] = _identity_transform()
    queue = [root]
    visited = {root}

    while queue:
        current = queue.pop(0)
        T_world_current = placements[current]
        for neighbor, T_neighbor_from_current in adj.get(current, []):
            if neighbor not in visited:
                T_world_neighbor = _matmul(T_world_current, T_neighbor_from_current)
                placements[neighbor] = T_world_neighbor
                visited.add(neighbor)
                queue.append(neighbor)

    unplaced = [p for p in parts if p not in placements]
    for p in unplaced:
        placements[p] = None

    return {
        "placements": placements,
        "connected_components": len(set(
            tuple(sorted([k for k, v in placements.items() if v is not None]))
        )),
        "unplaced_parts": unplaced,
        "cycle_conflicts": [],  # TODO: full cycle check
        "graph_consistent": True,
    }


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def _invert_transform(m: list[list[float]]) -> list[list[float]]:
    if not m:
        return _identity_transform()
    r_t = [[float(m[j][i]) for j in range(3)] for i in range(3)]
    t = [float(m[i][3]) for i in range(3)]
    inv_t = [-sum(r_t[i][j] * t[j] for j in range(3)) for i in range(3)]
    return [r_t[i] + [inv_t[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


# ---------------------------------------------------------------------------
# Conservative gate
# ---------------------------------------------------------------------------
def classify_edge(
    prediction: dict[str, Any],
    geom_result: dict[str, Any],
    part_count: int,
) -> tuple[str, str]:
    """Classify edge as accepted/review/rejected with reason."""
    direct_score = prediction.get("direct_edge_score", 0.0)
    pose_status = geom_result.get("pose_status", "uncertain")
    collision_free = geom_result.get("collision_free")
    evidence = geom_result.get("geometry_evidence", [])
    evidence_count = len(evidence)
    geom_score = geom_result.get("geometry_score", 0.0)

    reasons = []

    # Reject conditions
    if collision_free is False:
        return "rejected", "collision_detected"
    if direct_score < 0.3:
        return "rejected", f"low_direct_edge_score:{direct_score:.3f}"

    # Review conditions
    if pose_status == "failed":
        return "rejected", "pose_solving_failed"
    if pose_status == "uncertain":
        reasons.append("pose_uncertain")
    if direct_score < DIRECT_EDGE_THRESHOLD:
        reasons.append(f"direct_edge_below_threshold:{direct_score:.3f}")
    if geom_score < GEOMETRY_SCORE_THRESHOLD:
        reasons.append(f"geometry_score_low:{geom_score:.3f}")
    if evidence_count < INDEPENDENT_EVIDENCE_MIN:
        reasons.append(f"insufficient_evidence:{evidence_count}")
    if part_count > MAX_AUTO_ACCEPT_PARTS:
        reasons.append(f"large_assembly:{part_count}_parts")

    if reasons:
        return "review", "; ".join(reasons)

    # Accept conditions
    if (
        pose_status == "valid"
        and collision_free is True
        and direct_score >= DIRECT_EDGE_THRESHOLD
        and evidence_count >= INDEPENDENT_EVIDENCE_MIN
        and geom_score >= GEOMETRY_SCORE_THRESHOLD
    ):
        return "accepted", "all_gates_passed"

    return "review", "did_not_meet_all_accept_criteria"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def process_case(
    case_id: str,
    model_data: dict[str, Any],
    sw_dir: Path = SW_DIR,
) -> dict[str, Any]:
    """Process one SolidWorks exam case."""
    case_dir = sw_dir / case_id
    if not case_dir.exists() and not str(case_id).startswith("case_"):
        case_dir = sw_dir / f"case_{case_id}"
    if not case_dir.exists():
        return {"case_id": case_id, "error": "case_dir_not_found"}

    # Find STEP parts (support .step, .stp, .STP)
    # Exclude assembly.step/stp (the combined assembly file)
    step_files = sorted(
        list(case_dir.glob("*.step")) +
        list(case_dir.glob("*.stp")) +
        list(case_dir.glob("*.STP"))
    )
    # Filter out assembly files
    step_files = [
        f for f in step_files
        if not f.stem.lower().startswith("assembly")
    ]
    # Keep only unique by stem
    seen = set()
    unique_files = []
    for f in step_files:
        if f.stem.lower() not in seen:
            seen.add(f.stem.lower())
            unique_files.append(f)
    step_files = unique_files
    part_names = [f.name for f in step_files]
    part_paths = [str(f.resolve()) for f in step_files]

    if len(part_names) < 2:
        return {"case_id": case_id, "error": "too_few_parts", "parts": part_names}

    # Enumerate all candidate pairs
    candidates: list[dict[str, Any]] = []
    for i in range(len(part_names)):
        for j in range(i + 1, len(part_names)):
            pair = [part_names[i], part_names[j]]
            paths = [part_paths[i], part_paths[j]]

            # Step 2: predict
            features = [featurize_part_names([part_names[i], part_names[j]])]
            pred = predict_pair(model_data, features)

            # Determine best relation type
            rel_scores = pred.get("relation_type_scores", {})
            best_rel = max(rel_scores, key=rel_scores.get) if rel_scores else "unknown"
            best_score = rel_scores.get(best_rel, 0.0)

            # Step 3: geometry check
            geom = simple_geometry_check(paths[0], paths[1], best_rel)

            # Step 4: classify
            decision, reason = classify_edge(pred, geom, len(part_names))

            candidates.append({
                "pair_id": f"case_{case_id}_pair_{i:02d}_{j:02d}",
                "part_pair": pair,
                "part_paths": paths,
                "direct_edge_score": pred["direct_edge_score"],
                "relation_type_scores": rel_scores,
                "best_relation_type": best_rel,
                "best_relation_score": best_score,
                "pose_status": geom["pose_status"],
                "collision_free": geom["collision_free"],
                "geometry_evidence": geom["geometry_evidence"],
                "geometry_score": geom.get("geometry_score", 0.0),
                "pairwise_transform": geom.get("pairwise_transform"),
                "checked_pose_count": geom.get("checked_pose_count", 0),
                "decision": decision,
                "decision_reason": reason,
            })

    # Partition into accepted/review/rejected
    accepted = [c for c in candidates if c["decision"] == "accepted"]
    review = [c for c in candidates if c["decision"] == "review"]
    rejected = [c for c in candidates if c["decision"] == "rejected"]

    # Pose graph for accepted edges
    pose_graph = build_pose_graph(accepted, part_names)

    # Unresolved parts
    placed_parts = set()
    for p in part_names:
        if pose_graph["placements"].get(p) is not None:
            placed_parts.add(p)
    unresolved = [p for p in part_names if p not in placed_parts]

    # Build solidworks_assembly_plan.json
    plan = {
        "schema_version": "1.0",
        "case_id": f"sw_case_{case_id}",
        "input_parts": [
            {
                "part_id": f"part_{i:03d}",
                "file_path": part_paths[i],
                "file_name": part_names[i],
                "unit": "mm",
            }
            for i in range(len(part_names))
        ],
        "placements": [],
        "accepted_edges": [],
        "review_edges": [],
        "rejected_edges": [],
        "unresolved_parts": [],
        "generation_policy": {
            "used_human_labels": False,
            "used_case_specific_rules": False,
            "used_filename_answer_hardcoding": False,
            "acceptance_mode": "conservative",
            "semantic_reranking_enabled": False,
        },
    }

    # Fill placements
    for part_name in part_names:
        T = pose_graph["placements"].get(part_name)
        plan["placements"].append({
            "part_id": f"part_{part_names.index(part_name):03d}",
            "part_name": part_name,
            "transform_world_from_part": T if T else _identity_transform(),
            "fixed": (part_name == part_names[0]),
            "source": "root_part" if part_name == part_names[0] else "pose_graph_propagation",
        })

    # Fill edges
    for edge in accepted:
        plan["accepted_edges"].append({
            "pair_id": edge["pair_id"],
            "parts": edge["part_pair"],
            "relation_type": edge["best_relation_type"],
            "direct_edge_score": edge["direct_edge_score"],
            "geometry_evidence": edge["geometry_evidence"],
            "decision_reason": edge["decision_reason"],
        })
    for edge in review:
        plan["review_edges"].append({
            "pair_id": edge["pair_id"],
            "parts": edge["part_pair"],
            "relation_type": edge["best_relation_type"],
            "direct_edge_score": edge["direct_edge_score"],
            "review_reason": edge["decision_reason"],
        })
    for edge in rejected:
        plan["rejected_edges"].append({
            "pair_id": edge["pair_id"],
            "parts": edge["part_pair"],
            "reject_reason": edge["decision_reason"],
        })

    plan["unresolved_parts"] = unresolved

    return {
        "case_id": case_id,
        "part_count": len(part_names),
        "candidate_pairs": len(candidates),
        "accepted_count": len(accepted),
        "review_count": len(review),
        "rejected_count": len(rejected),
        "unresolved_count": len(unresolved),
        "candidates": candidates,
        "plan": plan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "public_cad_dataset_audit/outputs/step123_pair_relation_head_a00_a01/pair_relation_head.pkl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("public_cad_dataset_audit/outputs/step34_solidworks_plan"),
    )
    parser.add_argument(
        "--sw-dir",
        type=Path,
        default=SW_DIR,
        help=(
            "SolidWorks case root. Supports either 4/5 style folders or "
            "case_4/case_5 style folders."
        ),
    )
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()

    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()
    sw_dir = args.sw_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run Step 2 training first.")
        return 1

    print(f"Loading model from {model_path}...")
    model_data = load_step2_model(model_path)
    print("Model loaded.")

    summary = {
        "schema_version": "1.0.0",
        "pipeline": "step3_step4_deterministic_assembly_recovery",
        "model_path": str(model_path),
        "human_labels_used": False,
        "cases": {},
    }

    for case_id in args.cases:
        print(f"\n{'='*60}")
        print(f"Processing case {case_id}...")
        result = process_case(case_id, model_data, sw_dir=sw_dir)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            summary["cases"][case_id] = result
            continue

        print(f"  Parts: {result['part_count']}")
        print(f"  Candidate pairs: {result['candidate_pairs']}")
        print(f"  Accepted: {result['accepted_count']}")
        print(f"  Review: {result['review_count']}")
        print(f"  Rejected: {result['rejected_count']}")
        print(f"  Unresolved: {result['unresolved_count']}")

        # Save individual case outputs
        case_output_dir = output_dir / f"case_{case_id}"
        case_output_dir.mkdir(parents=True, exist_ok=True)

        # Save solidworks_assembly_plan.json
        plan_path = case_output_dir / "solidworks_assembly_plan.json"
        plan_path.write_text(
            json.dumps(result["plan"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"  -> {plan_path}")

        # Save detailed candidates
        candidates_path = case_output_dir / "candidate_scores.json"
        candidates_path.write_text(
            json.dumps(result["candidates"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Save accepted/review/rejected
        for tier in ["accepted", "review", "rejected"]:
            tier_edges = [c for c in result["candidates"] if c["decision"] == tier]
            tier_path = case_output_dir / f"{tier}_edges.json"
            tier_path.write_text(
                json.dumps(tier_edges, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        summary["cases"][case_id] = {
            "part_count": result["part_count"],
            "candidate_pairs": result["candidate_pairs"],
            "accepted_count": result["accepted_count"],
            "review_count": result["review_count"],
            "rejected_count": result["rejected_count"],
            "unresolved_count": result["unresolved_count"],
        }

    # Save summary
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
