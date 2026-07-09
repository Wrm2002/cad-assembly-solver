"""
joinable_inference.py — Run JoinABLe GNN inference on converted STEP pairs.

Pipeline:
  1. Convert STEP pair → B-Rep OBJ+JSON (step_to_fusion360_brep)
  2. Build joint graph from JSON graphs
  3. Load pretrained JoinABLe model
  4. Run GNN inference to get joint entity predictions
  5. Extract joint axes from top predictions
  6. Optionally run simplex pose search

Uses JoinABLe source code directly (models/joinable.py, datasets/).
"""
from __future__ import annotations

import sys, json, math, os
from pathlib import Path
from typing import Any
import numpy as np
import torch

# Add JoinABLe source to path
_JOINABLE_ROOT = Path(__file__).parent.parent / "joinable_source"
if str(_JOINABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_JOINABLE_ROOT))


def _build_single_graph(json_path):
    """Build a PyG Data object from our B-Rep JSON.
    
    Only includes features used by the pretrained model:
    entity_types, length, face_reversed, edge_reversed.
    Grid point features are NOT used by this checkpoint."""
    from torch_geometric.data import Data

    with open(json_path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    links = data["links"]
    face_count = sum(1 for n in nodes if "surface_type" in n)

    num_nodes = len(nodes)
    grid_size = 10

    # Feature tensors (grid will be empty — model doesn't use it)
    x = torch.zeros(num_nodes, grid_size, grid_size, 7)
    entity_types = torch.zeros(num_nodes, 16)
    is_face = torch.zeros(num_nodes, dtype=torch.long)
    area = torch.zeros(num_nodes)
    length = torch.zeros(num_nodes)
    face_reversed = torch.zeros(num_nodes, dtype=torch.long)
    edge_reversed = torch.zeros(num_nodes, dtype=torch.long)
    convexity = torch.zeros(num_nodes, 6)
    dihedral_angle = torch.zeros(num_nodes)
    radius = torch.zeros(num_nodes)

    type_to_idx = {
        "PlaneSurfaceType": 0, "CylinderSurfaceType": 1, "ConeSurfaceType": 2,
        "SphereSurfaceType": 3, "TorusSurfaceType": 4,
        "EllipticalCylinderSurfaceType": 5, "EllipticalConeSurfaceType": 6,
        "NurbsSurfaceType": 7,
        "Line3DCurveType": 8, "Arc3DCurveType": 9, "Circle3DCurveType": 10,
        "Ellipse3DCurveType": 11, "EllipticalArc3DCurveType": 12,
        "InfiniteLine3DCurveType": 13, "NurbsCurve3DCurveType": 14,
    }

    for i, node in enumerate(nodes):
        is_face[i] = 1 if "surface_type" in node else 0
        tname = node.get("surface_type") or node.get("curve_type") or "NurbsSurfaceType"
        tidx = type_to_idx.get(tname, 7)
        entity_types[i, tidx] = 1.0
        area[i] = node.get("area", 0) or 0
        length[i] = node.get("length", 0) or 0
        if node.get("reversed", False):
            if is_face[i]:
                face_reversed[i] = 1
            else:
                edge_reversed[i] = 1
        radius[i] = node.get("radius", 0) or 0
        conv = node.get("convexity", "None")
        conv_idx = {"None": 0, "Convex": 1, "Concave": 2, "Smooth": 3}.get(conv, 0)
        convexity[i, conv_idx] = 1.0

    # Edge index
    if links:
        src = [l["source"] for l in links]
        dst = [l["target"] for l in links]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Center+scale area/length (approximate)
    if area.sum() > 0:
        scale = 1.0 / (area.max().sqrt().item() + 1e-8)
        area *= scale
        length *= scale

    g = Data(
        x=x, edge_index=edge_index,
        entity_types=entity_types, is_face=is_face,
        area=area, length=length,
        face_reversed=face_reversed, edge_reversed=edge_reversed,
        convexity=convexity, dihedral_angle=dihedral_angle,
        radius=radius,
    )
    return g, face_count


def _build_joint_graph(g1, g2):
    """Build joint graph connecting all nodes of g1 to all nodes of g2."""
    from torch_geometric.data import Data

    n1 = g1.num_nodes
    n2 = g2.num_nodes

    # Cross-part edges: all g1 nodes → all g2 nodes
    src = torch.arange(n1).repeat_interleave(n2)
    dst = torch.arange(n2).repeat(n1)
    edge_index = torch.stack([src, dst], dim=0)

    # Edge labels (all 0 = unknown, for inference)
    edge_attr = torch.zeros(n1 * n2, dtype=torch.long)

    jg = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    jg.num_nodes_graph1 = n1
    jg.num_nodes_graph2 = n2
    return jg


def load_joinable_model(checkpoint_path=None):
    """Load pretrained JoinABLe model by direct state_dict loading.
    Bypasses pytorch_lightning compatibility issues."""
    from models.joinable import JoinABLe

    if checkpoint_path is None:
        checkpoint_path = _JOINABLE_ROOT / "pretrained/paper/last_run_0.ckpt"

    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False)
    hp = ckpt["hyper_parameters"]["args"]

    # Build model with same architecture
    model = JoinABLe(
        hidden_dim=hp.hidden,
        input_features=hp.input_features,
        dropout=hp.dropout,
        mpn=hp.mpn,
        batch_norm=hp.batch_norm,
        reduction=hp.reduction,
        post_net=hp.post_net,
        pre_net=hp.pre_net,
    )

    # Load state dict (strip 'model.' prefix)
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[6:]] = v  # remove 'model.' prefix
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_with_model(model, g1, g2, jg):
    """Run model inference, handling PyG API compatibility."""
    import torch.nn.functional as F

    with torch.no_grad():
        # Run pre-nets (face + edge)
        x1_face, x2_face = model.pre_face(g1, g2)
        x1_edge, x2_edge = model.pre_edge(g1, g2)
        x1 = x1_edge + x1_face
        x2 = x2_edge + x2_face
        # Message passing
        x1 = model.mpn(x1, g1.edge_index)
        x2 = model.mpn(x2, g2.edge_index)
        # Post-net: interleave features (single pair, no batching needed)
        x = torch.cat([x1, x2], dim=0)
        logits = model.post.mpn(x, jg.edge_index)
        probs = F.softmax(logits, dim=0)

    return probs


def predict_joint_entities(model, g1, g2, jg, top_k=10):
    """Run GNN inference, return top-K (face1_idx, face2_idx, probability)."""
    probs = _predict_with_model(model, g1, g2, jg)

    # Reshape to n1 x n2 matrix
    n1 = g1.num_nodes
    n2 = g2.num_nodes
    prob_mat = probs.view(n1, n2)

    # Get face-only indices
    face1_mask = g1.is_face > 0.5
    face2_mask = g2.is_face > 0.5

    # Get top-K face-face predictions
    face_probs = prob_mat[face1_mask][:, face2_mask]
    face1_idx = torch.where(face1_mask)[0]
    face2_idx = torch.where(face2_mask)[0]

    flat_probs = face_probs.flatten()
    top_values, top_indices = torch.topk(flat_probs, min(top_k, len(flat_probs)))

    results = []
    for val, idx in zip(top_values, top_indices):
        fi = idx // face2_idx.shape[0]
        fj = idx % face2_idx.shape[0]
        results.append({
            "face1_idx": int(face1_idx[fi]),
            "face2_idx": int(face2_idx[fj]),
            "probability": float(val),
        })
    return results


def extract_joint_axis(g, face_idx, json_nodes):
    """Extract joint axis (origin, direction) from a face node."""
    node = json_nodes[face_idx]
    stype = node.get("surface_type") or node.get("curve_type") or ""

    if "Cylinder" in stype or "Cone" in stype:
        origin = node.get("origin", {"x": 0, "y": 0, "z": 0})
        axis = node.get("axis", {"x": 0, "y": 0, "z": 1})
        return (
            np.array([origin["x"], origin["y"], origin["z"]]),
            np.array([axis["x"], axis["y"], axis["z"]]),
        )
    elif "Plane" in stype:
        origin = node.get("origin", {"x": 0, "y": 0, "z": 0})
        normal = node.get("normal", {"x": 0, "y": 0, "z": 1})
        return (
            np.array([origin["x"], origin["y"], origin["z"]]),
            np.array([normal["x"], normal["y"], normal["z"]]),
        )
    elif "Circle" in stype:
        origin = node.get("origin", {"x": 0, "y": 0, "z": 0})
        axis = node.get("axis", {"x": 0, "y": 0, "z": 1})
        return (
            np.array([origin["x"], origin["y"], origin["z"]]),
            np.array([axis["x"], axis["y"], axis["z"]]),
        )
    elif "Line" in stype:
        origin = node.get("origin", {"x": 0, "y": 0, "z": 0})
        direction = node.get("direction", {"x": 0, "y": 0, "z": 1})
        return (
            np.array([origin["x"], origin["y"], origin["z"]]),
            np.array([direction["x"], direction["y"], direction["z"]]),
        )
    return (np.zeros(3), np.array([0.0, 0.0, 1.0]))


def run_inference_on_pair(step1, step2, output_dir, model=None, top_k=5):
    """Run full JoinABLe inference on a pair of STEP files.

    Returns list of {face1_idx, face2_idx, probability, axis1, axis2}
    """
    from step_to_fusion360_brep import convert_step_to_brep

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert both parts
    r1 = convert_step_to_brep(step1, output_dir)
    r2 = convert_step_to_brep(step2, output_dir)

    # Build graphs
    g1, fc1 = _build_single_graph(r1["json"])
    g2, fc2 = _build_single_graph(r2["json"])

    g1.num_nodes_graph1 = g1.num_nodes
    g2.num_nodes_graph1 = g2.num_nodes

    # Load JSON data for axis extraction
    with open(r1["json"]) as f:
        nodes1 = json.load(f)["nodes"]
    with open(r2["json"]) as f:
        nodes2 = json.load(f)["nodes"]

    # Build joint graph
    jg = _build_joint_graph(g1, g2)

    # Load model
    if model is None:
        model = load_joinable_model()

    # Predict
    predictions = predict_joint_entities(model, g1, g2, jg, top_k=top_k)

    # Extract axes
    for pred in predictions:
        pred["axis1"] = extract_joint_axis(g1, pred["face1_idx"], nodes1)
        pred["axis2"] = extract_joint_axis(g2, pred["face2_idx"], nodes2)

    return predictions


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("step1", help="First STEP file")
    parser.add_argument("step2", help="Second STEP file")
    parser.add_argument("--output", "-o", default="brep_output", help="Output dir")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    preds = run_inference_on_pair(args.step1, args.step2, args.output, top_k=args.top_k)

    print(f"\n=== Top {len(preds)} Joint Entity Predictions ===")
    for i, p in enumerate(preds):
        a1 = p["axis1"]
        a2 = p["axis2"]
        print(f"  [{i}] face1={p['face1_idx']} face2={p['face2_idx']} prob={p['probability']:.4f}")
        print(f"       axis1: origin={a1[0].round(2)} dir={a1[1].round(2)}")
        print(f"       axis2: origin={a2[0].round(2)} dir={a2[1].round(2)}")
