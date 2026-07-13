"""Inference adapter from JoinABLe entity candidates to learned pose hypotheses.

This is deliberately a pair-level adapter.  It returns multiple relative
transforms and scores; the existing multi-part SE(3) solver remains responsible
for selecting a globally consistent combination and OCCT remains the exact
physical gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .hypothesis import frame_from_entity


PATCH_POINT_COUNT = 32


def _torch() -> Any:
    import torch
    return torch


def _rotation_6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = np.asarray(value[:3], dtype=float)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.asarray(value[3:6], dtype=float)
    second -= float(np.dot(first, second)) * first
    second /= max(float(np.linalg.norm(second)), 1e-12)
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def _relative_transform(
    frame_a: np.ndarray, frame_b: np.ndarray, pose_vector: np.ndarray, scale_mm: float
) -> np.ndarray:
    local = np.eye(4)
    local[:3, :3] = _rotation_6d_to_matrix(pose_vector[3:])
    local[:3, 3] = np.asarray(pose_vector[:3], dtype=float) * float(scale_mm)
    return frame_a @ local @ np.linalg.inv(frame_b)


def _one_ring_embedding(embedding: Any, graph: Any, index: int) -> Any:
    edges = graph.edge_index
    neighbours = edges[1, edges[0] == int(index)]
    pooled = embedding[index] if neighbours.numel() == 0 else embedding[neighbours].mean(dim=0)
    return _torch().cat((embedding[index], pooled), dim=0)


def _local_entity_patch(raw_graph: dict[str, Any], index: int, frame: np.ndarray, scale_mm: float) -> np.ndarray:
    """Return the selected entity's sampled B-Rep patch in its local frame.

    The samples are extracted from the STEP B-Rep itself by the graph adapter;
    no analytic part-family or filename rule is used here.
    """
    by_index = {int(node["joinable_node_index"]): node for node in raw_graph.get("nodes") or []}
    node = by_index.get(int(index))
    if node is None:
        raise ValueError("selected_entity_not_in_raw_graph")
    points = np.asarray(node.get("patch_points") or [], dtype=float).reshape(-1, 3)
    directions = np.asarray(node.get("patch_directions") or [], dtype=float).reshape(-1, 3)
    if len(points) == 0 or len(directions) == 0:
        raise ValueError("selected_entity_patch_unavailable")
    if len(directions) != len(points):
        sample = np.linspace(0, len(directions) - 1, len(points)).round().astype(int)
        directions = directions[sample]
    sample = np.linspace(0, len(points) - 1, PATCH_POINT_COUNT).round().astype(int)
    points, directions = points[sample], directions[sample]
    rotation, origin = frame[:3, :3], frame[:3, 3]
    local_points = ((points - origin) @ rotation) / max(float(scale_mm), 1e-9)
    local_directions = directions @ rotation
    norms = np.linalg.norm(local_directions, axis=1, keepdims=True)
    local_directions = np.divide(
        local_directions, norms, out=np.zeros_like(local_directions), where=norms > 1e-9
    )
    return np.concatenate((local_points, local_directions), axis=1).astype(np.float32)


def _patch_interface_frame(raw_graph: dict[str, Any], index: int) -> np.ndarray:
    """Construct the same geometry-only face frame used by strong training.

    The released JoinABLe local frame is still used by its candidate generator.
    The new pose head, however, is trained in a deterministic sampled B-Rep
    interface frame (centroid, mean normal, in-face PCA direction).  Rebuild
    that frame from the cached STEP patch here so a learned 9D vector never
    crosses coordinate conventions.
    """
    by_index = {int(node["joinable_node_index"]): node for node in raw_graph.get("nodes") or []}
    node = by_index.get(int(index))
    if node is None or str(node.get("entity_type")) != "face":
        raise ValueError("strong_contact_pose_requires_face_entity")
    points = np.asarray(node.get("patch_points") or [], dtype=float).reshape(-1, 3)
    normals = np.asarray(node.get("patch_directions") or [], dtype=float).reshape(-1, 3)
    if len(points) < 3 or len(normals) == 0:
        raise ValueError("selected_entity_patch_unavailable")
    if len(normals) != len(points):
        normals = normals[np.linspace(0, len(normals) - 1, len(points)).round().astype(int)]
    origin = points.mean(axis=0)
    z_axis = normals.mean(axis=0)
    if np.linalg.norm(z_axis) < 1e-8:
        _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
        z_axis = vh[-1]
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    centered = points - origin
    projected = centered - np.outer(centered @ z_axis, z_axis)
    if np.linalg.norm(projected) > 1e-8:
        _, _, vh = np.linalg.svd(projected, full_matrices=False)
        x_axis = vh[0]
    else:
        reference = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < .9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(reference, z_axis)
    x_axis -= z_axis * float(x_axis @ z_axis)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))
    frame[:3, 3] = origin
    return frame


def load_pose_heads(path: Path, *, device: str = "cpu") -> Any:
    """Load a checkpoint produced by ``train_joinable_pose_heads.py``."""

    torch = _torch()
    from .pose_learning import CADPairPoseModel, CADPairPosePatchModel

    payload = torch.load(path, map_location=device, weights_only=False)
    model_class = CADPairPosePatchModel if bool(payload.get("patch_geometry")) else CADPairPoseModel
    model = model_class(embedding_dim=int(payload["embedding_dim"]), modes=int(payload["modes"]))
    # Contact-aware v3 has an additional predictor.  Older v2 patch heads
    # remain readable for reproducibility, but cannot claim contact outputs.
    contact_target_enabled = bool(payload.get("contact_target"))
    model.load_state_dict(payload["state_dict"], strict=contact_target_enabled)
    model.contact_target_enabled = contact_target_enabled
    # Strong-contact supervision can deliberately train without the frozen
    # JoinABLe embedding.  At inference the same zero vector is mandatory;
    # otherwise an unobserved feature distribution would silently leak into
    # the local B-Rep pose head.
    model.geometry_only_zero_embedding = bool(payload.get("geometry_only_zero_embedding", False))
    return model.to(device).eval(), payload


def propose_for_joinable_candidate(
    *,
    joinable_model: Any,
    graph_a: Any,
    graph_b: Any,
    candidate: dict[str, Any],
    pose_model: Any,
    pair_extent_mm: float,
    raw_graph_a: dict[str, Any] | None = None,
    raw_graph_b: dict[str, Any] | None = None,
    device: str = "cpu",
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Create learned SE(3) hypotheses for a single JoinABLe entity pair.

    ``candidate`` must be the geometry-only public candidate emitted by
    ``sw.joinable_e2e.run_gnn_inference``.  Its part ids/ranks are bookkeeping;
    they are not passed to either neural model.
    """

    torch = _torch()
    from .pose_learning import matrix_to_rotation_6d, rotation_6d_to_matrix

    if top_k < 1:
        raise ValueError("top_k_must_be_positive")
    entity_a, entity_b = candidate["node_a"], candidate["node_b"]
    is_patch_model = pose_model.__class__.__name__ == "CADPairPosePatchModel"
    geometry_only = bool(getattr(pose_model, "geometry_only_zero_embedding", False))
    try:
        if geometry_only:
            if raw_graph_a is None or raw_graph_b is None:
                return []
            frame_a = _patch_interface_frame(raw_graph_a, int(entity_a["joinable_node_index"]))
            frame_b = _patch_interface_frame(raw_graph_b, int(entity_b["joinable_node_index"]))
        else:
            frame_a, frame_b = frame_from_entity(entity_a), frame_from_entity(entity_b)
    except (KeyError, ValueError):
        return []
    if frame_a is None or frame_b is None:
        return []
    index_a = int(entity_a["joinable_node_index"])
    index_b = int(entity_b["joinable_node_index"])
    target_device = torch.device(device)
    graph_a, graph_b = graph_a.to(target_device), graph_b.to(target_device)
    joinable_model = joinable_model.to(target_device).eval()
    pose_model = pose_model.to(target_device).eval()
    with torch.inference_mode():
        face_a, face_b = joinable_model.pre_face(graph_a, graph_b)
        edge_a, edge_b = joinable_model.pre_edge(graph_a, graph_b)
        embedding_a = joinable_model.mpn(face_a + edge_a, graph_a.edge_index)
        embedding_b = joinable_model.mpn(face_b + edge_b, graph_b.edge_index)
        feature = torch.cat((
            _one_ring_embedding(embedding_a, graph_a, index_a),
            _one_ring_embedding(embedding_b, graph_b, index_b),
        )).unsqueeze(0)
        patch_a = patch_b = None
        if is_patch_model:
            if raw_graph_a is None or raw_graph_b is None:
                raise ValueError("patch_pose_model_requires_raw_brep_graphs")
            # A JoinABLe frontier can include edge entities while the first
            # contact-trained head is intentionally face-patch based.  This
            # is not an inference failure for the whole pair: discard only
            # the unsupported entity candidate and let the remaining top-k
            # face candidates reach the global solver.
            try:
                patch_a = torch.from_numpy(
                    _local_entity_patch(raw_graph_a, index_a, frame_a, pair_extent_mm)
                ).unsqueeze(0).to(target_device)
                patch_b = torch.from_numpy(
                    _local_entity_patch(raw_graph_b, index_b, frame_b, pair_extent_mm)
                ).unsqueeze(0).to(target_device)
            except (KeyError, ValueError):
                return []
        if geometry_only:
            feature = torch.zeros(
                (1, int(pose_model.embedding_dim)), dtype=feature.dtype, device=target_device
            )
        proposal = pose_model.propose(feature, patch_a, patch_b) if is_patch_model else pose_model.propose(feature)
        poses = proposal["pose_modes"][0]
        pose_logits = proposal["mode_logits"][0]
        interface_logits = (
            pose_model.score(feature, patch_a, patch_b, poses.unsqueeze(0))[0]
            if is_patch_model else pose_model.score(feature, poses.unsqueeze(0))[0]
        )
        contact_predictions = (
            pose_model.predict_contact(feature, patch_a, patch_b, poses.unsqueeze(0))[0]
            if is_patch_model and bool(getattr(pose_model, "contact_target_enabled", False)) else None
        )
        combined = pose_logits + interface_logits
        order = torch.argsort(combined, descending=True)[: min(top_k, poses.shape[0])]
        free_dof_probability = torch.sigmoid(proposal["free_dof_logits"][0]).detach().cpu().numpy()
    results = []
    for rank, mode in enumerate(order.detach().cpu().tolist(), 1):
        vector = poses[mode].detach().cpu().numpy()
        # Reproject rotation once more to make JSON transforms numerically safe.
        rotation6 = matrix_to_rotation_6d(rotation_6d_to_matrix(poses[mode, 3:])).detach().cpu().numpy()
        vector[3:] = rotation6
        transform = _relative_transform(frame_a, frame_b, vector, pair_extent_mm)
        results.append({
            "rank": rank,
            "mode_index": int(mode),
            "pose_logit": float(pose_logits[mode].detach().cpu()),
            "interface_logit": float(interface_logits[mode].detach().cpu()),
            "combined_logit": float(combined[mode].detach().cpu()),
            "local_pose_vector": vector.tolist(),
            "predicted_free_dof_probability": free_dof_probability.tolist(),
            "relative_transform": transform.tolist(),
            "predicted_contact": (
                {
                    "normalised_gap": float(contact_predictions[mode, 0].detach().cpu()),
                    "coverage": float(contact_predictions[mode, 1].detach().cpu()),
                    "normal_mismatch": float(contact_predictions[mode, 2].detach().cpu()),
                }
                if contact_predictions is not None else None
            ),
            "model_contract": (
                ("local sampled B-Rep patch only; " if geometry_only
                 else "B-Rep entity, one-ring embedding, and local sampled B-Rep patch only; ") +
                "no names/case tokens"
            ),
        })
    return results
