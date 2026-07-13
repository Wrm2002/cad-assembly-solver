"""Build leakage-safe supervision for CAD pair-pose and interface-score heads.

The output stores frozen JoinABLe B-Rep embeddings plus relative-pose labels
expressed in the two selected local interface frames.  Paths, CAD names, case
identifiers, joint type strings, and assembly ids never enter the model arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINABLE_ROOT = PROJECT_ROOT / "joinable_source"
if str(JOINABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(JOINABLE_ROOT))

from datasets.joint_graph_dataset import JointGraphDataset  # noqa: E402
from models.joinable import JoinABLe  # noqa: E402


FORBIDDEN_TEXT_FIELDS = ("case", "name", "filename", "bom", "solidworks", "joint_type")
PATCH_POINT_COUNT = 32


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _frame_matrix(frame: dict[str, Any]) -> np.ndarray:
    axes = np.column_stack((
        np.asarray(frame["primary_axis"], dtype=np.float64),
        np.asarray(frame["secondary_axis"], dtype=np.float64),
        np.asarray(frame["tertiary_axis"], dtype=np.float64),
    ))
    origin = np.asarray(frame["origin"], dtype=np.float64)
    if axes.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("invalid_interface_frame")
    # Polar decomposition is overkill here; SVD gives a safe right-handed frame
    # when an exported tangent has accumulated minor floating-point drift.
    left, _, right = np.linalg.svd(axes)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = origin
    return result


def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate((rotation[:, 0], rotation[:, 1])).astype(np.float32)


def _rotation_6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = np.asarray(value[:3], dtype=np.float64)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.asarray(value[3:6], dtype=np.float64)
    second -= first * float(np.dot(first, second))
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return np.column_stack((first, second, np.cross(first, second)))


def _contact_reference(patch_a: np.ndarray, patch_b: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Automatic local contact target at the recorded Fusion assembly pose.

    This is geometry supervision, not a mate-name label.  The final flag means
    that the recorded entity pair has sufficiently dense, close support to be
    used for *contact* ranking; other valid joints remain available for Pose
    supervision without falsely assuming they should have zero gap.
    """
    rotation = _rotation_6d_to_matrix(pose[3:])
    points_a, directions_a = patch_a[:, :3], patch_a[:, 3:]
    points_b, directions_b = patch_b[:, :3], patch_b[:, 3:]
    moved_points_b = points_b @ rotation.T + pose[:3]
    moved_directions_b = directions_b @ rotation.T
    distances = np.linalg.norm(points_a[:, None, :] - moved_points_b[None, :, :], axis=-1)
    min_a, indices_b = distances.min(axis=1), distances.min(axis=0)
    gap = float((min_a.mean() + indices_b.mean()) / 2.0)
    tolerance = 0.015
    coverage = float(((min_a < tolerance).mean() + (indices_b < tolerance).mean()) / 2.0)
    nearest_b = moved_directions_b[distances.argmin(axis=1)]
    valid = (np.linalg.norm(directions_a, axis=1) > 0.5) & (np.linalg.norm(nearest_b, axis=1) > 0.5)
    normal_error = float(np.abs(1.0 + (directions_a[valid] * nearest_b[valid]).sum(axis=1)).mean()) if valid.any() else 0.0
    contact_bearing = float(gap < 0.04 and coverage > 0.10)
    return np.asarray((gap, coverage, normal_error, contact_bearing), dtype=np.float32)


def _bbox_scale(properties: dict[str, Any]) -> float:
    box = properties.get("bounding_box") or {}
    lower, upper = box.get("min_point") or {}, box.get("max_point") or {}
    try:
        lo = np.asarray([lower["x"], lower["y"], lower["z"]], dtype=float)
        hi = np.asarray([upper["x"], upper["y"], upper["z"]], dtype=float)
    except KeyError:
        return 1.0
    return float(max(np.linalg.norm(hi - lo), 1e-4))


def _entity_node_index(entity: dict[str, Any], face_count: int) -> int:
    index = int(entity["topology_index"])
    return index if entity["entity_kind"] == "face" else face_count + index


def _entity_patch(
    raw_graph: dict[str, Any], node_index: int, frame: dict[str, Any], scale_mm: float
) -> np.ndarray:
    """Sample one selected B-Rep entity in its own local interface frame."""

    node = raw_graph["nodes"][node_index]
    points = np.asarray(node.get("points") or [], dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(
        node.get("normals") or node.get("tangents") or [], dtype=np.float64
    ).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("selected_entity_has_no_samples")
    if len(directions) == 0:
        directions = np.zeros_like(points)
    if len(directions) != len(points):
        # Edge samples may be shorter than a tiled face grid.  Deterministic
        # nearest-index resampling preserves the entity curve without names.
        indices = np.linspace(0, len(directions) - 1, len(points)).round().astype(int)
        directions = directions[indices]
    indices = np.linspace(0, len(points) - 1, PATCH_POINT_COUNT).round().astype(int)
    points, directions = points[indices], directions[indices]
    local_frame = _frame_matrix(frame)
    rotation, origin = local_frame[:3, :3], local_frame[:3, 3]
    local_points = ((points - origin) @ rotation) / float(scale_mm)
    local_directions = directions @ rotation
    norms = np.linalg.norm(local_directions, axis=1, keepdims=True)
    local_directions = np.divide(
        local_directions, norms, out=np.zeros_like(local_directions), where=norms > 1e-9
    )
    return np.concatenate((local_points, local_directions), axis=1).astype(np.float32)


def _one_ring_embedding(embedding: torch.Tensor, graph: Any, index: int) -> torch.Tensor:
    edge_index = graph.edge_index
    neighbours = edge_index[1, edge_index[0] == int(index)]
    if neighbours.numel() == 0:
        pooled = embedding[index]
    else:
        pooled = embedding[neighbours].mean(dim=0)
    return torch.cat((embedding[index], pooled), dim=0)


def _load_pretrained_encoder(checkpoint: Path, device: torch.device) -> Any:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload["hyper_parameters"]["args"]
    # Do not instantiate the repository's Lightning wrapper: newer
    # torchmetrics removed its legacy ``IoU`` helper, although the actual
    # JoinABLe model and checkpoint remain compatible.
    model = JoinABLe(
        args.hidden,
        args.input_features,
        dropout=args.dropout,
        mpn=args.mpn,
        batch_norm=args.batch_norm,
        reduction=args.reduction,
        post_net=args.post_net,
        pre_net=args.pre_net,
    )
    state = {
        key.removeprefix("model."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _encode_graph_pair(model: Any, graph_a: Any, graph_b: Any) -> tuple[torch.Tensor, torch.Tensor]:
    graph_a = graph_a.to(next(model.parameters()).device)
    graph_b = graph_b.to(next(model.parameters()).device)
    with torch.no_grad():
        face_a, face_b = model.pre_face(graph_a, graph_b)
        edge_a, edge_b = model.pre_edge(graph_a, graph_b)
        embedding_a = model.mpn(face_a + edge_a, graph_a.edge_index)
        embedding_b = model.mpn(face_b + edge_b, graph_b.edge_index)
    return embedding_a, embedding_b


def _converter() -> JointGraphDataset:
    # The original loader has all feature maps as class attributes.  A minimal
    # instance lets us reuse precisely the paper's graph conversion code without
    # loading its unrelated train/val split or labels.
    result = JointGraphDataset.__new__(JointGraphDataset)
    result.center_and_scale = True
    result.max_node_count = 0
    # PyG >= 2.6 already concatenates homogeneous node attributes in
    # ``from_networkx``.  The paper's loader expected a list and calls
    # ``torch.cat`` a second time.  Keep this compatibility shim local to the
    # new builder rather than editing the vendored research repository.
    def reshape_graph_features_compatible(self: JointGraphDataset, graph: Any) -> Any:
        if torch.is_tensor(graph.x):
            graph.x = graph.x.reshape((-1, self.grid_size, self.grid_size, self.grid_channels))
            if torch.is_tensor(graph.entity_types):
                graph.entity_types = graph.entity_types.reshape((-1, len(self.entity_type_map)))
            if torch.is_tensor(graph.convexity):
                graph.convexity = graph.convexity.reshape((-1, len(self.convexity_type_map)))
            return graph
        return JointGraphDataset.reshape_graph_features(self, graph)
    result.reshape_graph_features = types.MethodType(reshape_graph_features_compatible, result)
    return result


def _record_uses_forbidden_model_text(record: dict[str, Any]) -> bool:
    # Only audit fields that could conceivably be copied to model tensors.  The
    # source paths in ``storage`` are loader metadata and never emitted.
    model_input = json.dumps(record.get("model_input", {}), ensure_ascii=False).lower()
    return any(token in model_input for token in FORBIDDEN_TEXT_FIELDS)


def build_split(
    records: list[dict[str, Any]],
    model: Any,
    *,
    limit: int,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    converter = _converter()
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dofs: list[np.ndarray] = []
    scales: list[float] = []
    patches_a: list[np.ndarray] = []
    patches_b: list[np.ndarray] = []
    contact_references: list[np.ndarray] = []
    counts = {
        "records": len(records), "supervisions": 0, "emitted": 0,
        "skipped": 0, "skip_reasons": {},
    }

    def note_skip(exc: Exception, stage: str) -> None:
        # Do not silently reduce supervision coverage: reason counts are part
        # of the dataset audit and are checked before training can be trusted.
        message = str(exc).replace("\n", " ")[:120]
        key = f"{stage}:{type(exc).__name__}:{message}" if message else f"{stage}:{type(exc).__name__}"
        counts["skip_reasons"][key] = int(counts["skip_reasons"].get(key, 0)) + 1

    def reraise_if_cuda_poisoned(exc: Exception) -> None:
        # A CUDA illegal-access error poisons the context.  Treating every
        # subsequent graph as a bad sample silently corrupts coverage audit.
        message = str(exc).lower()
        if "cuda error" in message or "cublas_status" in message:
            raise RuntimeError("fatal_cuda_encoder_error") from exc
    for record in records:
        if limit and counts["emitted"] >= limit:
            break
        if _record_uses_forbidden_model_text(record):
            raise RuntimeError("forbidden_text_in_model_input_contract")
        storage = record["storage"]
        path_a, path_b = Path(storage["body_graph_a"]), Path(storage["body_graph_b"])
        try:
            converter.root_dir = path_a.parent
            graph_a, graph_data_a, faces_a, _, _ = converter.load_graph_body(path_a.stem)
            converter.root_dir = path_b.parent
            graph_b, graph_data_b, faces_b, _, _ = converter.load_graph_body(path_b.stem)
            if graph_a is None or graph_b is None:
                raise ValueError("invalid_brep_graph")
            converter.scale_geometry(
                graph_a.x, graph_b.x,
                area_features1=graph_a.area, area_features2=graph_b.area,
                length_features1=graph_a.length, length_features2=graph_b.length,
            )
            embedding_a, embedding_b = _encode_graph_pair(model, graph_a, graph_b)
            raw_graph_a = json.loads(path_a.read_text(encoding="utf-8"))
            raw_graph_b = json.loads(path_b.read_text(encoding="utf-8"))
        except Exception as exc:
            reraise_if_cuda_poisoned(exc)
            counts["skipped"] += 1
            note_skip(exc, "graph_or_embedding")
            continue
        scale_mm = max(_bbox_scale(graph_data_a["properties"]), _bbox_scale(graph_data_b["properties"]))
        for supervision in record.get("supervision", []):
            counts["supervisions"] += 1
            try:
                index_a = _entity_node_index(supervision["entity_a"], faces_a)
                index_b = _entity_node_index(supervision["entity_b"], faces_b)
                if index_a >= embedding_a.shape[0] or index_b >= embedding_b.shape[0]:
                    raise ValueError("entity_index_out_of_range")
                local_a = _frame_matrix(supervision["frame_a"])
                local_b = _frame_matrix(supervision["frame_b"])
                relative = np.asarray(supervision["relative_pose"], dtype=np.float64)
                if relative.shape != (4, 4):
                    raise ValueError("relative_pose_shape")
                local_pose = np.linalg.inv(local_a) @ relative @ local_b
                vector = np.concatenate((
                    (local_pose[:3, 3] / scale_mm).astype(np.float32),
                    _rotation_6d(local_pose[:3, :3]),
                ))
                if not np.all(np.isfinite(vector)):
                    raise ValueError("nonfinite_pose_target")
                pair_embedding = torch.cat((
                    _one_ring_embedding(embedding_a, graph_a.to(embedding_a.device), index_a),
                    _one_ring_embedding(embedding_b, graph_b.to(embedding_b.device), index_b),
                )).detach().cpu().numpy().astype(np.float32)
                patch_a = _entity_patch(
                    raw_graph_a, index_a, supervision["frame_a"], scale_mm
                )
                patch_b = _entity_patch(
                    raw_graph_b, index_b, supervision["frame_b"], scale_mm
                )
                contact_reference = _contact_reference(patch_a, patch_b, vector)
                features.append(pair_embedding)
                targets.append(vector)
                dofs.append(np.asarray(supervision["free_dof_mask"], dtype=np.float32))
                scales.append(float(scale_mm))
                patches_a.append(patch_a)
                patches_b.append(patch_b)
                contact_references.append(contact_reference)
                counts["emitted"] += 1
                if progress_every and counts["emitted"] % progress_every == 0:
                    print(json.dumps({"stage": "embedding", "emitted": counts["emitted"], "skipped": counts["skipped"]}), flush=True)
                if limit and counts["emitted"] >= limit:
                    break
            except Exception as exc:
                reraise_if_cuda_poisoned(exc)
                counts["skipped"] += 1
                note_skip(exc, "supervision_or_patch")
    if not features:
        raise RuntimeError("no_pose_training_examples_emitted")
    arrays = {
        "pair_embedding": np.stack(features),
        "target_pose": np.stack(targets),
        "free_dof_mask": np.stack(dofs),
        "translation_scale_mm": np.asarray(scales, dtype=np.float32),
        "patch_a": np.stack(patches_a),
        "patch_b": np.stack(patches_b),
        "contact_reference": np.stack(contact_references),
    }
    return arrays, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pure_brep_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "joinable_source" / "pretrained" / "paper" / "last_run_0.ckpt")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "dev", "test"),
        default=("train", "dev", "test"),
        help="Build only selected splits.  This permits CUDA-safe restart between splits.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep a completed .npz split and do not recompute it.",
    )
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help="Record an existing split produced before resumable auditing was added.",
    )
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _load_pretrained_encoder(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "split_build_audit.json"
    if audit_path.exists():
        audit: dict[str, Any] = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        audit = {
        "schema_version": "joinable_pose_head_dataset.v1",
        "encoder_checkpoint": str(args.checkpoint.resolve()),
        "encoder_frozen": True,
        "model_input": "frozen_joinable_selected_entity_plus_one_ring_embeddings",
        "forbidden_model_fields": ["file_name", "part_name", "case_id", "bom", "solidworks_answer", "joint_type"],
        "splits": {},
        }
    for split in args.splits:
        output_path = args.output_dir / f"{split}.npz"
        if args.skip_existing and output_path.exists():
            if args.adopt_existing and split not in audit["splits"]:
                with np.load(output_path) as existing:
                    emitted = int(existing["pair_embedding"].shape[0])
                    feature_dim = int(existing["pair_embedding"].shape[1])
                audit["splits"][split] = {
                    "records": len(_read_jsonl(args.pure_brep_dir / f"fusion360_pure_brep_{split}.jsonl")),
                    "supervisions": "unknown_recovered_from_pre_resume_run",
                    "emitted": emitted,
                    "skipped": "unknown_recovered_from_pre_resume_run",
                    "feature_dim": feature_dim,
                    "audit_status": "recovered_existing_split",
                }
            print(json.dumps({"stage": "skip_existing", "split": split}), flush=True)
            continue
        records = _read_jsonl(args.pure_brep_dir / f"fusion360_pure_brep_{split}.jsonl")
        arrays, counts = build_split(
            records, model, limit=args.limit_per_split, progress_every=args.progress_every
        )
        np.savez_compressed(output_path, **arrays)
        contact = arrays["contact_reference"]
        counts.update({
            "feature_dim": int(arrays["pair_embedding"].shape[1]),
            "contact_target": {
                "definition": "recorded_pose_local_gap_coverage_normal_mismatch",
                "contact_bearing_count": int((contact[:, 3] > 0.5).sum()),
                "contact_bearing_rate": float((contact[:, 3] > 0.5).mean()),
                "median_normalised_gap": float(np.median(contact[:, 0])),
                "median_coverage": float(np.median(contact[:, 1])),
            },
        })
        audit["splits"][split] = counts
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit["passed"] = all(
        split in audit["splits"] and audit["splits"][split]["emitted"] > 0
        for split in ("train", "dev", "test")
    )
    # The completed audit is deliberately absent until all three split files
    # exist; downstream training therefore cannot accidentally consume a
    # partial dataset after an interrupted CUDA extraction.
    if audit["passed"]:
        (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
