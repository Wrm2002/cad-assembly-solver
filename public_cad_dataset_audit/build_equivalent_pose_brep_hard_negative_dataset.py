"""Build multi-positive CAD pose data and B-Rep-measured hard negatives.

This builder is deliberately narrower than an assembly reconstruction system.
It takes the leakage-safe Fusion 360 joint records and creates tensors for the
two learned pair-level heads only:

* each input entity pair has a *set* of valid local poses induced by its
  declared free degrees of freedom;
* equivalent B-Rep entities become additional input examples in their own
  local coordinate frames (never extra labels in the primary frame);
* gap/slip/rotation candidates are kept as negatives only after their local
  B-Rep patch measurements deteriorate relative to the occurrence pose.

Paths, record ids, auxiliary joint labels, names and split hashes are retained
only in the audit report.  They never enter an ``.npz`` model tensor.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from public_cad_dataset_audit.build_joinable_pose_head_dataset import (  # noqa: E402
    _bbox_scale,
    _contact_reference,
    _encode_graph_pair,
    _entity_node_index,
    _entity_patch,
    _frame_matrix,
    _load_pretrained_encoder,
    _one_ring_embedding,
    _read_jsonl,
    _rotation_6d,
    _rotation_6d_to_matrix,
    _converter,
)


POSE_DIM = 9
MAX_EQUIVALENT_POSES = 7
MAX_ENTITY_VARIANTS = 3
HARD_NEGATIVE_COUNT = 8


def _normalised(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return np.asarray(value / norm if norm > 1e-9 else fallback, dtype=np.float64)


def _canonical_entity_frame(raw_graph: dict[str, Any], node_index: int) -> dict[str, list[float]]:
    """Construct a deterministic right-handed frame from a raw B-Rep entity.

    This is used only for alternate entities which do not have a recorded
    Joint-Dataset interface frame.  Its input is sampled B-Rep geometry, not a
    name or mate category.  The frame is a coordinate convention; the actual
    positive pose still comes from the occurrence-relative body transform.
    """

    node = raw_graph["nodes"][node_index]
    points = np.asarray(node.get("points") or [], dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("equivalent_entity_has_no_points")
    origin = points.mean(axis=0)
    vectors = np.asarray(node.get("normals") or node.get("tangents") or [], dtype=np.float64)
    vectors = vectors.reshape(-1, 3) if vectors.size else np.empty((0, 3), dtype=np.float64)
    if len(vectors):
        primary = _normalised(vectors.mean(axis=0), np.array((0.0, 0.0, 1.0)))
    else:
        centered = points - origin
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        primary = _normalised(right[-1], np.array((0.0, 0.0, 1.0)))
    centered = points - origin
    tangent_cloud = centered - np.outer(centered @ primary, primary)
    if np.linalg.norm(tangent_cloud) > 1e-9:
        _, _, right = np.linalg.svd(tangent_cloud, full_matrices=False)
        secondary = right[0]
    else:
        basis = np.eye(3)[int(np.argmin(np.abs(primary)))]
        secondary = np.cross(primary, basis)
    secondary = secondary - primary * float(np.dot(primary, secondary))
    secondary = _normalised(secondary, np.array((1.0, 0.0, 0.0)))
    tertiary = _normalised(np.cross(primary, secondary), np.array((0.0, 1.0, 0.0)))
    secondary = _normalised(np.cross(tertiary, primary), secondary)
    return {
        "origin": origin.tolist(),
        "primary_axis": primary.tolist(),
        "secondary_axis": secondary.tolist(),
        "tertiary_axis": tertiary.tolist(),
        "axis_origin": origin.tolist(),
        "axis_direction": primary.tolist(),
    }


def _entity_key(entity: dict[str, Any]) -> tuple[str, int]:
    return str(entity.get("entity_kind")), int(entity.get("topology_index", -1))


def _equivalent_pairs(supervision: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select a small deterministic set of alternative compatible entities.

    The export supplies two *sets* rather than bijective correspondence.  A
    full Cartesian product would assert unverified pairings.  We only retain
    alternatives with the same entity kind and analytic geometry class on both
    parts, and cap them.  This makes the augmentation conservative.
    """

    primary_a, primary_b = supervision["entity_a"], supervision["entity_b"]
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = [(primary_a, primary_b)]
    left = sorted(supervision.get("equivalent_entities_a") or [], key=_entity_key)
    right = sorted(supervision.get("equivalent_entities_b") or [], key=_entity_key)
    for entity_a in left:
        for entity_b in right:
            if entity_a.get("entity_kind") != entity_b.get("entity_kind"):
                continue
            if entity_a.get("geometry_type") != entity_b.get("geometry_type"):
                continue
            pair = (entity_a, entity_b)
            if _entity_key(entity_a) == _entity_key(primary_a) and _entity_key(entity_b) == _entity_key(primary_b):
                continue
            rows.append(pair)
            if len(rows) >= MAX_ENTITY_VARIANTS:
                return rows
    return rows


def _local_pose(relative: np.ndarray, frame_a: dict[str, Any], frame_b: dict[str, Any], scale: float) -> np.ndarray:
    result = np.linalg.inv(_frame_matrix(frame_a)) @ relative @ _frame_matrix(frame_b)
    return np.concatenate(((result[:3, 3] / float(scale)).astype(np.float32), _rotation_6d(result[:3, :3])))


def _dedupe_pose(rows: Iterable[np.ndarray], maximum: int = MAX_EQUIVALENT_POSES) -> tuple[np.ndarray, np.ndarray]:
    kept: list[np.ndarray] = []
    for row in rows:
        value = np.asarray(row, dtype=np.float32)
        if not np.all(np.isfinite(value)):
            continue
        if not any(np.allclose(value, previous, rtol=0.0, atol=1e-5) for previous in kept):
            kept.append(value)
        if len(kept) >= maximum:
            break
    if not kept:
        raise ValueError("no_valid_equivalent_pose")
    output = np.zeros((maximum, POSE_DIM), dtype=np.float32)
    mask = np.zeros((maximum,), dtype=np.bool_)
    output[:len(kept)] = np.stack(kept)
    mask[:len(kept)] = True
    return output, mask


def _free_dof_pose_modes(target: np.ndarray, dof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate valid representatives only along the declared free manifold."""

    target = np.asarray(target, dtype=np.float32)
    dof = np.asarray(dof, dtype=np.float32)
    rows: list[np.ndarray] = [target]
    # Local translations are normalised by the joint pair bbox.  The values are
    # intentionally moderate: they represent the same sliding manifold rather
    # than a new arbitrary assembly location.
    for axis in range(3):
        if dof[axis] > 0.5:
            for distance in (-0.10, 0.10):
                value = target.copy()
                value[axis] += distance
                rows.append(value)
    original = _rotation_6d_to_matrix(target[3:])
    for axis in range(3):
        if dof[axis + 3] > 0.5:
            for degrees in (90.0, 180.0):
                radians = np.deg2rad(degrees)
                cosine, sine = float(np.cos(radians)), float(np.sin(radians))
                delta = np.eye(3, dtype=np.float64)
                first, second = (axis + 1) % 3, (axis + 2) % 3
                delta[first, first], delta[second, second] = cosine, cosine
                delta[first, second], delta[second, first] = -sine, sine
                value = target.copy()
                value[3:] = _rotation_6d(delta @ original)
                rows.append(value)
    return _dedupe_pose(rows)


def _hard_negative_candidates(
    target: np.ndarray,
    dof: np.ndarray,
    patch_a: np.ndarray,
    patch_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only perturbations whose B-Rep contact actually worsens.

    ``measurement`` contains [gap, coverage, normal_mismatch] and masks
    distinguish a difficult measured negative from a symmetry/free-DOF
    perturbation that should not be assigned a negative label.
    """

    candidates = np.zeros((HARD_NEGATIVE_COUNT, POSE_DIM), dtype=np.float32)
    mask = np.zeros((HARD_NEGATIVE_COUNT,), dtype=np.bool_)
    measurement = np.zeros((HARD_NEGATIVE_COUNT, 3), dtype=np.float32)
    positive = _contact_reference(patch_a, patch_b, target)[:3]
    positive_quality = 0.70 * positive[1] - positive[0] - 0.20 * positive[2]
    translation_constrained = 1.0 - np.asarray(dof[:3], dtype=np.float32)
    rotation_constrained = bool((1.0 - np.asarray(dof[3:], dtype=np.float32)).max() > 0.5)
    available = np.flatnonzero(translation_constrained > 0.5)
    original_rotation = _rotation_6d_to_matrix(target[3:])
    magnitudes = (0.0125, 0.03, 0.075, 0.15)
    angles = (10.0, 25.0, 55.0, 180.0)
    for index in range(HARD_NEGATIVE_COUNT):
        value = target.copy()
        if len(available):
            axis = int(available[index % len(available)])
            value[axis] += (-1.0 if (index // 3) % 2 else 1.0) * magnitudes[index % len(magnitudes)]
        else:
            # No constrained translation: only a constrained orientation can
            # constitute a valid negative.  Fully free poses produce no fake
            # negatives and remain proposal-only training examples.
            axis = index % 3
        if index % 2 and rotation_constrained:
            radians = np.deg2rad(angles[index % len(angles)])
            cosine, sine = float(np.cos(radians)), float(np.sin(radians))
            delta = np.eye(3, dtype=np.float64)
            first, second = (axis + 1) % 3, (axis + 2) % 3
            delta[first, first], delta[second, second] = cosine, cosine
            delta[first, second], delta[second, first] = -sine, sine
            value[3:] = _rotation_6d(delta @ original_rotation)
        observed = _contact_reference(patch_a, patch_b, value)[:3]
        quality = 0.70 * observed[1] - observed[0] - 0.20 * observed[2]
        # B-Rep-measured gate.  A candidate that is indistinguishable at the
        # sampled interface can be an actual symmetry or a weak observation;
        # it is not silently labelled wrong.
        deteriorated = bool(quality < positive_quality - 0.025 or
                            observed[0] > positive[0] + 0.0125 or
                            observed[1] < positive[1] - 0.10 or
                            observed[2] > positive[2] + 0.15)
        candidates[index], measurement[index], mask[index] = value, observed, deteriorated
    return candidates, mask, measurement


def _append_sample(
    arrays: dict[str, list[np.ndarray]],
    *,
    pair_embedding: np.ndarray,
    target: np.ndarray,
    dof: np.ndarray,
    scale: float,
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    equivalent_entity_support: int,
) -> tuple[int, int]:
    modes, mode_mask = _free_dof_pose_modes(target, dof)
    negative, negative_mask, negative_measurement = _hard_negative_candidates(target, dof, patch_a, patch_b)
    arrays["pair_embedding"].append(pair_embedding)
    arrays["target_pose"].append(target)
    arrays["target_pose_modes"].append(modes)
    arrays["target_pose_mode_mask"].append(mode_mask)
    arrays["free_dof_mask"].append(dof)
    arrays["translation_scale_mm"].append(np.asarray(scale, dtype=np.float32))
    arrays["patch_a"].append(patch_a)
    arrays["patch_b"].append(patch_b)
    arrays["contact_reference"].append(_contact_reference(patch_a, patch_b, target))
    arrays["hard_negative_pose"].append(negative)
    arrays["hard_negative_mask"].append(negative_mask)
    arrays["hard_negative_contact"].append(negative_measurement)
    arrays["equivalent_entity_support"].append(np.asarray(equivalent_entity_support, dtype=np.int16))
    return int(mode_mask.sum()), int(negative_mask.sum())


def _empty_arrays() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in (
        "pair_embedding", "target_pose", "target_pose_modes", "target_pose_mode_mask", "free_dof_mask",
        "translation_scale_mm", "patch_a", "patch_b", "contact_reference", "hard_negative_pose",
        "hard_negative_mask", "hard_negative_contact", "equivalent_entity_support",
    )}


def _load_equivalence_manifest(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load the separately audited equivalence registry for consistency checks.

    The registry deliberately remains metadata rather than a model feature.
    Re-reading it here prevents a later source-record change from silently
    changing which alternatives are called equivalent.
    """

    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            result[(str(row["split"]), str(row["record_id"]), int(row["supervision_index"]))] = row
    if not result:
        raise ValueError("empty_pose_equivalence_manifest")
    return result


def build_split(
    records: list[dict[str, Any]], model: Any, *, split: str,
    equivalence_manifest: dict[tuple[str, str, int], dict[str, Any]],
    limit_records: int, progress_every: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    converter = _converter()
    arrays = _empty_arrays()
    counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    for record_index, record in enumerate(records):
        if limit_records and record_index >= limit_records:
            break
        storage = record["storage"]
        path_a, path_b = Path(storage["body_graph_a"]), Path(storage["body_graph_b"])
        try:
            converter.root_dir = path_a.parent
            graph_a, graph_data_a, faces_a, _, _ = converter.load_graph_body(path_a.stem)
            converter.root_dir = path_b.parent
            graph_b, graph_data_b, faces_b, _, _ = converter.load_graph_body(path_b.stem)
            if graph_a is None or graph_b is None:
                raise ValueError("invalid_brep_graph")
            converter.scale_geometry(graph_a.x, graph_b.x, area_features1=graph_a.area, area_features2=graph_b.area,
                                     length_features1=graph_a.length, length_features2=graph_b.length)
            embedding_a, embedding_b = _encode_graph_pair(model, graph_a, graph_b)
            raw_a, raw_b = json.loads(path_a.read_text(encoding="utf-8")), json.loads(path_b.read_text(encoding="utf-8"))
        except Exception as exc:
            # CUDA illegal access poisons this process.  Continuing would turn
            # every later valid record into a false "skip" and silently create
            # a biased partial dataset.  The chunk runner restarts a clean
            # worker instead.
            message = str(exc).lower()
            if "cuda error" in message or "illegal memory access" in message or "cublas_status" in message:
                raise RuntimeError("fatal_cuda_encoder_error") from exc
            counts["skipped_records"] += 1
            skip_reasons[f"graph:{type(exc).__name__}:{str(exc)[:80]}"] += 1
            continue
        scale = max(_bbox_scale(graph_data_a["properties"]), _bbox_scale(graph_data_b["properties"]))
        relative_cache: dict[int, np.ndarray] = {}
        for supervision_index, supervision in enumerate(record.get("supervision") or []):
            counts["source_supervisions"] += 1
            try:
                seed = equivalence_manifest.get((split, str(record.get("record_id")), supervision_index))
                if seed is None:
                    raise ValueError("missing_equivalence_manifest_seed")
                if list(seed.get("free_dof_mask") or []) != [int(value) for value in supervision.get("free_dof_mask") or []]:
                    raise ValueError("equivalence_manifest_dof_mismatch")
                counts["manifest_validated_supervisions"] += 1
                relative = relative_cache.setdefault(
                    supervision_index, np.asarray(supervision["relative_pose"], dtype=np.float64)
                )
                if relative.shape != (4, 4):
                    raise ValueError("relative_pose_shape")
                dof = np.asarray(supervision["free_dof_mask"], dtype=np.float32)
                if dof.shape != (6,):
                    raise ValueError("free_dof_shape")
                variants = _equivalent_pairs(supervision)
                for variant_index, (entity_a, entity_b) in enumerate(variants):
                    primary = variant_index == 0
                    frame_a = supervision["frame_a"] if primary else _canonical_entity_frame(raw_a, _entity_node_index(entity_a, faces_a))
                    frame_b = supervision["frame_b"] if primary else _canonical_entity_frame(raw_b, _entity_node_index(entity_b, faces_b))
                    index_a, index_b = _entity_node_index(entity_a, faces_a), _entity_node_index(entity_b, faces_b)
                    if index_a >= embedding_a.shape[0] or index_b >= embedding_b.shape[0]:
                        raise ValueError("equivalent_entity_index_out_of_range")
                    target = _local_pose(relative, frame_a, frame_b, scale)
                    patch_a = _entity_patch(raw_a, index_a, frame_a, scale)
                    patch_b = _entity_patch(raw_b, index_b, frame_b, scale)
                    pair_embedding = torch.cat((
                        _one_ring_embedding(embedding_a, graph_a.to(embedding_a.device), index_a),
                        _one_ring_embedding(embedding_b, graph_b.to(embedding_b.device), index_b),
                    )).detach().cpu().numpy().astype(np.float32)
                    mode_count, negative_count = _append_sample(
                        arrays, pair_embedding=pair_embedding, target=target, dof=dof, scale=scale,
                        patch_a=patch_a, patch_b=patch_b, equivalent_entity_support=len(variants),
                    )
                    counts["emitted_primary" if primary else "emitted_equivalent_entity"] += 1
                    counts[f"positive_modes_{mode_count}"] += 1
                    counts["measured_hard_negatives"] += negative_count
            except Exception as exc:
                counts["skipped_supervisions"] += 1
                skip_reasons[f"supervision:{type(exc).__name__}:{str(exc)[:80]}"] += 1
        if progress_every and (record_index + 1) % progress_every == 0:
            print(json.dumps({"stage": "build", "records": record_index + 1, "samples": len(arrays["target_pose"]),
                              "hard_negatives": counts["measured_hard_negatives"]}), flush=True)
    if not arrays["target_pose"]:
        raise RuntimeError("no_examples_emitted")
    output = {key: np.stack(value) for key, value in arrays.items()}
    audit = {
        "source_records": min(len(records), limit_records) if limit_records else len(records),
        "counts": dict(counts),
        "skip_reasons": dict(skip_reasons),
        "samples": int(output["target_pose"].shape[0]),
        "mean_positive_pose_modes": float(output["target_pose_mode_mask"].sum(axis=1).mean()),
        "mean_measured_hard_negatives": float(output["hard_negative_mask"].sum(axis=1).mean()),
        "equivalent_entity_examples": int(counts["emitted_equivalent_entity"]),
        "primary_examples": int(counts["emitted_primary"]),
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pure_brep_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "joinable_source" / "pretrained" / "paper" / "last_run_0.ckpt")
    parser.add_argument("--equivalence-manifest", type=Path,
                        default=Path("D:/Model_match_public_data/fusion360_pose_equivalence_v1/pose_equivalence_manifest.jsonl"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    parser.add_argument("--limit-records", type=int, default=0, help="Audit-only pilot cap; zero builds the full split.")
    parser.add_argument("--record-start", type=int, default=0,
                        help="Zero-based record offset for failure-isolated chunk construction.")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = _load_pretrained_encoder(args.checkpoint, device)
    equivalence_manifest = _load_equivalence_manifest(args.equivalence_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "schema_version": "fusion360_equivalent_pose_brep_hard_negative.v1",
        "split_contract": "Official Fusion assembly split is retained; all entity variants stay within their source split.",
        "model_input_contract": "B-Rep patches + frozen JoinABLe embeddings only; no names, ids, paths, joint types or test cases.",
        "negative_contract": "Only generic local SE(3) perturbations whose B-Rep contact measurement deteriorates are negative.",
        "occt_claim": "No OCCT full-body collision label is inferred by this local patch builder.",
        "device": str(device), "equivalence_manifest": str(args.equivalence_manifest.resolve()),
        "equivalence_manifest_entries": len(equivalence_manifest), "splits": {},
    }
    for split in args.splits:
        source_records = _read_jsonl(args.pure_brep_dir / f"fusion360_pure_brep_{split}.jsonl")
        if args.record_start < 0 or args.record_start >= len(source_records):
            parser.error(f"--record-start out of range for {split}")
        end = args.record_start + args.limit_records if args.limit_records else len(source_records)
        records = source_records[args.record_start:end]
        data, split_audit = build_split(
            records, model, split=split, equivalence_manifest=equivalence_manifest,
            limit_records=args.limit_records, progress_every=args.progress_every,
        )
        split_audit["source_record_range"] = {
            "start": args.record_start,
            "end_exclusive": min(end, len(source_records)),
            "total_in_split": len(source_records),
        }
        np.savez_compressed(args.output_dir / f"{split}.npz", **data)
        audit["splits"][split] = split_audit
        (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"split": split, **split_audit}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
