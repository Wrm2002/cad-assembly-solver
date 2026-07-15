"""Build strong contact-Pose supervision from complete Fusion360 assemblies.

Unlike the earlier Joint Dataset route, positives here are only recorded
cross-body *contact faces* whose transformed OBJ surface samples pass a strict
geometric gate.  The target Pose is the true occurrence-body relative SE(3)
transform.  File names, component names, joint labels and SolidWorks files are
never written to model arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
from fusion360_common import build_parts, discover_assembly_files, entity_part_id, load_json  # noqa: E402


PATCH_POINTS = 48
EMBEDDING_DIM = 1536  # kept for a common model interface; values are zero.


def _parse_obj(path: Path, cache: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if path in cache:
        return cache[path]
    vertices: list[list[float]] = []
    groups: dict[int, list[tuple[int, int, int]]] = {}
    current: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                try: vertices.append([float(v) for v in line.split()[1:4]])
                except ValueError: pass
            elif line.startswith("g face "):
                try:
                    current = int(line.split()[2]); groups.setdefault(current, [])
                except (IndexError, ValueError): current = None
            elif line.startswith("f ") and current is not None:
                indices = []
                for token in line.split()[1:]:
                    try:
                        index = int(token.split("/")[0]); indices.append(index - 1 if index > 0 else len(vertices) + index)
                    except ValueError: pass
                if len(indices) >= 3:
                    for offset in range(1, len(indices) - 1): groups[current].append((indices[0], indices[offset], indices[offset + 1]))
    values = np.asarray(vertices, dtype=np.float64)
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for index, triangles in groups.items():
        ids = np.unique(np.asarray(triangles, dtype=int).reshape(-1))
        points = values[ids]
        normal_sum = np.zeros_like(points)
        position = {int(vertex): local for local, vertex in enumerate(ids)}
        for first, second, third in triangles:
            cross = np.cross(values[second] - values[first], values[third] - values[first])
            for vertex in (first, second, third): normal_sum[position[vertex]] += cross
        norms = np.linalg.norm(normal_sum, axis=1, keepdims=True)
        normals = np.divide(normal_sum, norms, out=np.zeros_like(normal_sum), where=norms > 1e-12)
        result[index] = (points, normals)
    cache[path] = result
    return result


def _obj_path(part: dict[str, Any]) -> Path | None:
    for candidate in (part.get("geometry") or {}).get("candidates") or []:
        if candidate.get("format") == "obj" and candidate.get("exists"):
            return Path(candidate["path"])
    return None


def _part_extent(mesh: dict[int, tuple[np.ndarray, np.ndarray]]) -> float:
    """Return a part-scale normaliser, not a contact-face-scale normaliser.

    The inference path normalises a selected B-Rep patch using its parent
    part's bounding-box diagonal.  Mirroring that rule here avoids teaching a
    translation convention that only works for unusually small contact faces.
    """
    clouds = [values[0] for values in mesh.values() if len(values[0])]
    if not clouds:
        return 1e-4
    points = np.concatenate(clouds, axis=0)
    return max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1e-4)


def _cached_part_extent(
    path: Path, mesh: dict[int, tuple[np.ndarray, np.ndarray]], cache: dict[Path, float]
) -> float:
    if path not in cache:
        cache[path] = _part_extent(mesh)
    return cache[path]


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]


def _transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    output = (matrix[:3, :3] @ normals.T).T
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    return np.divide(output, norms, out=np.zeros_like(output), where=norms > 1e-12)


def _sample(points: np.ndarray, normals: np.ndarray, count: int = PATCH_POINTS) -> tuple[np.ndarray, np.ndarray]:
    ids = np.linspace(0, len(points) - 1, count).round().astype(int)
    return points[ids], normals[ids]


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate((matrix[:3, 0], matrix[:3, 1])).astype(np.float32)


def _face_frame(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """A deterministic local SE(3) frame from a B-Rep face sample.

    No surface/mate name participates.  The origin is the sampled-face
    centroid, its z-axis is the mean exported face normal, and the x-axis is
    the strongest in-face PCA direction.  It gives the same coordinate
    *meaning* as the selected-entity frame used by the STEP inference path:
    a pose is interface-B expressed in interface-A, rather than a pose between
    arbitrary modelling origins.
    """
    origin = points.mean(axis=0)
    z_axis = normals.mean(axis=0)
    if np.linalg.norm(z_axis) < 1e-8:
        centered = points - origin
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        z_axis = vh[-1]
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-12)
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
    y_axis = np.cross(z_axis, x_axis)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    frame[:3, 3] = origin
    return frame


def _patch_in_frame(points: np.ndarray, normals: np.ndarray, frame: np.ndarray, scale: float) -> np.ndarray:
    rotation, origin = frame[:3, :3], frame[:3, 3]
    local_points = ((points - origin) @ rotation) / max(float(scale), 1e-9)
    local_normals = normals @ rotation
    local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-12)
    return np.concatenate((local_points, local_normals), axis=1).astype(np.float32)


def _contact_metrics(a_points: np.ndarray, a_normals: np.ndarray, b_points: np.ndarray, b_normals: np.ndarray, relative: np.ndarray, scale: float) -> tuple[float, float, float, float]:
    moved_b = _transform_points(b_points, relative)
    moved_normals_b = _transform_normals(b_normals, relative)
    distances = np.linalg.norm(a_points[:, None] - moved_b[None, :], axis=-1)
    min_a, min_b = distances.min(axis=1), distances.min(axis=0)
    mean_gap = float((min_a.mean() + min_b.mean()) / 2.0)
    tolerance = max(0.002, min(0.02, 0.01 * scale))  # cm; mesh-resolution aware
    coverage = float(((min_a <= tolerance).mean() + (min_b <= tolerance).mean()) / 2.0)
    nearest = moved_normals_b[distances.argmin(axis=1)]
    valid = (np.linalg.norm(a_normals, axis=1) > .5) & (np.linalg.norm(nearest, axis=1) > .5)
    normal_error = float(np.abs(1.0 + (a_normals[valid] * nearest[valid]).sum(axis=1)).mean()) if valid.any() else 0.0
    return mean_gap, coverage, normal_error, float(min(min_a.min(), min_b.min()))


def _split(assembly_id: str) -> str:
    bucket = int(hashlib.sha1(assembly_id.encode()).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else ("dev" if bucket == 1 else "train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", nargs="+", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Maximum accepted contacts; zero means all.")
    parser.add_argument("--max-mean-gap-cm", type=float, default=.10)
    parser.add_argument("--max-minimum-gap-cm", type=float, default=.01)
    args = parser.parse_args()
    output: dict[str, dict[str, list[np.ndarray]]] = {split: {key: [] for key in ("pair_embedding", "target_pose", "free_dof_mask", "patch_a", "patch_b", "contact_reference")} for split in ("train", "dev", "test")}
    rows: dict[str, list[dict[str, Any]]] = {split: [] for split in output}
    cache: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    extent_cache: dict[Path, float] = {}
    reasons, accepted = Counter(), 0
    files = [file for root in args.input_root for file in discover_assembly_files(root)]
    for assembly_file in files:
        if args.limit and accepted >= args.limit: break
        data = load_json(assembly_file)
        parts, failures = build_parts(data, assembly_file.parent, visible_only=True)
        if failures: reasons["assembly_part_mapping_warning"] += 1
        by_id = {part["part_id"]: part for part in parts}
        split = _split(assembly_file.parent.name)
        for contact_index, contact in enumerate(data.get("contacts") or []):
            if args.limit and accepted >= args.limit: break
            left, right = contact.get("entity_one") or {}, contact.get("entity_two") or {}
            part_a, part_b = entity_part_id(left), entity_part_id(right)
            if part_a not in by_id or part_b not in by_id or left.get("type") != "BRepFace" or right.get("type") != "BRepFace": reasons["contact_mapping_unavailable"] += 1; continue
            path_a, path_b = _obj_path(by_id[part_a]), _obj_path(by_id[part_b])
            if path_a is None or path_b is None: reasons["obj_missing"] += 1; continue
            all_mesh_a, all_mesh_b = _parse_obj(path_a, cache), _parse_obj(path_b, cache)
            mesh_a = all_mesh_a.get(int(left.get("index", -1))); mesh_b = all_mesh_b.get(int(right.get("index", -1)))
            if mesh_a is None or mesh_b is None: reasons["face_index_unavailable"] += 1; continue
            points_a, normals_a = mesh_a; points_b, normals_b = mesh_b
            if min(len(points_a), len(points_b)) < 3: reasons["face_too_small"] += 1; continue
            world_a, world_b = np.asarray(by_id[part_a]["transform"], dtype=float), np.asarray(by_id[part_b]["transform"], dtype=float)
            relative = np.linalg.inv(world_a) @ world_b
            scale = max(
                _cached_part_extent(path_a, all_mesh_a, extent_cache),
                _cached_part_extent(path_b, all_mesh_b, extent_cache),
            )
            sample_a, normal_a = _sample(points_a, normals_a); sample_b, normal_b = _sample(points_b, normals_b)
            mean_gap, coverage, normal_error, minimum = _contact_metrics(sample_a, normal_a, sample_b, normal_b, relative, scale)
            if mean_gap > args.max_mean_gap_cm or minimum > args.max_minimum_gap_cm:
                reasons["contact_geometry_gate_failed"] += 1; continue
            frame_a, frame_b = _face_frame(sample_a, normal_a), _face_frame(sample_b, normal_b)
            interface_relative = np.linalg.inv(frame_a) @ relative @ frame_b
            patch_a = _patch_in_frame(sample_a, normal_a, frame_a, scale)
            patch_b = _patch_in_frame(sample_b, normal_b, frame_b, scale)
            vector = np.concatenate(((interface_relative[:3, 3] / scale).astype(np.float32), _rotation_6d(interface_relative[:3, :3])))
            output[split]["pair_embedding"].append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
            output[split]["target_pose"].append(vector); output[split]["free_dof_mask"].append(np.zeros(6, dtype=np.float32))
            output[split]["patch_a"].append(patch_a); output[split]["patch_b"].append(patch_b)
            # The contact head uses [gap, coverage, normal mismatch] in
            # [0, 1].  This is a measured target at the recorded occurrence
            # pose, not a mate/frame label.
            output[split]["contact_reference"].append(np.asarray((
                min(1.0, mean_gap / max(scale, 1e-9)),
                min(1.0, coverage),
                min(1.0, normal_error / 2.0),
                1.0,
            ), dtype=np.float32))
            rows[split].append({"assembly_id": assembly_file.parent.name, "contact_index": contact_index, "surface_types": [left.get("surface_type"), right.get("surface_type")], "mean_gap_cm": mean_gap, "coverage": coverage, "normal_error": normal_error})
            accepted += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "fusion360_strong_contact_pose.v2", "accepted": accepted, "input_contract": "local OBJ B-Rep face samples in deterministic interface frames plus occurrence relative transform; no names, joint labels or case data", "gate": {"max_mean_gap_cm": args.max_mean_gap_cm, "max_minimum_gap_cm": args.max_minimum_gap_cm}, "rejections": dict(reasons), "splits": {}}
    for split, arrays in output.items():
        packed = {key: np.stack(values) if values else np.empty((0,), dtype=np.float32) for key, values in arrays.items()}
        np.savez_compressed(args.output_dir / f"{split}.npz", **packed)
        (args.output_dir / f"{split}_provenance.json").write_text(json.dumps(rows[split], ensure_ascii=False) + "\n", encoding="utf-8")
        surface_counts = Counter(
            "|".join(str(value or "unknown") for value in row["surface_types"])
            for row in rows[split]
        )
        summary["splits"][split] = {
            "examples": len(rows[split]),
            "assembly_count": len({row["assembly_id"] for row in rows[split]}),
            "surface_pair_counts": dict(surface_counts),
        }
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if all(summary["splits"][split]["examples"] > 0 for split in summary["splits"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
