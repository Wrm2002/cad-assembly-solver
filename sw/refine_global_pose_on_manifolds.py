"""Generate subassembly Pose candidates along unresolved joint manifolds.

This is the multi-part analogue of JoinABLe's offset/rotation search.  For a
tree edge with free axial translation, removing the edge splits the assembly
graph into two rigid subassemblies.  The non-anchor side is translated as a
whole along the predicted interface axis to generic projected-support contact
events.  Optional phase samples rotate that same subassembly about the axis.

No part name, file token, case id or mechanical-family label is inspected.
The output remains review-only until exact OCCT validation succeeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "sw") not in sys.path:
    sys.path.insert(0, str(ROOT / "sw"))
from learned_joint.mesh_residuals import _load_mesh  # noqa: E402


def _part(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--part requires PART_ID=STEP_PATH")
    key, raw = value.split("=", 1)
    path = Path(raw).resolve()
    if not key or not path.is_file():
        raise argparse.ArgumentTypeError("--part requires an existing STEP path")
    return key, path


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _factor_rows(manifest: Path) -> dict[str, dict[str, Any]]:
    payload = _read(manifest)
    result: dict[str, dict[str, Any]] = {}
    for record in payload.get("records") or []:
        if record.get("status") != "success":
            continue
        rows = (_read(Path(record["result_path"])).get("joint_hypotheses") or {}).get("rows") or []
        for index, row in enumerate(rows):
            key = f"{record['source']}:{record['target']}:manifold:{index:04d}"
            result[key] = dict(row) | {"source": str(record["source"]), "target": str(record["target"])}
    return result


def _component_without_anchor(
    parts: list[str], edges: list[tuple[str, str]], removed: tuple[str, str], anchor: str
) -> set[str]:
    adjacency = {part: set() for part in parts}
    removed_key = tuple(sorted(removed))
    for first, second in edges:
        if tuple(sorted((first, second))) == removed_key:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)
    remaining = set(parts)
    remaining.remove(anchor)
    stack, fixed = [anchor], {anchor}
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor not in fixed:
                fixed.add(neighbor)
                stack.append(neighbor)
    return set(parts) - fixed


def _axis_rotation(axis: np.ndarray, angle_degrees: float, origin: np.ndarray) -> np.ndarray:
    angle = math.radians(float(angle_degrees))
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = origin - rotation @ origin
    return result


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pair-frontier-manifest", type=Path, required=True)
    parser.add_argument("--part", action="append", type=_part, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--phase-step-degrees", type=float, default=90.0)
    parser.add_argument("--clearance-mm", type=float, default=.02)
    args = parser.parse_args()
    source = _read(args.input)
    part_sources = dict(args.part)
    parts = list(source.get("part_ids") or sorted(part_sources))
    anchor = str(source.get("anchor_id") or parts[0])
    rows = _factor_rows(args.pair_frontier_manifest)
    vertices = {}
    for part, step_path in part_sources.items():
        stl = step_path.with_suffix(".stl")
        if not stl.is_file():
            raise FileNotFoundError(f"sibling_stl_required:{stl}")
        vertices[part] = _load_mesh(stl).vertices
    output_rows = []
    seen: set[tuple[float, ...]] = set()
    for parent_index, parent in enumerate((source.get("hypotheses") or [])[: max(1, args.top_n)]):
        poses = {part: np.asarray(value, dtype=float) for part, value in parent["part_poses"].items()}
        tree_ids = list(parent.get("tree_candidate_ids") or [])
        tree_factors = [rows[candidate] for candidate in tree_ids if candidate in rows]
        tree_edges = [(str(row["source"]), str(row["target"])) for row in tree_factors]
        for candidate_id, factor in zip(tree_ids, tree_factors):
            free = np.asarray(factor.get("free_dof_mask") or [], dtype=int)
            if free.shape != (6,) or free[2] != 1:
                continue
            source_part, target_part = str(factor["source"]), str(factor["target"])
            moving = _component_without_anchor(parts, tree_edges, (source_part, target_part), anchor)
            if not moving or moving == set(parts):
                continue
            fixed_endpoint = source_part if source_part not in moving else target_part
            moving_endpoint = target_part if target_part in moving else source_part
            frame = np.asarray(
                factor["frame_a"] if fixed_endpoint == source_part else factor["frame_b"], dtype=float
            )
            fixed_frame_world = poses[fixed_endpoint] @ frame
            axis = fixed_frame_world[:3, 2]
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            origin = fixed_frame_world[:3, 3]
            fixed_points = _transform_points(vertices[fixed_endpoint], poses[fixed_endpoint])
            moving_points = _transform_points(vertices[moving_endpoint], poses[moving_endpoint])
            fixed_projection, moving_projection = fixed_points @ axis, moving_points @ axis
            offsets = {
                0.0,
                float(fixed_projection.min() - moving_projection.max() - args.clearance_mm),
                float(fixed_projection.max() - moving_projection.min() + args.clearance_mm),
            }
            # Include nearby support events so a tessellation/numerical
            # tolerance cannot be the only reason an exact contact is missed.
            base_offsets = list(offsets)
            for value in base_offsets[1:]:
                offsets.update((value - args.clearance_mm, value + args.clearance_mm))
            phases = [0.0]
            if free[5] == 1 and args.phase_step_degrees > 0:
                phases = list(np.arange(0.0, 360.0, args.phase_step_degrees))
            for offset in sorted(offsets):
                for phase in phases:
                    delta = _axis_rotation(axis, phase, origin)
                    delta[:3, 3] += axis * offset
                    candidate_poses = {
                        part: (delta @ pose if part in moving else pose.copy())
                        for part, pose in poses.items()
                    }
                    signature = tuple(
                        np.round(np.concatenate([candidate_poses[part][:3, :4].reshape(-1) for part in sorted(parts)]), 5)
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    row = copy.deepcopy(parent)
                    row["hypothesis_id"] = (
                        f"{parent.get('hypothesis_id', parent_index)}:manifold_refine:"
                        f"{candidate_id}:{offset:.6f}:{phase:.3f}"
                    )
                    row["part_poses"] = {part: pose.tolist() for part, pose in candidate_poses.items()}
                    row["exact_validation"] = {"status": "not_checked"}
                    row["accepted"] = False
                    row["review_required"] = True
                    row["manifold_refinement"] = {
                        "parent_index": parent_index,
                        "cut_candidate_id": candidate_id,
                        "moving_subassembly": sorted(moving),
                        "offset_mm": offset,
                        "phase_degrees": phase,
                        "event": "projected_support_contact",
                        "case_specific_rule": False,
                    }
                    output_rows.append(row)
    result = {
        "schema_version": "constraint_manifold_subassembly_refinement.v1",
        "status": "review_required",
        "accepted": False,
        "review_required": True,
        "part_ids": parts,
        "anchor_id": anchor,
        "source_hypothesis_count": min(len(source.get("hypotheses") or []), max(1, args.top_n)),
        "hypothesis_count": len(output_rows),
        "hypotheses": output_rows,
        "input_contract": "poses, selected interface frames, free DOFs and STL geometry only",
        "acceptance_boundary": "Generated candidates require exact OCCT validation and functional review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hypothesis_count": len(output_rows), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
