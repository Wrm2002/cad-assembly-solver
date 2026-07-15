"""Add rigid Pose hypotheses from a convex prism and a concave three-face slot.

The proposal is purely geometric: no part names, paths, case identifiers,
relation labels, or mechanical-family vocabulary reach the decision.  A
finite convex three-axis planar prism is matched to a topology-backed slot;
the slot walls, bottom and longitudinal cylinder axis define a local frame.
All symmetry-equivalent frame polarities remain candidates for global closure
and exact OCCT validation.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "sw") not in sys.path:
    sys.path.insert(0, str(ROOT / "sw"))

from pose_search.key_slot_features import extract_key_slot_evidence  # noqa: E402
from pose_search.prismatic_key_features import extract_prismatic_key_feature  # noqa: E402
from pose_search.transforms import unit  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _inverse_rigid(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def _nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("node_id")): row for row in graph.get("nodes") or []}


def _dominant_axis(graph: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    rows = []
    for node in graph.get("nodes") or []:
        if node.get("entity_type") != "face" or node.get("axis_direction") is None:
            continue
        try:
            radius = float(node.get("radius") or 0.0)
            area = float(node.get("area") or 0.0)
            direction = unit(np.asarray(node["axis_direction"], dtype=float))
            origin = np.asarray(node.get("axis_origin") or node.get("centroid"), dtype=float)
        except (TypeError, ValueError, KeyError):
            continue
        if radius > 0.0 and area > 0.0 and origin.shape == (3,):
            rows.append((area, radius, origin, direction))
    if not rows:
        return None
    _, _, origin, direction = max(rows, key=lambda row: (row[0], row[1]))
    return origin, direction


def _slot_descriptions(graph: dict[str, Any]) -> list[dict[str, Any]]:
    axis = _dominant_axis(graph)
    if axis is None:
        return []
    origin, direction = axis
    evidence = extract_key_slot_evidence(graph, origin, direction)
    nodes = _nodes(graph)
    result = []
    for row in evidence.get("candidates") or []:
        try:
            left = nodes[str(row["wall_face_ids"][0])]
            right = nodes[str(row["wall_face_ids"][1])]
            bottom = nodes[str(row["bottom_face_id"])]
            longitudinal = unit(direction)
            wall = unit(np.asarray(left["normal"], dtype=float))
            wall = unit(wall - longitudinal * float(wall @ longitudinal))
            bottom_normal = unit(np.asarray(bottom["normal"], dtype=float))
            bottom_normal = unit(
                bottom_normal
                - longitudinal * float(bottom_normal @ longitudinal)
                - wall * float(bottom_normal @ wall)
            )
            wall_midpoint = .5 * (
                np.asarray(left["centroid"], dtype=float)
                + np.asarray(right["centroid"], dtype=float)
            )
            bottom_center = np.asarray(bottom["centroid"], dtype=float)
            base = bottom_center + wall * float((wall_midpoint - bottom_center) @ wall)
        except (KeyError, TypeError, ValueError):
            continue
        result.append({
            "candidate": row,
            "axis": longitudinal,
            "wall": wall,
            "bottom": bottom_normal,
            "base": base,
        })
    return result


def _prism_description(graph: dict[str, Any]) -> dict[str, Any] | None:
    evidence = extract_prismatic_key_feature(graph)
    if evidence.get("status") != "detected" or not evidence.get("feature"):
        return None
    feature = evidence["feature"]
    directions = [unit(np.asarray(value, dtype=float)) for value in feature["principal_directions"]]
    dimensions = [float(value) for value in feature["dimensions_mm"]]
    return {
        "feature": feature,
        "center": np.asarray(feature["center_point_mm"], dtype=float),
        "directions": directions,
        "dimensions": dimensions,
        "long_index": int(np.argmax(dimensions)),
    }


def _rotation(target: np.ndarray, source: np.ndarray) -> np.ndarray | None:
    raw = target @ source.T
    u, _, vt = np.linalg.svd(raw)
    value = u @ vt
    if np.linalg.det(value) < 0.0:
        u[:, -1] *= -1.0
        value = u @ vt
    return value if np.linalg.det(value) > .999 else None


def _proposals_key_to_slot(
    key_graph: dict[str, Any], slot_graph: dict[str, Any], *, clearance_mm: float
) -> list[dict[str, Any]]:
    key = _prism_description(key_graph)
    if key is None:
        return []
    cross_indices = [index for index in range(3) if index != key["long_index"]]
    proposals: dict[tuple[float, ...], dict[str, Any]] = {}
    for slot in _slot_descriptions(slot_graph):
        width = float(slot["candidate"]["wall_separation_mm"])
        for width_index, depth_index in (cross_indices, cross_indices[::-1]):
            width_dimension = key["dimensions"][width_index]
            depth_dimension = key["dimensions"][depth_index]
            if width_dimension > width + max(.05, clearance_mm) or width_dimension < .5 * width:
                continue
            key_basis_raw = np.column_stack((
                key["directions"][key["long_index"]],
                key["directions"][width_index],
                key["directions"][depth_index],
            ))
            for signs in itertools.product((-1.0, 1.0), repeat=3):
                key_basis = key_basis_raw @ np.diag(signs)
                for opening_sign in (-1.0, 1.0):
                    target_center = slot["base"] + opening_sign * slot["bottom"] * (
                        .5 * depth_dimension + clearance_mm
                    )
                    slot_basis = np.column_stack((
                        slot["axis"], slot["wall"], opening_sign * slot["bottom"]
                    ))
                    rotation = _rotation(key_basis, slot_basis)
                    if rotation is None:
                        continue
                    transform = np.eye(4)
                    transform[:3, :3] = rotation
                    transform[:3, 3] = key["center"] - rotation @ target_center
                    signature = tuple(np.round(transform[:3, :4].reshape(-1), 5))
                    proposals.setdefault(signature, {
                        "transform_slot_in_key": transform,
                        "key_feature_id": key["feature"]["feature_id"],
                        "slot_candidate_id": slot["candidate"]["candidate_id"],
                        "slot_width_mm": width,
                        "prism_width_mm": width_dimension,
                        "prism_depth_mm": depth_dimension,
                        "fit_ratio": width_dimension / max(width, 1e-9),
                        "clearance_mm": float(clearance_mm),
                        "opening_sign": opening_sign,
                        "frame_signs": list(signs),
                        "independent_evidence_count": int(
                            slot["candidate"].get("evidence_count", 5)
                        ) + 3,
                    })
    return sorted(
        proposals.values(),
        key=lambda row: (-row["fit_ratio"], np.linalg.norm(row["transform_slot_in_key"][:3, 3])),
    )


def _make_row(
    transform_b_in_a: np.ndarray, proposal: dict[str, Any], index: int, direction: str
) -> dict[str, Any]:
    evidence_count = int(proposal["independent_evidence_count"])
    confidence = min(.995, .55 + .05 * evidence_count + .04 * float(proposal["fit_ratio"]))
    public = {key: value for key, value in proposal.items() if key != "transform_slot_in_key"}
    return {
        "entity_a": f"compound_prismatic_set_a_{index:03d}",
        "entity_b": f"compound_prismatic_set_b_{index:03d}",
        "rank": index + 1,
        "manifold_type": "compound_prismatic_insertion_rigid",
        "frame_a": np.eye(4).tolist(),
        "frame_b": _inverse_rigid(transform_b_in_a).tolist(),
        "initial_pose_b_in_a": transform_b_in_a.tolist(),
        "free_dof_mask": [0, 0, 0, 0, 0, 0],
        "confidence": confidence,
        "provenance": {
            "multi_interface_prismatic": True,
            "independent_evidence_count": evidence_count,
            "geometry_only": True,
            "case_specific_override": False,
            "mapping_direction": direction,
            **public,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_result", type=Path)
    parser.add_argument("graph_a", type=Path)
    parser.add_argument("graph_b", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--clearance-mm",
        type=float,
        default=0.02,
        help=(
            "Small geometry-kernel clearance at slot walls/bottom; this is "
            "not an acceptance tolerance."
        ),
    )
    parser.add_argument("--maximum-additions", type=int, default=32)
    args = parser.parse_args()
    result, graph_a, graph_b = _read(args.base_result), _read(args.graph_a), _read(args.graph_b)
    proposals = []
    for proposal in _proposals_key_to_slot(graph_a, graph_b, clearance_mm=args.clearance_mm):
        proposals.append((proposal["transform_slot_in_key"], proposal, "a_prism_b_slot"))
    for proposal in _proposals_key_to_slot(graph_b, graph_a, clearance_mm=args.clearance_mm):
        proposals.append((
            _inverse_rigid(proposal["transform_slot_in_key"]), proposal, "a_slot_b_prism"
        ))
    additions = [
        _make_row(transform, proposal, index, direction)
        for index, (transform, proposal, direction) in enumerate(
            proposals[: max(1, args.maximum_additions)]
        )
    ]
    output = copy.deepcopy(result)
    output.setdefault("joint_hypotheses", {}).setdefault("rows", []).extend(additions)
    output["joint_hypotheses"]["multi_interface_prismatic"] = {
        "added_rows": len(additions),
        "geometry_only": True,
        "case_specific_override": False,
        "acceptance_boundary": "Proposal only; global closure and exact OCCT validation remain mandatory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"added_rows": len(additions), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
