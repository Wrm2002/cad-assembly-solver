"""Leakage-safe Fusion 360 B-Rep joint supervision.

Storage locators are intentionally kept outside ``model_input``.  A data
loader may use them to open the raw B-Rep graph, but the projection function
below is the only supported path from raw JSON to tensors.  Names, paths,
case identifiers, BOM text, CAD authoring metadata, and SolidWorks outputs are
therefore not reachable by the model.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "pure_brep_joint_supervision.v1"

ALLOWED_NODE_FIELDS = (
    "id",
    "points",
    "normals",
    "tangents",
    "trimming_mask",
    "surface_type",
    "curve_type",
    "reversed",
    "area",
    "length",
    "radius",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "origin_x",
    "origin_y",
    "origin_z",
    "axis_x",
    "axis_y",
    "axis_z",
    "point_on_face_x",
    "point_on_face_y",
    "point_on_face_z",
    "max_tangent_x",
    "max_tangent_y",
    "max_tangent_z",
    "max_curvature",
    "min_curvature",
    "convexity",
    "dihedral_angle",
)
ALLOWED_LINK_FIELDS = ("source", "target")

FORBIDDEN_MODEL_KEY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(^|_)file(_|$)",
        r"(^|_)path(_|$)",
        r"(^|_)name(_|$)",
        r"case(_|$)",
        r"bom",
        r"solidworks",
        r"source_id",
        r"assembly_id",
        r"part_role",
        r"assembly_family",
    )
)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def project_brep_graph(raw: dict[str, Any]) -> dict[str, Any]:
    """Return only numeric/analytic B-Rep topology usable by a model."""

    nodes = []
    for raw_node in raw.get("nodes", []):
        node = {
            key: raw_node[key]
            for key in ALLOWED_NODE_FIELDS
            if key in raw_node
        }
        # Reject arbitrary nested strings rather than silently carrying CAD
        # metadata.  Surface/curve type are the only categorical strings.
        for key, value in tuple(node.items()):
            if key in {"surface_type", "curve_type", "convexity"}:
                continue
            if isinstance(value, list):
                node[key] = [float(item) for item in value if _finite_number(item)]
            elif _finite_number(value) or isinstance(value, bool):
                node[key] = value
            else:
                node.pop(key)
        nodes.append(node)
    links = [
        {key: row[key] for key in ALLOWED_LINK_FIELDS if key in row}
        for row in raw.get("links", [])
    ]
    return {
        "directed": bool(raw.get("directed", False)),
        "multigraph": bool(raw.get("multigraph", False)),
        "nodes": nodes,
        "links": links,
    }


def _vector(value: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    result = [value.get(axis) for axis in ("x", "y", "z")]
    if not all(_finite_number(item) for item in result):
        return None
    return [float(item) for item in result]


def _unit(value: list[float] | None) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(array))
    return None if norm <= 1e-12 else (array / norm).tolist()


def transform_matrix(value: dict[str, Any] | None) -> list[list[float]] | None:
    if not isinstance(value, dict):
        return None
    origin = _vector(value.get("origin"))
    axes = [_unit(_vector(value.get(f"{axis}_axis"))) for axis in "xyz"]
    if origin is None or any(axis is None for axis in axes):
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(axes, dtype=float).T
    matrix[:3, 3] = origin
    if abs(float(np.linalg.det(matrix[:3, :3])) - 1.0) > 1e-4:
        return None
    return matrix.tolist()


def _entity(entity: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(entity, dict):
        return None
    kind = str(entity.get("type", ""))
    if kind not in {"BRepFace", "BRepEdge"}:
        return None
    geometry_type = entity.get("surface_type", entity.get("curve_type"))
    return {
        "entity_kind": "face" if kind == "BRepFace" else "edge",
        "topology_index": int(entity["index"]),
        "geometry_type": str(geometry_type),
    }


def _joint_frame(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "origin": _vector(geometry.get("origin")),
        "primary_axis": _unit(_vector(geometry.get("primary_axis_vector"))),
        "secondary_axis": _unit(_vector(geometry.get("secondary_axis_vector"))),
        "tertiary_axis": _unit(_vector(geometry.get("tertiary_axis_vector"))),
        "axis_origin": _vector((geometry.get("axis_line") or {}).get("origin")),
        "axis_direction": _unit(_vector((geometry.get("axis_line") or {}).get("direction"))),
    }


def free_dof_mask(joint_type: str) -> list[int] | None:
    """Return [tx,ty,tz,rx,ry,rz] in the Fusion joint-local frame."""

    mapping = {
        "RigidJointType": (0, 0, 0, 0, 0, 0),
        "RevoluteJointType": (0, 0, 0, 0, 0, 1),
        "SliderJointType": (0, 0, 1, 0, 0, 0),
        "CylindricalJointType": (0, 0, 1, 0, 0, 1),
        "PinSlotJointType": (1, 0, 0, 0, 0, 1),
        "PlanarJointType": (1, 1, 0, 0, 0, 1),
        "BallJointType": (0, 0, 0, 1, 1, 1),
    }
    value = mapping.get(str(joint_type))
    return list(value) if value is not None else None


def _equivalents(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for row in geometry.get("entity_one_equivalents") or []:
        parsed = _entity(row)
        if parsed is not None:
            values.append(parsed)
    return values


def _joint_supervision(joint: dict[str, Any]) -> dict[str, Any] | None:
    left = joint.get("geometry_or_origin_one")
    right = joint.get("geometry_or_origin_two")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    entity_a, entity_b = _entity(left.get("entity_one")), _entity(right.get("entity_one"))
    if entity_a is None or entity_b is None:
        return None
    transform_a = transform_matrix(left.get("transform"))
    transform_b = transform_matrix(right.get("transform"))
    relative = None
    if transform_a is not None and transform_b is not None:
        relative = (
            np.linalg.inv(np.asarray(transform_a)) @ np.asarray(transform_b)
        ).tolist()
    motion = joint.get("joint_motion") or {}
    joint_type = str(motion.get("joint_type", "UnknownJointType"))
    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "frame_a": _joint_frame(left),
        "frame_b": _joint_frame(right),
        "relative_pose": relative,
        "free_dof_mask": free_dof_mask(joint_type),
        "equivalent_entities_a": _equivalents(left),
        "equivalent_entities_b": _equivalents(right),
        "offset": float((joint.get("offset") or {}).get("value", 0.0)),
        "angle": float((joint.get("angle") or {}).get("value", 0.0)),
        "is_flipped": bool(joint.get("is_flipped", False)),
        # Auxiliary target only: never fed into the solver as a named rule.
        "auxiliary_joint_type": joint_type,
    }


def design_group_from_body_token(token: str) -> str:
    """Recover the public dataset design group for split isolation only."""

    fields = str(token).split("_")
    return "_".join(fields[:2]) if len(fields) >= 2 else str(token)


def deterministic_split(group_token: str) -> str:
    bucket = int(sha256(group_token.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "dev" if bucket < 90 else "test"


def make_record(joint_path: Path, raw: dict[str, Any]) -> dict[str, Any] | None:
    body_a, body_b = str(raw.get("body_one", "")), str(raw.get("body_two", ""))
    if not body_a or not body_b:
        return None
    group_a, group_b = design_group_from_body_token(body_a), design_group_from_body_token(body_b)
    if group_a != group_b:
        # Cross-design samples would invalidate assembly-isolated splitting.
        return None
    supervision = [
        row for joint in raw.get("joints") or []
        if (row := _joint_supervision(joint)) is not None
    ]
    if not supervision:
        return None
    record_id = sha256(
        f"{body_a}|{body_b}|{joint_path.name}".encode("utf-8")
    ).hexdigest()[:24]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "split": deterministic_split(group_a),
        "storage": {
            "joint_json": str(joint_path.resolve()),
            "body_graph_a": str((joint_path.parent / f"{body_a}.json").resolve()),
            "body_graph_b": str((joint_path.parent / f"{body_b}.json").resolve()),
            "split_group_hash": sha256(group_a.encode("utf-8")).hexdigest()[:16],
        },
        "model_input": {
            "representation": "paired_brep_topology",
            "allowed_node_fields": list(ALLOWED_NODE_FIELDS),
            "allowed_link_fields": list(ALLOWED_LINK_FIELDS),
            "normalization": "pairwise_center_and_isotropic_scale",
        },
        "supervision": supervision,
    }


def iter_forbidden_model_keys(value: Any, prefix: str = "model_input") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if any(pattern.search(str(key)) for pattern in FORBIDDEN_MODEL_KEY_PATTERNS):
                yield path
            yield from iter_forbidden_model_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_forbidden_model_keys(child, f"{prefix}[{index}]")


def audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    forbidden = []
    group_splits: dict[str, set[str]] = {}
    type_counts: dict[str, int] = {}
    for record in rows:
        forbidden.extend(
            {"record_id": record.get("record_id"), "path": path}
            for path in iter_forbidden_model_keys(record.get("model_input", {}))
        )
        group_hash = record["storage"]["split_group_hash"]
        group_splits.setdefault(group_hash, set()).add(record["split"])
        for joint in record.get("supervision", []):
            key = str(joint.get("auxiliary_joint_type"))
            type_counts[key] = type_counts.get(key, 0) + 1
    overlaps = {
        key: sorted(value) for key, value in group_splits.items() if len(value) > 1
    }
    return {
        "schema_version": "pure_brep_contract_audit.v1",
        "record_count": len(rows),
        "joint_supervision_count": sum(len(row.get("supervision", [])) for row in rows),
        "split_counts": {
            split: sum(row.get("split") == split for row in rows)
            for split in ("train", "dev", "test")
        },
        "auxiliary_joint_type_counts": dict(sorted(type_counts.items())),
        "forbidden_model_key_count": len(forbidden),
        "forbidden_model_keys": forbidden[:100],
        "split_group_overlap_count": len(overlaps),
        "split_group_overlaps": overlaps,
        "passed": not forbidden and not overlaps and bool(rows),
        "model_input_boundary": (
            "Only project_brep_graph(raw) and model_input.allowed_* fields may reach tensors; "
            "storage locators and auxiliary labels are audit/supervision metadata."
        ),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
