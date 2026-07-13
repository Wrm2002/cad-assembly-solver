"""Shared, dependency-light readers for Fusion Joint/JoinABLe records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_joint_files(root: Path) -> list[Path]:
    patterns = ("joint_set_[0-9][0-9][0-9][0-9][0-9].json",)
    files: set[Path] = set()
    for pattern in patterns:
        files.update(root.rglob(pattern))
    return sorted(files)


def graph_file(joint_file: Path, body_id: str) -> Path:
    return joint_file.parent / f"{body_id}.json"


def geometry_file(joint_file: Path, body_id: str) -> Path | None:
    for suffix in (".smt", ".step", ".stp", ".obj"):
        candidate = joint_file.parent / f"{body_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def transform_matrix(transform: Any) -> list[list[float]] | None:
    if not isinstance(transform, dict):
        return None
    required = ("x_axis", "y_axis", "z_axis", "origin")
    if not all(isinstance(transform.get(key), dict) for key in required):
        return None
    try:
        columns = []
        for key in required:
            value = transform[key]
            columns.append([
                float(value["x"]),
                float(value["y"]),
                float(value["z"]),
                0.0 if key != "origin" else 1.0,
            ])
        return [
            [columns[column][row] for column in range(4)]
            for row in range(4)
        ]
    except (KeyError, TypeError, ValueError):
        return None


def relative_transform(
    transform_a: list[list[float]] | None,
    transform_b: list[list[float]] | None,
) -> list[list[float]] | None:
    if transform_a is None or transform_b is None:
        return None
    try:
        # Fusion transforms are rigid.  Avoid numpy.linalg here because the
        # project's Windows BLAS runtime is not reliable after the OS rebuild.
        rotation_b_t = [
            [float(transform_b[column][row]) for column in range(3)]
            for row in range(3)
        ]
        translation_b = [
            float(transform_b[row][3]) for row in range(3)
        ]
        inverse_b = [
            rotation_b_t[row]
            + [-sum(
                rotation_b_t[row][k] * translation_b[k]
                for k in range(3)
            )]
            for row in range(3)
        ]
        inverse_b.append([0.0, 0.0, 0.0, 1.0])
        return [[
            sum(
                inverse_b[row][k] * float(transform_a[k][column])
                for k in range(4)
            )
            for column in range(4)
        ] for row in range(4)]
    except (IndexError, TypeError, ValueError):
        return None


def selected_entity(geometry: Any) -> dict[str, Any] | None:
    if not isinstance(geometry, dict):
        return None
    entity = geometry.get("entity_one")
    return entity if isinstance(entity, dict) else None


def graph_node_for_entity(
    graph: dict[str, Any] | None,
    entity: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, int | None]:
    if graph is None or entity is None:
        return None, None
    try:
        source_index = int(entity["index"])
    except (KeyError, TypeError, ValueError):
        return None, None
    entity_type = entity.get("type")
    properties = graph.get("properties") or {}
    if entity_type == "BRepFace":
        node_index = source_index
    elif entity_type == "BRepEdge":
        node_index = int(properties.get("face_count", 0)) + source_index
    else:
        return None, None
    nodes = graph.get("nodes") or []
    if node_index < 0 or node_index >= len(nodes):
        return None, node_index
    return nodes[node_index], node_index


def _xyz(value: Any) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return [
            float(value["x"]),
            float(value["y"]),
            float(value["z"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None


def _point(node: dict[str, Any], key: str) -> list[float] | None:
    direct = _xyz(node.get(key))
    if direct is not None:
        return direct
    try:
        return [
            float(node[f"{key}_x"]),
            float(node[f"{key}_y"]),
            float(node[f"{key}_z"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None


def _normalized(vector: list[float] | None) -> list[float] | None:
    if vector is None:
        return None
    length = sum(float(value) ** 2 for value in vector) ** 0.5
    if length <= 1e-12:
        return None
    return [float(value) / length for value in vector]


def axis_from_graph_node(
    node: dict[str, Any] | None,
) -> tuple[list[float] | None, list[float] | None]:
    """Mirror JoinABLe's entity-to-axis rules without importing torch."""
    if not node:
        return None, None
    surface = node.get("surface_type")
    curve = node.get("curve_type")
    if surface == "PlaneSurfaceType":
        return _point(node, "centroid"), _normalized(
            _point(node, "normal")
        )
    if surface in {
        "CylinderSurfaceType",
        "EllipticalCylinderSurfaceType",
        "ConeSurfaceType",
        "EllipticalConeSurfaceType",
        "TorusSurfaceType",
    }:
        return _point(node, "origin"), _normalized(_point(node, "axis"))
    if surface == "SphereSurfaceType":
        return _point(node, "origin"), [0.0, 0.0, 1.0]
    if curve == "Line3DCurveType":
        start = _point(node, "start_point")
        end = _point(node, "end_point")
        if start is None or end is None:
            return None, None
        return start, _normalized([
            end[index] - start[index] for index in range(3)
        ])
    if curve in {
        "Arc3DCurveType",
        "Circle3DCurveType",
        "Ellipse3DCurveType",
        "EllipticalArc3DCurveType",
    }:
        return _point(node, "center"), _normalized(
            _point(node, "normal")
        )
    return None, None


def joint_type(joint: dict[str, Any]) -> str:
    motion = joint.get("joint_motion")
    if isinstance(motion, dict) and motion.get("joint_type"):
        return str(motion["joint_type"])
    for key in ("joint_type", "type", "name"):
        if joint.get(key):
            return str(joint[key])
    return "unknown"


def interface_record(
    geometry: Any,
    graph: dict[str, Any] | None,
    graph_path: Path,
) -> dict[str, Any]:
    entity = selected_entity(geometry)
    node, node_index = graph_node_for_entity(graph, entity)
    origin, direction = axis_from_graph_node(node)
    missing = []
    if entity is None:
        missing.append("selected_brep_entity")
    if node is None:
        missing.append("selected_entity_graph_node")
    if origin is None or direction is None:
        missing.append("axis_origin_or_direction")
    entity_type = entity.get("type") if entity else None
    source_index = entity.get("index") if entity else None
    return {
        "entity_type": entity_type,
        "entity_ids": (
            [{
                "source_entity_type": entity_type,
                "source_entity_index": source_index,
                "joinable_node_index": node_index,
            }]
            if entity is not None else []
        ),
        "surface_or_curve_type": (
            node.get("surface_type") or node.get("curve_type")
            if node else None
        ),
        "semantic_role": "designer_selected_joint_origin",
        "axis_origin": origin,
        "axis_direction": direction,
        "brep_graph_path": str(graph_path.resolve()),
        "source_node_features": {
            key: node.get(key)
            for key in (
                "surface_type",
                "curve_type",
                "area",
                "length",
                "radius",
                "reversed",
                "convexity",
                "dihedral_angle",
            )
            if node and key in node
        },
        "failure_reasons": [],
        "unavailable_fields": missing,
    }


def _contact_entities(contact: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for first_key, second_key in (
        ("entity_one", "entity_two"),
        ("entityOne", "entityTwo"),
    ):
        first = contact.get(first_key)
        second = contact.get(second_key)
        if isinstance(first, dict) and isinstance(second, dict):
            records.append({
                "part_a_entity_ids": [first],
                "part_b_entity_ids": [second],
            })
            break
    return records


def build_common_sample(
    joint_file: Path,
    joint_index: int,
) -> dict[str, Any]:
    data = load_json(joint_file)
    joints = data.get("joints") or []
    if joint_index < 0 or joint_index >= len(joints):
        raise IndexError(f"joint_index_out_of_range:{joint_index}")
    joint = joints[joint_index]
    body_a = str(data.get("body_one", ""))
    body_b = str(data.get("body_two", ""))
    graph_a_path = graph_file(joint_file, body_a)
    graph_b_path = graph_file(joint_file, body_b)
    graph_a = load_json(graph_a_path) if graph_a_path.is_file() else None
    graph_b = load_json(graph_b_path) if graph_b_path.is_file() else None
    geometry_a = geometry_file(joint_file, body_a)
    geometry_b = geometry_file(joint_file, body_b)
    geo_a = joint.get("geometry_or_origin_one")
    geo_b = joint.get("geometry_or_origin_two")
    interface_a = interface_record(geo_a, graph_a, graph_a_path)
    interface_b = interface_record(geo_b, graph_b, graph_b_path)
    transform_a = transform_matrix(
        geo_a.get("transform") if isinstance(geo_a, dict) else None
    )
    transform_b = transform_matrix(
        geo_b.get("transform") if isinstance(geo_b, dict) else None
    )
    unavailable = []
    contacts = []
    for contact in data.get("contacts") or []:
        if isinstance(contact, dict):
            contact_joint_index = contact.get("joint_index")
            if contact_joint_index is not None:
                try:
                    if int(contact_joint_index) != joint_index:
                        continue
                except (TypeError, ValueError):
                    unavailable.append(
                        "contacts.invalid_joint_index"
                    )
            contacts.extend(_contact_entities(contact))
    if graph_a is None:
        unavailable.append("part_a.brep_graph_path")
    if graph_b is None:
        unavailable.append("part_b.brep_graph_path")
    if geometry_a is None:
        unavailable.append("part_a.geometry_path")
    if geometry_b is None:
        unavailable.append("part_b.geometry_path")
    if transform_a is None or transform_b is None:
        unavailable.append("relation.transform_a_to_b")
    if not contacts:
        unavailable.append("contacts")
    if not data.get("holes"):
        unavailable.append("holes")
    unavailable.extend(
        f"interface_a.{field}"
        for field in interface_a["unavailable_fields"]
    )
    unavailable.extend(
        f"interface_b.{field}"
        for field in interface_b["unavailable_fields"]
    )
    explicit_assembly_id = (
        data.get("assembly_id") or data.get("design_id")
    )
    body_a_base = body_a.rsplit("_", 1)[0] if "_" in body_a else body_a
    body_b_base = body_b.rsplit("_", 1)[0] if "_" in body_b else body_b
    inferred_assembly_id = (
        body_a_base if body_a_base == body_b_base else joint_file.stem
    )
    if explicit_assembly_id is None:
        unavailable.append("source_explicit_assembly_id")
    assembly_id = str(explicit_assembly_id or inferred_assembly_id)
    return {
        "schema_version": "1.0.0",
        "sample_id": f"{joint_file.stem}:joint:{joint_index}",
        "source_dataset": "fusion360_joinable_joint",
        "assembly_id": assembly_id,
        "part_a": {
            "part_id": body_a,
            "geometry_path": (
                str(geometry_a.resolve()) if geometry_a else None
            ),
            "brep_graph_path": str(graph_a_path.resolve()),
            "source_body_role": "body_one",
        },
        "part_b": {
            "part_id": body_b,
            "geometry_path": (
                str(geometry_b.resolve()) if geometry_b else None
            ),
            "brep_graph_path": str(graph_b_path.resolve()),
            "source_body_role": "body_two",
        },
        "relation": {
            "has_joint": True,
            "label_semantics": "designer_selected_joint",
            "compatibility_label": "positive",
            "joint_type": joint_type(joint),
            "axis_origin": interface_a["axis_origin"],
            "axis_direction": interface_a["axis_direction"],
            "axis_origin_b": interface_b["axis_origin"],
            "axis_direction_b": interface_b["axis_direction"],
            "transform_a_to_world": transform_a,
            "transform_b_to_world": transform_b,
            "transform_a_to_b": relative_transform(
                transform_a, transform_b
            ),
            "offset": joint.get("offset"),
            "angle": joint.get("angle"),
            "is_flipped": joint.get("is_flipped"),
        },
        "interface_a": interface_a,
        "interface_b": interface_b,
        "contacts": contacts,
        "holes": data.get("holes") or [],
        "metadata": {
            "unit": "cm",
            "source_file": str(joint_file.resolve()),
            "source_sha256": sha256_file(joint_file),
            "source_joint_index": joint_index,
            "assembly_id_semantics": (
                "source_explicit"
                if explicit_assembly_id is not None
                else "inferred_from_common_body_name_prefix"
            ),
            "conversion_status": "success",
            "failure_reason": None,
            "failure_reasons": [],
            "unavailable_fields": sorted(set(unavailable)),
        },
        "failure_reasons": [],
        "unavailable_fields": sorted(set(unavailable)),
    }


def all_joint_records(
    files: Iterable[Path],
) -> Iterable[tuple[Path, int]]:
    for path in files:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        for index, _ in enumerate(data.get("joints") or []):
            yield path, index
