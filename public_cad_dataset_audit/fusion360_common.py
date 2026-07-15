"""Shared, dependency-free helpers for Fusion 360 Assembly JSON files."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_assembly_files(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file() and path.name == "assembly.json":
        return [path]
    if not path.exists():
        return []
    return sorted(path.glob("**/assembly.json"))


def _identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matrix_multiply(
    first: list[list[float]], second: list[list[float]]
) -> list[list[float]]:
    return [
        [
            sum(first[row][index] * second[index][column]
                for index in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def transform_matrix(transform: dict[str, Any] | None) -> list[list[float]]:
    if not transform:
        return _identity()
    axes = [
        transform["x_axis"],
        transform["y_axis"],
        transform["z_axis"],
    ]
    origin = transform["origin"]
    return [
        [
            float(axes[column][axis])
            for column in range(3)
        ] + [float(origin[axis])]
        for axis in ("x", "y", "z")
    ] + [[0.0, 0.0, 0.0, 1.0]]


def _geometry_candidates(
    body: dict[str, Any], assembly_dir: Path
) -> list[dict[str, Any]]:
    candidates = []
    for field, geometry_format in (
        ("smt", "smt"),
        ("step", "step"),
        ("obj", "obj"),
    ):
        declared = body.get(field)
        path = assembly_dir / declared if declared else None
        candidates.append({
            "format": geometry_format,
            "declared_path": declared,
            "path": str(path.resolve()) if path else None,
            "exists": bool(path and path.is_file()),
        })
    return candidates


def build_parts(
    data: dict[str, Any],
    assembly_dir: Path,
    *,
    visible_only: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return occurrence-body instances, not only unique body definitions."""
    parts: list[dict[str, Any]] = []
    failures: list[str] = []
    bodies = data.get("bodies") or {}

    def add_part(
        body_id: str,
        occurrence_id: str | None,
        occurrence: dict[str, Any] | None,
        matrix: list[list[float]],
        is_visible: bool,
    ) -> None:
        body = bodies.get(body_id)
        if body is None:
            failures.append(f"body_definition_missing:{body_id}")
            return
        part_id = (
            body_id if occurrence_id is None
            else f"{occurrence_id}_{body_id}"
        )
        candidates = _geometry_candidates(body, assembly_dir)
        preferred = next(
            (
                candidate for candidate in candidates
                if candidate["exists"]
                and candidate["format"] in {"smt", "step"}
            ),
            next(
                (
                    candidate for candidate in candidates
                    if candidate["exists"]
                ),
                None,
            ),
        )
        parts.append({
            "part_id": part_id,
            "body_id": body_id,
            "occurrence_id": occurrence_id,
            "part_name": (
                occurrence.get("name") if occurrence else body.get("name")
            ),
            "body_name": body.get("name"),
            "visible": is_visible,
            "transform": matrix,
            "geometry": {
                "path": preferred["path"] if preferred else None,
                "format": preferred["format"] if preferred else None,
                "available": preferred is not None,
                "candidates": candidates,
                "unavailable_reason": (
                    None if preferred else "declared_geometry_file_missing"
                ),
            },
        })

    root_bodies = (data.get("root") or {}).get("bodies") or {}
    for body_id, state in root_bodies.items():
        visible = bool(state.get("is_visible", True))
        if visible_only and not visible:
            continue
        add_part(body_id, None, None, _identity(), visible)

    occurrences = data.get("occurrences") or {}

    def walk(
        subtree: dict[str, Any],
        parent_matrix: list[list[float]],
    ) -> None:
        for occurrence_id, child_tree in subtree.items():
            occurrence = occurrences.get(occurrence_id)
            if occurrence is None:
                failures.append(
                    f"tree_occurrence_missing:{occurrence_id}"
                )
                continue
            occurrence_visible = bool(
                occurrence.get("is_visible", True)
            )
            matrix = _matrix_multiply(
                parent_matrix,
                transform_matrix(occurrence.get("transform")),
            )
            for body_id, state in (
                occurrence.get("bodies") or {}
            ).items():
                visible = (
                    occurrence_visible
                    and bool(state.get("is_visible", True))
                )
                if not visible_only or visible:
                    add_part(
                        body_id,
                        occurrence_id,
                        occurrence,
                        matrix,
                        visible,
                    )
            walk(child_tree or {}, matrix)

    tree_root = ((data.get("tree") or {}).get("root") or {})
    walk(tree_root, _identity())
    duplicate_ids = len(parts) - len({part["part_id"] for part in parts})
    if duplicate_ids:
        failures.append(f"duplicate_part_instance_ids:{duplicate_ids}")
    return parts, failures


def entity_part_id(entity: dict[str, Any]) -> str | None:
    body = entity.get("body")
    if not body:
        return None
    occurrence = entity.get("occurrence")
    return f"{occurrence}_{body}" if occurrence else body


def interface_reference(
    entity: dict[str, Any] | None,
    expected_part_id: str | None,
    parts_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not entity:
        return {
            "part_id": expected_part_id,
            "mapping_status": "unavailable",
            "failure_reason": "interface_entity_missing",
            "entity_type": None,
            "entity_id": None,
            "local_geometry": None,
        }
    part_id = entity_part_id(entity) or expected_part_id
    entity_type = entity.get("type")
    entity_index = entity.get("index")
    part_exists = part_id in parts_by_id
    index_available = isinstance(entity_index, int) and entity_index >= 0
    status = "mapped" if part_exists else "unmapped"
    failure_reason = (
        None if part_exists else f"part_instance_not_found:{part_id}"
    )
    return {
        "part_id": part_id,
        "mapping_status": status,
        "failure_reason": failure_reason,
        "entity_type": entity_type,
        "entity_id": (
            f"{entity_type}:{entity_index}"
            if entity_type and index_available else None
        ),
        "body_id": entity.get("body"),
        "occurrence_id": entity.get("occurrence"),
        "topology_index": entity_index,
        "topology_index_available": index_available,
        "geometry_file_available": bool(
            part_exists
            and parts_by_id[part_id]["geometry"]["available"]
        ),
        "local_geometry": {
            "surface_type": entity.get("surface_type"),
            "point_on_entity": entity.get("point_on_entity"),
            "bounding_box": entity.get("bounding_box"),
        },
        "unavailable_fields": [
            field for field, value in (
                ("topology_index", entity_index),
                ("surface_type", entity.get("surface_type")),
                ("point_on_entity", entity.get("point_on_entity")),
                ("bounding_box", entity.get("bounding_box")),
            )
            if value is None
        ],
    }


def _relation(
    source_id: str,
    relation_kind: str,
    relation_type: str | None,
    first_entity: dict[str, Any] | None,
    second_entity: dict[str, Any] | None,
    parts_by_id: dict[str, dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    first_id = entity_part_id(first_entity or {})
    second_id = entity_part_id(second_entity or {})
    first_ref = interface_reference(first_entity, first_id, parts_by_id)
    second_ref = interface_reference(
        second_entity, second_id, parts_by_id
    )
    first_id = first_ref["part_id"]
    second_id = second_ref["part_id"]
    if not first_id or not second_id:
        return None, f"{relation_kind}:{source_id}:part_endpoint_missing"
    if first_id == second_id:
        return None, f"{relation_kind}:{source_id}:self_pair"
    pair = sorted((first_id, second_id))
    mapped = (
        first_ref["mapping_status"] == "mapped"
        and second_ref["mapping_status"] == "mapped"
    )
    relation = {
        "source_relation_id": source_id,
        "relation_kind": relation_kind,
        "relation_type": relation_type or "unknown",
        "part_pair": pair,
        "interface_entities": [first_ref, second_ref],
        "part_pair_mapping_status": (
            "mapped" if mapped else "unmapped"
        ),
        "failure_reasons": [
            reason for reason in (
                first_ref.get("failure_reason"),
                second_ref.get("failure_reason"),
            )
            if reason
        ],
        "source_payload": extra or {},
    }
    return relation, None if mapped else (
        f"{relation_kind}:{source_id}:part_pair_unmapped"
    )


def extract_relations(
    data: dict[str, Any],
    parts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    parts_by_id = {part["part_id"]: part for part in parts}
    relations: list[dict[str, Any]] = []
    failures: list[str] = []

    for joint_id, joint in (data.get("joints") or {}).items():
        try:
            first = joint["geometry_or_origin_one"].get("entity_one")
            second = joint["geometry_or_origin_two"].get("entity_one")
        except (KeyError, TypeError):
            first = second = None
        relation, failure = _relation(
            joint_id,
            "joint",
            (joint.get("joint_motion") or {}).get("joint_type"),
            first,
            second,
            parts_by_id,
            {
                "name": joint.get("name"),
                "geometry_type_one": (
                    joint.get("geometry_or_origin_one") or {}
                ).get("geometry_type"),
                "geometry_type_two": (
                    joint.get("geometry_or_origin_two") or {}
                ).get("geometry_type"),
            },
        )
        if relation:
            relations.append(relation)
        if failure:
            failures.append(failure)

    occurrences = data.get("occurrences") or {}
    for joint_id, joint in (
        data.get("as_built_joints") or {}
    ).items():
        occurrence_ids = [
            joint.get("occurrence_one"),
            joint.get("occurrence_two"),
        ]
        entities = []
        geometry_entity = (
            (joint.get("joint_geometry") or {}).get("entity_one")
        )
        for occurrence_id in occurrence_ids:
            body_ids = list(
                (occurrences.get(occurrence_id) or {})
                .get("bodies", {})
            )
            if (
                geometry_entity
                and geometry_entity.get("occurrence") == occurrence_id
            ):
                entities.append(geometry_entity)
            elif len(body_ids) == 1:
                entities.append({
                    "type": None,
                    "occurrence": occurrence_id,
                    "body": body_ids[0],
                })
            else:
                entities.append(None)
        relation, failure = _relation(
            joint_id,
            "as_built_joint",
            (joint.get("joint_motion") or {}).get("joint_type"),
            entities[0],
            entities[1],
            parts_by_id,
            {"name": joint.get("name")},
        )
        if relation:
            relations.append(relation)
        if failure:
            failures.append(failure)

    for contact_index, contact in enumerate(data.get("contacts") or []):
        contact_id = str(contact.get("id", contact_index))
        relation, failure = _relation(
            contact_id,
            "contact",
            "contact",
            contact.get("entity_one"),
            contact.get("entity_two"),
            parts_by_id,
            {
                "contact_area": contact.get("contact_area"),
                "contact_volume": contact.get("contact_volume"),
            },
        )
        if relation:
            relations.append(relation)
        if failure:
            failures.append(failure)
    return relations, failures


def aggregate_edges(
    relations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        grouped[tuple(relation["part_pair"])].append(relation)
    edges = []
    for index, pair in enumerate(sorted(grouped), start=1):
        rows = grouped[pair]
        edges.append({
            "edge_id": f"positive_{index:06d}",
            "part_pair": list(pair),
            "relation_types": sorted({
                row["relation_type"] for row in rows
            }),
            "relation_kinds": sorted({
                row["relation_kind"] for row in rows
            }),
            "relations": rows,
            "source_dataset": "fusion360_gallery_assembly",
            "failure_reasons": sorted({
                failure for row in rows
                for failure in row.get("failure_reasons", [])
            }),
            "unavailable_fields": sorted({
                field for row in rows
                for entity in row.get("interface_entities", [])
                for field in entity.get("unavailable_fields", [])
            }),
        })
    return edges


def negative_edges(
    part_ids: list[str],
    positive_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    positive_pairs = {
        tuple(edge["part_pair"]) for edge in positive_edges
    }
    negatives = []
    for first, second in combinations(sorted(part_ids), 2):
        if (first, second) in positive_pairs:
            continue
        negatives.append({
            "edge_id": f"negative_{len(negatives) + 1:06d}",
            "part_pair": [first, second],
            "relation_type": "none_observed",
            "source_dataset": "fusion360_gallery_assembly",
            "negative_definition": (
                "No joint, as-built joint, or contact is recorded between "
                "this visible occurrence-body pair in this assembly."
            ),
            "failure_reasons": [],
            "unavailable_fields": [
                "proof_of_physical_non_interaction"
            ],
        })
    return negatives


def convert_assembly(
    assembly_file: Path,
) -> dict[str, Any]:
    data = load_json(assembly_file)
    parts, part_failures = build_parts(
        data, assembly_file.parent, visible_only=True
    )
    relations, relation_failures = extract_relations(data, parts)
    positives = aggregate_edges(relations)
    negatives = negative_edges(
        [part["part_id"] for part in parts], positives
    )
    unavailable = set()
    for part in parts:
        if not part["geometry"]["available"]:
            unavailable.add(
                f"part_geometry:{part['part_id']}"
            )
    for edge in positives:
        unavailable.update(edge["unavailable_fields"])
    failure_reasons = sorted(
        set(part_failures + relation_failures)
    )
    unavailable_fields = sorted(unavailable)
    return {
        "schema_version": "1.0.0",
        "assembly_id": assembly_file.parent.name,
        "source_dataset": "fusion360_gallery_assembly",
        "source_record_path": str(assembly_file.resolve()),
        "source_record_sha256": sha256_file(assembly_file),
        "units": {"length": "cm", "angle": "radian"},
        "parts": parts,
        "positive_part_pair_edges": positives,
        "negative_part_pair_edges": negatives,
        "failure_reasons": failure_reasons,
        "unavailable_fields": unavailable_fields,
        "quality": {
            "status": (
                "usable" if len(parts) >= 2 and positives
                else "insufficient"
            ),
            "failure_reasons": failure_reasons,
            "unavailable_fields": unavailable_fields,
            "part_count": len(parts),
            "positive_pair_count": len(positives),
            "negative_pair_count": len(negatives),
            "relation_count": len(relations),
        },
    }
