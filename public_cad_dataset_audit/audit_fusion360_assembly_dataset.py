"""Audit real Fusion 360 Gallery Assembly folders and label mappings."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from fusion360_common import (
    build_parts,
    discover_assembly_files,
    extract_relations,
    load_json,
    write_json,
)


def _increment(counter: Counter, key: Any, amount: int = 1) -> None:
    counter[str(key if key is not None else "unavailable")] += amount


def _obj_topology_indices(
    path: Path, cache: dict[Path, dict[str, set[int]]]
) -> dict[str, set[int]]:
    if path in cache:
        return cache[path]
    result = {"BRepFace": set(), "BRepEdge": set()}
    if path.is_file():
        with path.open(
            "r", encoding="utf-8", errors="ignore"
        ) as handle:
            for line in handle:
                if line.startswith("g face "):
                    try:
                        result["BRepFace"].add(
                            int(line.split()[2])
                        )
                    except (IndexError, ValueError):
                        continue
                elif line.startswith("g halfedge "):
                    tokens = line.split()
                    try:
                        edge_position = tokens.index("edge") + 1
                        result["BRepEdge"].add(
                            int(tokens[edge_position])
                        )
                    except (ValueError, IndexError):
                        continue
    cache[path] = result
    return result


def _verify_entity_in_obj(
    entity: dict[str, Any],
    part: dict[str, Any] | None,
    cache: dict[Path, dict[str, set[int]]],
) -> str:
    if part is None:
        return "part_unmapped"
    entity_type = entity.get("entity_type")
    index = entity.get("topology_index")
    if entity_type not in {"BRepFace", "BRepEdge"}:
        return "entity_type_not_verifiable_in_obj"
    if not isinstance(index, int):
        return "topology_index_unavailable"
    obj = next(
        (
            candidate for candidate
            in part["geometry"]["candidates"]
            if candidate["format"] == "obj"
            and candidate["exists"]
        ),
        None,
    )
    if obj is None:
        return "obj_geometry_unavailable"
    indices = _obj_topology_indices(Path(obj["path"]), cache)
    return (
        "verified"
        if index in indices[entity_type]
        else "index_not_found_in_obj"
    )


def audit_dataset(input_path: Path, *, limit: int | None = None):
    files = discover_assembly_files(input_path)
    if limit is not None:
        files = files[:limit]
    totals = Counter()
    relation_kinds = Counter()
    relation_types = Counter()
    entity_types = Counter()
    surface_types = Counter()
    per_assembly = []
    global_failures = []
    obj_cache: dict[Path, dict[str, set[int]]] = {}
    for assembly_file in files:
        try:
            data = load_json(assembly_file)
        except Exception as exc:
            global_failures.append(
                f"{assembly_file}:json_load_failed:{type(exc).__name__}:{exc}"
            )
            continue
        visible_parts, part_failures = build_parts(
            data, assembly_file.parent, visible_only=True
        )
        all_parts, all_part_failures = build_parts(
            data, assembly_file.parent, visible_only=False
        )
        relations, relation_failures = extract_relations(
            data, visible_parts
        )
        joints = data.get("joints") or {}
        as_built = data.get("as_built_joints") or {}
        contacts = data.get("contacts") or []
        bodies = data.get("bodies") or {}
        totals["assemblies_loaded"] += 1
        totals["body_definitions"] += len(bodies)
        totals["part_instances_all"] += len(all_parts)
        totals["part_instances_visible"] += len(visible_parts)
        totals["joints"] += len(joints)
        totals["as_built_joints"] += len(as_built)
        totals["contacts"] += len(contacts)
        totals["relations_extracted"] += len(relations)
        totals["parts_with_local_geometry"] += sum(
            part["geometry"]["available"] for part in visible_parts
        )
        totals["parts_without_local_geometry"] += sum(
            not part["geometry"]["available"] for part in visible_parts
        )
        mapped = 0
        face_or_edge_addressable = 0
        contact_face_addressable = 0
        local_geometry_addressable = 0
        obj_verified = 0
        obj_mismatch = 0
        obj_unavailable = 0
        parts_by_id = {
            part["part_id"]: part for part in visible_parts
        }
        for relation in relations:
            _increment(relation_kinds, relation["relation_kind"])
            _increment(relation_types, relation["relation_type"])
            if relation["part_pair_mapping_status"] == "mapped":
                mapped += 1
            for entity in relation["interface_entities"]:
                _increment(entity_types, entity.get("entity_type"))
                _increment(
                    surface_types,
                    (entity.get("local_geometry") or {}).get(
                        "surface_type"
                    ),
                )
                addressable = (
                    entity.get("mapping_status") == "mapped"
                    and entity.get("topology_index_available")
                    and entity.get("entity_type")
                    in {"BRepFace", "BRepEdge", "BRepVertex"}
                )
                if addressable:
                    face_or_edge_addressable += 1
                    if relation["relation_kind"] == "contact":
                        contact_face_addressable += 1
                local_geometry = entity.get("local_geometry") or {}
                if (
                    local_geometry.get("point_on_entity") is not None
                    or local_geometry.get("bounding_box") is not None
                ):
                    local_geometry_addressable += 1
                verification = _verify_entity_in_obj(
                    entity,
                    parts_by_id.get(entity.get("part_id")),
                    obj_cache,
                )
                if verification == "verified":
                    obj_verified += 1
                elif verification == "index_not_found_in_obj":
                    obj_mismatch += 1
                else:
                    obj_unavailable += 1
        totals["relations_mapped_to_part_pair"] += mapped
        totals["relations_not_mapped_to_part_pair"] += (
            len(relations) - mapped
        )
        totals["interface_entities_addressable_by_topology_index"] += (
            face_or_edge_addressable
        )
        totals["contact_entities_addressable_as_brep_face"] += (
            contact_face_addressable
        )
        totals["interface_entities_with_local_geometry"] += (
            local_geometry_addressable
        )
        totals["interface_entities_verified_in_indexed_obj"] += (
            obj_verified
        )
        totals["interface_entity_indices_missing_from_obj"] += (
            obj_mismatch
        )
        totals["interface_entities_not_obj_verifiable"] += (
            obj_unavailable
        )
        failures = sorted(set(
            part_failures
            + all_part_failures
            + relation_failures
        ))
        per_assembly.append({
            "assembly_id": assembly_file.parent.name,
            "source_path": str(assembly_file.resolve()),
            "body_definition_count": len(bodies),
            "part_instance_count": len(visible_parts),
            "joint_count": len(joints),
            "as_built_joint_count": len(as_built),
            "contact_count": len(contacts),
            "relation_count": len(relations),
            "mapped_relation_count": mapped,
            "local_brep_or_mesh_part_count": sum(
                part["geometry"]["available"]
                for part in visible_parts
            ),
            "interface_entities_verified_in_indexed_obj": obj_verified,
            "interface_entity_indices_missing_from_obj": obj_mismatch,
            "failure_reasons": failures,
            "unavailable_fields": sorted({
                field
                for relation in relations
                for entity in relation["interface_entities"]
                for field in entity.get("unavailable_fields", [])
            }),
        })
    assembly_count = len(per_assembly)
    mapped_rate = (
        totals["relations_mapped_to_part_pair"]
        / max(totals["relations_extracted"], 1)
    )
    geometry_rate = (
        totals["parts_with_local_geometry"]
        / max(totals["part_instances_visible"], 1)
    )
    obj_verification_rate = (
        totals["interface_entities_verified_in_indexed_obj"]
        / max(
            totals["interface_entities_verified_in_indexed_obj"]
            + totals["interface_entity_indices_missing_from_obj"],
            1,
        )
    )
    suitable = (
        assembly_count >= 10
        and totals["relations_extracted"] > 0
        and mapped_rate >= 0.9
        and geometry_rate >= 0.9
        and obj_verification_rate >= 0.99
    )
    failure_reasons = list(global_failures)
    if assembly_count < 10:
        failure_reasons.append("fewer_than_10_assemblies_were_audited")
    if totals["relations_extracted"] == 0:
        failure_reasons.append("no_joint_or_contact_relations_extracted")
    if mapped_rate < 0.9:
        failure_reasons.append("part_pair_mapping_rate_below_0.90")
    if geometry_rate < 0.9:
        failure_reasons.append("local_geometry_availability_below_0.90")
    if obj_verification_rate < 0.99:
        failure_reasons.append(
            "indexed_obj_interface_verification_rate_below_0.99"
        )
    report = {
        "schema_version": "1.0.0",
        "dataset": "Autodesk Fusion 360 Gallery Assembly Dataset",
        "audit_status": (
            "success" if assembly_count == len(files) else "partial"
        ),
        "audit_scope": {
            "input_path": str(input_path.resolve()),
            "assembly_files_discovered": len(
                discover_assembly_files(input_path)
            ),
            "assembly_files_audited": assembly_count,
            "sample_is_full_dataset": False,
        },
        "official_dataset_metadata": {
            "assembly_count": 8251,
            "part_count": 154468,
            "compressed_size_gb": 18.8,
            "uncompressed_size_gb": 146.53,
            "license": (
                "Autodesk custom license: non-commercial research only; "
                "full-dataset redistribution prohibited"
            ),
            "units": {"length": "cm", "angle": "radian"},
            "source": (
                "https://github.com/AutodeskAILab/"
                "Fusion360GalleryDataset"
            ),
        },
        "observed_counts": dict(sorted(totals.items())),
        "relation_kind_counts": dict(sorted(relation_kinds.items())),
        "relation_type_counts": dict(sorted(relation_types.items())),
        "interface_entity_type_counts": dict(
            sorted(entity_types.items())
        ),
        "surface_type_counts": dict(sorted(surface_types.items())),
        "mapping_quality": {
            "joint_contact_to_part_pair_rate": mapped_rate,
            "local_geometry_availability_rate": geometry_rate,
            "indexed_obj_interface_verification_rate": (
                obj_verification_rate
            ),
            "face_or_edge_mapping_semantics": (
                "Fusion JSON body/occurrence plus topology index is "
                "structurally mappable to native SMT and indexed OBJ. "
                "Neutral STEP topology numbering is not guaranteed."
            ),
            "contact_quality_warning": (
                "Linkify reports that original Fusion contacts are often "
                "missing, incomplete, or erroneous."
            ),
        },
        "suitability": {
            "verdict": (
                "suitable_as_primary_source_with_contact_caveat"
                if suitable else "not_yet_demonstrated_by_local_sample"
            ),
            "suitable_for_part_pair_classification": suitable,
            "suitable_for_joint_type_supervision": (
                totals["joints"] + totals["as_built_joints"] > 0
            ),
            "suitable_for_interface_supervision": (
                totals[
                    "interface_entities_addressable_by_topology_index"
                ] > 0
            ),
            "reasons": [
                "Occurrence-body part instances and pair relations are explicit.",
                "Joints carry B-Rep face/edge references and joint types.",
                "Contacts carry B-Rep face indices and local geometry fields.",
                "Native SMT and neutral STEP geometry are declared per body.",
                "Original contact labels require filtering or correction.",
            ],
        },
        "failure_reasons": failure_reasons,
        "unavailable_fields": [
            "full_dataset_empirical_counts_not_computed_from_sample",
            "neutral_step_topology_index_equivalence_not_guaranteed",
            "original_contact_completeness_not_guaranteed",
        ],
        "assemblies": per_assembly,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument(
        "--output", default="outputs/fusion360_audit_report.json"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = audit_dataset(Path(args.input), limit=args.limit)
    write_json(Path(args.output), report)
    print(
        f"Fusion assemblies audited: "
        f"{report['audit_scope']['assembly_files_audited']}"
    )
    print(f"Verdict: {report['suitability']['verdict']}")
    return 0 if not report["failure_reasons"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
