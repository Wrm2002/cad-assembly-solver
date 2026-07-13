"""Strictly audit converted Fusion assembly graphs for mixed-pool use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_pair(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    first, second = (str(value[0]), str(value[1]))
    if first == second:
        return None
    return tuple(sorted((first, second)))


def is_matrix_4x4_finite(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
        and all(
            isinstance(number, (int, float)) and math.isfinite(float(number))
            for row in value
            for number in row
        )
    )


def components(
    part_ids: Iterable[str], pairs: Iterable[tuple[str, str]]
) -> list[list[str]]:
    adjacency = {part_id: set() for part_id in part_ids}
    for first, second in pairs:
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = set(adjacency)
    result: list[list[str]] = []
    while remaining:
        root = min(remaining)
        queue = deque([root])
        remaining.remove(root)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        result.append(sorted(component))
    return sorted(result, key=lambda row: (-len(row), row))


def relation_layer(edge: dict[str, Any]) -> str:
    kinds = set(edge.get("relation_kinds") or [])
    if kinds & {"joint", "as_built_joint"}:
        return "designer_joint"
    if "contact" in kinds:
        return "observed_contact"
    return "unknown_positive"


def audit_graph(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    unavailable: set[str] = set()
    try:
        graph = read_json(path)
    except Exception as exc:
        error = f"json_load_failed:{type(exc).__name__}:{exc}"
        return (
            {
                "assembly_id": path.stem,
                "status": "rejected",
                "failure_reasons": [error],
                "unavailable_fields": ["assembly_graph"],
            },
            {},
        )

    assembly_id = str(graph.get("assembly_id") or path.stem)
    if graph.get("schema_version") != "1.0.0":
        errors.append(
            f"unexpected_schema_version:{graph.get('schema_version')}"
        )
    if path.stem != assembly_id:
        errors.append(f"filename_assembly_id_mismatch:{path.stem}")

    parts = graph.get("parts") or []
    part_ids = [str(part.get("part_id")) for part in parts]
    part_id_set = set(part_ids)
    if len(part_ids) < 2:
        errors.append("fewer_than_two_parts")
    duplicate_part_ids = len(part_ids) - len(part_id_set)
    if duplicate_part_ids:
        errors.append(f"duplicate_part_ids:{duplicate_part_ids}")

    missing_geometry = 0
    missing_step = 0
    invalid_transforms = 0
    repeated_body_definitions = Counter()
    geometry_paths: list[str] = []
    step_paths_by_part: dict[str, str] = {}
    for part in parts:
        part_id = str(part.get("part_id"))
        repeated_body_definitions[str(part.get("body_id"))] += 1
        if not is_matrix_4x4_finite(part.get("transform")):
            invalid_transforms += 1
            errors.append(f"invalid_transform:{part_id}")
        geometry = part.get("geometry") or {}
        geometry_path = geometry.get("path")
        if not geometry_path or not Path(geometry_path).is_file():
            missing_geometry += 1
            errors.append(f"preferred_geometry_missing:{part_id}")
        else:
            geometry_paths.append(str(Path(geometry_path).resolve()))
        step_path = None
        for candidate in geometry.get("candidates") or []:
            if (
                str(candidate.get("format")).lower() == "step"
                and candidate.get("path")
                and Path(candidate["path"]).is_file()
            ):
                step_path = str(Path(candidate["path"]).resolve())
                break
        if step_path is None:
            missing_step += 1
            errors.append(f"step_geometry_missing:{part_id}")
        else:
            step_paths_by_part[part_id] = step_path

    positive_edges = graph.get("positive_part_pair_edges") or []
    negative_edges = graph.get("negative_part_pair_edges") or []
    positive_pairs: set[tuple[str, str]] = set()
    negative_pairs: set[tuple[str, str]] = set()
    relation_counts: Counter[str] = Counter()
    layer_counts: Counter[str] = Counter()
    mapped_relation_count = 0
    interface_entity_count = 0
    mapped_interface_entity_count = 0
    local_geometry_entity_count = 0
    duplicate_positive_pairs = 0
    duplicate_negative_pairs = 0

    for edge in positive_edges:
        pair = canonical_pair(edge.get("part_pair"))
        if pair is None:
            errors.append(
                f"invalid_positive_pair:{edge.get('edge_id')}"
            )
            continue
        if any(part_id not in part_id_set for part_id in pair):
            errors.append(
                f"positive_pair_unknown_part:{edge.get('edge_id')}"
            )
        if pair in positive_pairs:
            duplicate_positive_pairs += 1
        positive_pairs.add(pair)
        layer_counts[relation_layer(edge)] += 1
        for relation in edge.get("relations") or []:
            kind = str(relation.get("relation_kind") or "unknown")
            relation_counts[kind] += 1
            if relation.get("part_pair_mapping_status") == "mapped":
                mapped_relation_count += 1
            else:
                errors.append(
                    f"unmapped_relation:{edge.get('edge_id')}:"
                    f"{relation.get('source_relation_id')}"
                )
            for entity in relation.get("interface_entities") or []:
                interface_entity_count += 1
                if entity.get("mapping_status") == "mapped":
                    mapped_interface_entity_count += 1
                else:
                    errors.append(
                        f"unmapped_interface_entity:"
                        f"{edge.get('edge_id')}:"
                        f"{entity.get('entity_id')}"
                    )
                if entity.get("local_geometry"):
                    local_geometry_entity_count += 1

    for edge in negative_edges:
        pair = canonical_pair(edge.get("part_pair"))
        if pair is None:
            errors.append(
                f"invalid_negative_pair:{edge.get('edge_id')}"
            )
            continue
        if any(part_id not in part_id_set for part_id in pair):
            errors.append(
                f"negative_pair_unknown_part:{edge.get('edge_id')}"
            )
        if pair in negative_pairs:
            duplicate_negative_pairs += 1
        negative_pairs.add(pair)

    if duplicate_positive_pairs:
        errors.append(
            f"duplicate_positive_pairs:{duplicate_positive_pairs}"
        )
    if duplicate_negative_pairs:
        errors.append(
            f"duplicate_negative_pairs:{duplicate_negative_pairs}"
        )
    overlap = positive_pairs & negative_pairs
    if overlap:
        errors.append(f"positive_negative_pair_overlap:{len(overlap)}")
    expected_pairs = set(combinations(sorted(part_id_set), 2))
    observed_pairs = positive_pairs | negative_pairs
    missing_pair_partition = expected_pairs - observed_pairs
    extra_pair_partition = observed_pairs - expected_pairs
    if missing_pair_partition:
        errors.append(
            f"pair_partition_missing:{len(missing_pair_partition)}"
        )
    if extra_pair_partition:
        errors.append(
            f"pair_partition_extra:{len(extra_pair_partition)}"
        )
    if not positive_pairs:
        errors.append("no_observed_positive_pair")

    source_path_value = graph.get("source_record_path")
    source_path = Path(source_path_value) if source_path_value else None
    source_hash_matches = None
    if source_path and source_path.is_file():
        source_hash_matches = (
            sha256_file(source_path) == graph.get("source_record_sha256")
        )
        if not source_hash_matches:
            errors.append("source_record_hash_mismatch")
    else:
        warnings.append("source_record_not_available_for_rehash")
        unavailable.add("source_record_rehash")

    joint_pairs = {
        canonical_pair(edge.get("part_pair"))
        for edge in positive_edges
        if relation_layer(edge) == "designer_joint"
    }
    joint_pairs.discard(None)
    all_components = components(part_ids, positive_pairs)
    joint_components = components(part_ids, joint_pairs)
    repeated_bodies = {
        body_id: count
        for body_id, count in repeated_body_definitions.items()
        if body_id and body_id != "None" and count > 1
    }
    if not joint_pairs:
        warnings.append("no_designer_joint_edges_contact_only_truth")
        unavailable.add("designer_joint_supervision")
    unavailable.add("functional_semantic_group_truth")
    unavailable.add("proof_that_non_edge_is_mechanically_invalid")

    status = "usable" if not errors else "rejected"
    record = {
        "assembly_id": assembly_id,
        "status": status,
        "source_graph_path": str(path.resolve()),
        "source_graph_sha256": sha256_file(path),
        "source_record_path": (
            str(source_path.resolve()) if source_path else None
        ),
        "source_record_hash_matches": source_hash_matches,
        "part_count": len(parts),
        "unique_body_definition_count": len(
            repeated_body_definitions
        ),
        "repeated_body_definitions": repeated_bodies,
        "positive_pair_count": len(positive_pairs),
        "negative_pair_count": len(negative_pairs),
        "expected_complete_pair_count": len(expected_pairs),
        "pair_partition_complete": (
            not missing_pair_partition
            and not extra_pair_partition
            and not overlap
        ),
        "designer_joint_pair_count": layer_counts["designer_joint"],
        "contact_only_pair_count": layer_counts["observed_contact"],
        "unknown_positive_pair_count": layer_counts["unknown_positive"],
        "relation_counts": dict(sorted(relation_counts.items())),
        "relation_count": sum(relation_counts.values()),
        "mapped_relation_count": mapped_relation_count,
        "interface_entity_count": interface_entity_count,
        "mapped_interface_entity_count": mapped_interface_entity_count,
        "local_geometry_entity_count": local_geometry_entity_count,
        "preferred_geometry_missing_count": missing_geometry,
        "step_geometry_missing_count": missing_step,
        "invalid_transform_count": invalid_transforms,
        "positive_component_sizes": [
            len(component) for component in all_components
        ],
        "joint_component_sizes": [
            len(component) for component in joint_components
        ],
        "usable_for_mixed_pool_provenance": (
            status == "usable"
            and len(parts) >= 2
            and bool(positive_pairs)
            and missing_step == 0
        ),
        "usable_for_direct_interface_supervision": (
            status == "usable"
            and mapped_relation_count == sum(relation_counts.values())
            and mapped_interface_entity_count == interface_entity_count
        ),
        "functional_semantic_truth_available": False,
        "warnings": sorted(set(warnings)),
        "failure_reasons": sorted(set(errors)),
        "unavailable_fields": sorted(unavailable),
    }
    internal = {
        "graph": graph,
        "step_paths_by_part": step_paths_by_part,
        "positive_pairs": positive_pairs,
        "joint_pairs": joint_pairs,
    }
    return record, internal


def write_inventory_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "assembly_id",
        "status",
        "part_count",
        "unique_body_definition_count",
        "positive_pair_count",
        "designer_joint_pair_count",
        "contact_only_pair_count",
        "negative_pair_count",
        "relation_count",
        "mapped_relation_count",
        "interface_entity_count",
        "mapped_interface_entity_count",
        "step_geometry_missing_count",
        "pair_partition_complete",
        "usable_for_mixed_pool_provenance",
        "usable_for_direct_interface_supervision",
        "functional_semantic_truth_available",
        "source_graph_sha256",
        "failure_reasons",
        "warnings",
        "unavailable_fields",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(field) for field in fields}
            for field in (
                "failure_reasons",
                "warnings",
                "unavailable_fields",
            ):
                output[field] = "|".join(output[field] or [])
            writer.writerow(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase123_real_assembly_dataset"),
    )
    args = parser.parse_args()
    graph_dir = args.graph_dir.resolve()
    output_dir = args.output_dir.resolve()
    graph_files = sorted(
        path
        for path in graph_dir.glob("*.json")
        if path.name != "conversion_manifest.json"
    )
    rows = []
    assembly_records = []
    seen_assembly_ids: set[str] = set()
    cross_file_failures: list[str] = []
    for graph_file in graph_files:
        row, _ = audit_graph(graph_file)
        rows.append(row)
        assembly_id = str(row["assembly_id"])
        if assembly_id in seen_assembly_ids:
            cross_file_failures.append(
                f"duplicate_assembly_id:{assembly_id}"
            )
        seen_assembly_ids.add(assembly_id)
        if row["status"] == "usable":
            assembly_records.append(
                {
                    "assembly_id": assembly_id,
                    "graph_path": row["source_graph_path"],
                    "graph_sha256": row["source_graph_sha256"],
                    "part_count": row["part_count"],
                    "positive_pair_count": row["positive_pair_count"],
                    "designer_joint_pair_count": row[
                        "designer_joint_pair_count"
                    ],
                    "contact_only_pair_count": row[
                        "contact_only_pair_count"
                    ],
                    "step_geometry_missing_count": row[
                        "step_geometry_missing_count"
                    ],
                    "warnings": row["warnings"],
                    "failure_reasons": row["failure_reasons"],
                    "unavailable_fields": row["unavailable_fields"],
                }
            )

    totals = {
        key: sum(int(row.get(key) or 0) for row in rows)
        for key in (
            "part_count",
            "positive_pair_count",
            "negative_pair_count",
            "designer_joint_pair_count",
            "contact_only_pair_count",
            "relation_count",
            "mapped_relation_count",
            "interface_entity_count",
            "mapped_interface_entity_count",
            "step_geometry_missing_count",
        )
    }
    rejected = [row for row in rows if row["status"] != "usable"]
    dataset_manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "fusion360_real_assembly_10_v1",
        "source_dataset": "Autodesk Fusion 360 Gallery Assembly Dataset",
        "graph_schema": "assembly_graph_schema.md",
        "assembly_count": len(assembly_records),
        "assemblies": assembly_records,
        "truth_layers": {
            "group_provenance": (
                "Visible part occurrences from one source assembly_id."
            ),
            "high_confidence_direct_edge": (
                "Designer joint or as-built joint mapped to a part pair."
            ),
            "observed_direct_edge": (
                "Recorded contact mapped to a part pair; completeness and "
                "semantic correctness are not guaranteed."
            ),
            "non_edge": (
                "No recorded joint/contact in this source assembly; this is "
                "not proof of mechanical or semantic incompatibility."
            ),
        },
        "split_policy": (
            "All train/validation/test and mixed-pool splits must be made by "
            "assembly_id before part-pair sampling."
        ),
        "usable_for_mixed_pool_construction": (
            len(assembly_records) >= 10
            and not rejected
            and not cross_file_failures
        ),
        "functional_semantic_truth_available": False,
        "failure_reasons": cross_file_failures
        + [
            f"rejected_assembly:{row['assembly_id']}"
            for row in rejected
        ],
        "unavailable_fields": [
            "part_role",
            "assembly_family",
            "functional_relation",
            "proof_that_cross_assembly_pairs_cannot_function_together",
        ],
    }
    report = {
        "schema_version": "1.0.0",
        "purpose": (
            "Strict assembly/occurrence/edge/interface quality audit before "
            "mixed-pool construction."
        ),
        "source_graph_dir": str(graph_dir),
        "graph_file_count": len(graph_files),
        "usable_assembly_count": len(assembly_records),
        "rejected_assembly_count": len(rejected),
        "totals": totals,
        "all_pair_partitions_complete": all(
            bool(row.get("pair_partition_complete")) for row in rows
        ),
        "all_step_geometry_available": totals[
            "step_geometry_missing_count"
        ]
        == 0,
        "all_relations_mapped": (
            totals["relation_count"] == totals["mapped_relation_count"]
        ),
        "all_interface_entities_mapped": (
            totals["interface_entity_count"]
            == totals["mapped_interface_entity_count"]
        ),
        "truth_policy": dataset_manifest["truth_layers"],
        "assemblies": rows,
        "failure_reasons": dataset_manifest["failure_reasons"],
        "unavailable_fields": dataset_manifest["unavailable_fields"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "assembly_dataset_manifest.json", dataset_manifest)
    write_json(output_dir / "assembly_graph_quality_report.json", report)
    write_inventory_csv(output_dir / "assembly_graph_inventory.csv", rows)
    print(
        f"Assembly graphs usable: {len(assembly_records)}/{len(graph_files)}; "
        f"parts={totals['part_count']}, "
        f"joint-pairs={totals['designer_joint_pair_count']}, "
        f"contact-only-pairs={totals['contact_only_pair_count']}"
    )
    return (
        0
        if dataset_manifest["usable_for_mixed_pool_construction"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
