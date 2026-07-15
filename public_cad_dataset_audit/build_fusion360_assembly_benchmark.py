"""Build assembly-level and pair-level Fusion360 deterministic assembly benchmark.

This is Step 1 data engineering for deterministic CAD assembly relation
recovery.  The script:

* reads Fusion360 Assembly Dataset `assembly.json` files only;
* does not read SolidWorks exam labels or human_labels.json;
* emits assembly-level samples plus derived same-assembly pair samples;
* labels negatives as `same_assembly_non_edge`, never as physically impossible;
* splits strictly by `assembly_id`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from fusion360_common import convert_assembly, discover_assembly_files, write_json
from map_fusion360_to_solidworks_exam import (
    COMMON_LABELS,
    map_fusion_edge_to_common_labels,
    step_candidate,
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_split(assembly_id: str) -> str:
    bucket = int(hashlib.sha256(assembly_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "dev"
    return "train"


def mat_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
        for i in range(4)
    ]


def rigid_inverse(m: list[list[float]] | None) -> list[list[float]] | None:
    if not m or len(m) != 4 or any(len(row) != 4 for row in m):
        return None
    r_t = [[float(m[j][i]) for j in range(3)] for i in range(3)]
    t = [float(m[i][3]) for i in range(3)]
    inv_t = [-sum(r_t[i][j] * t[j] for j in range(3)) for i in range(3)]
    return [r_t[i] + [inv_t[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def pairwise_transform(
    transform_world_from_a: list[list[float]] | None,
    transform_world_from_b: list[list[float]] | None,
) -> dict[str, Any]:
    inv_a = rigid_inverse(transform_world_from_a)
    if inv_a is None or transform_world_from_b is None:
        return {
            "available": False,
            "transform_part_b_from_part_a": None,
            "failure_reasons": ["part_transform_unavailable"],
        }
    return {
        "available": True,
        "transform_part_b_from_part_a": matmul(inv_a, transform_world_from_b),
        "failure_reasons": [],
    }


def compact_part(part: dict[str, Any]) -> dict[str, Any]:
    step = step_candidate(part)
    return {
        "part_id": part.get("part_id"),
        "body_id": part.get("body_id"),
        "occurrence_id": part.get("occurrence_id"),
        "part_name": part.get("part_name"),
        "body_name": part.get("body_name"),
        "visible": part.get("visible"),
        "transform_world_from_part": part.get("transform") or mat_identity(),
        "geometry": {
            "preferred_path": (part.get("geometry") or {}).get("path"),
            "preferred_format": (part.get("geometry") or {}).get("format"),
            "available": (part.get("geometry") or {}).get("available"),
            "step_path": step.get("path"),
            "step_exists": step.get("exists"),
            "candidates": (part.get("geometry") or {}).get("candidates") or [],
        },
    }


def interface_entities(edge: dict[str, Any]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for relation in edge.get("relations") or []:
        for entity in relation.get("interface_entities") or []:
            entities.append(
                {
                    "part_id": entity.get("part_id"),
                    "entity_type": entity.get("entity_type"),
                    "entity_id": entity.get("entity_id"),
                    "body_id": entity.get("body_id"),
                    "occurrence_id": entity.get("occurrence_id"),
                    "topology_index": entity.get("topology_index"),
                    "topology_index_available": entity.get("topology_index_available"),
                    "geometry_file_available": entity.get("geometry_file_available"),
                    "local_geometry": entity.get("local_geometry"),
                    "mapping_status": entity.get("mapping_status"),
                    "failure_reason": entity.get("failure_reason"),
                }
            )
    return entities


def part_name_evidence(pair: list[str], parts_by_id: dict[str, dict[str, Any]]) -> list[str]:
    evidence = []
    for part_id in pair:
        part = parts_by_id[part_id]
        evidence.append(
            " ".join(
                str(part.get(key) or "") for key in ("part_name", "body_name")
            ).strip()
        )
    return evidence


def positive_pair_row(
    *,
    assembly_id: str,
    split: str,
    edge: dict[str, Any],
    parts_by_id: dict[str, dict[str, Any]],
    graph_path: Path,
) -> dict[str, Any]:
    pair = edge.get("part_pair") or []
    names = part_name_evidence(pair, parts_by_id)
    labels, primary, confidence, reasons, unavailable = map_fusion_edge_to_common_labels(
        edge,
        names,
    )
    source_relation_types = [str(value) for value in edge.get("relation_types") or []]
    if "PlanarJointType" in source_relation_types:
        if "planar_align" not in labels:
            labels.append("planar_align")
        if primary in {None, "planar_mate"}:
            primary = "planar_align"
        reasons.append(
            "Fusion PlanarJointType maps to planar_align; planar face evidence may also support planar_mate."
        )
        confidence = "medium" if confidence == "unmapped" else confidence
        labels = [label for label in COMMON_LABELS if label in set(labels)]
    step_paths = [step_candidate(parts_by_id[part_id]) for part_id in pair]
    transform = pairwise_transform(
        parts_by_id[pair[0]].get("transform"),
        parts_by_id[pair[1]].get("transform"),
    )
    return {
        "schema_version": "1.0.0",
        "sample_id": f"fusion360_assembly:{assembly_id}:{edge.get('edge_id')}",
        "assembly_id": assembly_id,
        "split": split,
        "source_dataset": "fusion360_gallery_assembly",
        "source_graph_path": str(graph_path.resolve()),
        "ground_truth_scope": "development_only_same_assembly_direct_edge",
        "do_not_use_for_solidworks_exam": True,
        "pair_id": edge.get("edge_id"),
        "part_pair": pair,
        "part_names": [parts_by_id[part_id].get("part_name") for part_id in pair],
        "part_name_evidence": names,
        "solidworks_compatible_geometry_paths": [row.get("path") for row in step_paths],
        "solidworks_compatible_geometry_exists": all(row.get("exists") for row in step_paths),
        "direct_edge": True,
        "direct_connection": True,
        "negative_type": None,
        "source_relation_types": source_relation_types,
        "source_relation_kinds": edge.get("relation_kinds") or [],
        "joint_contact_records": edge.get("relations") or [],
        "interface_entities": interface_entities(edge),
        "mapped_relation_types": labels,
        "primary_relation_type": primary,
        "mapping_confidence": confidence,
        "weak_label": confidence == "low" or "pocket_mate_human_audit" in unavailable,
        "mapping_reasons": reasons,
        "pairwise_transform": transform,
        "failure_reasons": edge.get("failure_reasons") or [],
        "unavailable_fields": sorted(set((edge.get("unavailable_fields") or []) + unavailable)),
    }


def non_edge_pair_row(
    *,
    assembly_id: str,
    split: str,
    edge: dict[str, Any],
    parts_by_id: dict[str, dict[str, Any]],
    graph_path: Path,
) -> dict[str, Any]:
    pair = edge.get("part_pair") or []
    step_paths = [step_candidate(parts_by_id[part_id]) for part_id in pair]
    transform = pairwise_transform(
        parts_by_id[pair[0]].get("transform"),
        parts_by_id[pair[1]].get("transform"),
    )
    return {
        "schema_version": "1.0.0",
        "sample_id": f"fusion360_assembly:{assembly_id}:{edge.get('edge_id')}",
        "assembly_id": assembly_id,
        "split": split,
        "source_dataset": "fusion360_gallery_assembly",
        "source_graph_path": str(graph_path.resolve()),
        "ground_truth_scope": "development_only_same_assembly_non_edge",
        "do_not_use_for_solidworks_exam": True,
        "pair_id": edge.get("edge_id"),
        "part_pair": pair,
        "part_names": [parts_by_id[part_id].get("part_name") for part_id in pair],
        "part_name_evidence": part_name_evidence(pair, parts_by_id),
        "solidworks_compatible_geometry_paths": [row.get("path") for row in step_paths],
        "solidworks_compatible_geometry_exists": all(row.get("exists") for row in step_paths),
        "direct_edge": False,
        "direct_connection": False,
        "negative_type": "same_assembly_non_edge",
        "negative_definition": (
            "Both parts are in the same Fusion360 assembly, but no direct "
            "joint/as-built joint/contact edge is recorded between them. "
            "This does not mean physically impossible."
        ),
        "source_relation_types": ["same_assembly_non_edge"],
        "source_relation_kinds": [],
        "joint_contact_records": [],
        "interface_entities": [],
        "mapped_relation_types": [],
        "primary_relation_type": None,
        "mapping_confidence": "same_assembly_non_edge",
        "weak_label": False,
        "mapping_reasons": [
            "No direct Fusion360 joint/as-built joint/contact record exists for this same-assembly pair."
        ],
        "pairwise_transform": transform,
        "failure_reasons": edge.get("failure_reasons") or [],
        "unavailable_fields": sorted(
            set((edge.get("unavailable_fields") or []) + ["direct_interface_label"])
        ),
    }


def build_benchmarks(
    input_roots: list[Path],
    *,
    limit: int | None,
    maximum_parts: int,
    maximum_pairs_per_assembly: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assembly_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    graph_cache_dir = Path("public_cad_dataset_audit/outputs/fusion360_assembly_benchmark/_converted_graphs").resolve()
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for input_root in input_roots:
        files.extend(discover_assembly_files(input_root))
    files = sorted(set(files))
    converted_count = 0
    seen_assembly_ids: set[str] = set()
    for assembly_file in files:
        if limit is not None and converted_count >= limit:
            break
        try:
            graph = convert_assembly(assembly_file)
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {
                    "source_path": str(assembly_file.resolve()),
                    "failure_reasons": [f"conversion_exception:{type(exc).__name__}:{exc}"],
                }
            )
            continue

        quality = graph.get("quality") or {}
        assembly_id = graph.get("assembly_id") or assembly_file.parent.name
        if assembly_id in seen_assembly_ids:
            rejected.append(
                {
                    "assembly_id": assembly_id,
                    "source_path": str(assembly_file.resolve()),
                    "failure_reasons": ["duplicate_assembly_id_skipped"],
                }
            )
            continue
        seen_assembly_ids.add(assembly_id)
        parts = graph.get("parts") or []
        positive_edges = graph.get("positive_part_pair_edges") or []
        non_edges = graph.get("negative_part_pair_edges") or []
        pair_count = len(positive_edges) + len(non_edges)
        if len(parts) < 2 or not positive_edges:
            rejected.append(
                {
                    "assembly_id": assembly_id,
                    "source_path": str(assembly_file.resolve()),
                    "failure_reasons": ["fewer_than_two_parts_or_no_positive_edge"],
                    "part_count": len(parts),
                    "positive_edge_count": len(positive_edges),
                }
            )
            continue
        if len(parts) > maximum_parts:
            rejected.append(
                {
                    "assembly_id": assembly_id,
                    "source_path": str(assembly_file.resolve()),
                    "failure_reasons": [f"part_count_exceeds_limit:{len(parts)}"],
                    "part_count": len(parts),
                }
            )
            continue
        if pair_count > maximum_pairs_per_assembly:
            rejected.append(
                {
                    "assembly_id": assembly_id,
                    "source_path": str(assembly_file.resolve()),
                    "failure_reasons": [f"pair_count_exceeds_limit:{pair_count}"],
                    "part_count": len(parts),
                    "pair_count": pair_count,
                }
            )
            continue

        split = stable_split(assembly_id)
        graph_path = graph_cache_dir / f"{assembly_id}.json"
        write_json(graph_path, graph)
        parts_by_id = {part["part_id"]: part for part in parts}
        compact_parts = [compact_part(part) for part in parts]
        edge_failures: list[str] = []

        def edge_is_usable(edge: dict[str, Any], kind: str) -> bool:
            pair = edge.get("part_pair") or []
            if len(pair) != 2:
                edge_failures.append(f"{kind}:{edge.get('edge_id')}:bad_pair_arity")
                return False
            missing = [part_id for part_id in pair if part_id not in parts_by_id]
            if missing:
                edge_failures.append(
                    f"{kind}:{edge.get('edge_id')}:part_id_not_in_visible_parts:{','.join(missing)}"
                )
                return False
            return True

        usable_positive_edges = [
            edge for edge in positive_edges if edge_is_usable(edge, "positive_edge")
        ]
        usable_non_edges = [
            edge for edge in non_edges if edge_is_usable(edge, "same_assembly_non_edge")
        ]
        if not usable_positive_edges:
            rejected.append(
                {
                    "assembly_id": assembly_id,
                    "source_path": str(assembly_file.resolve()),
                    "failure_reasons": edge_failures + ["no_usable_positive_edges_after_filter"],
                    "part_count": len(parts),
                    "positive_edge_count": len(positive_edges),
                }
            )
            continue
        positive_pair_rows = [
            positive_pair_row(
                assembly_id=assembly_id,
                split=split,
                edge=edge,
                parts_by_id=parts_by_id,
                graph_path=graph_path,
            )
            for edge in usable_positive_edges
        ]
        non_edge_pair_rows = [
            non_edge_pair_row(
                assembly_id=assembly_id,
                split=split,
                edge=edge,
                parts_by_id=parts_by_id,
                graph_path=graph_path,
            )
            for edge in usable_non_edges
        ]
        all_pair_rows = positive_pair_rows + non_edge_pair_rows
        pair_rows.extend(all_pair_rows)
        label_counts = Counter(
            label
            for row in positive_pair_rows
            for label in row.get("mapped_relation_types", [])
        )
        assembly_rows.append(
            {
                "schema_version": "1.0.0",
                "assembly_id": assembly_id,
                "split": split,
                "source_dataset": "fusion360_gallery_assembly",
                "source_record_path": str(assembly_file.resolve()),
                "source_graph_path": str(graph_path.resolve()),
                "ground_truth_scope": "development_only_deterministic_assembly_graph",
                "do_not_use_for_solidworks_exam": True,
                "units": graph.get("units"),
                "parts": compact_parts,
                "all_candidate_pairs": [
                    {
                        "sample_id": row["sample_id"],
                        "pair_id": row["pair_id"],
                        "part_pair": row["part_pair"],
                        "direct_edge": row["direct_edge"],
                        "negative_type": row["negative_type"],
                        "mapped_relation_types": row["mapped_relation_types"],
                        "primary_relation_type": row["primary_relation_type"],
                        "pairwise_transform": row["pairwise_transform"],
                    }
                    for row in all_pair_rows
                ],
                "positive_edges": [
                    {
                        "sample_id": row["sample_id"],
                        "pair_id": row["pair_id"],
                        "part_pair": row["part_pair"],
                        "mapped_relation_types": row["mapped_relation_types"],
                        "primary_relation_type": row["primary_relation_type"],
                        "interface_entities": row["interface_entities"],
                        "pairwise_transform": row["pairwise_transform"],
                        "source_relation_types": row["source_relation_types"],
                        "source_relation_kinds": row["source_relation_kinds"],
                        "mapping_confidence": row["mapping_confidence"],
                        "weak_label": row["weak_label"],
                        "failure_reasons": row["failure_reasons"],
                        "unavailable_fields": row["unavailable_fields"],
                    }
                    for row in positive_pair_rows
                ],
                "same_assembly_non_edges": [
                    {
                        "sample_id": row["sample_id"],
                        "pair_id": row["pair_id"],
                        "part_pair": row["part_pair"],
                        "negative_type": "same_assembly_non_edge",
                        "pairwise_transform": row["pairwise_transform"],
                        "negative_definition": row["negative_definition"],
                    }
                    for row in non_edge_pair_rows
                ],
                "quality": {
                    "part_count": len(compact_parts),
                    "candidate_pair_count": len(all_pair_rows),
                    "positive_edge_count": len(positive_pair_rows),
                    "same_assembly_non_edge_count": len(non_edge_pair_rows),
                    "mapped_label_counts": dict(sorted(label_counts.items())),
                    "source_quality": quality,
                },
                "failure_reasons": (graph.get("failure_reasons") or []) + edge_failures,
                "unavailable_fields": graph.get("unavailable_fields") or [],
            }
        )
        converted_count += 1

    repair_rare_label_splits(assembly_rows, pair_rows)
    summary = summarize(assembly_rows, pair_rows, rejected, len(files))
    return assembly_rows, pair_rows, rejected, summary


def summarize(
    assembly_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    discovered_assembly_json_count: int,
) -> dict[str, Any]:
    split_assembly_counts: dict[str, set[str]] = defaultdict(set)
    split_pair_counts: Counter[str] = Counter()
    split_positive_counts: Counter[str] = Counter()
    split_non_edge_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assembly_rows:
        split_assembly_counts[row["split"]].add(row["assembly_id"])
    for row in pair_rows:
        split = row["split"]
        split_pair_counts[split] += 1
        if row["direct_edge"]:
            split_positive_counts[split] += 1
            for label in row.get("mapped_relation_types", []):
                label_counts[label] += 1
                split_label_counts[split][label] += 1
        else:
            split_non_edge_counts[split] += 1
    split_repairs = []
    for row in assembly_rows:
        split_repairs.extend(row.get("split_repair_reasons") or [])
    return {
        "schema_version": "1.0.0",
        "source_dataset": "fusion360_gallery_assembly",
        "discovered_assembly_json_count": discovered_assembly_json_count,
        "assembly_count": len(assembly_rows),
        "pair_sample_count": len(pair_rows),
        "positive_edge_count": sum(1 for row in pair_rows if row["direct_edge"]),
        "same_assembly_non_edge_count": sum(1 for row in pair_rows if not row["direct_edge"]),
        "mapped_label_counts": dict(sorted(label_counts.items())),
        "missing_common_labels": [
            label for label in COMMON_LABELS if label_counts.get(label, 0) == 0
        ],
        "pocket_mate_candidate_count": sum(
            1 for row in pair_rows if "pocket_mate" in row.get("mapped_relation_types", [])
        ),
        "weak_label_count": sum(1 for row in pair_rows if row.get("weak_label")),
        "geometry_step_missing_pair_count": sum(
            1 for row in pair_rows if not row.get("solidworks_compatible_geometry_exists")
        ),
        "split": {
            split: {
                "assembly_count": len(split_assembly_counts[split]),
                "pair_count": split_pair_counts[split],
                "positive_edge_count": split_positive_counts[split],
                "same_assembly_non_edge_count": split_non_edge_counts[split],
                "label_counts": dict(sorted(split_label_counts[split].items())),
            }
            for split in ("train", "dev", "test")
        },
        "rejected_count": len(rejected),
        "split_repairs": split_repairs,
        "rejected_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rejected
                    for reason in row.get("failure_reasons", [])
                ).items()
            )
        ),
        "negative_policy": (
            "Negative pair samples are same_assembly_non_edge only: both parts "
            "belong to the same assembly, but no direct joint/as-built joint/contact "
            "edge is recorded. They are not labelled physically impossible."
        ),
        "solidworks_exam_isolation": {
            "reads_human_labels": False,
            "reads_solidworks_exam_cases": False,
            "uses_case_specific_rules": False,
        },
    }


def write_label_mapping_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Label mapping report",
        "",
        "## Scope",
        "",
        "This report describes Fusion360 Assembly Dataset conversion rules. SolidWorks human labels are not read.",
        "",
        "## Relation schema",
        "",
        "- `coaxial`: revolute/cylindrical joint types or cylindrical/conical/circular interface evidence.",
        "- `planar_mate`: planar B-Rep face contact/joint evidence.",
        "- `clearance`: slider/pin/cylindrical insertion or sliding freedom.",
        "- `pocket_mate`: mined slot/cavity/rail/channel style candidates; weak unless strongly supported.",
        "- `planar_align`: reserved for planar alignment without mandatory contact; currently sparse in Assembly Dataset mapping.",
        "",
        "## Negative policy",
        "",
        summary["negative_policy"],
        "",
        "## Counts",
        "",
        f"- assembly_count: {summary['assembly_count']}",
        f"- pair_sample_count: {summary['pair_sample_count']}",
        f"- positive_edge_count: {summary['positive_edge_count']}",
        f"- same_assembly_non_edge_count: {summary['same_assembly_non_edge_count']}",
        f"- mapped_label_counts: `{summary['mapped_label_counts']}`",
        f"- missing_common_labels: `{summary['missing_common_labels']}`",
        f"- pocket_mate_candidate_count: {summary['pocket_mate_candidate_count']}",
        f"- weak_label_count: {summary['weak_label_count']}",
        "",
        "## Weak label policy",
        "",
        "`pocket_mate` candidates mined from names/contact context are marked weak and exported for human audit. They should not be treated as high-confidence functional truth before review.",
        "",
        "## Known limitations",
        "",
        "- Fusion360 contact/joint labels may be incomplete.",
        "- `same_assembly_non_edge` means no recorded direct edge, not physical impossibility.",
        "- Pairwise transforms are derived from Fusion360 occurrence transforms when available.",
        "- Full SolidWorks exam isolation is preserved: no `human_labels.json` is read here.",
        "",
    ]
    (output_dir / "label_mapping_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_split_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Split report",
        "",
        "Initial split key: stable hash of `assembly_id`. Rare-label repair may move whole assemblies between splits. No assembly appears in more than one split.",
        "",
        "| split | assemblies | pairs | positive_edges | same_assembly_non_edges | labels |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for split in ("train", "dev", "test"):
        row = summary["split"][split]
        lines.append(
            f"| {split} | {row['assembly_count']} | {row['pair_count']} | "
            f"{row['positive_edge_count']} | {row['same_assembly_non_edge_count']} | "
            f"`{row['label_counts']}` |"
        )
    lines += [
        "",
        "## Rejections",
        "",
        f"- rejected_count: {summary['rejected_count']}",
        f"- rejected_reason_counts: `{summary['rejected_reason_counts']}`",
        f"- split_repairs: `{summary.get('split_repairs', [])}`",
        "",
        "## Isolation",
        "",
        "- SolidWorks `human_labels.json` is not read.",
        "- SolidWorks case IDs and filenames are not used as answer hints.",
        "- Threshold tuning should use train/dev only; test is evaluation-only.",
        "",
    ]
    (output_dir / "split_report.md").write_text("\n".join(lines), encoding="utf-8")


def repair_rare_label_splits(
    assembly_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    """Move whole assemblies so rare labels appear in dev/test when possible.

    This preserves strict assembly_id isolation.  It is not threshold tuning; it
    only prevents a label from being completely absent from dev/test when there
    are enough assemblies containing that label.
    """

    pair_rows_by_assembly: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_rows_by_assembly[row["assembly_id"]].append(row)

    assembly_by_id = {row["assembly_id"]: row for row in assembly_rows}
    label_to_assemblies: dict[str, set[str]] = defaultdict(set)
    label_to_split_assemblies: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in pair_rows:
        if not row.get("direct_edge"):
            continue
        for label in row.get("mapped_relation_types") or []:
            label_to_assemblies[label].add(row["assembly_id"])
            label_to_split_assemblies[label][row["split"]].add(row["assembly_id"])

    planned_moves: list[tuple[str, str, str, str]] = []
    for label in COMMON_LABELS:
        assemblies = sorted(label_to_assemblies.get(label) or [])
        if len(assemblies) < 3:
            continue
        for target_split in ("dev", "test"):
            if label_to_split_assemblies[label].get(target_split):
                continue
            candidates = [
                assembly_id
                for assembly_id in assemblies
                if assembly_by_id[assembly_id]["split"] == "train"
                and assembly_id not in {move[0] for move in planned_moves}
            ]
            if not candidates:
                continue
            chosen = sorted(
                candidates,
                key=lambda assembly_id: (
                    len(pair_rows_by_assembly[assembly_id]),
                    assembly_id,
                ),
            )[0]
            planned_moves.append(
                (
                    chosen,
                    assembly_by_id[chosen]["split"],
                    target_split,
                    f"ensure_{label}_support_in_{target_split}",
                )
            )
            label_to_split_assemblies[label][target_split].add(chosen)

    for assembly_id, old_split, new_split, reason in planned_moves:
        assembly = assembly_by_id[assembly_id]
        assembly["split"] = new_split
        assembly.setdefault("split_repair_reasons", []).append(
            {
                "from": old_split,
                "to": new_split,
                "reason": reason,
            }
        )
        for pair in pair_rows_by_assembly[assembly_id]:
            pair["split"] = new_split
            pair.setdefault("split_repair_reasons", []).append(
                {
                    "from": old_split,
                    "to": new_split,
                    "reason": reason,
                }
            )


def write_pocket_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "assembly_id",
        "split",
        "part_pair",
        "part_name_evidence",
        "source_relation_types",
        "source_relation_kinds",
        "mapped_relation_types",
        "mapping_confidence",
        "weak_label",
        "mapping_reasons",
        "source_graph_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field)
                    for field in fields
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        nargs="+",
        default=[Path(r"D:\Model_match_public_data\fusion360")],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("public_cad_dataset_audit/outputs/fusion360_assembly_benchmark"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--maximum-parts", type=int, default=300)
    parser.add_argument("--maximum-pairs-per-assembly", type=int, default=50000)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assembly_rows, pair_rows, rejected, summary = build_benchmarks(
        [path.resolve() for path in args.input_root],
        limit=args.limit,
        maximum_parts=args.maximum_parts,
        maximum_pairs_per_assembly=args.maximum_pairs_per_assembly,
    )
    split_pair_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        split_pair_rows[row["split"]].append(row)

    write_jsonl(output_dir / "fusion360_assembly_benchmark.jsonl", assembly_rows)
    write_jsonl(output_dir / "fusion360_pair_relation_benchmark.jsonl", pair_rows)
    for split in ("train", "dev", "test"):
        write_jsonl(output_dir / f"fusion360_{split}.jsonl", split_pair_rows[split])
    pocket_rows = [
        row for row in pair_rows if "pocket_mate" in row.get("mapped_relation_types", [])
    ]
    write_json(output_dir / "fusion360_pocket_mate_candidates.json", pocket_rows)
    write_pocket_csv(output_dir / "fusion360_pocket_mate_candidates.csv", pocket_rows)
    write_json(output_dir / "fusion360_assembly_benchmark_summary.json", summary)
    write_json(output_dir / "rejected_assemblies.json", rejected)
    write_label_mapping_report(output_dir, summary)
    write_split_report(output_dir, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Non-zero only when there is no usable benchmark at all.
    return 0 if assembly_rows and pair_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
