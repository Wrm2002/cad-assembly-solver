"""Build reproducible anonymized mixed pools from real Fusion assemblies.

The source assembly id is treated as provenance truth, not as proof of
universal functional incompatibility.  Direct joint/contact edges and
same-group membership are deliberately kept as separate labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
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


def canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def edge_pair(edge: dict[str, Any]) -> tuple[str, str] | None:
    pair = edge.get("part_pair")
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    if pair[0] == pair[1]:
        return None
    return canonical_pair(pair[0], pair[1])


def relation_layer(edge: dict[str, Any]) -> str:
    kinds = set(edge.get("relation_kinds") or [])
    if kinds & {"joint", "as_built_joint"}:
        return "designer_joint"
    if "contact" in kinds:
        return "observed_contact"
    return "unknown_positive"


def connected_components(
    part_ids: Iterable[str],
    positive_pairs: Iterable[tuple[str, str]],
) -> list[list[str]]:
    adjacency = {part_id: set() for part_id in part_ids}
    for first, second in positive_pairs:
        if first in adjacency and second in adjacency:
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = set(adjacency)
    output = []
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
        output.append(sorted(component))
    return sorted(output, key=lambda row: (-len(row), row))


def choose_connected_subset(
    graph: dict[str, Any],
    target_size: int,
    salt: str,
) -> list[str]:
    part_ids = [str(part["part_id"]) for part in graph.get("parts", [])]
    positive_pairs = {
        pair
        for edge in graph.get("positive_part_pair_edges", [])
        if (pair := edge_pair(edge)) is not None
    }
    components = [
        component
        for component in connected_components(part_ids, positive_pairs)
        if len(component) >= 2
    ]
    if not components:
        return []
    salt_value = int(
        hashlib.sha256(salt.encode("utf-8")).hexdigest()[:16], 16
    )
    component = components[salt_value % len(components)]
    target_size = min(max(2, target_size), 5, len(component))
    allowed = set(component)
    adjacency = {part_id: set() for part_id in component}
    for first, second in positive_pairs:
        if first in allowed and second in allowed:
            adjacency[first].add(second)
            adjacency[second].add(first)
    roots = sorted(
        component,
        key=lambda value: hashlib.sha256(
            f"{salt}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    root = roots[0]
    selected = []
    seen = {root}
    queue = deque([root])
    while queue and len(selected) < target_size:
        current = queue.popleft()
        selected.append(current)
        neighbors = sorted(
            adjacency[current],
            key=lambda value: hashlib.sha256(
                f"{salt}:{current}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        for neighbor in neighbors:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    if len(selected) < target_size:
        return []
    return selected


def step_path(part: dict[str, Any]) -> Path | None:
    for candidate in (part.get("geometry") or {}).get("candidates") or []:
        value = candidate.get("path")
        if (
            str(candidate.get("format")).lower() == "step"
            and value
            and Path(value).is_file()
        ):
            return Path(value).resolve()
    return None


STEP_TOKENS = (
    "ADVANCED_FACE",
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "CONICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "B_SPLINE_SURFACE",
    "CIRCLE",
)


def step_signature(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    counters = Counter()
    with path.open("r", encoding="latin-1", errors="ignore") as stream:
        for line in stream:
            upper = line.upper()
            for token in STEP_TOKENS:
                counters[token] += upper.count(token)
    result = {
        "file_bytes": path.stat().st_size,
        "entity_counts": {
            token.lower(): counters[token] for token in STEP_TOKENS
        },
    }
    cache[key] = result
    return result


def signature_distance(
    first: dict[str, Any], second: dict[str, Any]
) -> float:
    first_bytes = max(1, int(first["file_bytes"]))
    second_bytes = max(1, int(second["file_bytes"]))
    distance = abs(math.log(first_bytes / second_bytes))
    first_counts = first["entity_counts"]
    second_counts = second["entity_counts"]
    for token in (value.lower() for value in STEP_TOKENS):
        distance += 0.2 * abs(
            math.log1p(int(first_counts.get(token, 0)))
            - math.log1p(int(second_counts.get(token, 0)))
        )
    return distance


def build_split_assignment(
    assembly_ids: list[str], seed: int
) -> dict[str, list[str]]:
    if len(assembly_ids) < 10:
        raise ValueError(
            f"at_least_10_assemblies_required:{len(assembly_ids)}"
        )
    shuffled = sorted(assembly_ids)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": sorted(shuffled[:6]),
        "validation": sorted(shuffled[6:8]),
        "test": sorted(shuffled[8:]),
    }


def source_combinations(
    source_ids: list[str],
    count: int,
    groups_per_pool: int,
    seed: int,
    split: str,
) -> list[list[str]]:
    if len(source_ids) < groups_per_pool:
        raise ValueError(
            f"not_enough_source_assemblies:{split}:"
            f"{len(source_ids)}<{groups_per_pool}"
        )
    rng = random.Random(f"{seed}:{split}")
    output = []
    for pool_index in range(count):
        rotated = source_ids[pool_index % len(source_ids) :] + source_ids[
            : pool_index % len(source_ids)
        ]
        candidates = list(rotated)
        rng.shuffle(candidates)
        output.append(sorted(candidates[:groups_per_pool]))
    return output


def build_pool(
    pool_id: str,
    split: str,
    source_ids: list[str],
    graphs: dict[str, dict[str, Any]],
    output_root: Path,
    seed: int,
    signature_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pool_dir = output_root / pool_id
    if pool_dir.exists():
        raise FileExistsError(f"pool_output_already_exists:{pool_dir}")
    parts_dir = pool_dir / "parts"
    parts_dir.mkdir(parents=True)
    failures: list[str] = []
    unavailable = {
        "functional_semantic_validity",
        "physical_compatibility_of_cross_group_pairs",
        "interchangeable_part_labels",
    }

    selected_groups = []
    selected_records = []
    selected_part_count = 0
    for group_index, assembly_id in enumerate(source_ids, start=1):
        graph = graphs[assembly_id]
        raw_desired_size = 2 + (
            int(
                hashlib.sha256(
                    f"{seed}:{pool_id}:{assembly_id}".encode("utf-8")
                ).hexdigest()[:8],
                16,
            )
            % 4
        )
        remaining_group_count = len(source_ids) - group_index
        maximum_size = min(
            5,
            12 - selected_part_count - 2 * remaining_group_count,
        )
        minimum_size = max(
            2,
            5 - selected_part_count
            if remaining_group_count == 0
            else 2,
        )
        desired_size = min(
            maximum_size, max(minimum_size, raw_desired_size)
        )
        selected = choose_connected_subset(
            graph,
            desired_size,
            f"{seed}:{pool_id}:{assembly_id}",
        )
        if len(selected) < 2:
            failures.append(
                f"no_connected_subset:{assembly_id}:{desired_size}"
            )
            continue
        parts_by_id = {
            str(part["part_id"]): part for part in graph.get("parts", [])
        }
        selected_set = set(selected)
        selected_edges = [
            edge
            for edge in graph.get("positive_part_pair_edges", [])
            if (
                (pair := edge_pair(edge)) is not None
                and pair[0] in selected_set
                and pair[1] in selected_set
            )
        ]
        selected_groups.append(
            {
                "group_id": f"G{group_index:02d}",
                "source_assembly_id": assembly_id,
                "source_graph_path": graph["_source_graph_path"],
                "source_part_ids": selected,
                "source_visible_part_count": len(graph.get("parts", [])),
                "is_partial_source_assembly": (
                    len(selected) < len(graph.get("parts", []))
                ),
                "source_direct_edges": selected_edges,
            }
        )
        selected_part_count += len(selected)
        for source_part_id in selected:
            part = parts_by_id[source_part_id]
            path = step_path(part)
            if path is None:
                failures.append(
                    f"step_path_missing:{assembly_id}:{source_part_id}"
                )
                continue
            selected_records.append(
                {
                    "group_id": f"G{group_index:02d}",
                    "source_assembly_id": assembly_id,
                    "source_part_id": source_part_id,
                    "source_part_name": part.get("part_name"),
                    "source_body_id": part.get("body_id"),
                    "source_occurrence_id": part.get("occurrence_id"),
                    "source_step_path": path,
                    "source_transform": part.get("transform"),
                }
            )

    if failures:
        raise RuntimeError(";".join(failures))
    random.Random(f"{seed}:{pool_id}:part-order").shuffle(selected_records)
    source_to_pool: dict[tuple[str, str], str] = {}
    input_parts = []
    gt_parts = []
    signatures = {}
    for part_index, record in enumerate(selected_records, start=1):
        pool_part_id = f"P{part_index:03d}"
        destination = parts_dir / f"{pool_part_id}.step"
        shutil.copy2(record["source_step_path"], destination)
        signature = step_signature(
            record["source_step_path"], signature_cache
        )
        signatures[pool_part_id] = signature
        source_to_pool[
            (record["source_assembly_id"], record["source_part_id"])
        ] = pool_part_id
        input_parts.append(
            {
                "part_id": pool_part_id,
                "geometry_path": f"parts/{pool_part_id}.step",
                "geometry_format": "step",
                "geometry_sha256": sha256_file(destination),
                "geometry_bytes": destination.stat().st_size,
                "geometry_signature": signature,
                "source_dataset": "fusion360_gallery_assembly",
            }
        )
        gt_parts.append(
            {
                "part_id": pool_part_id,
                "true_group_id": record["group_id"],
                "source_assembly_id": record["source_assembly_id"],
                "source_part_id": record["source_part_id"],
                "source_part_name": record["source_part_name"],
                "source_body_id": record["source_body_id"],
                "source_occurrence_id": record["source_occurrence_id"],
                "source_step_path": str(record["source_step_path"]),
                "source_transform": record["source_transform"],
            }
        )

    true_groups = []
    direct_positive_pairs = []
    same_group_non_edges = []
    for group in selected_groups:
        assembly_id = group["source_assembly_id"]
        pool_ids = [
            source_to_pool[(assembly_id, source_part_id)]
            for source_part_id in group["source_part_ids"]
        ]
        selected_source_pairs = set()
        mapped_edges = []
        for edge in group["source_direct_edges"]:
            pair = edge_pair(edge)
            if pair is None:
                continue
            selected_source_pairs.add(pair)
            pool_pair = canonical_pair(
                source_to_pool[(assembly_id, pair[0])],
                source_to_pool[(assembly_id, pair[1])],
            )
            mapped = {
                "part_pair": list(pool_pair),
                "source_part_pair": list(pair),
                "evidence_layer": relation_layer(edge),
                "relation_kinds": edge.get("relation_kinds", []),
                "relation_types": edge.get("relation_types", []),
                "source_edge_id": edge.get("edge_id"),
                "interface_entities": [
                    entity
                    for relation in edge.get("relations", [])
                    for entity in relation.get("interface_entities", [])
                ],
            }
            mapped_edges.append(mapped)
            direct_positive_pairs.append(
                {
                    "sample_id": (
                        f"{pool_id}:direct_positive:"
                        f"{pool_pair[0]}:{pool_pair[1]}"
                    ),
                    "part_pair": list(pool_pair),
                    "label": 1,
                    "label_task": "direct_joint_or_contact",
                    "evidence_layer": relation_layer(edge),
                    "true_group_id": group["group_id"],
                    "failure_reasons": [],
                    "unavailable_fields": (
                        []
                        if relation_layer(edge) == "designer_joint"
                        else ["designer_intent_for_contact_only_edge"]
                    ),
                }
            )
        for first, second in combinations(group["source_part_ids"], 2):
            source_pair = canonical_pair(first, second)
            if source_pair in selected_source_pairs:
                continue
            pool_pair = canonical_pair(
                source_to_pool[(assembly_id, first)],
                source_to_pool[(assembly_id, second)],
            )
            same_group_non_edges.append(
                {
                    "part_pair": list(pool_pair),
                    "true_group_id": group["group_id"],
                    "label_task": "same_provenance_group_non_edge",
                    "direct_joinability_label": "unknown",
                    "reason": (
                        "Parts share source assembly provenance but have no "
                        "recorded direct joint/contact in the selected graph."
                    ),
                }
            )
        true_groups.append(
            {
                "group_id": group["group_id"],
                "part_ids": sorted(pool_ids),
                "source_assembly_id": assembly_id,
                "source_graph_path": group["source_graph_path"],
                "source_visible_part_count": group[
                    "source_visible_part_count"
                ],
                "selected_part_count": len(pool_ids),
                "is_partial_source_assembly": group[
                    "is_partial_source_assembly"
                ],
                "direct_edges": mapped_edges,
                "group_truth_type": "source_assembly_provenance",
                "functional_completeness_proven": False,
            }
        )

    group_by_part = {
        part_id: group["group_id"]
        for group in true_groups
        for part_id in group["part_ids"]
    }
    cross_group_negatives = []
    for first, second in combinations(sorted(group_by_part), 2):
        if group_by_part[first] == group_by_part[second]:
            continue
        pair = canonical_pair(first, second)
        cross_group_negatives.append(
            {
                "sample_id": (
                    f"{pool_id}:cross_group_negative:"
                    f"{pair[0]}:{pair[1]}"
                ),
                "part_pair": list(pair),
                "label": 0,
                "label_task": "same_source_assembly_membership",
                "negative_tier": "cross_assembly_provenance_negative",
                "reason": "The parts originate from different source assemblies.",
                "physical_incompatibility_proven": False,
                "functional_incompatibility_proven": False,
                "failure_reasons": [],
                "unavailable_fields": [
                    "physical_incompatibility_proof",
                    "functional_incompatibility_proof",
                ],
            }
        )

    similarity_candidates = []
    for row in cross_group_negatives:
        first, second = row["part_pair"]
        similarity_candidates.append(
            {
                "part_pair": [first, second],
                "signature_distance": signature_distance(
                    signatures[first], signatures[second]
                ),
                "negative_tier": (
                    "geometric_similarity_candidate_negative"
                ),
                "provenance_negative": True,
                "geometry_compatibility_verified": False,
                "functional_incompatibility_verified": False,
                "reason": (
                    "Cross-assembly pair with a similar coarse STEP entity "
                    "signature; retained for later physical hard-negative "
                    "verification."
                ),
            }
        )
    similarity_candidates.sort(
        key=lambda row: (
            row["signature_distance"],
            row["part_pair"],
        )
    )
    similarity_candidates = similarity_candidates[: min(
        10, len(similarity_candidates)
    )]

    pool_input = {
        "schema_version": "1.0.0",
        "pool_id": pool_id,
        "split": split,
        "part_count": len(input_parts),
        "parts": sorted(input_parts, key=lambda row: row["part_id"]),
        "candidate_part_pair_count": (
            len(input_parts) * (len(input_parts) - 1) // 2
        ),
        "anonymization": {
            "source_assembly_id_exposed": False,
            "source_part_id_exposed": False,
            "source_part_name_exposed": False,
            "filenames_anonymized": True,
        },
        "failure_reasons": [],
        "unavailable_fields": sorted(unavailable),
    }
    pool_gt = {
        "schema_version": "1.0.0",
        "pool_id": pool_id,
        "split": split,
        "parts": sorted(gt_parts, key=lambda row: row["part_id"]),
        "true_groups": true_groups,
        "direct_positive_pairs": direct_positive_pairs,
        "same_group_non_edges": same_group_non_edges,
        "cross_group_negative_pairs": cross_group_negatives,
        "truth_policy": {
            "group_label": "source_assembly_provenance",
            "direct_edge_label": (
                "mapped designer joint or observed contact"
            ),
            "negative_label": (
                "different source assembly provenance, not universal "
                "mechanical or semantic incompatibility"
            ),
        },
        "failure_reasons": [],
        "unavailable_fields": sorted(unavailable),
    }
    hard_negatives = {
        "schema_version": "1.0.0",
        "pool_id": pool_id,
        "verified_easy_negatives": [],
        "verified_geometric_hard_negatives": [],
        "verified_semantic_hard_negatives": [],
        "geometric_similarity_candidates": similarity_candidates,
        "failure_reasons": [],
        "unavailable_fields": [
            "collision_validated_easy_negatives",
            "pose_validated_geometric_hard_negatives",
            "role_labeled_semantic_hard_negatives",
        ],
    }
    input_text = json.dumps(
        pool_input, ensure_ascii=False, sort_keys=True
    )
    leaked_values = []
    for part in gt_parts:
        for field in (
            "source_assembly_id",
            "source_part_id",
            "source_part_name",
            "source_body_id",
            "source_occurrence_id",
        ):
            value = part.get(field)
            if value and str(value) in input_text:
                leaked_values.append(f"{field}:{value}")
    leakage_audit = {
        "schema_version": "1.0.0",
        "pool_id": pool_id,
        "status": "passed" if not leaked_values else "failed",
        "source_identity_leaks_in_pool_input": leaked_values,
        "anonymized_part_filename_count": len(input_parts),
        "failure_reasons": (
            []
            if not leaked_values
            else ["source_identity_exposed_in_pool_input"]
        ),
        "unavailable_fields": [],
    }
    write_json(pool_dir / "pool_input.json", pool_input)
    write_json(pool_dir / "pool_gt.json", pool_gt)
    write_json(pool_dir / "hard_negative_candidates.json", hard_negatives)
    write_json(pool_dir / "leakage_audit.json", leakage_audit)
    return {
        "pool_id": pool_id,
        "split": split,
        "pool_path": str(pool_dir.resolve()),
        "pool_input_path": str((pool_dir / "pool_input.json").resolve()),
        "pool_gt_path": str((pool_dir / "pool_gt.json").resolve()),
        "part_count": len(input_parts),
        "true_group_count": len(true_groups),
        "group_sizes": sorted(
            len(group["part_ids"]) for group in true_groups
        ),
        "direct_positive_pair_count": len(direct_positive_pairs),
        "same_group_non_edge_count": len(same_group_non_edges),
        "cross_group_negative_pair_count": len(cross_group_negatives),
        "geometric_similarity_candidate_count": len(
            similarity_candidates
        ),
        "leakage_audit_status": leakage_audit["status"],
        "failure_reasons": leakage_audit["failure_reasons"],
        "unavailable_fields": sorted(unavailable),
    }


def write_pair_csv(
    output_path: Path, output_root: Path, pools: list[dict[str, Any]]
) -> None:
    fields = [
        "sample_id",
        "pool_id",
        "split",
        "part_a",
        "part_b",
        "label",
        "label_task",
        "evidence_layer",
        "negative_tier",
        "failure_reasons",
        "unavailable_fields",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for pool in pools:
            gt = read_json(Path(pool["pool_gt_path"]))
            rows = list(gt["direct_positive_pairs"]) + list(
                gt["cross_group_negative_pairs"]
            )
            for row in rows:
                first, second = row["part_pair"]
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "pool_id": pool["pool_id"],
                        "split": pool["split"],
                        "part_a": first,
                        "part_b": second,
                        "label": row["label"],
                        "label_task": row["label_task"],
                        "evidence_layer": row.get(
                            "evidence_layer", ""
                        ),
                        "negative_tier": row.get(
                            "negative_tier", ""
                        ),
                        "failure_reasons": "|".join(
                            row.get("failure_reasons") or []
                        ),
                        "unavailable_fields": "|".join(
                            row.get("unavailable_fields") or []
                        ),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly_dataset_manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--train-pools", type=int, default=4)
    parser.add_argument("--validation-pools", type=int, default=1)
    parser.add_argument("--test-pools", type=int, default=1)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output_root_must_be_new_or_empty:{output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_manifest = read_json(
        args.assembly_dataset_manifest.resolve()
    )
    graphs: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for record in dataset_manifest.get("assemblies", []):
        graph_path = Path(record["graph_path"])
        if not graph_path.is_file():
            failures.append(
                f"source_graph_missing:{record['assembly_id']}"
            )
            continue
        if sha256_file(graph_path) != record.get("graph_sha256"):
            failures.append(
                f"source_graph_hash_mismatch:{record['assembly_id']}"
            )
            continue
        graph = read_json(graph_path)
        graph["_source_graph_path"] = str(graph_path.resolve())
        graphs[str(record["assembly_id"])] = graph
    if failures:
        raise RuntimeError(";".join(failures))

    splits = build_split_assignment(sorted(graphs), args.seed)
    split_overlaps = {
        "train_validation": sorted(
            set(splits["train"]) & set(splits["validation"])
        ),
        "train_test": sorted(
            set(splits["train"]) & set(splits["test"])
        ),
        "validation_test": sorted(
            set(splits["validation"]) & set(splits["test"])
        ),
    }
    pool_specs = []
    counts = {
        "train": args.train_pools,
        "validation": args.validation_pools,
        "test": args.test_pools,
    }
    global_index = 1
    for split in ("train", "validation", "test"):
        groups_per_pool = 3 if split == "train" else 2
        combinations_for_split = source_combinations(
            splits[split],
            counts[split],
            groups_per_pool,
            args.seed,
            split,
        )
        for source_ids in combinations_for_split:
            pool_specs.append(
                {
                    "pool_id": f"pool_{global_index:03d}",
                    "split": split,
                    "source_assembly_ids": source_ids,
                }
            )
            global_index += 1

    signature_cache: dict[str, dict[str, Any]] = {}
    pools = []
    for spec in pool_specs:
        pools.append(
            build_pool(
                spec["pool_id"],
                spec["split"],
                spec["source_assembly_ids"],
                graphs,
                output_root,
                args.seed,
                signature_cache,
            )
        )

    total_parts = sum(pool["part_count"] for pool in pools)
    total_direct_positives = sum(
        pool["direct_positive_pair_count"] for pool in pools
    )
    total_cross_negatives = sum(
        pool["cross_group_negative_pair_count"] for pool in pools
    )
    leakage_passed = all(
        pool["leakage_audit_status"] == "passed" for pool in pools
    )
    no_split_overlap = not any(split_overlaps.values())
    pair_manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "fusion360_real_mixed_pools_v1",
        "task_layers": {
            "direct_interface_prediction": (
                "positive only when a joint/contact edge is observed"
            ),
            "group_membership": (
                "positive when parts share source assembly provenance"
            ),
        },
        "positive_direct_pair_count": total_direct_positives,
        "negative_cross_group_pair_count": total_cross_negatives,
        "same_group_non_edges_excluded_from_direct_negative_label": True,
        "csv_path": str((output_root / "pair_samples.csv").resolve()),
        "failure_reasons": [],
        "unavailable_fields": [
            "proof_that_cross_group_pair_is_mechanically_incompatible",
            "proof_that_cross_group_pair_is_functionally_incompatible",
        ],
    }
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "fusion360_real_mixed_pools_v1",
        "source_dataset_manifest": str(
            args.assembly_dataset_manifest.resolve()
        ),
        "seed": args.seed,
        "split_assignment": splits,
        "split_overlap_audit": split_overlaps,
        "source_assembly_count": len(graphs),
        "pool_count": len(pools),
        "part_instance_count_across_pools": total_parts,
        "direct_positive_pair_count": total_direct_positives,
        "cross_group_negative_pair_count": total_cross_negatives,
        "pools": pools,
        "truth_policy": {
            "true_group": "same source assembly provenance",
            "direct_positive": "mapped joint/contact",
            "cross_group_negative": (
                "different source assembly provenance only"
            ),
            "same_group_non_edge": (
                "unknown direct relation; never forced to negative"
            ),
        },
        "quality_gates": {
            "no_source_assembly_overlap_between_splits": no_split_overlap,
            "all_pool_inputs_anonymized": leakage_passed,
            "all_groups_have_2_to_5_parts": all(
                all(2 <= size <= 5 for size in pool["group_sizes"])
                for pool in pools
            ),
            "all_pools_have_5_to_12_parts": all(
                5 <= pool["part_count"] <= 12 for pool in pools
            ),
            "positive_and_negative_pair_samples_available": (
                total_direct_positives > 0
                and total_cross_negatives > 0
            ),
        },
        "failure_reasons": [
            reason
            for pool in pools
            for reason in pool["failure_reasons"]
        ]
        + (
            []
            if no_split_overlap
            else ["source_assembly_split_overlap"]
        ),
        "unavailable_fields": [
            "functional_semantic_group_truth",
            "verified_geometric_hard_negatives",
            "verified_semantic_hard_negatives",
            "interchangeable_part_labels",
        ],
    }
    write_pair_csv(
        output_root / "pair_samples.csv", output_root, pools
    )
    write_json(output_root / "pair_dataset_manifest.json", pair_manifest)
    write_json(output_root / "mixed_pool_manifest.json", manifest)
    print(
        f"Mixed pools: {len(pools)}, parts={total_parts}, "
        f"direct-positive={total_direct_positives}, "
        f"cross-group-negative={total_cross_negatives}"
    )
    all_gates = all(manifest["quality_gates"].values())
    return 0 if all_gates and not manifest["failure_reasons"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
