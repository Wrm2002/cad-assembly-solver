"""Independently validate anonymized real Fusion mixed-pool artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


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
    first, second = str(value[0]), str(value[1])
    if first == second:
        return None
    return tuple(sorted((first, second)))


def validate_pool(pool_record: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    pool_id = str(pool_record["pool_id"])
    pool_dir = Path(pool_record["pool_path"]).resolve()
    input_path = pool_dir / "pool_input.json"
    gt_path = pool_dir / "pool_gt.json"
    hard_path = pool_dir / "hard_negative_candidates.json"
    leakage_path = pool_dir / "leakage_audit.json"
    required = [input_path, gt_path, hard_path, leakage_path]
    for path in required:
        if not path.is_file():
            failures.append(f"required_file_missing:{path.name}")
    if failures:
        return {
            "pool_id": pool_id,
            "status": "failed",
            "failure_reasons": failures,
            "unavailable_fields": ["pool_content_validation"],
        }

    pool_input = read_json(input_path)
    pool_gt = read_json(gt_path)
    hard = read_json(hard_path)
    leakage = read_json(leakage_path)
    if pool_input.get("pool_id") != pool_id:
        failures.append("pool_input_id_mismatch")
    if pool_gt.get("pool_id") != pool_id:
        failures.append("pool_gt_id_mismatch")
    if pool_input.get("split") != pool_record.get("split"):
        failures.append("pool_input_split_mismatch")
    if pool_gt.get("split") != pool_record.get("split"):
        failures.append("pool_gt_split_mismatch")

    input_parts = pool_input.get("parts") or []
    input_part_ids = [str(part.get("part_id")) for part in input_parts]
    if len(input_part_ids) != len(set(input_part_ids)):
        failures.append("duplicate_input_part_ids")
    if not 5 <= len(input_part_ids) <= 12:
        failures.append(
            f"pool_part_count_outside_5_12:{len(input_part_ids)}"
        )
    geometry_failures = []
    for part in input_parts:
        relative_value = str(part.get("geometry_path") or "")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            geometry_failures.append(
                f"unsafe_geometry_path:{part.get('part_id')}"
            )
            continue
        geometry_path = (pool_dir / relative).resolve()
        try:
            geometry_path.relative_to(pool_dir)
        except ValueError:
            geometry_failures.append(
                f"geometry_path_escapes_pool:{part.get('part_id')}"
            )
            continue
        if not geometry_path.is_file():
            geometry_failures.append(
                f"geometry_missing:{part.get('part_id')}"
            )
        elif sha256_file(geometry_path) != part.get("geometry_sha256"):
            geometry_failures.append(
                f"geometry_hash_mismatch:{part.get('part_id')}"
            )
    failures.extend(geometry_failures)

    gt_parts = pool_gt.get("parts") or []
    gt_part_ids = [str(part.get("part_id")) for part in gt_parts]
    if set(gt_part_ids) != set(input_part_ids):
        failures.append("pool_input_gt_part_set_mismatch")
    true_groups = pool_gt.get("true_groups") or []
    grouped_ids = [
        str(part_id)
        for group in true_groups
        for part_id in group.get("part_ids", [])
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        failures.append("true_group_part_overlap")
    if set(grouped_ids) != set(input_part_ids):
        failures.append("true_groups_do_not_partition_pool")
    group_sizes = [
        len(group.get("part_ids") or []) for group in true_groups
    ]
    if not group_sizes or any(size < 2 or size > 5 for size in group_sizes):
        failures.append(f"true_group_size_outside_2_5:{group_sizes}")
    source_assembly_ids = [
        str(group.get("source_assembly_id")) for group in true_groups
    ]
    if len(source_assembly_ids) != len(set(source_assembly_ids)):
        failures.append("duplicate_source_assembly_group_in_pool")

    group_by_part = {
        str(part_id): str(group.get("group_id"))
        for group in true_groups
        for part_id in group.get("part_ids", [])
    }
    expected_same_pairs = set()
    expected_cross_pairs = set()
    for first, second in combinations(sorted(input_part_ids), 2):
        pair = (first, second)
        if group_by_part.get(first) == group_by_part.get(second):
            expected_same_pairs.add(pair)
        else:
            expected_cross_pairs.add(pair)

    direct_rows = pool_gt.get("direct_positive_pairs") or []
    direct_pairs = {
        pair
        for row in direct_rows
        if (pair := canonical_pair(row.get("part_pair"))) is not None
    }
    same_nonedge_rows = pool_gt.get("same_group_non_edges") or []
    same_nonedge_pairs = {
        pair
        for row in same_nonedge_rows
        if (pair := canonical_pair(row.get("part_pair"))) is not None
    }
    cross_rows = pool_gt.get("cross_group_negative_pairs") or []
    cross_pairs = {
        pair
        for row in cross_rows
        if (pair := canonical_pair(row.get("part_pair"))) is not None
    }
    if len(direct_pairs) != len(direct_rows):
        failures.append("duplicate_or_invalid_direct_positive_pair")
    if len(same_nonedge_pairs) != len(same_nonedge_rows):
        failures.append("duplicate_or_invalid_same_group_nonedge_pair")
    if len(cross_pairs) != len(cross_rows):
        failures.append("duplicate_or_invalid_cross_group_negative_pair")
    if not direct_pairs <= expected_same_pairs:
        failures.append("direct_positive_crosses_true_group")
    if not same_nonedge_pairs <= expected_same_pairs:
        failures.append("same_group_nonedge_crosses_true_group")
    if direct_pairs & same_nonedge_pairs:
        failures.append("direct_positive_same_group_nonedge_overlap")
    if direct_pairs | same_nonedge_pairs != expected_same_pairs:
        failures.append("same_group_pair_partition_incomplete")
    if cross_pairs != expected_cross_pairs:
        failures.append("cross_group_negative_partition_incomplete")
    if any(
        row.get("label") != 1
        or row.get("label_task") != "direct_joint_or_contact"
        for row in direct_rows
    ):
        failures.append("invalid_direct_positive_label_semantics")
    if any(
        row.get("label") != 0
        or row.get("label_task") != "same_source_assembly_membership"
        or row.get("physical_incompatibility_proven") is not False
        or row.get("functional_incompatibility_proven") is not False
        for row in cross_rows
    ):
        failures.append("invalid_cross_group_negative_label_semantics")

    similarity_rows = hard.get("geometric_similarity_candidates") or []
    for row in similarity_rows:
        pair = canonical_pair(row.get("part_pair"))
        if pair not in expected_cross_pairs:
            failures.append(
                f"similarity_candidate_not_cross_group:{row.get('part_pair')}"
            )
        if (
            row.get("geometry_compatibility_verified") is not False
            or row.get("functional_incompatibility_verified") is not False
        ):
            failures.append(
                f"unverified_similarity_candidate_overclaimed:{pair}"
            )
    if hard.get("verified_geometric_hard_negatives"):
        warnings.append(
            "verified_geometric_hard_negatives_present_review_required"
        )
    if hard.get("verified_semantic_hard_negatives"):
        warnings.append(
            "verified_semantic_hard_negatives_present_review_required"
        )

    input_serialized = json.dumps(
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
            if value and str(value) in input_serialized:
                leaked_values.append(f"{field}:{value}")
    if leaked_values:
        failures.append(
            f"source_identity_leak_count:{len(leaked_values)}"
        )
    if leakage.get("status") != "passed":
        failures.append("builder_leakage_audit_not_passed")

    return {
        "pool_id": pool_id,
        "split": pool_record.get("split"),
        "status": "passed" if not failures else "failed",
        "part_count": len(input_part_ids),
        "true_group_count": len(true_groups),
        "group_sizes": group_sizes,
        "direct_positive_pair_count": len(direct_pairs),
        "same_group_nonedge_count": len(same_nonedge_pairs),
        "cross_group_negative_pair_count": len(cross_pairs),
        "geometry_file_count": len(input_parts),
        "geometry_hash_match_count": (
            len(input_parts) - len(geometry_failures)
        ),
        "similarity_candidate_count": len(similarity_rows),
        "source_identity_leak_count": len(leaked_values),
        "warnings": warnings,
        "failure_reasons": failures,
        "unavailable_fields": [
            "functional_semantic_validity",
            "verified_geometric_hard_negatives",
            "verified_semantic_hard_negatives",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.mixed_pool_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "mixed_pool_validation_report.json"
    )
    manifest_path = root / "mixed_pool_manifest.json"
    manifest = read_json(manifest_path)
    rows = [validate_pool(pool) for pool in manifest.get("pools", [])]
    splits = manifest.get("split_assignment") or {}
    split_sets = {
        split: set(values) for split, values in splits.items()
    }
    split_overlaps = {
        "train_validation": sorted(
            split_sets.get("train", set())
            & split_sets.get("validation", set())
        ),
        "train_test": sorted(
            split_sets.get("train", set())
            & split_sets.get("test", set())
        ),
        "validation_test": sorted(
            split_sets.get("validation", set())
            & split_sets.get("test", set())
        ),
    }
    failures = [
        f"{row['pool_id']}:{reason}"
        for row in rows
        for reason in row["failure_reasons"]
    ]
    if any(split_overlaps.values()):
        failures.append("source_assembly_split_overlap")
    totals = {
        field: sum(int(row.get(field) or 0) for row in rows)
        for field in (
            "part_count",
            "true_group_count",
            "direct_positive_pair_count",
            "same_group_nonedge_count",
            "cross_group_negative_pair_count",
            "geometry_file_count",
            "geometry_hash_match_count",
            "similarity_candidate_count",
            "source_identity_leak_count",
        )
    }
    report = {
        "schema_version": "1.0.0",
        "dataset_id": manifest.get("dataset_id"),
        "status": "passed" if not failures else "failed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "pool_count": len(rows),
        "passed_pool_count": sum(
            row["status"] == "passed" for row in rows
        ),
        "source_assembly_split_overlap": split_overlaps,
        "totals": totals,
        "label_semantics": {
            "direct_positive": "observed joint/contact",
            "same_group_nonedge": (
                "same provenance group with unknown direct interface"
            ),
            "cross_group_negative": (
                "different provenance group; physical and functional "
                "incompatibility are not claimed"
            ),
        },
        "pools": rows,
        "failure_reasons": failures,
        "unavailable_fields": [
            "functional_semantic_group_truth",
            "verified_geometric_hard_negatives",
            "verified_semantic_hard_negatives",
            "interchangeable_part_labels",
        ],
    }
    write_json(output, report)
    print(
        f"Mixed-pool validation: {report['status']} "
        f"({report['passed_pool_count']}/{report['pool_count']} pools)"
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
