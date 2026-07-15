"""Audit AutoMate parquet metadata without training or loading private CAD."""

from __future__ import annotations

import argparse
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from fusion360_common import write_json


REQUIRED_FILES = (
    "assemblies.parquet",
    "parts.parquet",
    "mates.parquet",
)


def _count_unique_pairs(mates_path: Path) -> tuple[int, Counter, Counter]:
    mate_types = Counter()
    data_quality = Counter()
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "pairs.sqlite"
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE pairs (first TEXT NOT NULL, second TEXT NOT NULL, "
            "PRIMARY KEY (first, second)) WITHOUT ROWID"
        )
        parquet = pq.ParquetFile(mates_path)
        for batch in parquet.iter_batches(
            batch_size=100_000,
            columns=["mateType", "parts", "has_step", "ps_has_errors"],
        ):
            rows = batch.to_pydict()
            pairs_to_insert = []
            for mate_type, parts, has_step, ps_error in zip(
                rows["mateType"],
                rows["parts"],
                rows["has_step"],
                rows["ps_has_errors"],
            ):
                mate_types[str(mate_type or "unavailable")] += 1
                data_quality["mate_rows"] += 1
                data_quality["mate_rows_with_step_pair"] += bool(has_step)
                data_quality["mate_rows_with_parasolid_error"] += bool(
                    ps_error
                )
                if not parts or len(parts) != 2 or not all(parts):
                    data_quality["mate_rows_with_invalid_part_pair"] += 1
                    continue
                first, second = sorted((str(parts[0]), str(parts[1])))
                if first == second:
                    data_quality["mate_rows_with_self_pair"] += 1
                    continue
                pairs_to_insert.append((first, second))
            connection.executemany(
                "INSERT OR IGNORE INTO pairs VALUES (?, ?)",
                pairs_to_insert,
            )
            connection.commit()
        unique_pairs = connection.execute(
            "SELECT COUNT(*) FROM pairs"
        ).fetchone()[0]
        connection.close()
    return unique_pairs, mate_types, data_quality


def _sum_boolean(table_path: Path, column: str) -> int:
    total = 0
    parquet = pq.ParquetFile(table_path)
    for batch in parquet.iter_batches(
        batch_size=200_000, columns=[column]
    ):
        total += sum(bool(value) for value in batch[column].to_pylist())
    return total


def _part_counts(parts_path: Path) -> dict[str, int]:
    counters = Counter()
    parquet = pq.ParquetFile(parts_path)
    columns = ["has_step", "readable", "has_error", "n_faces"]
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=columns
    ):
        rows = batch.to_pydict()
        for has_step, readable, has_error, n_faces in zip(
            rows["has_step"],
            rows["readable"],
            rows["has_error"],
            rows["n_faces"],
        ):
            counters["part_rows"] += 1
            counters["declared_step_brep_count"] += bool(has_step)
            counters["readable_parasolid_count"] += bool(readable)
            counters["parasolid_without_reported_error_count"] += (
                bool(readable) and not bool(has_error)
            )
            counters["parts_with_positive_face_count"] += (
                n_faces is not None and int(n_faces) > 0
            )
    return dict(counters)


def _assembly_counts(path: Path) -> dict[str, int]:
    counters = Counter()
    parquet = pq.ParquetFile(path)
    columns = [
        "n_parts",
        "n_step",
        "n_mates",
        "n_step_mates",
        "is_subassembly",
    ]
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=columns
    ):
        rows = batch.to_pydict()
        for n_parts, n_step, n_mates, n_step_mates, is_sub in zip(
            rows["n_parts"],
            rows["n_step"],
            rows["n_mates"],
            rows["n_step_mates"],
            rows["is_subassembly"],
        ):
            counters["assembly_rows"] += 1
            counters["root_assembly_rows"] += not bool(is_sub)
            counters["subassembly_rows"] += bool(is_sub)
            counters["assemblies_with_at_least_two_parts"] += (
                int(n_parts) >= 2
            )
            counters["assemblies_with_mates"] += int(n_mates) > 0
            counters["assemblies_with_step_mates"] += (
                int(n_step_mates) > 0
            )
            counters["assemblies_with_all_step_parts"] += (
                int(n_parts) > 0 and int(n_step) == int(n_parts)
            )
    return dict(counters)


def audit_dataset(metadata_root: Path) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_FILES
        if not (metadata_root / name).is_file()
    ]
    if missing:
        return {
            "schema_version": "1.0.0",
            "dataset": "AutoMate",
            "audit_status": "failed",
            "failure_reasons": [
                f"required_file_missing:{name}" for name in missing
            ],
            "unavailable_fields": [
                "mate_type_counts",
                "part_pair_count",
                "brep_availability",
            ],
        }
    assemblies_path = metadata_root / "assemblies.parquet"
    parts_path = metadata_root / "parts.parquet"
    mates_path = metadata_root / "mates.parquet"
    unique_pairs, mate_types, mate_quality = _count_unique_pairs(
        mates_path
    )
    part_counts = _part_counts(parts_path)
    assembly_counts = _assembly_counts(assemblies_path)
    mate_type_count = len(mate_types)
    expected_types = {
        "FASTENED",
        "REVOLUTE",
        "SLIDER",
        "PLANAR",
        "CYLINDRICAL",
        "PIN_SLOT",
        "BALL",
        "PARALLEL",
    }
    observed_types = set(mate_types)
    report = {
        "schema_version": "1.0.0",
        "dataset": "AutoMate",
        "audit_status": "success",
        "audit_scope": {
            "metadata_root": str(metadata_root.resolve()),
            "metadata_tables_read": list(REQUIRED_FILES),
            "step_archive_downloaded": False,
            "parasolid_archive_downloaded": False,
            "assembly_json_archive_downloaded": False,
        },
        "official_dataset_metadata": {
            "license": "CC0-1.0",
            "step_archive_size_gb": 13.2,
            "parasolid_archive_size_gb": 20.5,
            "assembly_json_archive_size_mb": 911.4,
            "source": "https://zenodo.org/records/7776208",
        },
        "observed_counts": {
            **assembly_counts,
            **part_counts,
            **dict(mate_quality),
            "unique_positive_part_pair_count": unique_pairs,
            "mate_type_count": mate_type_count,
        },
        "mate_type_counts": dict(sorted(mate_types.items())),
        "mate_type_schema_check": {
            "expected_types": sorted(expected_types),
            "observed_types": sorted(observed_types),
            "missing_expected_types": sorted(
                expected_types - observed_types
            ),
            "additional_types": sorted(
                observed_types - expected_types
            ),
        },
        "brep_availability": {
            "declared_step_brep_count": part_counts[
                "declared_step_brep_count"
            ],
            "readable_parasolid_count": part_counts[
                "readable_parasolid_count"
            ],
            "locally_present_step_count": 0,
            "locally_present_parasolid_count": 0,
            "reason": (
                "The 13.2 GB STEP and 20.5 GB Parasolid archives were not "
                "required for metadata-format auditing."
            ),
        },
        "mate_prediction_sample_assessment": {
            "mate_type_prediction_possible": (
                mate_quality["mate_rows"] > 0
                and mate_type_count > 1
            ),
            "positive_part_pair_prediction_possible": unique_pairs > 0,
            "negative_part_pair_generation_possible_from_full_release": True,
            "negative_part_pair_generation_possible_from_local_metadata_only": False,
            "face_or_edge_level_supervision_directly_available": False,
            "reasoning": [
                "Mate rows contain type, two part ids, two mate coordinate frames, and STEP availability.",
                "Assembly JSON contains occurrence membership needed for same-assembly negatives.",
                "Published mate records do not contain direct B-Rep face or edge ids.",
                "Equal part ids can represent two occurrences of one repeated part; parquet alone cannot disambiguate those occurrences.",
            ],
        },
        "suitability": {
            "verdict": (
                "suitable_secondary_source_for_mate_type_and_pair_tasks"
            ),
            "suitable_as_primary_interface_face_source": False,
            "suitable_as_primary_part_pair_source": True,
            "reasons": [
                "Very large real assembly and mate corpus.",
                "Both Parasolid and STEP geometry are released.",
                "Mate types and coordinate frames are explicit.",
                "Direct interface topology ids are unavailable.",
                "Full geometry download and preprocessing cost is substantial.",
            ],
        },
        "failure_reasons": [],
        "unavailable_fields": [
            "direct_mate_brep_face_ids",
            "direct_mate_brep_edge_ids",
            "local_geometry_files_not_downloaded",
            "same_assembly_negative_pairs_not_built_without_assemblies_zip",
            "occurrence_identity_for_equal_part_id_mates_not_in_parquet",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_root")
    parser.add_argument(
        "--output", default="outputs/automate_audit_report.json"
    )
    args = parser.parse_args()
    report = audit_dataset(Path(args.metadata_root))
    write_json(Path(args.output), report)
    print(f"AutoMate audit: {report['audit_status']}")
    if report["audit_status"] == "success":
        print(
            "Unique mate part pairs: "
            f"{report['observed_counts']['unique_positive_part_pair_count']}"
        )
    return 0 if report["audit_status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
