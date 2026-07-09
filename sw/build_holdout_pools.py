"""Build anonymous per-case pools from the locked functional CAD holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _anonymous(pool_id: str, case_id: str, part_id: str) -> str:
    digest = hashlib.sha256(
        f"{pool_id}:{case_id}:{part_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"part_{digest}.step"


def _mate_edges(
    metadata: dict[str, Any], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    rows = []
    for mate in metadata["functional_mates"]:
        ids = mate["part_ids"]
        mate_type = mate["mate_type"]
        pairs: list[tuple[str, str, str]] = []
        if mate_type in {
            "circular_registered_fastening",
            "registered_end_retention",
        }:
            pairs.append((ids[0], ids[1], "planar_mate"))
            if len(ids) > 2:
                pairs.append((ids[0], ids[2], "planar_mate"))
        elif mate_type in {
            "single_dowel_clocking",
        }:
            pairs.extend(
                [
                    (ids[0], ids[2], "clearance"),
                    (ids[1], ids[2], "clearance"),
                ]
            )
        elif mate_type in {
            "coaxial_shoulder_location",
            "outer_race_cartridge_seat",
            "inner_race_shaft_fit",
        }:
            pairs.append((ids[0], ids[1], "clearance"))
        elif mate_type == "keyed_torque_transfer":
            pairs.extend(
                [
                    (ids[0], ids[2], "planar_mate"),
                    (ids[1], ids[2], "planar_mate"),
                ]
            )
        elif mate_type == "optional_axial_retention":
            pairs.extend(
                [
                    (ids[0], ids[2], "clearance"),
                    (ids[1], ids[2], "planar_mate"),
                ]
            )
        for left, right, candidate_type in pairs:
            rows.append(
                {
                    "part_a": mapping[left],
                    "part_b": mapping[right],
                    "type": candidate_type,
                    "required_interface_type": mate_type,
                }
            )
    unique = {}
    for row in rows:
        key = (
            tuple(sorted((row["part_a"], row["part_b"]))),
            row["type"],
        )
        unique[key] = row
    return list(unique.values())


def build(
    holdout_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    holdout = Path(holdout_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    manifest = _load(holdout / "dataset_manifest.json")
    lock_sha = hashlib.sha256(
        (holdout / "holdout_lock.json").read_bytes()
    ).hexdigest()
    pool_rows = []

    for index, case in enumerate(manifest["cases"], start=1):
        case_dir = holdout / case["case_id"]
        metadata = _load(case_dir / "metadata.json")
        pool_id = f"holdout_pool_{index:03d}"
        pool = output / pool_id
        parts_dir = pool / "parts"
        parts_dir.mkdir(parents=True)
        mapping: dict[str, str] = {}
        semantics = {}
        production_parts = []

        for part in metadata["parts"] + metadata["negative_parts"]:
            part_id = part["part_id"]
            anonymous = _anonymous(
                pool_id, metadata["case_id"], part_id
            )
            mapping[part_id] = anonymous
            shutil.copy2(case_dir / part["file"], parts_dir / anonymous)
            production_parts.append(
                {"part_id": anonymous, "file": f"parts/{anonymous}"}
            )
            relations = sorted(
                {
                    mate["functional_relation"]
                    for mate in metadata["functional_mates"]
                    if part_id in mate["part_ids"]
                }
            )
            is_positive = part in metadata["parts"]
            semantics[anonymous] = {
                "semantic_schema_version": "2.0.0",
                "part_id": anonymous,
                "part_name": part["part_name"],
                "file_name": anonymous,
                "part_role": part["part_role"],
                "interface_type": part["interface_types"],
                "assembly_family": (
                    metadata["assembly_family"]
                    if is_positive
                    else "unassigned_distractor"
                ),
                "functional_relation": relations if is_positive else [],
                "functions": part.get("functions", []),
                "source": "locked_holdout_metadata_evaluation_only",
                "source_template_disclosed_for_evaluation_only": False,
            }

        mate_edges = _mate_edges(metadata, mapping)
        true_groups = []
        for group_index, valid_group in enumerate(
            metadata["valid_groups"], start=1
        ):
            remapped = [mapping[part_id] for part_id in valid_group]
            remapped_set = set(remapped)
            true_groups.append(
                {
                    "group_id": f"G{group_index:02d}",
                    "parts": sorted(remapped),
                    "assembly_family": metadata["assembly_family"],
                    "engineering_name": metadata["engineering_name"],
                    "truth_basis": (
                        "provisional_functional_validity_pending_engineer_signoff"
                    ),
                    "functional_validity": "provisional_valid",
                    "true_mates": [
                        row
                        for row in mate_edges
                        if {
                            row["part_a"],
                            row["part_b"],
                        }
                        <= remapped_set
                    ],
                    "functional_mates": [
                        {
                            **mate,
                            "part_ids": [
                                mapping[part_id]
                                for part_id in mate["part_ids"]
                            ],
                        }
                        for mate in metadata["functional_mates"]
                        if set(mate["part_ids"]) <= set(valid_group)
                    ],
                    "placements": {
                        mapping[part["part_id"]]: part[
                            "assembly_placement"
                        ]
                        for part in metadata["parts"]
                        if part["part_id"] in valid_group
                    },
                    "interchangeable_parts": [
                        [mapping[part_id] for part_id in group]
                        for group in metadata["interchangeable_parts"]
                        if set(group) <= set(valid_group)
                    ],
                }
            )

        negative_groups = [
            {
                **negative,
                "parts": [
                    mapping[part_id] for part_id in negative["parts"]
                ],
                "source_case_is_truth": False,
                "engineer_confirmation_pending": True,
            }
            for negative in metadata["negative_groups"]
        ]
        all_parts = sorted(mapping.values())
        pool_gt = {
            "schema_version": "2.0.0",
            "pool_id": pool_id,
            "split": "locked_holdout",
            "truth_basis": (
                "provisional_functional_validity_pending_engineer_signoff"
            ),
            "source_id_is_production_truth": False,
            "used_for_rule_tuning": False,
            "holdout_lock_sha256": lock_sha,
            "parts": all_parts,
            "true_groups": true_groups,
            "functional_negative_groups": negative_groups,
            "distractors": sorted(
                mapping[part["part_id"]]
                for part in metadata["negative_parts"]
            ),
            "naming_policy": "anonymous deterministic hashes",
        }
        (pool / "pool_gt.json").write_text(
            json.dumps(pool_gt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (pool / "pool_input.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "pool_id": pool_id,
                    "parts": production_parts,
                    "source_identity_available": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (pool / "part_semantics.json").write_text(
            json.dumps(semantics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pool_rows.append(
            {
                "pool_id": pool_id,
                "source_case_id": metadata["case_id"],
                "assembly_family": metadata["assembly_family"],
                "part_count": len(all_parts),
                "true_group_count": len(true_groups),
                "functional_negative_count": len(negative_groups),
                "used_for_rule_tuning": False,
            }
        )

    output_manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "functional_cad_holdout_pools_v1",
        "pool_count": len(pool_rows),
        "truth_basis": (
            "provisional_functional_validity_pending_engineer_signoff"
        ),
        "used_for_rule_tuning": False,
        "holdout_lock_sha256": lock_sha,
        "pools": pool_rows,
    }
    (output / "mixed_pool_manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_manifest


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-root",
        default=str(here / "data" / "functional_cad_holdout_v1"),
    )
    parser.add_argument(
        "--output-root",
        default=str(here / "data" / "functional_cad_holdout_pools_v1"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.holdout_root, args.output_root),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
