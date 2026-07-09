"""Build frozen mixed pools from D0 function-grounded assembly cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


FAMILIES = ("cover_base", "shaft_hub_key", "bearing_housing")
SPLITS = {1: "train", 2: "calibration", 3: "test"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _anonymous_name(pool_id: str, token: str) -> str:
    digest = hashlib.sha256(f"{pool_id}:{token}".encode("utf-8")).hexdigest()
    return f"part_{digest[:12]}.step"


def _geometry_mates(
    family: str,
    role_to_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if family == "cover_base":
        base = role_to_ids["base"][0]
        cover = role_to_ids["cover"][0]
        rows = [
            {
                "part_a": base,
                "part_b": cover,
                "type": "planar_mate",
                "required_interface_type": "planar_registered_fastening",
            }
        ]
        for pin in role_to_ids["locating_pin"]:
            rows.extend(
                [
                    {
                        "part_a": base,
                        "part_b": pin,
                        "type": "clearance",
                        "required_interface_type": "locating_pin_fit",
                    },
                    {
                        "part_a": cover,
                        "part_b": pin,
                        "type": "clearance",
                        "required_interface_type": "locating_pin_fit",
                    },
                ]
            )
        return rows
    if family == "shaft_hub_key":
        shaft = role_to_ids["shaft"][0]
        hub = role_to_ids["hub"][0]
        key = role_to_ids["key"][0]
        return [
            {
                "part_a": shaft,
                "part_b": hub,
                "type": "clearance",
                "required_interface_type": "coaxial_radius_fit",
            },
            {
                "part_a": shaft,
                "part_b": key,
                "type": "planar_mate",
                "required_interface_type": "shaft_keyway_contact",
            },
            {
                "part_a": hub,
                "part_b": key,
                "type": "planar_mate",
                "required_interface_type": "hub_keyway_contact",
            },
        ]
    housing = role_to_ids["housing"][0]
    bearing = role_to_ids["bearing"][0]
    shaft = role_to_ids["shaft"][0]
    cover = role_to_ids["end_cover"][0]
    return [
        {
            "part_a": housing,
            "part_b": bearing,
            "type": "clearance",
            "required_interface_type": "outer_race_seat",
        },
        {
            "part_a": bearing,
            "part_b": shaft,
            "type": "clearance",
            "required_interface_type": "inner_race_shaft_fit",
        },
        {
            "part_a": housing,
            "part_b": cover,
            "type": "planar_mate",
            "required_interface_type": "registered_fastened_cover",
        },
    ]


def build_pools(
    dataset_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    pool_rows = []
    for variant, split in SPLITS.items():
        pool_id = f"functional_pool_{variant:03d}"
        pool = output / pool_id
        parts_dir = pool / "parts"
        parts_dir.mkdir(parents=True)
        pool_parts = []
        part_semantics = {}
        true_groups = []
        functional_negatives = []
        production_parts = []
        for family_index, family in enumerate(FAMILIES, start=1):
            case_id = f"{family}_{variant:02d}"
            case_dir = dataset / case_id
            metadata = _load(case_dir / "metadata.json")
            mapping = {}
            role_to_ids: dict[str, list[str]] = {}
            for part in metadata["parts"]:
                anonymous = _anonymous_name(
                    pool_id, f"{case_id}:{part['part_id']}"
                )
                mapping[part["part_id"]] = anonymous
                shutil.copy2(case_dir / part["file"], parts_dir / anonymous)
                pool_parts.append(anonymous)
                production_parts.append(
                    {"part_id": anonymous, "file": f"parts/{anonymous}"}
                )
                role_to_ids.setdefault(part["part_role"], []).append(anonymous)
                relations = sorted(
                    {
                        mate["functional_relation"]
                        for mate in metadata["functional_mates"]
                        if part["part_id"] in mate["part_ids"]
                    }
                )
                part_semantics[anonymous] = {
                    "semantic_schema_version": "2.0.0",
                    "part_id": anonymous,
                    "part_name": part["part_name"],
                    "file_name": anonymous,
                    "part_role": part["part_role"],
                    "interface_type": part["interface_types"],
                    "assembly_family": family,
                    "functional_relation": relations,
                    "functions": part.get("functions", []),
                    "source": "function_grounded_metadata",
                    "source_template_disclosed_for_evaluation_only": False,
                }
            geometry_mates = _geometry_mates(family, role_to_ids)
            true_groups.append(
                {
                    "group_id": f"G{family_index:02d}",
                    "parts": sorted(mapping.values()),
                    "assembly_family": family,
                    "engineering_name": metadata["engineering_name"],
                    "truth_basis": "functional_validity",
                    "functional_validity": "valid",
                    "true_mates": geometry_mates,
                    "functional_mates": [
                        {
                            **mate,
                            "part_ids": [
                                mapping[part_id]
                                for part_id in mate["part_ids"]
                            ],
                        }
                        for mate in metadata["functional_mates"]
                    ],
                    "placements": {
                        mapping[part["part_id"]]: part["assembly_placement"]
                        for part in metadata["parts"]
                    },
                    "interchangeable_parts": [
                        [mapping[part_id] for part_id in group]
                        for group in metadata["interchangeable_parts"]
                    ],
                }
            )

            # One controlled negative tier per family yields three distractors
            # per pool without turning the benchmark into a negative-only set.
            negative_part_id = f"N0{family_index}"
            negative = next(
                part
                for part in metadata["negative_parts"]
                if part["part_id"] == negative_part_id
            )
            anonymous_negative = _anonymous_name(
                pool_id, f"{case_id}:{negative_part_id}"
            )
            shutil.copy2(
                case_dir / negative["file"],
                parts_dir / anonymous_negative,
            )
            pool_parts.append(anonymous_negative)
            production_parts.append(
                {
                    "part_id": anonymous_negative,
                    "file": f"parts/{anonymous_negative}",
                }
            )
            part_semantics[anonymous_negative] = {
                "semantic_schema_version": "2.0.0",
                "part_id": anonymous_negative,
                "part_name": negative["part_name"],
                "file_name": anonymous_negative,
                "part_role": negative["part_role"],
                "interface_type": negative["interface_types"],
                "assembly_family": "unassigned_distractor",
                "functional_relation": [],
                "functions": [],
                "source": "function_grounded_metadata",
                "source_template_disclosed_for_evaluation_only": False,
            }
            negative_group = next(
                row
                for row in metadata["negative_groups"]
                if row["negative_id"]
                == (
                    "NEG_EASY"
                    if family_index == 1
                    else (
                        "NEG_GEOMETRIC_HARD"
                        if family_index == 2
                        else "NEG_SEMANTIC_HARD"
                    )
                )
            )
            remapped_negative_parts = [
                (
                    anonymous_negative
                    if part_id == negative_part_id
                    else mapping[part_id]
                )
                for part_id in negative_group["parts"]
            ]
            functional_negatives.append(
                {
                    **negative_group,
                    "parts": remapped_negative_parts,
                    "source_case_is_truth": False,
                }
            )

        pool_parts = sorted(pool_parts)
        pool_gt = {
            "schema_version": "2.0.0",
            "pool_id": pool_id,
            "split": split,
            "truth_basis": "functional_validity",
            "source_id_is_production_truth": False,
            "parts": pool_parts,
            "true_groups": true_groups,
            "functional_negative_groups": functional_negatives,
            "distractors": sorted(
                set(pool_parts)
                - {
                    part
                    for group in true_groups
                    for part in group["parts"]
                }
            ),
            "naming_policy": "anonymous deterministic hashes",
        }
        (pool / "pool_gt.json").write_text(
            json.dumps(pool_gt, indent=2, ensure_ascii=False) + "\n",
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
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (pool / "part_semantics.json").write_text(
            json.dumps(part_semantics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pool_rows.append(
            {
                "pool_id": pool_id,
                "split": split,
                "part_count": len(pool_parts),
                "true_group_count": len(true_groups),
                "functional_negative_count": len(functional_negatives),
                "source_case_overlap_with_other_splits": False,
            }
        )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "functional_mixed_pool_benchmark_v1",
        "truth_basis": "functional_validity",
        "source_id_is_production_truth": False,
        "pool_count": len(pool_rows),
        "pools": pool_rows,
        "split_policy": "variant-disjoint train/calibration/test",
        "failure_reasons": [],
    }
    (output / "mixed_pool_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "functional_dataset_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "functional_mixed_pools_v1"
        ),
    )
    args = parser.parse_args()
    result = build_pools(args.dataset_root, args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
