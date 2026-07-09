"""Generate a locked, topology-varied CAD holdout and engineer review pack.

The holdout is intentionally separate from functional_dataset_v1 and must not
be used to tune thresholds or ranking.  It adds one novel modeled topology per
supported family and leaves engineering sign-off explicitly pending.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from functional_dataset_generator import (
    FAMILIES,
    NegativePart,
    Part,
    _bolt_pattern,
    _box,
    _compound,
    _contact_sheet,
    _cut,
    _cut_holes,
    _cylinder,
    _fuse,
    _metadata,
    _render,
    _ring,
    _shape_stats,
    _translate,
    _write_step,
    validate_metadata,
)


def _round_cover_holdout() -> tuple[
    list[Part], list[NegativePart], dict[str, Any]
]:
    radius = 34.0
    base_height = 14.0
    cover_height = 5.0
    bolt_holes = _bolt_pattern(26.0, 6, 2.2)
    dowel = [(15.0, 0.0, 2.55)]
    base = _cut(
        _cylinder(radius, base_height),
        _cylinder(27.0, 10.0, 0.0, 0.0, 4.0),
    )
    base = _cut_holes(base, bolt_holes + dowel, base_height + 2.0)
    cover = _cylinder(radius, cover_height)
    cover = _cut_holes(cover, bolt_holes + dowel, cover_height + 2.0)
    circular_register = _translate(
        _ring(26.6, 23.0, 2.0), [0.0, 0.0, -2.0]
    )
    cover = _fuse(cover, circular_register)
    parts = [
        Part(
            "P01",
            "base",
            base,
            [
                "planar_seating",
                "circular_register",
                "six_hole_fastener_pattern",
                "locating_pin_hole",
            ],
            [0.0, 0.0, 0.0],
            ["support_components", "locate_cover"],
            "round instrument enclosure base",
        ),
        Part(
            "P02",
            "cover",
            cover,
            [
                "planar_seating",
                "circular_register",
                "six_hole_fastener_pattern",
                "locating_pin_hole",
            ],
            [0.0, 0.0, base_height],
            ["close_enclosure", "protect_instrument"],
            "registered round inspection cover",
        ),
        Part(
            "P03",
            "locating_pin",
            _cylinder(2.35, 10.0),
            ["dowel_pin", "cylindrical_pin_fit"],
            [15.0, 0.0, base_height - 3.0],
            ["clock_cover"],
            "single clocking dowel",
        ),
    ]
    negatives = [
        NegativePart(
            "N01",
            "oversize_pin",
            _cylinder(4.2, 10.0),
            ["oversize_cylindrical_pin"],
            "oversize clocking pin",
        ),
        NegativePart(
            "N02",
            "unregistered_blank_cover",
            _cut_holes(
                _cylinder(radius, cover_height),
                [(15.0, 0.0, 2.55)],
                cover_height + 2.0,
            ),
            ["single_planar_contact", "single_round_hole"],
            "blank disk with one coincident hole but no register or pattern",
        ),
        NegativePart(
            "N03",
            "bearing_ring",
            _ring(26.5, 23.0, 5.0),
            ["circular_register_like_surface"],
            "bearing-like ring that fits the cover register",
        ),
    ]
    relations = {
        "engineering_name": "Round registered instrument inspection cover",
        "functional_mates": [
            {
                "part_ids": ["P01", "P02"],
                "roles": ["base", "cover"],
                "mate_type": "circular_registered_fastening",
                "interface_types": [
                    "planar_seating",
                    "circular_register",
                    "six_hole_fastener_pattern",
                ],
                "functional_relation": (
                    "The circular pilot centers the cover while the repeated "
                    "hole pattern fastens and clocks it."
                ),
                "independent_evidence": [
                    "planar_seating",
                    "circular_register_fit",
                    "six_hole_pattern",
                ],
            },
            {
                "part_ids": ["P01", "P02", "P03"],
                "roles": ["base", "cover", "locating_pin"],
                "mate_type": "single_dowel_clocking",
                "interface_types": [
                    "locating_pin_hole",
                    "dowel_pin",
                ],
                "functional_relation": (
                    "The dowel removes rotational ambiguity left by the "
                    "circular register."
                ),
                "independent_evidence": [
                    "dowel_radius_fit",
                    "off_axis_clocking_position",
                ],
            },
        ],
        "interchangeable_parts": [],
        "invalid_role_combinations": [
            ["cover", "bearing_ring"],
            ["base", "oversize_pin"],
        ],
    }
    return parts, negatives, relations


def _retained_key_drive_holdout() -> tuple[
    list[Part], list[NegativePart], dict[str, Any]
]:
    shaft_radius = 10.0
    hub_inner = 10.25
    hub_height = 24.0
    hub_z = 24.0
    shaft = _fuse(
        _cylinder(shaft_radius, 76.0),
        _cylinder(14.0, 6.0, 0.0, 0.0, 18.0),
    )
    shaft = _cut(
        shaft,
        _box(5.0, 6.0, 32.0, 7.0, -3.0, hub_z - 4.0),
    )
    hub = _fuse(
        _ring(30.0, hub_inner, hub_height),
        _translate(_ring(38.0, hub_inner, 6.0), [0.0, 0.0, 9.0]),
    )
    hub = _cut(
        hub,
        _box(7.0, 6.25, hub_height + 2.0, 9.8, -3.125, -1.0),
    )
    key = _box(6.0, 5.8, hub_height - 2.0, -3.0, -2.9, -11.0)
    collar = _ring(18.0, shaft_radius + 0.3, 6.0)
    parts = [
        Part(
            "P01",
            "stepped_shaft",
            shaft,
            ["stepped_cylindrical_shaft", "shaft_keyway", "shaft_shoulder"],
            [0.0, 0.0, 0.0],
            ["transmit_torque", "axially_locate_hub"],
            "stepped keyed drive shaft",
        ),
        Part(
            "P02",
            "flanged_hub",
            hub,
            ["hub_bore", "hub_keyway", "mounting_flange"],
            [0.0, 0.0, hub_z],
            ["receive_torque", "mount_rotor"],
            "flanged keyed hub",
        ),
        Part(
            "P03",
            "key",
            key,
            ["parallel_key", "shaft_keyway", "hub_keyway"],
            [10.0, 0.0, hub_z + hub_height / 2.0],
            ["transfer_torque"],
            "parallel drive key",
        ),
        Part(
            "P04",
            "axial_retainer",
            collar,
            ["shaft_collar_bore", "axial_retaining_face"],
            [0.0, 0.0, hub_z + hub_height],
            ["limit_axial_motion"],
            "optional axial retaining collar",
        ),
    ]
    negatives = [
        NegativePart(
            "N01",
            "undersize_collar",
            _ring(20.0, 8.0, 8.0),
            ["undersize_bore"],
            "collar that cannot enter the shaft",
        ),
        NegativePart(
            "N02",
            "plain_flanged_hub",
            _fuse(
                _ring(30.0, hub_inner, hub_height),
                _translate(
                    _ring(38.0, hub_inner, 6.0), [0.0, 0.0, 9.0]
                ),
            ),
            ["hub_bore", "single_coaxial_fit"],
            "flanged hub with no torque-transfer keyway",
        ),
        NegativePart(
            "N03",
            "sensor_mount_plate",
            _cut(
                _box(72.0, 54.0, 5.0, -36.0, -27.0, 0.0),
                _cylinder(hub_inner, 7.0, 0.0, 0.0, -1.0),
            ),
            ["central_shaft_hole", "planar_mount"],
            "sensor plate with a coincident shaft-sized hole",
        ),
    ]
    relations = {
        "engineering_name": "Stepped shaft with flanged keyed hub and retainer",
        "functional_mates": [
            {
                "part_ids": ["P01", "P02"],
                "roles": ["stepped_shaft", "flanged_hub"],
                "mate_type": "coaxial_shoulder_location",
                "interface_types": [
                    "stepped_cylindrical_shaft",
                    "hub_bore",
                    "shaft_shoulder",
                ],
                "functional_relation": (
                    "The shaft centers the flanged hub and its shoulder "
                    "provides one axial datum."
                ),
                "independent_evidence": [
                    "coaxial_radius_fit",
                    "shoulder_face_location",
                ],
            },
            {
                "part_ids": ["P01", "P02", "P03"],
                "roles": ["stepped_shaft", "flanged_hub", "key"],
                "mate_type": "keyed_torque_transfer",
                "interface_types": [
                    "shaft_keyway",
                    "hub_keyway",
                    "parallel_key",
                ],
                "functional_relation": (
                    "The key couples the shaft and hub for torque transfer."
                ),
                "independent_evidence": [
                    "paired_keyway_width",
                    "key_radial_engagement",
                ],
            },
            {
                "part_ids": ["P01", "P02", "P04"],
                "roles": [
                    "stepped_shaft",
                    "flanged_hub",
                    "axial_retainer",
                ],
                "mate_type": "optional_axial_retention",
                "interface_types": [
                    "shaft_collar_bore",
                    "axial_retaining_face",
                ],
                "functional_relation": (
                    "The optional collar closes the axial retention stack."
                ),
                "independent_evidence": [
                    "collar_bore_fit",
                    "retaining_face_contact",
                ],
            },
        ],
        "interchangeable_parts": [],
        "invalid_role_combinations": [
            ["flanged_hub", "sensor_mount_plate"],
            ["key", "cover"],
        ],
    }
    return parts, negatives, relations


def _cartridge_bearing_holdout() -> tuple[
    list[Part], list[NegativePart], dict[str, Any]
]:
    shaft_radius = 8.0
    bearing_inner = 8.2
    bearing_outer = 22.0
    housing_bore = 22.25
    housing_height = 28.0
    flange_holes = _bolt_pattern(30.0, 4, 2.5)
    housing = _fuse(
        _ring(34.0, housing_bore, housing_height),
        _ring(42.0, housing_bore, 6.0),
    )
    housing = _cut_holes(housing, flange_holes, 8.0)
    bearing = _ring(bearing_outer, bearing_inner, 14.0)
    shaft = _fuse(
        _cylinder(shaft_radius, 82.0),
        _cylinder(10.5, 5.0, 0.0, 0.0, 19.0),
    )
    cover = _ring(34.0, shaft_radius + 1.0, 5.0)
    cover = _cut_holes(cover, flange_holes, 7.0)
    cover = _fuse(
        cover,
        _translate(
            _ring(bearing_outer + 1.5, shaft_radius + 1.0, 2.0),
            [0.0, 0.0, -2.0],
        ),
    )
    retainer = _ring(bearing_outer + 0.8, bearing_inner, 2.5)
    parts = [
        Part(
            "P01",
            "cartridge_housing",
            housing,
            [
                "housing_bore",
                "bearing_outer_race_seat",
                "flange_hole_pattern",
            ],
            [0.0, 0.0, 0.0],
            ["locate_bearing", "mount_to_frame"],
            "flanged cartridge bearing housing",
        ),
        Part(
            "P02",
            "bearing",
            bearing,
            ["bearing_outer_race", "bearing_inner_race"],
            [0.0, 0.0, 6.0],
            ["support_rotating_shaft", "reduce_friction"],
            "cartridge rolling bearing",
        ),
        Part(
            "P03",
            "stepped_shaft",
            shaft,
            ["shaft_bearing_seat", "shaft_shoulder"],
            [0.0, 0.0, -24.0],
            ["rotate", "transmit_torque"],
            "shouldered bearing shaft",
        ),
        Part(
            "P04",
            "end_cover",
            cover,
            [
                "end_cover_register",
                "flange_hole_pattern",
                "shaft_clearance_hole",
            ],
            [0.0, 0.0, housing_height],
            ["close_housing", "retain_bearing"],
            "registered cartridge end cover",
        ),
        Part(
            "P05",
            "bearing_retainer",
            retainer,
            ["outer_race_retaining_face", "shaft_clearance_hole"],
            [0.0, 0.0, 19.5],
            ["retain_outer_race"],
            "thin outer-race retaining ring",
        ),
    ]
    negatives = [
        NegativePart(
            "N01",
            "oversize_bearing",
            _ring(housing_bore + 3.0, bearing_inner, 14.0),
            ["oversize_bearing_outer_race"],
            "bearing too large for the cartridge bore",
        ),
        NegativePart(
            "N02",
            "plain_spacer",
            _ring(bearing_outer, bearing_inner, 8.0),
            ["single_coaxial_fit"],
            "plain spacer matching the bearing diameters",
        ),
        NegativePart(
            "N03",
            "pipe_flange",
            _cut_holes(
                _ring(42.0, shaft_radius + 1.0, 5.0),
                flange_holes,
                7.0,
            ),
            ["flange_hole_pattern", "shaft_clearance_hole"],
            "pipe flange that geometrically matches the end face",
        ),
    ]
    relations = {
        "engineering_name": "Flanged cartridge bearing support with retainer",
        "functional_mates": [
            {
                "part_ids": ["P01", "P02"],
                "roles": ["cartridge_housing", "bearing"],
                "mate_type": "outer_race_cartridge_seat",
                "interface_types": [
                    "housing_bore",
                    "bearing_outer_race",
                ],
                "functional_relation": (
                    "The cartridge bore locates the bearing outer race."
                ),
                "independent_evidence": [
                    "outer_race_radius_fit",
                    "axial_seat_depth",
                ],
            },
            {
                "part_ids": ["P02", "P03"],
                "roles": ["bearing", "stepped_shaft"],
                "mate_type": "inner_race_shaft_fit",
                "interface_types": [
                    "bearing_inner_race",
                    "shaft_bearing_seat",
                ],
                "functional_relation": (
                    "The bearing supports the shouldered rotating shaft."
                ),
                "independent_evidence": [
                    "inner_race_radius_fit",
                    "shaft_shoulder_location",
                ],
            },
            {
                "part_ids": ["P01", "P04", "P05"],
                "roles": [
                    "cartridge_housing",
                    "end_cover",
                    "bearing_retainer",
                ],
                "mate_type": "registered_end_retention",
                "interface_types": [
                    "end_cover_register",
                    "flange_hole_pattern",
                    "outer_race_retaining_face",
                ],
                "functional_relation": (
                    "The registered cover and retainer close the outer-race "
                    "axial stack."
                ),
                "independent_evidence": [
                    "register_fit",
                    "four_hole_pattern",
                    "retaining_face_contact",
                ],
            },
        ],
        "interchangeable_parts": [],
        "invalid_role_combinations": [
            ["cartridge_housing", "pipe_flange"],
            ["bearing", "plain_spacer"],
        ],
    }
    return parts, negatives, relations


HOLDOUT_BUILDERS: dict[
    str, Callable[[], tuple[list[Part], list[NegativePart], dict[str, Any]]]
] = {
    "cover_base": _round_cover_holdout,
    "shaft_hub_key": _retained_key_drive_holdout,
    "bearing_housing": _cartridge_bearing_holdout,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate_case(
    output: Path,
    family: str,
    builder: Callable[
        [], tuple[list[Part], list[NegativePart], dict[str, Any]]
    ],
) -> dict[str, Any]:
    case_id = {
        "cover_base": "holdout_round_cover_01",
        "shaft_hub_key": "holdout_retained_key_drive_01",
        "bearing_housing": "holdout_cartridge_bearing_01",
    }[family]
    case_dir = output / case_id
    (case_dir / "parts").mkdir(parents=True)
    (case_dir / "negatives").mkdir()
    parts, negatives, relations = builder()
    stats: dict[str, dict[str, Any]] = {}
    for part in parts:
        stats[part.part_id] = _shape_stats(part.shape)
        _write_step(
            part.shape, case_dir / "parts" / f"{part.part_id}.step"
        )
    for negative in negatives:
        stats[negative.part_id] = _shape_stats(negative.shape)
        _write_step(
            negative.shape,
            case_dir / "negatives" / f"{negative.part_id}.step",
        )
    _write_step(_compound(parts), case_dir / "assembly_gt.step")
    metadata = _metadata(
        case_id, family, parts, negatives, relations, stats
    )
    metadata["holdout_policy"] = {
        "used_for_rule_tuning": False,
        "topology_seen_in_training_benchmark": False,
        "engineer_signoff_required": True,
    }
    if family == "shaft_hub_key":
        metadata["optional_parts"] = ["P04"]
        metadata["valid_groups"] = [
            ["P01", "P02", "P03"],
            ["P01", "P02", "P03", "P04"],
        ]
    failures = validate_metadata(metadata, case_dir)
    metadata["validation"] = {
        "status": "passed" if not failures else "failed",
        "failure_reasons": failures,
    }
    if failures:
        raise ValueError(f"{case_id}:{failures}")
    _render(
        [
            (part.part_id, _translate(part.shape, part.placement))
            for part in parts
        ],
        case_dir / "preview.png",
        relations["engineering_name"],
    )
    exploded_shapes = []
    spacing = 75.0
    center_offset = spacing * (len(parts) - 1) / 2.0
    for index, part in enumerate(parts):
        exploded_shapes.append(
            (
                part.part_id,
                _translate(
                    part.shape,
                    [
                        part.placement[0]
                        + index * spacing
                        - center_offset,
                        part.placement[1],
                        part.placement[2],
                    ],
                ),
            )
        )
    _render(
        exploded_shapes,
        case_dir / "exploded_preview.png",
        f"{relations['engineering_name']} | exploded parts",
    )
    review_assets = {
        "positive": {
            "assembled": "preview.png",
            "exploded": "exploded_preview.png",
        },
        "negatives": {},
    }
    part_by_id = {part.part_id: part for part in parts}
    negative_by_id = {
        negative.part_id: negative for negative in negatives
    }
    for negative_group in metadata["negative_groups"]:
        left_id, right_id = negative_group["parts"]
        left_part = part_by_id[left_id]
        right_part = negative_by_id[right_id]
        preview_name = (
            f"review_{negative_group['negative_id'].lower()}.png"
        )
        _render(
            [
                (
                    left_id,
                    _translate(left_part.shape, [-45.0, 0.0, 0.0]),
                ),
                (
                    right_id,
                    _translate(right_part.shape, [45.0, 0.0, 0.0]),
                ),
            ],
            case_dir / preview_name,
            (
                f"{case_id} | "
                f"{negative_group['negative_type']} candidate"
            ),
        )
        review_assets["negatives"][
            negative_group["negative_id"]
        ] = preview_name
    metadata["engineering_review_assets"] = review_assets
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "case_id": case_id,
        "assembly_family": family,
        "engineering_name": relations["engineering_name"],
        "part_count": len(parts),
        "negative_count": len(negatives),
        "preview": str((case_dir / "preview.png").resolve()),
    }


def _write_review_pack(output: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "sample_id",
        "assembly_family",
        "proposed_label",
        "assembled_asset",
        "exploded_asset",
        "negative_pair_asset",
        "engineer_verdict",
        "recognizable_engineering_system",
        "functional_validity_confirmed",
        "reviewer_name",
        "mechanical_engineering_qualification",
        "review_date",
        "comments",
        "signature",
    ]
    rows = []
    for case in cases:
        metadata = json.loads(
            (output / case["case_id"] / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "sample_id": "POSITIVE",
                "assembly_family": case["assembly_family"],
                "proposed_label": "functionally_valid_positive",
                "assembled_asset": str(
                    (
                        output
                        / case["case_id"]
                        / metadata["engineering_review_assets"]["positive"][
                            "assembled"
                        ]
                    ).resolve()
                ),
                "exploded_asset": str(
                    (
                        output
                        / case["case_id"]
                        / metadata["engineering_review_assets"]["positive"][
                            "exploded"
                        ]
                    ).resolve()
                ),
            }
        )
        for negative in metadata["negative_groups"]:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "sample_id": negative["negative_id"],
                    "assembly_family": case["assembly_family"],
                    "proposed_label": negative["negative_type"],
                    "negative_pair_asset": str(
                        (
                            output
                            / case["case_id"]
                            / metadata["engineering_review_assets"][
                                "negatives"
                            ][negative["negative_id"]]
                        ).resolve()
                    ),
                }
            )
    with (output / "ENGINEERING_REVIEW_FORM.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    (output / "ENGINEERING_REVIEW_INSTRUCTIONS.md").write_text(
        "\n".join(
            [
                "# Mechanical Engineering Holdout Review",
                "",
                "This holdout is locked and must not be used for threshold "
                "or ranking changes before sign-off.",
                "",
                "For every row in `ENGINEERING_REVIEW_FORM.csv`:",
                "",
                "1. Open the assembled, exploded, or negative-pair PNG and "
                "the corresponding STEP files.",
                "2. Set `engineer_verdict` to `confirm`, `reject`, or `uncertain`.",
                "3. Record whether the positive resembles a real engineering "
                "assembly or whether the negative is functionally invalid.",
                "4. Fill reviewer identity, mechanical-engineering "
                "qualification, date, comments, and signature.",
                "",
                "All 12 rows require a non-uncertain verdict before the "
                "engineering sign-off gate can pass.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "engineering_signoff.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "pending_external_mechanical_engineer_review",
                "required_review_rows": len(rows),
                "confirmed_rows": 0,
                "gate_passed": False,
                "reason": (
                    "No qualified mechanical engineer signature has been "
                    "provided."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generate(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    cases = [
        _generate_case(output, family, HOLDOUT_BUILDERS[family])
        for family in FAMILIES
    ]
    _contact_sheet(cases, output / "holdout_contact_sheet.png")
    _write_review_pack(output, cases)
    locked_files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name
        not in {
            "ENGINEERING_REVIEW_FORM.csv",
            "engineering_signoff.json",
            "holdout_lock.json",
            "dataset_manifest.json",
        }
    )
    lock = {
        "schema_version": "1.0.0",
        "policy": "evaluation_only_never_tune_on_holdout",
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): _sha256(path)
            for path in locked_files
        },
    }
    (output / "holdout_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "functional_cad_holdout_v1",
        "dataset_purpose": (
            "Topology-varied modeled CAD holdout requiring external "
            "mechanical-engineer confirmation"
        ),
        "truth_basis": "functional_validity_pending_engineer_confirmation",
        "used_for_rule_tuning": False,
        "deepseek_enabled": False,
        "case_count": len(cases),
        "review_sample_count": len(cases) * 4,
        "cases": cases,
        "engineering_signoff_status": (
            "pending_external_mechanical_engineer_review"
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "functional_cad_holdout_v1"
        ),
    )
    args = parser.parse_args()
    manifest = generate(args.output_root)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
