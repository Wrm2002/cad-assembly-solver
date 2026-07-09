"""Generate a small, function-grounded STEP assembly benchmark.

The generator intentionally supports only the three D0 families.  It creates
real interface features (registers, repeated holes, keyways, bearing seats),
not anonymous primitive stacks.  Ground truth is functional validity rather
than source-case identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from build_human_semantic_review_pack import _render


SCHEMA_VERSION = "1.0.0"
FAMILIES = ("cover_base", "shaft_hub_key", "bearing_housing")
COLORS = {
    "base": "#4C78A8",
    "cover": "#F58518",
    "locating_pin": "#54A24B",
    "shaft": "#4C78A8",
    "hub": "#F58518",
    "key": "#54A24B",
    "housing": "#4C78A8",
    "bearing": "#F58518",
    "end_cover": "#54A24B",
}


@dataclass
class Part:
    part_id: str
    role: str
    shape: Any
    interface_types: list[str]
    placement: list[float]
    functions: list[str]
    name: str


@dataclass
class NegativePart:
    part_id: str
    role: str
    shape: Any
    interface_types: list[str]
    name: str


def _box(
    dx: float,
    dy: float,
    dz: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
):
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def _cylinder(
    radius: float,
    height: float,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
):
    axis = gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0.0, 0.0, 1.0))
    return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()


def _cut(shape, tool):
    operation = BRepAlgoAPI_Cut(shape, tool)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError("OCCT cut failed")
    return operation.Shape()


def _fuse(first, second):
    operation = BRepAlgoAPI_Fuse(first, second)
    operation.Build()
    if not operation.IsDone():
        raise RuntimeError("OCCT fuse failed")
    return operation.Shape()


def _ring(outer_radius: float, inner_radius: float, height: float):
    return _cut(
        _cylinder(outer_radius, height),
        _cylinder(inner_radius, height),
    )


def _translate(shape, xyz: list[float]):
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(*[float(value) for value in xyz]))
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _cut_holes(shape, holes: list[tuple[float, float, float]], height: float):
    result = shape
    for x, y, radius in holes:
        result = _cut(result, _cylinder(radius, height, x, y, -1.0))
    return result


def _bolt_pattern(
    radius: float, count: int, hole_radius: float
) -> list[tuple[float, float, float]]:
    return [
        (
            radius * math.cos(2.0 * math.pi * index / count),
            radius * math.sin(2.0 * math.pi * index / count),
            hole_radius,
        )
        for index in range(count)
    ]


def _shape_stats(shape) -> dict[str, Any]:
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        raise ValueError("invalid_generated_shape")
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_count = 0
    while explorer.More():
        solid_count += 1
        explorer.Next()
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    volume = float(properties.Mass())
    if solid_count < 1 or volume <= 0.0:
        raise ValueError(
            f"empty_generated_shape:solids={solid_count}:volume={volume}"
        )
    return {
        "solid_count": solid_count,
        "volume_mm3": round(volume, 6),
        "shape_valid": True,
    }


def _write_step(shape, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"empty STEP output: {path}")


def _compound(parts: list[Part]):
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for part in parts:
        builder.Add(compound, _translate(part.shape, part.placement))
    return compound


def _cover_base(variant: int) -> tuple[list[Part], list[NegativePart], dict]:
    width = 78.0 + 4.0 * variant
    depth = 58.0 + 2.0 * variant
    base_height = 12.0
    wall = 5.0
    cover_thickness = 5.0
    bolt_radius = min(width, depth) * 0.37
    holes = _bolt_pattern(bolt_radius, 4, 2.2)
    locating = [(-width * 0.28, 0.0, 2.55), (width * 0.28, 0.0, 2.55)]

    base = _box(width, depth, base_height, -width / 2, -depth / 2, 0.0)
    cavity = _box(
        width - 2 * wall,
        depth - 2 * wall,
        base_height - 4.0,
        -width / 2 + wall,
        -depth / 2 + wall,
        4.0,
    )
    base = _cut(base, cavity)
    base = _cut_holes(base, holes + locating, base_height + 2.0)

    cover = _box(
        width, depth, cover_thickness, -width / 2, -depth / 2, 0.0
    )
    cover = _cut_holes(cover, holes + locating, cover_thickness + 2.0)
    register_outer = _box(
        width - 2 * wall - 0.6,
        depth - 2 * wall - 0.6,
        2.0,
        -width / 2 + wall + 0.3,
        -depth / 2 + wall + 0.3,
        -2.0,
    )
    register_inner = _box(
        width - 2 * wall - 4.0,
        depth - 2 * wall - 4.0,
        2.2,
        -width / 2 + wall + 2.0,
        -depth / 2 + wall + 2.0,
        -2.1,
    )
    cover = _fuse(cover, _cut(register_outer, register_inner))

    pin_shape = _cylinder(2.35, 9.0)
    parts = [
        Part(
            "P01",
            "base",
            base,
            [
                "planar_seating",
                "rectangular_register",
                "fastener_hole_pattern",
                "locating_pin_hole",
            ],
            [0.0, 0.0, 0.0],
            ["support_components", "locate_cover"],
            "machined enclosure base",
        ),
        Part(
            "P02",
            "cover",
            cover,
            [
                "planar_seating",
                "rectangular_register",
                "fastener_hole_pattern",
                "locating_pin_hole",
            ],
            [0.0, 0.0, base_height],
            ["close_enclosure", "protect_internal_components"],
            "registered enclosure cover",
        ),
        Part(
            "P03",
            "locating_pin",
            pin_shape,
            ["dowel_pin", "cylindrical_pin_fit"],
            [locating[0][0], locating[0][1], base_height - 3.0],
            ["locate_cover"],
            "left locating dowel",
        ),
        Part(
            "P04",
            "locating_pin",
            pin_shape,
            ["dowel_pin", "cylindrical_pin_fit"],
            [locating[1][0], locating[1][1], base_height - 3.0],
            ["locate_cover"],
            "right locating dowel",
        ),
    ]
    negatives = [
        NegativePart(
            "N01",
            "oversize_pin",
            _cylinder(3.4, 9.0),
            ["oversize_cylindrical_pin"],
            "oversize dowel that cannot enter the locating hole",
        ),
        NegativePart(
            "N02",
            "service_plate",
            _cut(
                _box(38.0, 28.0, 4.0, -19.0, -14.0, 0.0),
                _cylinder(2.55, 6.0, 0.0, 0.0, -1.0),
            ),
            ["single_planar_contact", "single_round_hole"],
            "unrelated service plate with one coincident hole",
        ),
        NegativePart(
            "N03",
            "shaft",
            _cylinder(2.3, 25.0),
            ["cylindrical_shaft"],
            "unrelated slender shaft that happens to fit a pin hole",
        ),
    ]
    relations = {
        "engineering_name": "Dowel-located fastened enclosure cover",
        "functional_mates": [
            {
                "part_ids": ["P01", "P02"],
                "roles": ["base", "cover"],
                "mate_type": "planar_registered_fastening",
                "interface_types": [
                    "planar_seating",
                    "rectangular_register",
                    "fastener_hole_pattern",
                ],
                "functional_relation": (
                    "The cover closes the base and is located by the register "
                    "and repeated fastener pattern."
                ),
                "independent_evidence": [
                    "planar_seating",
                    "rectangular_register",
                    "fastener_hole_pattern",
                ],
            },
            {
                "part_ids": ["P01", "P02", "P03", "P04"],
                "roles": ["base", "cover", "locating_pin", "locating_pin"],
                "mate_type": "dual_dowel_location",
                "interface_types": ["locating_pin_hole", "dowel_pin"],
                "functional_relation": (
                    "Two interchangeable dowels prevent lateral cover shift."
                ),
                "independent_evidence": [
                    "left_dowel_fit",
                    "right_dowel_fit",
                ],
            },
        ],
        "interchangeable_parts": [["P03", "P04"]],
        "invalid_role_combinations": [
            ["shaft", "cover"],
            ["bearing", "service_plate"],
        ],
    }
    return parts, negatives, relations


def _shaft_hub_key(
    variant: int,
) -> tuple[list[Part], list[NegativePart], dict]:
    shaft_radius = 11.0 + 0.7 * variant
    shaft_length = 70.0
    key_width = 6.0
    key_height = 6.0
    hub_inner = shaft_radius + 0.25
    hub_outer = 27.0 + variant
    hub_height = 20.0
    hub_z = 25.0

    shaft = _cylinder(shaft_radius, shaft_length)
    shaft_slot = _box(
        5.0,
        key_width,
        hub_height + 8.0,
        shaft_radius - 3.0,
        -key_width / 2,
        hub_z - 4.0,
    )
    shaft = _cut(shaft, shaft_slot)

    hub = _ring(hub_outer, hub_inner, hub_height)
    hub_slot = _box(
        7.0,
        key_width + 0.25,
        hub_height + 2.0,
        shaft_radius - 0.2,
        -(key_width + 0.25) / 2,
        -1.0,
    )
    hub = _cut(hub, hub_slot)
    key = _box(
        key_height,
        key_width - 0.2,
        hub_height - 2.0,
        -key_height / 2,
        -(key_width - 0.2) / 2,
        -(hub_height - 2.0) / 2,
    )

    parts = [
        Part(
            "P01",
            "shaft",
            shaft,
            ["cylindrical_shaft", "shaft_keyway"],
            [0.0, 0.0, 0.0],
            ["transmit_torque", "support_hub"],
            "keyed transmission shaft",
        ),
        Part(
            "P02",
            "hub",
            hub,
            ["hub_bore", "hub_keyway", "axial_hub_face"],
            [0.0, 0.0, hub_z],
            ["receive_torque", "mount_rotating_element"],
            "keyed wheel hub",
        ),
        Part(
            "P03",
            "key",
            key,
            ["parallel_key", "shaft_keyway", "hub_keyway"],
            [shaft_radius, 0.0, hub_z + hub_height / 2],
            ["transfer_torque"],
            "parallel machine key",
        ),
    ]
    negatives = [
        NegativePart(
            "N01",
            "undersize_bore_collar",
            _ring(hub_outer - 3.0, shaft_radius - 1.5, hub_height),
            ["undersize_hub_bore"],
            "collar whose bore cannot accept the shaft",
        ),
        NegativePart(
            "N02",
            "plain_hub",
            _ring(hub_outer, hub_inner, hub_height),
            ["hub_bore", "single_coaxial_fit"],
            "plain hub that fits the shaft but has no torque-transfer keyway",
        ),
        NegativePart(
            "N03",
            "chassis_lid",
            _cut(
                _box(58.0, 58.0, 4.0, -29.0, -29.0, 0.0),
                _cylinder(hub_inner, 6.0, 0.0, 0.0, -1.0),
            ),
            ["central_round_hole", "planar_seating"],
            "chassis lid with a coincident shaft-sized hole",
        ),
    ]
    relations = {
        "engineering_name": "Parallel-key shaft and hub torque drive",
        "functional_mates": [
            {
                "part_ids": ["P01", "P02"],
                "roles": ["shaft", "hub"],
                "mate_type": "coaxial_insert",
                "interface_types": ["cylindrical_shaft", "hub_bore"],
                "functional_relation": "The shaft centers and supports the hub.",
                "independent_evidence": [
                    "coaxial_radius_fit",
                    "axial_engagement",
                ],
            },
            {
                "part_ids": ["P01", "P02", "P03"],
                "roles": ["shaft", "hub", "key"],
                "mate_type": "keyed_torque_transfer",
                "interface_types": [
                    "shaft_keyway",
                    "hub_keyway",
                    "parallel_key",
                ],
                "functional_relation": (
                    "The paired keyways and key transfer torque from shaft to hub."
                ),
                "independent_evidence": [
                    "paired_keyway_width",
                    "key_radial_engagement",
                ],
            },
        ],
        "interchangeable_parts": [],
        "invalid_role_combinations": [
            ["hub", "chassis_lid"],
            ["key", "cover"],
        ],
    }
    return parts, negatives, relations


def _bearing_housing(
    variant: int,
) -> tuple[list[Part], list[NegativePart], dict]:
    shaft_radius = 7.5 + 0.4 * variant
    bearing_inner = shaft_radius + 0.2
    bearing_outer = 19.0 + variant
    housing_bore = bearing_outer + 0.25
    width = 64.0 + 2.0 * variant
    depth = 54.0
    height = 24.0
    bolt_pattern = [
        (-width * 0.38, -depth * 0.34, 2.2),
        (width * 0.38, -depth * 0.34, 2.2),
        (-width * 0.38, depth * 0.34, 2.2),
        (width * 0.38, depth * 0.34, 2.2),
    ]

    housing = _box(width, depth, height, -width / 2, -depth / 2, 0.0)
    housing = _cut(housing, _cylinder(housing_bore, height + 2.0, 0, 0, -1))
    housing = _cut(
        housing,
        _cylinder(bearing_outer + 2.0, 4.0, 0.0, 0.0, height - 4.0),
    )
    housing = _cut_holes(housing, bolt_pattern, height + 2.0)
    left_foot = _box(14.0, depth + 12.0, 5.0, -width / 2 - 5.0, -depth / 2 - 6.0, 0.0)
    right_foot = _box(14.0, depth + 12.0, 5.0, width / 2 - 9.0, -depth / 2 - 6.0, 0.0)
    housing = _fuse(_fuse(housing, left_foot), right_foot)

    bearing = _ring(bearing_outer, bearing_inner, 12.0)
    shaft = _cylinder(shaft_radius, 70.0)
    cover = _box(width, depth, 5.0, -width / 2, -depth / 2, 0.0)
    cover = _cut(cover, _cylinder(shaft_radius + 1.0, 7.0, 0, 0, -1))
    cover = _cut_holes(cover, bolt_pattern, 7.0)
    register = _ring(bearing_outer + 1.6, shaft_radius + 1.0, 2.0)
    cover = _fuse(cover, _translate(register, [0.0, 0.0, -2.0]))

    parts = [
        Part(
            "P01",
            "housing",
            housing,
            [
                "housing_bore",
                "bearing_outer_race_seat",
                "end_cover_register",
                "cover_fastener_pattern",
            ],
            [0.0, 0.0, 0.0],
            ["locate_bearing", "react_radial_load"],
            "foot-mounted bearing housing",
        ),
        Part(
            "P02",
            "bearing",
            bearing,
            ["bearing_outer_race", "bearing_inner_race"],
            [0.0, 0.0, 6.0],
            ["support_rotating_shaft", "reduce_friction"],
            "simplified rolling bearing",
        ),
        Part(
            "P03",
            "shaft",
            shaft,
            ["shaft_bearing_seat", "cylindrical_shaft"],
            [0.0, 0.0, -20.0],
            ["rotate", "transmit_torque"],
            "bearing-supported shaft",
        ),
        Part(
            "P04",
            "end_cover",
            cover,
            [
                "end_cover_register",
                "cover_fastener_pattern",
                "shaft_clearance_hole",
            ],
            [0.0, 0.0, height],
            ["retain_bearing", "close_housing"],
            "registered bearing end cover",
        ),
    ]
    negatives = [
        NegativePart(
            "N01",
            "oversize_bearing",
            _ring(housing_bore + 2.5, bearing_inner, 12.0),
            ["oversize_bearing_outer_race"],
            "bearing whose outer race cannot enter the housing",
        ),
        NegativePart(
            "N02",
            "plain_spacer",
            _ring(bearing_outer - 1.0, bearing_inner, 8.0),
            ["single_coaxial_fit"],
            "plain spacer ring with no bearing or axial-retention function",
        ),
        NegativePart(
            "N03",
            "random_plate",
            _cut(
                _box(width, depth, 4.0, -width / 2, -depth / 2, 0.0),
                _cylinder(shaft_radius + 1.0, 6.0, 0.0, 0.0, -1.0),
            ),
            ["planar_seating", "shaft_clearance_hole"],
            "unrelated plate that can sit on the housing",
        ),
    ]
    relations = {
        "engineering_name": "Shaft bearing support with registered end cover",
        "functional_mates": [
            {
                "part_ids": ["P01", "P02"],
                "roles": ["housing", "bearing"],
                "mate_type": "outer_race_seat",
                "interface_types": [
                    "housing_bore",
                    "bearing_outer_race",
                ],
                "functional_relation": "The housing locates the bearing outer race.",
                "independent_evidence": [
                    "outer_race_radius_fit",
                    "housing_shoulder_retention",
                ],
            },
            {
                "part_ids": ["P02", "P03"],
                "roles": ["bearing", "shaft"],
                "mate_type": "inner_race_shaft_fit",
                "interface_types": [
                    "bearing_inner_race",
                    "shaft_bearing_seat",
                ],
                "functional_relation": "The bearing supports the rotating shaft.",
                "independent_evidence": [
                    "inner_race_radius_fit",
                    "coaxial_axis_alignment",
                ],
            },
            {
                "part_ids": ["P01", "P04"],
                "roles": ["housing", "end_cover"],
                "mate_type": "registered_fastened_cover",
                "interface_types": [
                    "end_cover_register",
                    "cover_fastener_pattern",
                ],
                "functional_relation": "The end cover retains and closes the bearing seat.",
                "independent_evidence": [
                    "register_fit",
                    "four_hole_pattern",
                ],
            },
        ],
        "interchangeable_parts": [],
        "invalid_role_combinations": [
            ["bearing", "random_plate"],
            ["housing", "chassis_lid"],
        ],
    }
    return parts, negatives, relations


BUILDERS: dict[
    str, Callable[[int], tuple[list[Part], list[NegativePart], dict]]
] = {
    "cover_base": _cover_base,
    "shaft_hub_key": _shaft_hub_key,
    "bearing_housing": _bearing_housing,
}


def _negative_groups(family: str, parts: list[Part]) -> list[dict[str, Any]]:
    anchor = parts[0].part_id
    secondary = parts[1].part_id
    semantic_pair = {
        "cover_base": [secondary, "N03"],
        "shaft_hub_key": [anchor, "N03"],
        "bearing_housing": [anchor, "N03"],
    }[family]
    return [
        {
            "negative_id": "NEG_EASY",
            "negative_type": "easy_negative",
            "parts": [anchor, "N01"],
            "geometry_feasible": False,
            "functional_validity": "invalid",
            "reason": "Critical size mismatch prevents insertion.",
        },
        {
            "negative_id": "NEG_GEOMETRIC_HARD",
            "negative_type": "geometric_hard_negative",
            "parts": [anchor, "N02"],
            "geometry_feasible": True,
            "functional_validity": "invalid",
            "weak_evidence_only": True,
            "reason": (
                "The parts admit one weak planar or coaxial interface but do "
                "not complete the required functional structure."
            ),
        },
        {
            "negative_id": "NEG_SEMANTIC_HARD",
            "negative_type": "semantic_hard_negative",
            "parts": semantic_pair,
            "geometry_feasible": True,
            "functional_validity": "invalid",
            "weak_evidence_only": False,
            "reason": "The geometry can be posed, but the functional roles are incompatible.",
        },
    ]


def _metadata(
    case_id: str,
    family: str,
    parts: list[Part],
    negatives: list[NegativePart],
    relations: dict[str, Any],
    stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "assembly_family": family,
        "engineering_name": relations["engineering_name"],
        "truth_basis": "functional_validity",
        "source_id_is_production_truth": False,
        "positive_generation_policy": "function_grounded_template_only",
        "parts": [
            {
                "part_id": part.part_id,
                "file": f"parts/{part.part_id}.step",
                "part_name": part.name,
                "part_role": part.role,
                "interface_types": part.interface_types,
                "functions": part.functions,
                "assembly_placement": {
                    "translate": part.placement,
                    "rotate": [],
                },
                **stats[part.part_id],
            }
            for part in parts
        ],
        "functional_mates": relations["functional_mates"],
        "valid_groups": [[part.part_id for part in parts]],
        "optional_parts": [],
        "interchangeable_parts": relations["interchangeable_parts"],
        "invalid_role_combinations": relations[
            "invalid_role_combinations"
        ],
        "negative_parts": [
            {
                "part_id": part.part_id,
                "file": f"negatives/{part.part_id}.step",
                "part_name": part.name,
                "part_role": part.role,
                "interface_types": part.interface_types,
                **stats[part.part_id],
            }
            for part in negatives
        ],
        "negative_groups": _negative_groups(family, parts),
        "minimum_independent_evidence_count": 2,
        "semantic_review_fields": [
            "part_name",
            "part_role",
            "interface_types",
            "assembly_family",
            "functional_relation",
        ],
        "units": {"length": "mm", "angle": "degree"},
    }


def validate_metadata(metadata: dict[str, Any], case_dir: Path) -> list[str]:
    failures = []
    required = {
        "case_id",
        "assembly_family",
        "engineering_name",
        "parts",
        "functional_mates",
        "valid_groups",
        "optional_parts",
        "interchangeable_parts",
        "invalid_role_combinations",
        "negative_groups",
    }
    missing = sorted(required - set(metadata))
    if missing:
        failures.append(f"missing_metadata_fields:{missing}")
    if metadata.get("assembly_family") not in FAMILIES:
        failures.append("unsupported_assembly_family")
    if metadata.get("truth_basis") != "functional_validity":
        failures.append("truth_basis_not_functional")
    if metadata.get("source_id_is_production_truth") is not False:
        failures.append("source_id_used_as_production_truth")
    part_ids = set()
    for part in metadata.get("parts", []):
        part_ids.add(part.get("part_id"))
        if not part.get("part_role"):
            failures.append(f"missing_part_role:{part.get('part_id')}")
        if not part.get("interface_types"):
            failures.append(f"missing_interface_types:{part.get('part_id')}")
        path = case_dir / str(part.get("file"))
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing_part_file:{path}")
    for mate in metadata.get("functional_mates", []):
        if not mate.get("functional_relation"):
            failures.append("missing_functional_relation")
        if len(mate.get("independent_evidence", [])) < 2:
            failures.append(
                f"insufficient_independent_evidence:{mate.get('mate_type')}"
            )
    for group in metadata.get("valid_groups", []):
        if not set(group) <= part_ids:
            failures.append(f"invalid_valid_group:{group}")
    negative_types = {
        row.get("negative_type")
        for row in metadata.get("negative_groups", [])
    }
    expected_negative_types = {
        "easy_negative",
        "geometric_hard_negative",
        "semantic_hard_negative",
    }
    if negative_types != expected_negative_types:
        failures.append(
            f"negative_type_coverage:{sorted(negative_types)}"
        )
    for part in metadata.get("negative_parts", []):
        path = case_dir / str(part.get("file"))
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"missing_negative_file:{path}")
    return failures


def generate_case(
    output_root: Path,
    family: str,
    variant: int,
    *,
    render: bool = True,
) -> dict[str, Any]:
    if family not in BUILDERS:
        raise ValueError(f"unsupported family: {family}")
    case_id = f"{family}_{variant:02d}"
    case_dir = output_root / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "parts").mkdir(parents=True)
    (case_dir / "negatives").mkdir()
    parts, negatives, relations = BUILDERS[family](variant)
    stats: dict[str, dict[str, Any]] = {}
    for part in parts:
        stats[part.part_id] = _shape_stats(part.shape)
        _write_step(part.shape, case_dir / "parts" / f"{part.part_id}.step")
    for part in negatives:
        stats[part.part_id] = _shape_stats(part.shape)
        _write_step(
            part.shape, case_dir / "negatives" / f"{part.part_id}.step"
        )
    assembly = _compound(parts)
    _shape_stats(assembly)
    _write_step(assembly, case_dir / "assembly_gt.step")
    metadata = _metadata(
        case_id, family, parts, negatives, relations, stats
    )
    failures = validate_metadata(metadata, case_dir)
    metadata["validation"] = {
        "status": "passed" if not failures else "failed",
        "failure_reasons": failures,
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise ValueError(f"{case_id}:{failures}")
    if render:
        _render(
            [
                (part.part_id, _translate(part.shape, part.placement))
                for part in parts
            ],
            case_dir / "preview.png",
            f"{relations['engineering_name']} ({case_id})",
        )
    return {
        "case_id": case_id,
        "assembly_family": family,
        "part_count": len(parts),
        "negative_part_count": len(negatives),
        "metadata": str((case_dir / "metadata.json").resolve()),
        "preview": (
            str((case_dir / "preview.png").resolve()) if render else None
        ),
        "status": "passed",
    }


def _contact_sheet(cases: list[dict[str, Any]], output: Path) -> None:
    cards = []
    for case in cases:
        preview = case.get("preview")
        if not preview:
            continue
        image = Image.open(preview).convert("RGB")
        image.thumbnail((520, 390))
        card = Image.new("RGB", (560, 450), "white")
        card.paste(image, ((560 - image.width) // 2, 20))
        draw = ImageDraw.Draw(card)
        draw.text(
            (20, 418),
            f"{case['case_id']} | {case['assembly_family']}",
            fill="black",
        )
        cards.append(card)
    columns = 3
    rows = math.ceil(len(cards) / columns)
    sheet = Image.new("RGB", (columns * 560, rows * 450), "#e8e8e8")
    for index, card in enumerate(cards):
        sheet.paste(card, ((index % columns) * 560, (index // columns) * 450))
    sheet.save(output)


def generate_dataset(
    output_root: str | Path,
    *,
    variants_per_family: int = 3,
    render: bool = True,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    cases = []
    for variant in range(1, variants_per_family + 1):
        for family in FAMILIES:
            cases.append(generate_case(output, family, variant, render=render))
    if render:
        _contact_sheet(cases, output / "functional_contact_sheet.png")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "functional_assembly_benchmark_v1",
        "dataset_purpose": (
            "Function-grounded positive assemblies and controlled hard negatives"
        ),
        "positive_generation_policy": "function_grounded_template_only",
        "legacy_primitive_stacks_allowed_as_positive": False,
        "truth_basis": "functional_validity",
        "source_id_is_production_truth": False,
        "assembly_families": list(FAMILIES),
        "case_count": len(cases),
        "variants_per_family": variants_per_family,
        "cases": cases,
        "validation": {
            "metadata_valid_case_count": len(cases),
            "failed_case_count": 0,
            "all_parts_have_roles": True,
            "all_mates_have_functional_relations": True,
            "all_mates_have_two_independent_evidence": True,
            "all_negative_tiers_present_per_case": True,
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
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
            / "functional_dataset_v1"
        ),
    )
    parser.add_argument("--variants-per-family", type=int, default=3)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(
        args.output_root,
        variants_per_family=args.variants_per_family,
        render=not args.no_render,
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "case_count": manifest["case_count"],
                "families": manifest["assembly_families"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
