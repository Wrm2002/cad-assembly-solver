"""Diverse parameterized mechanical-family specifications."""

from __future__ import annotations

import random


FAMILIES_BY_SIZE = {
    1: ("single_part",),
    2: ("shaft_bore", "pin_bushing", "flange_pair"),
    3: ("shaft_bearing_housing", "rotor_support", "stem_guide"),
    4: ("cover_base", "flange_stack", "clamp_module"),
    5: ("small_gearbox", "bearing_cartridge", "pump_end_module"),
    6: ("valve_like", "drive_train", "pump_module"),
}

FAMILY_SYSTEM_CLASS = {
    "single_part": "generic_component",
    "shaft_bore": "rotary_support",
    "pin_bushing": "pinned_joint",
    "flange_pair": "piping_connection",
    "shaft_bearing_housing": "rotary_support",
    "rotor_support": "rotary_support",
    "stem_guide": "flow_control",
    "cover_base": "enclosure",
    "flange_stack": "piping_connection",
    "clamp_module": "structural_clamp",
    "small_gearbox": "power_transmission",
    "bearing_cartridge": "rotary_support",
    "pump_end_module": "fluid_machine",
    "valve_like": "flow_control",
    "drive_train": "power_transmission",
    "pump_module": "fluid_machine",
}


def _part(
    name,
    shape,
    *,
    functional_role,
    functions,
    expected_interfaces,
    **parameters,
):
    return {
        "name": name,
        "shape": shape,
        "functional_role": functional_role,
        "functions": list(functions),
        "expected_interfaces": list(expected_interfaces),
        **parameters,
    }


def _roles_for_family(family):
    profiles = {
        "single_part": [
            ("inspection_blank", "cylinder", "generic_rotor", ["rotation"], ["outer_cylindrical_fit"]),
        ],
        "shaft_bore": [
            ("shaft", "cylinder", "drive_shaft", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("hub", "ring", "hub_sleeve", ["support_rotation"], ["inner_cylindrical_fit", "planar_mount"]),
        ],
        "pin_bushing": [
            ("pin", "cylinder", "pivot_pin", ["locate_joint"], ["outer_cylindrical_fit"]),
            ("bushing", "ring", "plain_bushing", ["reduce_friction"], ["inner_cylindrical_fit", "planar_mount"]),
        ],
        "flange_pair": [
            ("spigot", "cylinder", "centering_spigot", ["center_connection"], ["outer_cylindrical_fit"]),
            ("flange", "flange", "mounting_flange", ["connect_piping"], ["inner_cylindrical_fit", "bolt_pattern", "planar_mount"]),
        ],
        "shaft_bearing_housing": [
            ("shaft", "cylinder", "drive_shaft", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("bearing", "ring", "bearing_sleeve", ["radial_support"], ["inner_cylindrical_fit", "outer_cylindrical_fit"]),
            ("housing", "housing_bore", "bearing_housing", ["locate_bearing"], ["inner_cylindrical_fit", "planar_mount"]),
        ],
        "rotor_support": [
            ("rotor", "cylinder", "rotor_shaft", ["rotate"], ["outer_cylindrical_fit"]),
            ("journal", "ring", "journal_bearing", ["radial_support"], ["inner_cylindrical_fit"]),
            ("pedestal", "housing_bore", "support_pedestal", ["mount_bearing"], ["inner_cylindrical_fit", "planar_mount"]),
        ],
        "stem_guide": [
            ("stem", "cylinder", "valve_stem", ["transmit_linear_motion"], ["outer_cylindrical_fit"]),
            ("guide", "ring", "stem_guide", ["guide_translation"], ["inner_cylindrical_fit"]),
            ("bonnet", "flange", "valve_bonnet", ["seal_pressure_boundary"], ["inner_cylindrical_fit", "bolt_pattern"]),
        ],
        "cover_base": [
            ("shaft", "cylinder", "input_shaft", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("bore", "ring", "shaft_sleeve", ["radial_support"], ["inner_cylindrical_fit"]),
            ("base", "housing_bore", "enclosure_base", ["support_components"], ["inner_cylindrical_fit", "planar_mount"]),
            ("cover", "plate", "enclosure_cover", ["close_enclosure"], ["bolt_pattern", "planar_mount"]),
        ],
        "flange_stack": [
            ("pilot", "cylinder", "centering_pilot", ["center_connection"], ["outer_cylindrical_fit"]),
            ("hub", "ring", "flange_hub", ["carry_load"], ["inner_cylindrical_fit"]),
            ("flange", "flange", "pipe_flange", ["join_sections"], ["bolt_pattern", "planar_mount"]),
            ("gasket_plate", "plate", "gasket_retainer", ["seal_joint"], ["bolt_pattern", "planar_mount"]),
        ],
        "clamp_module": [
            ("pin", "cylinder", "clamp_pin", ["locate_clamp"], ["outer_cylindrical_fit"]),
            ("bushing", "ring", "clamp_bushing", ["guide_pin"], ["inner_cylindrical_fit"]),
            ("body", "housing_bore", "clamp_body", ["react_load"], ["inner_cylindrical_fit", "planar_mount"]),
            ("cap", "plate", "clamp_cap", ["retain_joint"], ["bolt_pattern", "planar_mount"]),
        ],
        "small_gearbox": [
            ("shaft", "cylinder", "gear_shaft", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("bearing", "ring", "shaft_bearing", ["radial_support"], ["inner_cylindrical_fit"]),
            ("case", "housing_bore", "gearbox_case", ["enclose_gears"], ["inner_cylindrical_fit", "planar_mount"]),
            ("end_plate", "plate", "bearing_end_plate", ["retain_bearing"], ["bolt_pattern", "planar_mount"]),
            ("flange", "flange", "mounting_flange", ["mount_gearbox"], ["bolt_pattern", "planar_mount"]),
        ],
        "bearing_cartridge": [
            ("shaft", "cylinder", "machine_shaft", ["rotate"], ["outer_cylindrical_fit"]),
            ("inner_sleeve", "ring", "inner_bearing_sleeve", ["radial_support"], ["inner_cylindrical_fit"]),
            ("cartridge", "housing_bore", "bearing_cartridge", ["locate_bearing"], ["inner_cylindrical_fit"]),
            ("retainer", "flange", "bearing_retainer", ["axial_retention"], ["bolt_pattern", "planar_mount"]),
            ("cover", "plate", "dust_cover", ["exclude_contamination"], ["bolt_pattern", "planar_mount"]),
        ],
        "pump_end_module": [
            ("spindle", "cylinder", "pump_spindle", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("sleeve", "ring", "seal_sleeve", ["protect_shaft"], ["inner_cylindrical_fit"]),
            ("casing", "housing_bore", "pump_casing", ["contain_fluid"], ["inner_cylindrical_fit", "planar_mount"]),
            ("seal_flange", "flange", "seal_flange", ["retain_seal"], ["bolt_pattern", "planar_mount"]),
            ("end_cover", "plate", "pump_end_cover", ["close_casing"], ["bolt_pattern", "planar_mount"]),
        ],
        "valve_like": [
            ("stem", "cylinder", "valve_stem", ["actuate_closure"], ["outer_cylindrical_fit"]),
            ("guide", "ring", "stem_guide", ["guide_translation"], ["inner_cylindrical_fit"]),
            ("body", "housing_bore", "valve_body", ["contain_pressure"], ["inner_cylindrical_fit", "planar_mount"]),
            ("bonnet", "flange", "valve_bonnet", ["close_body"], ["bolt_pattern", "planar_mount"]),
            ("gland", "ring", "packing_gland", ["compress_packing"], ["inner_cylindrical_fit", "planar_mount"]),
            ("cap", "plate", "bonnet_cap", ["retain_gland"], ["bolt_pattern", "planar_mount"]),
        ],
        "drive_train": [
            ("shaft", "cylinder", "drive_shaft", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("bearing", "ring", "shaft_bearing", ["radial_support"], ["inner_cylindrical_fit"]),
            ("housing", "housing_bore", "bearing_housing", ["locate_bearing"], ["inner_cylindrical_fit"]),
            ("coupling", "flange", "shaft_coupling", ["connect_shafts"], ["bolt_pattern", "planar_mount"]),
            ("spacer", "ring", "axial_spacer", ["set_axial_position"], ["inner_cylindrical_fit", "planar_mount"]),
            ("guard", "plate", "coupling_guard", ["protect_rotating_parts"], ["bolt_pattern", "planar_mount"]),
        ],
        "pump_module": [
            ("shaft", "cylinder", "pump_shaft", ["transmit_torque"], ["outer_cylindrical_fit"]),
            ("sleeve", "ring", "shaft_sleeve", ["protect_shaft"], ["inner_cylindrical_fit"]),
            ("casing", "housing_bore", "pump_casing", ["contain_fluid"], ["inner_cylindrical_fit"]),
            ("backplate", "plate", "pump_backplate", ["support_seal"], ["bolt_pattern", "planar_mount"]),
            ("flange", "flange", "connection_flange", ["connect_pipe"], ["bolt_pattern", "planar_mount"]),
            ("cover", "plate", "inspection_cover", ["provide_access"], ["bolt_pattern", "planar_mount"]),
        ],
    }
    return profiles[family]


def build_case_spec(group_size, seed, family=None):
    if group_size not in range(1, 7):
        raise ValueError("group_size must be 1..6")
    rng = random.Random(seed)
    families = FAMILIES_BY_SIZE[group_size]
    family = family or families[seed % len(families)]
    if family not in families:
        raise ValueError(f"family {family!r} is not valid for size {group_size}")

    shaft_radius = rng.uniform(8.0, 18.0)
    clearance = rng.uniform(0.12, 0.48)
    main_depth = rng.uniform(18.0, 34.0)
    bolt_count = rng.choice((4, 6))
    profiles = _roles_for_family(family)
    parts = []
    for index, (name, shape, role, functions, interfaces) in enumerate(profiles):
        depth = main_depth if index < 3 else rng.uniform(5.0, 12.0)
        common = {
            "functional_role": role,
            "functions": functions,
            "expected_interfaces": interfaces,
            "depth": depth,
        }
        if shape == "cylinder":
            part = _part(name, shape, radius=shaft_radius, **common)
        elif shape == "ring":
            part = _part(
                name,
                shape,
                inner_radius=shaft_radius + clearance,
                outer_radius=shaft_radius + rng.uniform(6.0, 13.0),
                **common,
            )
        elif shape == "flange":
            outer = shaft_radius + rng.uniform(18.0, 28.0)
            part = _part(
                name,
                shape,
                inner_radius=shaft_radius + clearance,
                outer_radius=outer,
                bolt_count=bolt_count,
                bolt_hole_radius=rng.uniform(2.0, 3.5),
                bolt_circle_radius=outer * 0.72,
                **common,
            )
        elif shape == "housing_bore":
            width = rng.uniform(58.0, 78.0)
            height = rng.uniform(52.0, 70.0)
            part = _part(
                name,
                shape,
                width=width,
                height=height,
                bore_radius=shaft_radius + clearance,
                **common,
            )
        elif shape == "plate":
            width = rng.uniform(58.0, 78.0)
            height = rng.uniform(52.0, 70.0)
            part = _part(
                name,
                shape,
                width=width,
                height=height,
                bolt_count=bolt_count,
                bolt_hole_radius=rng.uniform(2.0, 3.5),
                bolt_circle_radius=min(width, height) * 0.35,
                **common,
            )
        else:
            raise ValueError(f"unsupported family shape: {shape}")
        parts.append(part)

    placements = {}
    axial_end = main_depth / 2.0
    for index, part in enumerate(parts):
        if index < 2:
            center_z = 0.0
        else:
            center_z = axial_end + float(part["depth"]) / 2.0
            axial_end += float(part["depth"])
        placements[f"part_{index + 1:02d}.step"] = {
            "translate": [0.0, 0.0, center_z],
            "rotate": [],
        }
    true_mates = []
    if group_size >= 2:
        true_mates.append({
            "part_a": "part_01.step",
            "part_b": "part_02.step",
            "type": "clearance",
            "feature_a": "outer_cylinder",
            "feature_b": "bore_cylinder",
        })
    for index in range(2, group_size):
        true_mates.append({
            "part_a": f"part_{index:02d}.step",
            "part_b": f"part_{index + 1:02d}.step",
            "type": "planar_mate",
            "feature_a": "end_face",
            "feature_b": "start_face",
        })
    part_semantics = {
        f"part_{index + 1:02d}.step": {
            "semantic_schema_version": "1.0.0",
            "part_category": part["functional_role"],
            "functions": part["functions"],
            "expected_interfaces": part["expected_interfaces"],
            "system_class": FAMILY_SYSTEM_CLASS[family],
            "source": "generator_grounded_metadata",
            "group_identity_disclosed": False,
        }
        for index, part in enumerate(parts)
    }
    return {
        "group_size": group_size,
        "template": family,
        "dataset_intended_use": "geometry_smoke_only",
        "functional_positive_eligible": False,
        "functional_positive_exclusion_reason": (
            "Legacy primitive composition lacks validated functional assembly "
            "structure and must not be used as a D0 positive."
        ),
        "system_class": FAMILY_SYSTEM_CLASS[family],
        "parts": parts,
        "part_semantics": part_semantics,
        "placements": placements,
        "true_mates": true_mates,
        "parameters": {
            "shaft_radius": shaft_radius,
            "clearance": clearance,
            "depth": main_depth,
            "bolt_count": bolt_count,
        },
        "hard_negatives": [
            "near-radius cylinders",
            "similar-area planar faces",
            "cross-family role-compatible distractor",
            "bolt-pattern ambiguity",
            "randomized dimensions",
        ],
    }
