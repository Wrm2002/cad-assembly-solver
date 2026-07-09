"""Generate lightweight proxy STEP files for SolidWorks exam case 4 and case 5.

The original server/PCBA STEP files are too large for quick external scoring.
This script creates deterministic, low-detail proxy models that preserve the
interfaces needed by the assembly-relation task:

- case 4: PCBA + memory module + CPU package
- case 5: chassis + chassis ear + power-supply module

It does not overwrite the original ``sw/4`` or ``sw/5`` files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape


@dataclass(frozen=True)
class PartSpec:
    file_name: str
    part_id: str
    part_role: str
    interface_types: list[str]
    shape: TopoDS_Shape


def box(x: float, y: float, z: float, dx: float, dy: float, dz: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeBox(gp_Pnt(x, y, z), dx, dy, dz).Shape()


def cyl_x(x: float, y: float, z: float, radius: float, length: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(x, y, z), gp_Dir(1, 0, 0)), radius, length
    ).Shape()


def cyl_z(x: float, y: float, z: float, radius: float, length: float) -> TopoDS_Shape:
    return BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1)), radius, length
    ).Shape()


def fuse_all(shapes: Iterable[TopoDS_Shape]) -> TopoDS_Shape:
    shapes = list(shapes)
    if not shapes:
        raise ValueError("fuse_all requires at least one shape")
    result = shapes[0]
    for shape in shapes[1:]:
        op = BRepAlgoAPI_Fuse(result, shape)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("BRep fuse failed")
        result = op.Shape()
    return result


def cut_all(base: TopoDS_Shape, tools: Iterable[TopoDS_Shape]) -> TopoDS_Shape:
    result = base
    for tool in tools:
        op = BRepAlgoAPI_Cut(result, tool)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("BRep cut failed")
        result = op.Shape()
    return result


def compound(shapes: Iterable[TopoDS_Shape]) -> TopoDS_Shape:
    builder = BRep_Builder()
    result = TopoDS_Compound()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


def save_step(shape: TopoDS_Shape, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP242")
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError(f"STEP transfer failed: {path}")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")


def check_valid(shape: TopoDS_Shape, label: str) -> None:
    if shape.IsNull():
        raise RuntimeError(f"{label}: null shape")
    if not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError(f"{label}: invalid B-Rep")


def build_case4() -> tuple[list[PartSpec], dict]:
    # Board in assembled coordinates.  It intentionally contains both a CPU
    # socket frame and a memory-slot pocket, so the target relations are
    # supported by pocket + planar evidence rather than by arbitrary proximity.
    board = box(0, 0, 0, 180, 120, 2)
    board = cut_all(
        board,
        [
            cyl_z(15, 15, -1, 3, 5),
            cyl_z(165, 15, -1, 3, 5),
            cyl_z(15, 105, -1, 3, 5),
            cyl_z(165, 105, -1, 3, 5),
            box(40, 88, 1.1, 110, 20, 2.0),  # memory pocket
            box(72, 32, 1.1, 46, 46, 2.0),  # CPU socket pocket
        ],
    )
    pcba = fuse_all(
        [
            board,
            box(38, 86, 2, 114, 2.8, 4),  # memory slot rail
            box(38, 108, 2, 114, 2.8, 4),
            box(70, 30, 2, 50, 3, 4),  # CPU socket frame
            box(70, 78, 2, 50, 3, 4),
            box(70, 30, 2, 3, 51, 4),
            box(117, 30, 2, 3, 51, 4),
            box(5, 5, 2, 16, 8, 3),  # small connector proxy
            box(145, 12, 2, 25, 8, 5),
        ]
    )

    memory = fuse_all(
        [
            box(43, 90, 2.05, 104, 16, 2.2),
            box(47, 89.2, 1.2, 96, 1.2, 1.2),  # contact edge into slot
            box(82, 90, 4.25, 8, 16, 1.6),  # chip proxy
            box(100, 90, 4.25, 8, 16, 1.6),
        ]
    )

    cpu = fuse_all(
        [
            box(76, 36, 2.05, 38, 38, 4.0),
            box(80, 40, 6.05, 30, 30, 1.8),
        ]
    )

    parts = [
        PartSpec(
            "01-62DC24-MLB-PCBA.stp",
            "P01",
            "pcba",
            ["planar_mount_surface", "memory_slot_pocket", "cpu_socket_pocket", "mounting_holes"],
            pcba,
        ),
        PartSpec(
            "5-rd_rc_a_2rx4_1_ddrv.stp",
            "P02",
            "memory_module",
            ["planar_contact", "edge_insert_tongue"],
            memory,
        ),
        PartSpec(
            "Hygon-7400-3D.stp",
            "P03",
            "cpu_package",
            ["planar_contact", "socket_insert_block"],
            cpu,
        ),
    ]
    metadata = {
        "case_id": "case_4_lightweight",
        "source_case": "sw/4",
        "assembly_name": "PCBA with memory module and CPU package",
        "simplification_policy": "proxy model preserving exam-relevant interfaces only",
        "functional_mates": [
            {
                "parts": ["pcba", "memory_module"],
                "mate_type": "pocket_mate + planar_mate",
                "functional_relation": "memory module seats in PCBA memory slot",
            },
            {
                "parts": ["pcba", "cpu_package"],
                "mate_type": "pocket_mate + planar_mate",
                "functional_relation": "CPU package seats in PCBA CPU socket",
            },
        ],
        "expected_non_edges": [["memory_module", "cpu_package"]],
    }
    return parts, metadata


def build_case5() -> tuple[list[PartSpec], dict]:
    # Chassis: base tray + side wall + power bay rails + screw-hole interface.
    chassis_wall = cut_all(
        box(0, 0, 0, 4, 140, 70),
        [
            cyl_x(-1, 48, 28, 4.2, 7),
            cyl_x(-1, 92, 28, 4.2, 7),
        ],
    )
    chassis = fuse_all(
        [
            box(0, 0, 0, 220, 140, 4),  # base tray
            chassis_wall,
            box(0, 0, 0, 220, 4, 55),
            box(0, 136, 0, 220, 4, 55),
            box(60, 30, 4, 130, 4, 5),  # PSU pocket rails
            box(60, 106, 4, 130, 4, 5),
            box(60, 30, 4, 4, 80, 5),
            box(188, 30, 4, 4, 80, 5),
            box(205, 45, 4, 10, 50, 18),  # back stop
        ]
    )

    ear_plate = cut_all(
        box(-4.2, 32, 8, 4.2, 88, 48),
        [
            cyl_x(-5.2, 48, 28, 4.0, 7),
            cyl_x(-5.2, 92, 28, 4.0, 7),
        ],
    )
    ear = fuse_all(
        [
            ear_plate,
            box(-20, 32, 8, 16, 88, 4),  # L-bracket flange
            box(-20, 32, 52, 16, 88, 4),
        ]
    )

    psu = fuse_all(
        [
            box(66, 36, 4.05, 118, 68, 44),
            box(66, 46, 48.05, 18, 48, 8),  # handle/boss proxy
            box(180, 50, 16, 8, 40, 20),  # connector face proxy
        ]
    )

    parts = [
        PartSpec(
            "01-ASSY-CHASSIS-MODULE-R6250H0.stp",
            "P01",
            "chassis",
            ["planar_mount_surface", "coaxial_screw_holes", "psu_pocket_rails"],
            chassis,
        ),
        PartSpec(
            "01-ASSY-CHASSIS-EAR-L-R620.stp",
            "P02",
            "chassis_ear",
            ["planar_flange", "coaxial_screw_holes"],
            ear,
        ),
        PartSpec(
            "5-CRPS1300NC.stp",
            "P03",
            "power_supply_module",
            ["rectangular_insert_body", "planar_contact"],
            psu,
        ),
    ]
    metadata = {
        "case_id": "case_5_lightweight",
        "source_case": "sw/5",
        "assembly_name": "server chassis with chassis ear and CRPS power supply",
        "simplification_policy": "proxy model preserving exam-relevant interfaces only",
        "functional_mates": [
            {
                "parts": ["chassis", "chassis_ear"],
                "mate_type": "planar_mate + coaxial",
                "functional_relation": "chassis ear fastens to chassis side wall with aligned screw holes",
            },
            {
                "parts": ["chassis", "power_supply_module"],
                "mate_type": "pocket_mate + planar_mate",
                "functional_relation": "power supply slides into chassis bay",
            },
        ],
        "expected_non_edges": [["chassis_ear", "power_supply_module"]],
    }
    return parts, metadata


def write_case(out_dir: Path, parts: list[PartSpec], metadata: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        check_valid(part.shape, part.file_name)
        save_step(part.shape, out_dir / part.file_name)

    assembly = compound(part.shape for part in parts)
    check_valid(assembly, "assembly")
    save_step(assembly, out_dir / "assembly.step")

    manifest = {
        **metadata,
        "parts": [
            {
                "part_id": part.part_id,
                "file": part.file_name,
                "part_role": part.part_role,
                "interface_types": part.interface_types,
                "bytes": (out_dir / part.file_name).stat().st_size,
            }
            for part in parts
        ],
        "assembly_file": {
            "file": "assembly.step",
            "bytes": (out_dir / "assembly.step").stat().st_size,
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="sw/lightweight_exam",
        help="Output directory root. Original sw/4 and sw/5 are never overwritten.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    case4_parts, case4_meta = build_case4()
    case5_parts, case5_meta = build_case5()
    report = {
        "case_4": write_case(output_root / "case_4", case4_parts, case4_meta),
        "case_5": write_case(output_root / "case_5", case5_parts, case5_meta),
    }
    report_path = output_root / "lightweight_exam_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
