"""Create an auditable coarse STEP by retaining high-importance solids.

This is a resource proxy for oversized blind-exam inputs.  Selection uses only
geometry (volume, surface area and face count), never names or case labels.
The source file is read-only and every dropped component is reported.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Compound

from step_simplifier import load_step, save_step, shape_stats


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_faces(shape) -> int:
    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _mass(shape, surface: bool) -> float:
    props = GProp_GProps()
    if surface:
        brepgprop.SurfaceProperties(shape, props)
    else:
        brepgprop.VolumeProperties(shape, props)
    return max(0.0, float(props.Mass()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-solids", type=int, default=64)
    parser.add_argument("--max-total-faces", type=int, default=240)
    parser.add_argument("--minimum-solids", type=int, default=1)
    parser.add_argument("--target-importance-coverage", type=float, default=0.995)
    args = parser.parse_args()
    source = args.source.resolve()
    source_hash = _hash(source)
    shape = load_step(source)
    before = shape_stats(shape)
    components = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    index = 0
    while explorer.More():
        solid = explorer.Current()
        volume = _mass(solid, False)
        area = _mass(solid, True)
        components.append({
            "index": index,
            "shape": solid,
            "volume_mm3": volume,
            "surface_area_mm2": area,
            "face_count": _count_faces(solid),
            "importance": volume if volume > 1e-9 else area,
        })
        index += 1
        explorer.Next()
    if not components:
        raise RuntimeError("no_solid_components_found")
    components.sort(key=lambda row: (-row["importance"], row["face_count"], row["index"]))
    total_importance = sum(row["importance"] for row in components) or 1.0
    selected = []
    selected_faces = 0
    selected_importance = 0.0
    for row in components:
        if len(selected) >= args.max_solids:
            break
        exceeds_faces = selected_faces + row["face_count"] > args.max_total_faces
        if exceeds_faces and len(selected) >= args.minimum_solids:
            continue
        selected.append(row)
        selected_faces += row["face_count"]
        selected_importance += row["importance"]
        if (
            len(selected) >= args.minimum_solids
            and selected_importance / total_importance >= args.target_importance_coverage
        ):
            break
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for row in selected:
        builder.Add(compound, row["shape"])
    save_step(compound, args.output.resolve())
    roundtrip = load_step(args.output.resolve())
    after = shape_stats(roundtrip)
    if _hash(source) != source_hash:
        raise RuntimeError("source_hash_changed")
    report = {
        "schema_version": "geometry_component_proxy.v1",
        "source": str(source),
        "source_sha256": source_hash,
        "output": str(args.output.resolve()),
        "selection_uses_names_or_case_labels": False,
        "selection_policy": {
            "max_solids": args.max_solids,
            "max_total_faces": args.max_total_faces,
            "minimum_solids": args.minimum_solids,
            "target_importance_coverage": args.target_importance_coverage,
            "importance": "solid volume; surface area fallback for zero-volume solids",
        },
        "source_component_count": len(components),
        "selected_component_count": len(selected),
        "selected_original_indices": [row["index"] for row in selected],
        "selected_importance_coverage": selected_importance / total_importance,
        "selected_source_face_count": selected_faces,
        "before": before,
        "after_roundtrip": after,
        "dropped_component_count": len(components) - len(selected),
        "limitations": [
            "This is not lossless: low-importance solids are deliberately omitted.",
            "The proxy may be used for candidate generation/search only; final OCCT validation uses original STEP files.",
        ],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_components": len(components),
        "selected_components": len(selected),
        "importance_coverage": report["selected_importance_coverage"],
        "faces_after": after["topology"]["faces"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
