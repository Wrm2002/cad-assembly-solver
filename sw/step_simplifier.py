"""Create a geometry-preserving STEP proxy by unifying same-domain topology.

The source file is opened read-only and is never overwritten.  The generated
proxy is accepted only when a STEP roundtrip preserves solid count, volume,
bounding box, and B-Rep validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import BRepTools_ReShape
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.STEPControl import (
    STEPControl_AsIs,
    STEPControl_Reader,
    STEPControl_Writer,
)
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer


SHAPE_TYPES = {
    "compounds": TopAbs_COMPOUND,
    "compsolids": TopAbs_COMPSOLID,
    "solids": TopAbs_SOLID,
    "shells": TopAbs_SHELL,
    "faces": TopAbs_FACE,
    "wires": TopAbs_WIRE,
    "edges": TopAbs_EDGE,
    "vertices": TopAbs_VERTEX,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_step(path: Path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {path}")
    if reader.TransferRoots() <= 0:
        raise RuntimeError(f"STEP has no transferable roots: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP produced a null shape: {path}")
    return shape


def save_step(shape, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP242")
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError(f"STEP transfer failed: {path}")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"STEP write failed: {path}")


def _count(shape, shape_type: int) -> int:
    count = 0
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def shape_stats(shape) -> dict[str, Any]:
    box = Bnd_Box()
    box.SetGap(0.0)
    brepbndlib.Add(shape, box)
    if box.IsVoid():
        bounds = None
    else:
        bounds = [float(value) for value in box.Get()]
    properties = GProp_GProps()
    brepgprop.VolumeProperties(shape, properties)
    return {
        "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
        "volume_mm3": float(properties.Mass()),
        "bounds_mm": bounds,
        "topology": {
            name: _count(shape, shape_type)
            for name, shape_type in SHAPE_TYPES.items()
        },
    }


def simplify_shape(
    shape,
    *,
    linear_tolerance_mm: float,
    angular_tolerance_rad: float,
):
    # Running UnifySameDomain over a complete imported assembly can attempt
    # cross-component work and has caused native OCCT failures on real server
    # models.  Replace each solid independently so the original compound tree
    # and every component occurrence remain present.
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    if not solids:
        solids = [shape]
    reshaper = BRepTools_ReShape()
    for solid in solids:
        unifier = ShapeUpgrade_UnifySameDomain(
            solid, True, True, False
        )
        unifier.SetSafeInputMode(True)
        unifier.SetLinearTolerance(linear_tolerance_mm)
        unifier.SetAngularTolerance(angular_tolerance_rad)
        unifier.AllowInternalEdges(False)
        unifier.Build()
        unified = unifier.Shape()
        if unified.IsNull():
            raise RuntimeError(
                "same-domain unification produced a null component"
            )
        reshaper.Replace(solid, unified)
    result = reshaper.Apply(shape)
    if result.IsNull():
        raise RuntimeError("same-domain unification produced a null shape")
    return result


def preservation_checks(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    volume_relative_tolerance: float,
    bounds_absolute_tolerance_mm: float,
) -> dict[str, Any]:
    before_volume = float(before["volume_mm3"])
    after_volume = float(after["volume_mm3"])
    volume_scale = max(abs(before_volume), 1.0)
    volume_relative_error = abs(after_volume - before_volume) / volume_scale
    before_bounds = before.get("bounds_mm")
    after_bounds = after.get("bounds_mm")
    if before_bounds is None or after_bounds is None:
        bounds_max_error = None
        bounds_ok = before_bounds == after_bounds
    else:
        bounds_max_error = max(
            abs(float(first) - float(second))
            for first, second in zip(before_bounds, after_bounds)
        )
        bounds_ok = bounds_max_error <= bounds_absolute_tolerance_mm
    checks = {
        "source_was_valid": bool(before["valid"]),
        "output_is_valid": bool(after["valid"]),
        "validity_not_degraded": (
            bool(after["valid"]) if before["valid"] else True
        ),
        "solid_count_preserved": (
            before["topology"]["solids"]
            == after["topology"]["solids"]
        ),
        "volume_relative_error": volume_relative_error,
        "volume_preserved": (
            volume_relative_error <= volume_relative_tolerance
        ),
        "bounds_max_error_mm": bounds_max_error,
        "bounds_preserved": bounds_ok,
        "face_count_not_increased": (
            after["topology"]["faces"]
            <= before["topology"]["faces"]
        ),
    }
    checks["accepted"] = all(
        checks[key]
        for key in (
            "validity_not_degraded",
            "solid_count_preserved",
            "volume_preserved",
            "bounds_preserved",
            "face_count_not_increased",
        )
    )
    return checks


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(
    source: Path,
    output: Path,
    audit_path: Path,
    *,
    linear_tolerance_mm: float,
    angular_tolerance_rad: float,
    volume_relative_tolerance: float,
    bounds_absolute_tolerance_mm: float,
    overwrite: bool,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    audit_path = audit_path.resolve()
    if source == output:
        raise ValueError("refusing to overwrite the source STEP")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    started = time.perf_counter()
    source_hash_before = sha256_file(source)
    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    try:
        original_shape = load_step(source)
        before = shape_stats(original_shape)
        simplified_shape = simplify_shape(
            original_shape,
            linear_tolerance_mm=linear_tolerance_mm,
            angular_tolerance_rad=angular_tolerance_rad,
        )
        in_memory = shape_stats(simplified_shape)
        save_step(simplified_shape, temporary)
        roundtrip_shape = load_step(temporary)
        after = shape_stats(roundtrip_shape)
        checks = preservation_checks(
            before,
            after,
            volume_relative_tolerance=volume_relative_tolerance,
            bounds_absolute_tolerance_mm=bounds_absolute_tolerance_mm,
        )
        source_hash_after = sha256_file(source)
        checks["source_sha256_preserved"] = (
            source_hash_before == source_hash_after
        )
        checks["accepted"] = (
            checks["accepted"] and checks["source_sha256_preserved"]
        )
        if not checks["accepted"]:
            raise RuntimeError(
                "simplified STEP failed preservation checks: "
                + json.dumps(checks, ensure_ascii=False)
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        input_bytes = source.stat().st_size
        output_bytes = output.stat().st_size
        before_faces = before["topology"]["faces"]
        after_faces = after["topology"]["faces"]
        report = {
            "schema_version": "1.0.0",
            "status": "success",
            "method": (
                "per-solid OCCT ShapeUpgrade_UnifySameDomain with "
                "BRepTools_ReShape hierarchy replacement"
            ),
            "source": str(source),
            "output": str(output),
            "source_sha256": source_hash_before,
            "output_sha256": sha256_file(output),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "file_size_reduction_fraction": (
                1.0 - output_bytes / max(input_bytes, 1)
            ),
            "face_reduction_fraction": (
                1.0 - after_faces / max(before_faces, 1)
            ),
            "settings": {
                "linear_tolerance_mm": linear_tolerance_mm,
                "angular_tolerance_rad": angular_tolerance_rad,
                "unify_edges": True,
                "unify_faces": True,
                "concat_bsplines": False,
                "allow_internal_edges": False,
                "component_scope": "each solid independently",
            },
            "before": before,
            "after_in_memory": in_memory,
            "after_roundtrip": after,
            "checks": checks,
            "elapsed_seconds": time.perf_counter() - started,
            "metadata_limit": (
                "Geometry proxy only; SolidWorks feature history, colors, "
                "and product-tree labels are not guaranteed."
            ),
        }
        write_json(audit_path, report)
        return report
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if output.exists() and overwrite:
            output.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--linear-tolerance-mm", type=float, default=1e-7)
    parser.add_argument("--angular-tolerance-rad", type=float, default=1e-9)
    parser.add_argument(
        "--volume-relative-tolerance", type=float, default=1e-7
    )
    parser.add_argument(
        "--bounds-absolute-tolerance-mm", type=float, default=1e-5
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acknowledge-native-crash-risk",
        action="store_true",
        help=(
            "Required because OCCT 7.9 ShapeUpgrade has produced "
            "0xc0000005 on the real fan-cage STEP."
        ),
    )
    args = parser.parse_args()
    if not args.acknowledge_native_crash_risk:
        parser.error(
            "native STEP unification is quarantined after a reproducible "
            "OCCT access violation; use feature_proxy.py instead"
        )
    report = run(
        Path(args.source),
        Path(args.output),
        Path(args.audit),
        linear_tolerance_mm=args.linear_tolerance_mm,
        angular_tolerance_rad=args.angular_tolerance_rad,
        volume_relative_tolerance=args.volume_relative_tolerance,
        bounds_absolute_tolerance_mm=args.bounds_absolute_tolerance_mm,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "status": report["status"],
        "source": report["source"],
        "output": report["output"],
        "faces_before": report["before"]["topology"]["faces"],
        "faces_after": report["after_roundtrip"]["topology"]["faces"],
        "face_reduction_fraction": report["face_reduction_fraction"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
