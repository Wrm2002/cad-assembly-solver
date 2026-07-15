"""Build read-only, low-complexity interface proxies for STEP assemblies.

This module deliberately does *not* simplify or alter the source B-Rep.  It
reads only a source shape's bounding box and selected planar faces, then writes
new STEP files made from a small number of independent boxes.  The resulting
proxy can be used for inexpensive pose proposal; any pose must subsequently be
checked/rendered with the original STEP files.

The selection is purely geometric.  File names, colours, source ids, and case
ids are recorded for traceability but never used to select interfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _occ() -> dict[str, Any]:
    """Import OCC lazily so importing this module is harmless in light tests."""
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    return locals()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(list(vector), dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length geometric direction")
    return value / norm


def _bbox_from_shape(shape: Any, occ: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    box = occ["Bnd_Box"]()
    box.SetGap(0.0)
    occ["brepbndlib"].Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    lo = np.array([xmin, ymin, zmin], dtype=float)
    hi = np.array([xmax, ymax, zmax], dtype=float)
    if not np.all(np.isfinite(np.r_[lo, hi])) or np.any(hi - lo <= 1e-9):
        raise RuntimeError("invalid source bounding box")
    return lo, hi


def _bbox_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])],
        dtype=float,
    )


@dataclass(frozen=True)
class PlanePatch:
    face_index: int
    origin: np.ndarray
    normal: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    extent_u: float
    extent_v: float

    @property
    def area_proxy(self) -> float:
        return self.extent_u * self.extent_v

    def as_json(self) -> dict[str, Any]:
        return {
            "face_index": self.face_index,
            "origin": self.origin.round(6).tolist(),
            "normal": self.normal.round(8).tolist(),
            "axis_u": self.axis_u.round(8).tolist(),
            "axis_v": self.axis_v.round(8).tolist(),
            "extent_u": round(self.extent_u, 6),
            "extent_v": round(self.extent_v, 6),
            "area_proxy": round(self.area_proxy, 6),
        }


@dataclass(frozen=True)
class CylindricalPatch:
    """A read-only cylindrical-face proxy, suitable for screw-hole matching."""

    face_index: int
    centre: np.ndarray
    axis: np.ndarray
    radius: float
    axial_extent: float

    def as_json(self) -> dict[str, Any]:
        return {
            "face_index": self.face_index,
            "centre": self.centre.round(6).tolist(),
            "axis": self.axis.round(8).tolist(),
            "radius": round(self.radius, 6),
            "axial_extent": round(self.axial_extent, 6),
        }


@dataclass
class SourceSummary:
    path: Path
    sha256: str
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    planar_faces: list[PlanePatch]
    cylindrical_faces: list[CylindricalPatch]

    @property
    def dimensions(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.dimensions))


def extract_source_summary(path: Path, *, max_planar_faces: int = 2500) -> SourceSummary:
    """Read only the STEP shape and retain a bounded, useful planar-face set.

    Face dimensions are measured from each face bounding box projected into the
    plane's local axes.  This avoids area integration and topology healing.
    """
    occ = _occ()
    reader = occ["STEPControl_Reader"]()
    if reader.ReadFile(str(path)) != occ["IFSelect_RetDone"]:
        raise RuntimeError(f"cannot read STEP: {path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    source_min, source_max = _bbox_from_shape(shape, occ)

    candidates: list[PlanePatch] = []
    cylinders: list[CylindricalPatch] = []
    explorer = occ["TopExp_Explorer"](shape, occ["TopAbs_FACE"])
    face_index = 0
    while explorer.More():
        face_index += 1
        face = explorer.Current()
        explorer.Next()
        try:
            adaptor = occ["BRepAdaptor_Surface"](face, True)
            face_min, face_max = _bbox_from_shape(face, occ)
            face_centre = (face_min + face_max) / 2.0
            surface_type = adaptor.GetType()
            if surface_type == occ["GeomAbs_Plane"]:
                plane = adaptor.Plane()
                normal = _normalise((plane.Axis().Direction().X(), plane.Axis().Direction().Y(), plane.Axis().Direction().Z()))
                axis_u = _normalise((plane.XAxis().Direction().X(), plane.XAxis().Direction().Y(), plane.XAxis().Direction().Z()))
                axis_v = _normalise(np.cross(normal, axis_u))
                corners = _bbox_corners(face_min, face_max)
                extent_u = float(np.ptp(corners @ axis_u))
                extent_v = float(np.ptp(corners @ axis_v))
                if min(extent_u, extent_v) <= 0.02:
                    continue
                # A Geom_Plane location belongs to the *infinite* support plane
                # and is often the global origin, not the trimmed B-Rep patch.
                support_origin = np.array([plane.Location().X(), plane.Location().Y(), plane.Location().Z()], dtype=float)
                origin = face_centre - normal * float(np.dot(face_centre - support_origin, normal))
                candidates.append(PlanePatch(face_index, origin, normal, axis_u, axis_v, extent_u, extent_v))
            elif surface_type == occ["GeomAbs_Cylinder"]:
                cylinder = adaptor.Cylinder()
                axis = _normalise((cylinder.Axis().Direction().X(), cylinder.Axis().Direction().Y(), cylinder.Axis().Direction().Z()))
                support_origin = np.array([cylinder.Location().X(), cylinder.Location().Y(), cylinder.Location().Z()], dtype=float)
                centre = support_origin + axis * float(np.dot(face_centre - support_origin, axis))
                extent = float(np.ptp(_bbox_corners(face_min, face_max) @ axis))
                radius = float(cylinder.Radius())
                if radius > 0.1 and extent > 0.05:
                    cylinders.append(CylindricalPatch(face_index, centre, axis, radius, extent))
        except Exception:
            # A malformed individual face must not invalidate a read-only audit
            # of the rest of an otherwise valid imported STEP assembly.
            continue

    # Keep broad, mechanically useful planes.  Deterministic ordering makes the
    # proxy repeatable while avoiding a huge rail/hole-detail model.
    candidates.sort(key=lambda patch: (-patch.area_proxy, -max(patch.extent_u, patch.extent_v), patch.face_index))
    # Mechanical screw holes are normally among the smaller cylinders.  Keep a
    # deterministic bounded pool, without naming them holes until a matching
    # pattern provides evidence.
    small_radius_limit = max(3.0, min(source_max - source_min) * 0.10)
    cylinders = [patch for patch in cylinders if patch.radius <= small_radius_limit]
    # Keep fastener-scale bores ahead of microscopic embossing/thread detail.
    # The threshold is intentionally broad and is only a recall ordering; the
    # later matcher still requires two compatible interfaces.
    fastener_scale = [
        patch
        for patch in cylinders
        if 0.5 <= patch.radius <= min(12.0, small_radius_limit) and 0.08 <= patch.axial_extent <= 18.0
    ]
    # Preserve radius diversity.  A plain ascending-radius truncation retained
    # thousands of 0.5/1-mm cosmetic cylinders and silently dropped the actual
    # 2--4-mm mounting bores needed for a flange pattern.
    radius_buckets: dict[float, list[CylindricalPatch]] = {}
    for patch in fastener_scale:
        radius_buckets.setdefault(round(patch.radius * 4.0) / 4.0, []).append(patch)
    fastener_scale = []
    bucket_lists = [
        sorted(items, key=lambda patch: (-patch.axial_extent, patch.face_index))
        for _, items in sorted(radius_buckets.items())
    ]
    while bucket_lists and len(fastener_scale) < 2500:
        remaining: list[list[CylindricalPatch]] = []
        for items in bucket_lists:
            if items and len(fastener_scale) < 2500:
                fastener_scale.append(items.pop(0))
            if items:
                remaining.append(items)
        bucket_lists = remaining
    fastener_ids = {patch.face_index for patch in fastener_scale}
    non_fastener_scale = [patch for patch in cylinders if patch.face_index not in fastener_ids]
    non_fastener_scale.sort(key=lambda patch: (patch.radius, -patch.axial_extent, patch.face_index))
    return SourceSummary(
        path,
        _sha256(path),
        source_min,
        source_max,
        candidates[:max_planar_faces],
        (fastener_scale + non_fastener_scale)[:2500],
    )


def _relative_extent_error(patch: PlanePatch, moving_dimensions: np.ndarray) -> float:
    patch_dims = (max(patch.extent_u, 1e-6), max(patch.extent_v, 1e-6))
    values = []
    for first in patch_dims:
        for second in moving_dimensions:
            values.append(abs(math.log(max(first, 1e-6) / max(float(second), 1e-6))))
    return min(values)


def select_interface_planes(
    carrier: SourceSummary,
    moving_dimensions: np.ndarray,
    *,
    limit: int = 72,
) -> list[PlanePatch]:
    """Select large candidate rails/stops using only size and plane geometry.

    A plane is useful if at least one footprint dimension lies within a broad
    scale band of some moving-box dimension.  The broad band intentionally
    favours recall: false candidates are harmless at proposal stage and the
    final original-geometry validation remains conservative.
    """
    log_band = math.log(6.0)
    scored: list[tuple[float, float, int, PlanePatch]] = []
    for patch in carrier.planar_faces:
        error = _relative_extent_error(patch, moving_dimensions)
        if error <= log_band:
            # Prefer dimensional agreement, then retain large faces as likely
            # rails, side walls, or stop planes.
            scored.append((error, -patch.area_proxy, patch.face_index, patch))
    scored.sort(key=lambda row: row[:3])

    selected = [row[3] for row in scored[:limit]]
    # Preserve a few large planes even when their dimensions are outside the
    # moving-box band; they often encode a rear stop or chassis mounting wall.
    selected_ids = {patch.face_index for patch in selected}
    for patch in carrier.planar_faces:
        if len(selected) >= limit:
            break
        if patch.face_index not in selected_ids:
            selected.append(patch)
            selected_ids.add(patch.face_index)
    return selected


def _thin_box_from_patch(patch: PlanePatch, thickness: float, occ: dict[str, Any]) -> Any:
    normal = _normalise(patch.normal)
    axis_u = _normalise(patch.axis_u)
    axis_v = _normalise(np.cross(normal, axis_u))
    # gp_Ax2 defines Y as Z cross X.  Use this direction consistently when
    # centring the new box, even if the source UV orientation was reversed.
    centre = patch.origin
    lower = centre - axis_u * (patch.extent_u / 2.0) - axis_v * (patch.extent_v / 2.0) - normal * (thickness / 2.0)
    axes = occ["gp_Ax2"](
        occ["gp_Pnt"](*map(float, lower)),
        occ["gp_Dir"](*map(float, normal)),
        occ["gp_Dir"](*map(float, axis_u)),
    )
    return occ["BRepPrimAPI_MakeBox"](axes, float(patch.extent_u), float(patch.extent_v), float(thickness)).Shape()


def _axis_aligned_box(summary: SourceSummary, occ: dict[str, Any]) -> Any:
    dimensions = summary.dimensions
    return occ["BRepPrimAPI_MakeBox"](
        occ["gp_Pnt"](*map(float, summary.bbox_min)),
        *map(float, dimensions),
    ).Shape()


def _write_shape(shape: Any, path: Path, occ: dict[str, Any]) -> None:
    writer = occ["STEPControl_Writer"]()
    writer.Transfer(shape, occ["STEPControl_AsIs"])
    status = writer.Write(str(path))
    if int(status) != 1:
        raise RuntimeError(f"cannot write proxy STEP: {path}")


def build_interface_proxies(
    source_files: list[Path],
    output_dir: Path,
    *,
    max_planar_faces: int = 2500,
    max_interface_planes: int = 72,
) -> dict[str, Any]:
    """Create per-component proxy STEP files and a source-to-proxy audit.

    The largest bounding-box diagonal is treated as the stationary carrier only
    for proxy *proposal*.  This is not a functional assertion and the metadata
    records the assumption for later human review.
    """
    if len(source_files) < 2:
        raise ValueError("at least two source STEP files are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [extract_source_summary(path, max_planar_faces=max_planar_faces) for path in source_files]
    carrier_index = max(range(len(summaries)), key=lambda index: summaries[index].diagonal)
    carrier = summaries[carrier_index]

    selected_for_carrier: dict[int, PlanePatch] = {}
    plane_selection_audit: dict[str, list[dict[str, Any]]] = {}
    for index, summary in enumerate(summaries):
        if index == carrier_index:
            continue
        selected = select_interface_planes(carrier, summary.dimensions, limit=max_interface_planes)
        plane_selection_audit[summary.path.name] = [patch.as_json() for patch in selected]
        for patch in selected:
            selected_for_carrier[patch.face_index] = patch

    # The union over several moving parts must remain low complexity as well.
    # Retain the highest-area geometrically recalled plates deterministically.
    selected_carrier_patches = sorted(
        selected_for_carrier.values(),
        key=lambda patch: (-patch.area_proxy, patch.face_index),
    )[:max_interface_planes]

    occ = _occ()
    proxy_records: list[dict[str, Any]] = []
    for index, summary in enumerate(summaries):
        compound = occ["TopoDS_Compound"]()
        builder = occ["BRep_Builder"]()
        builder.MakeCompound(compound)
        proxy_kind: str
        selected_plane_ids: list[int] = []
        if index == carrier_index and selected_carrier_patches:
            # The carrier is intentionally represented by open interface plates,
            # not its outer bounding-box solid: an enclosing solid would falsely
            # occupy the bay into which another component should be inserted.
            thickness = max(0.25, min(summary.dimensions) * 0.003)
            for patch in selected_carrier_patches:
                builder.Add(compound, _thin_box_from_patch(patch, thickness, occ))
                selected_plane_ids.append(patch.face_index)
            proxy_kind = "open_interface_plate_set"
        else:
            builder.Add(compound, _axis_aligned_box(summary, occ))
            proxy_kind = "bounding_box"

        proxy_path = output_dir / summary.path.name
        _write_shape(compound, proxy_path, occ)
        proxy_records.append(
            {
                "source_file": str(summary.path.resolve()),
                "source_file_sha256": summary.sha256,
                "proxy_file": proxy_path.name,
                "proxy_kind": proxy_kind,
                "source_bbox_min": summary.bbox_min.round(6).tolist(),
                "source_bbox_max": summary.bbox_max.round(6).tolist(),
                "source_dimensions": summary.dimensions.round(6).tolist(),
                "selected_face_indices": selected_plane_ids,
                "source_unchanged_sha256_after": _sha256(summary.path),
            }
        )

    audit = {
        "schema": "interface_geometry_proxy/v1",
        "method": "read_only_bbox_and_planar_face_proxy",
        "topology_operations": "none_on_source_geometry",
        "selection_rule": "bbox scale compatibility plus planar footprint; no file-name, source-id, or colour feature",
        "carrier_proxy_assumption": {
            "source_file": carrier.path.name,
            "basis": "largest source bounding-box diagonal",
            "usage": "proposal-only; not an accepted assembly decision",
        },
        "components": proxy_records,
        "carrier_interface_planes_by_moving_component": plane_selection_audit,
        "cylindrical_interface_candidates": {
            summary.path.name: [patch.as_json() for patch in summary.cylindrical_faces]
            for summary in summaries
        },
        "final_pose_policy": "proxy pose is a review-only proposal; apply transform to originals and validate/render originals",
    }
    (output_dir / "interface_proxy_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _input_files(directory: Path) -> list[Path]:
    # Some supplied case folders also contain an already-exported aggregate
    # assembly.  It is not a source component and including it would make the
    # "largest carrier" proposal tautological, so exclude only this conventional
    # aggregate artifact (not any component based on its name).
    files = sorted(
        path
        for path in [*directory.glob("*.step"), *directory.glob("*.stp")]
        if path.stem.lower() not in {"assembly", "assembly_manifest"}
    )
    if not files:
        raise FileNotFoundError(f"no STEP files in {directory}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-planar-faces", type=int, default=2500)
    parser.add_argument("--max-interface-planes", type=int, default=72)
    args = parser.parse_args()
    audit = build_interface_proxies(
        _input_files(args.source_dir),
        args.output_dir,
        max_planar_faces=args.max_planar_faces,
        max_interface_planes=args.max_interface_planes,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "components": len(audit["components"])}, indent=2))


if __name__ == "__main__":
    main()
