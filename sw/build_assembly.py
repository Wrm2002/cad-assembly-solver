"""
build_assembly.py — Generic assembly builder.
Reads an assembly manifest JSON, loads STEP parts, applies transforms,
and exports a single assembly STEP file.
No hard-coded part names, file paths, or transform values.

Usage: python build_assembly.py <folder> [--use-parents]
  Reads  <folder>/assembly_manifest.json
  Writes <folder>/assembly.step
  --use-parents: use parent_source (full original files) instead of
                 decomposed sub-parts when available.
"""

import json
import math
import os
import sys
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Trsf, gp_Vec, gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRep import BRep_Builder


# ── geometry helpers ──────────────────────────────────────────────

def axis_angle_to_trsf(axis_x, axis_y, axis_z, angle_deg):
    """Build a gp_Trsf from axis + angle (degrees)."""
    trsf = gp_Trsf()
    axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(axis_x, axis_y, axis_z))
    trsf.SetRotation(axis, math.radians(angle_deg))
    return trsf


def align_axis_trsf(src_x, src_y, src_z, tgt_x, tgt_y, tgt_z):
    """
    Build a gp_Trsf that rotates the source direction to the target direction.
    Uses cross-product axis and dot-product angle.
    """
    src = gp_Dir(src_x, src_y, src_z)
    tgt = gp_Dir(tgt_x, tgt_y, tgt_z)

    # Already aligned
    if abs(src.Dot(tgt) - 1.0) < 1e-12:
        return gp_Trsf()

    # Anti-aligned (180°)
    if abs(src.Dot(tgt) + 1.0) < 1e-12:
        # Pick an arbitrary perpendicular axis
        if abs(tgt.X()) < 0.9:
            perp = gp_Dir(1, 0, 0)
        else:
            perp = gp_Dir(0, 1, 0)
        cross = tgt.Crossed(perp)
        trsf = gp_Trsf()
        trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), cross), math.pi)
        return trsf

    # General case
    cross = src.Crossed(tgt)
    axis = gp_Ax1(gp_Pnt(0, 0, 0), cross)
    angle = math.acos(max(-1.0, min(1.0, src.Dot(tgt))))

    trsf = gp_Trsf()
    trsf.SetRotation(axis, angle)
    return trsf


def build_transform(placement):
    """
    Build a combined gp_Trsf from a placement dict.
    Transform order applied to a point: rotate(s) → then translate.
    
    Supports:
      - translate: [dx, dy, dz]
      - rotate_axis_to: {"from": [fx,fy,fz], "to": [tx,ty,tz]}  (to defaults to Z)
      - rotate_axis_angle: [ax, ay, az, angle_deg]
      - rotate_sequence: ordered list of rotation specs, e.g.:
          [{"axis_angle": [0,1,0,90]}, {"axis_to": {"from":[1,0,0], "to":[0,0,-1]}}]
        Each spec applied left-to-right (first in list = first applied to point).
    
    OCCT Multiply semantics: P.Transformed(A*B) = P.Transformed(B).Transformed(A).
    So to get translate ∘ rot2 ∘ rot1 (apply rot1 first, translate last),
    we Multiply in order: translate, rot2, rot1.
    """
    trsf = gp_Trsf()

    # Collect all rotations in application order (first = first applied to point)
    rotations = []

    # New: rotate_sequence (ordered list)
    if 'rotate_sequence' in placement:
        for spec in placement['rotate_sequence']:
            if 'axis_angle' in spec:
                ra = spec['axis_angle']
                if abs(ra[3]) > 1e-9:
                    rotations.append(axis_angle_to_trsf(ra[0], ra[1], ra[2], ra[3]))
            elif 'axis_to' in spec:
                at = spec['axis_to']
                src = at['from']
                tgt = at.get('to', [0.0, 0.0, 1.0])
                rotations.append(align_axis_trsf(src[0], src[1], src[2], tgt[0], tgt[1], tgt[2]))
    else:
        # Legacy: single rotate_axis_to and rotate_axis_angle
        if 'rotate_axis_to' in placement:
            spec = placement['rotate_axis_to']
            src = spec['from']
            tgt = spec.get('to', [0.0, 0.0, 1.0])
            rotations.append(align_axis_trsf(src[0], src[1], src[2], tgt[0], tgt[1], tgt[2]))
        if 'rotate_axis_angle' in placement:
            ra = placement['rotate_axis_angle']
            if abs(ra[3]) > 1e-9:
                rotations.append(axis_angle_to_trsf(ra[0], ra[1], ra[2], ra[3]))

    # Build trsf: Multiply in reverse order so they apply correctly
    # We want: trsf = translate * rot_last * ... * rot_first
    # So Multiply(translate) first, then Multiply rotations in REVERSE
    
    # Translation (applied last to point, so Multiply first)
    t = placement.get('translate', [0.0, 0.0, 0.0])
    if any(abs(v) > 1e-12 for v in t):
        tsf_t = gp_Trsf()
        tsf_t.SetTranslation(gp_Vec(t[0], t[1], t[2]))
        trsf.Multiply(tsf_t)

    # Rotations: reverse order for Multiply
    for rot in reversed(rotations):
        trsf.Multiply(rot)

    return trsf


def transform_point(placement, x, y, z):
    """
    Apply the transform described by a placement dict to a single point.
    Uses OCCT gp_Trsf (same math as build_transform), so results are consistent.
    Returns (x', y', z').
    """
    trsf = build_transform(placement)
    pt = gp_Pnt(x, y, z)
    pt2 = pt.Transformed(trsf)
    return (pt2.X(), pt2.Y(), pt2.Z())


# ── STEP I/O ──────────────────────────────────────────────────────

def load_step(filepath):
    """Load a STEP file and return the TopoDS_Shape."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STEP file not found: {filepath}")
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {filepath}")
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def save_step(shape, filepath):
    """Save a TopoDS_Shape as a STEP file."""
    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", "AP242")
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(filepath)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP: {filepath}")
    return filepath


# ── main ──────────────────────────────────────────────────────────

def build_assembly(manifest_path, output_path, use_parents=False):
    """Build an assembly from a manifest file.
    
    If use_parents=True, uses parent_source (original full part) instead of
    decomposed sub-part when available. Same transform applies — sub-parts
    share their parent's coordinate system.
    """
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    components = manifest.get('components', [])
    print(f"Manifest: {manifest_path}")
    print(f"Assembly: {manifest.get('assembly_name', 'unnamed')}")
    print(f"Components: {len(components)}")
    if use_parents:
        print(f"Mode: use parent files (--use-parents)")

    # Build compound
    bb = BRep_Builder()
    compound = TopoDS_Compound()
    bb.MakeCompound(compound)

    for comp in components:
        cid = comp['id']
        # Choose source: prefer parent_source when --use-parents
        if use_parents and 'parent_source' in comp:
            source_rel = comp['parent_source']
            source = os.path.join(manifest_dir, source_rel)
            print(f"\n  [{cid}] {comp.get('label', cid)}")
            print(f"    Source: {source_rel}  (parent of {comp['source']})")
        else:
            source_rel = comp['source']
            source = os.path.join(manifest_dir, source_rel)
            print(f"\n  [{cid}] {comp.get('label', cid)}")
            print(f"    Source: {source_rel}")

        shape = load_step(source)
        print(f"    Loaded OK")

        placement = comp.get('placement', {})
        trsf = build_transform(placement)

        if trsf.Form() == 0:  # Identity
            print(f"    Transform: identity")
        else:
            shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
            print(f"    Transform applied")

        bb.Add(compound, shape)

    # Write output
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_step(compound, output_path)
    print(f"\n[OK] Output: {output_path}")

    # Verify
    load_step(output_path)
    print(f"[OK] Verification: re-read passed")


def main():
    args = sys.argv[1:]
    use_parents = False
    folder = None
    for a in args:
        if a == '--use-parents':
            use_parents = True
        elif folder is None:
            folder = a
    if folder is None:
        print("Usage: python build_assembly.py <folder> [--use-parents]")
        print("  Reads  <folder>/assembly_manifest.json")
        print("  Writes <folder>/assembly.step")
        sys.exit(1)

    folder = os.path.abspath(folder)
    manifest_path = os.path.join(folder, 'assembly_manifest.json')
    output_path = os.path.join(folder, 'assembly.step')
    build_assembly(manifest_path, output_path, use_parents=use_parents)


if __name__ == '__main__':
    main()
