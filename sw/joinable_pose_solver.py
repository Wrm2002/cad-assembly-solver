"""
joinable_pose_solver.py — JoinABLe SDF validation + rotation refinement.

Uses JoinABLe's SDF overlap/contact cost function (CVPR 2022) to:
  1. Validate placements from stop_plane_solver
  2. Refine rotation around joint axis via bounded scalar search
  3. Report overlap, contact, and quality metrics

Primary placement (axial position) is handled by stop_plane_solver.
JoinABLe components adapted from:
  search/search_simplex.py — simplex optimization
  joint/joint_environment.py — SDF cost function
  utils/util.py — point transform helpers
"""
from __future__ import annotations

import math, sys, json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.optimize
import trimesh
from pysdf import SDF


# ═══════════════════════════════════════════════════════════════
# Util helpers (from JoinABLe utils/util.py)
# ═══════════════════════════════════════════════════════════════

def _pad_pts(pts):
    return np.pad(pts, ((0, 0), (0, 1)), mode="constant", constant_values=1)

def _transform_pts(pts, matrix):
    if pts.shape[1] == 3:
        v = _pad_pts(pts).T
    else:
        v = pts.T
    v = matrix @ v
    return v.T[:, :3]


# ═══════════════════════════════════════════════════════════════
# SDF-based validation (from JoinABLe joint_environment.py)
# ═══════════════════════════════════════════════════════════════

def _sample_volume(mesh, num_samples, seed=42):
    """Sample points inside mesh volume."""
    np.random.seed(seed)
    if mesh.is_watertight:
        try:
            return trimesh.sample.volume_mesh(mesh, num_samples)
        except Exception:
            pass
    lo, hi = mesh.bounds
    pts = np.random.uniform(lo, hi, (num_samples * 10, 3))
    try:
        import igl
        wns = igl.fast_winding_number_for_meshes(mesh.vertices, mesh.faces, pts)
        inside = pts[wns > 0.5]
        if len(inside) >= num_samples:
            return inside[:num_samples]
    except ImportError:
        pass
    verts = mesh.vertices
    idx = np.random.choice(len(verts), num_samples)
    return verts[idx] + np.random.randn(num_samples, 3) * 0.1


def step_to_mesh(step_path):
    """Convert STEP file to trimesh via OCCT STL export."""
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    import tempfile, os

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5)
    mesh.Perform()

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        writer = StlAPI_Writer()
        writer.Write(shape, tmp_path)
        return trimesh.load(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def validate_placement(insert_mesh, receiver_mesh, transform_4x4,
                       num_samples=4096, seed=42):
    """Validate a placement using JoinABLe SDF overlap + contact.

    Returns: {overlap, contact, cost, status}
      status: 'good' | 'overlap' | 'clearance' | 'no_contact'
    """
    np.random.seed(seed)
    vol_pts = _sample_volume(insert_mesh, num_samples, seed)
    surf_pts, _ = trimesh.sample.sample_surface(insert_mesh, num_samples)

    vol_t = _transform_pts(vol_pts, transform_4x4)
    surf_t = _transform_pts(surf_pts, transform_4x4)

    sdf = SDF(receiver_mesh.vertices, receiver_mesh.faces)

    sv = sdf(vol_t[:, :3])
    overlap = float((sv > 0.01).sum() / num_samples)

    ss = sdf(surf_t[:, :3])
    contact = float((np.abs(ss) < 0.05).sum() / (num_samples * 0.1))
    contact = min(contact, 1.0)

    if overlap > 0.05:
        cost, status = overlap, "overlap"
    elif contact > 0.02:
        cost, status = -contact, "good"
    elif contact > 0.001:
        cost, status = 0.0, "clearance"
    else:
        cost, status = 0.0, "no_contact"

    return {"overlap": overlap, "contact": contact, "cost": cost, "status": status}


# ═══════════════════════════════════════════════════════════════
# Rotation refinement (from JoinABLe search approach)
# ═══════════════════════════════════════════════════════════════

def _rotation_matrix_about_axis(origin, direction, angle_deg):
    """4x4 affine matrix for rotation about axis through origin."""
    rad = np.deg2rad(angle_deg)
    d = np.array(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    o = np.array(origin, dtype=float)
    x, y, z = d
    c = math.cos(rad); s = math.sin(rad); C = 1 - c
    R = np.array([
        [x*x*C+c,  x*y*C-z*s, x*z*C+y*s],
        [x*y*C+z*s, y*y*C+c,  y*z*C-x*s],
        [x*z*C-y*s, y*z*C+x*s, z*z*C+c],
    ])
    aff = np.eye(4)
    aff[:3, :3] = R
    aff[:3, 3] = o - R @ o
    return aff


def refine_rotation(insert_mesh, receiver_mesh, joint_axis,
                    num_samples=4096, budget=50):
    """Refine rotation around joint axis to maximize contact (JoinABLe style).

    Returns: (best_angle_deg, overlap, contact, cost)
    """
    origin, direction = joint_axis

    np.random.seed(42)
    vol_pts = _sample_volume(insert_mesh, num_samples)
    surf_pts, _ = trimesh.sample.sample_surface(insert_mesh, num_samples)
    sdf = SDF(receiver_mesh.vertices, receiver_mesh.faces)

    def _cost(angle_deg):
        aff = _rotation_matrix_about_axis(origin, direction, angle_deg)
        v = _transform_pts(vol_pts, aff)
        s = _transform_pts(surf_pts, aff)
        sv = sdf(v[:, :3])
        ss = sdf(s[:, :3])
        ov = (sv > 0.01).sum() / num_samples
        ct = (np.abs(ss) < 0.05).sum() / (num_samples * 0.1)
        ct = min(ct, 1.0)
        if ov > 0.05:
            return ov * 10.0
        return -ct

    res = scipy.optimize.minimize_scalar(
        _cost, bounds=(-180, 180), method="bounded",
        options={"maxiter": budget, "xatol": 0.5},
    )

    best_angle = float(res.x)
    aff = _rotation_matrix_about_axis(origin, direction, best_angle)
    v = _transform_pts(vol_pts, aff)
    s = _transform_pts(surf_pts, aff)
    sv = sdf(v[:, :3])
    ss = sdf(s[:, :3])
    ov = float((sv > 0.01).sum() / num_samples)
    ct = float((np.abs(ss) < 0.05).sum() / (num_samples * 0.1))
    ct = min(ct, 1.0)

    return best_angle, ov, ct, float(res.fun)


# ═══════════════════════════════════════════════════════════════
# Integrated solver: stop_plane + JoinABLe validation
# ═══════════════════════════════════════════════════════════════

def solve_with_joinable_validation(
    case_dir: str | Path,
    refine: bool = True,
    num_samples: int = 4096,
) -> dict[str, Any]:
    """Run stop_plane_solver then validate/refine with JoinABLe SDF.

    Returns stop_plane result plus 'validation' dict.
    """
    case_dir = Path(case_dir)

    from stop_plane_solver import solve_stop_plane
    result = solve_stop_plane(case_dir)
    placements = result["placements"]
    receiver = result["receiver"]

    print("\n=== JoinABLe SDF Validation ===")
    step_files = sorted(
        p for p in case_dir.iterdir()
        if p.suffix.lower() in {".step", ".stp"}
        and not p.name.lower().startswith("assembly")
    )

    meshes = {}
    for p in step_files:
        try:
            meshes[p.name] = step_to_mesh(str(p))
            print(f"  mesh {p.stem}: {len(meshes[p.name].vertices)}v {len(meshes[p.name].faces)}f")
        except Exception as e:
            print(f"  mesh {p.stem}: FAILED ({e})")
            meshes[p.name] = None

    receiver_mesh = meshes.get(receiver)
    if receiver_mesh is None:
        print("Receiver mesh failed — skip")
        return result

    validation = {}
    for insert_name, plac in placements.items():
        if insert_name == receiver:
            continue
        insert_mesh = meshes.get(insert_name)
        if insert_mesh is None:
            continue

        # Build 4x4 transform
        translate = plac.get("translate", [0, 0, 0])
        rot_seq = plac.get("rotate_sequence", [])
        aff = np.eye(4)
        aff[:3, 3] = np.array(translate, dtype=float)
        for rot in reversed(rot_seq):
            aa = rot["axis_angle"]
            axis = np.array(aa[:3], dtype=float)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            rad = np.deg2rad(aa[3])
            x, y, z = axis
            c = math.cos(rad); s = math.sin(rad); C = 1 - c
            R = np.array([
                [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
                [x*y*C+z*s, y*y*C+c, y*z*C-x*s],
                [x*z*C-y*s, y*z*C+x*s, z*z*C+c],
            ])
            aff[:3, :3] = R @ aff[:3, :3]

        # Validate
        val = validate_placement(insert_mesh, receiver_mesh, aff, num_samples=num_samples)
        icon = {"good": "[OK]", "overlap": "[!!]", "clearance": "[~~]", "no_contact": "[--]"}.get(val["status"], "[??]")
        print(f"  {icon} {Path(insert_name).stem:30s} overlap={val['overlap']:.3f}"
              f" contact={val['contact']:.3f} status={val['status']}")

        validation[insert_name] = val

    result["validation"] = validation
    return result
