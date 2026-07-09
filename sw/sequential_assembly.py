"""
sequential_assembly.py — Greedy sequential part placement by contact maximization.

Algorithm:
  1. Select the "core" part (most connected, largest bbox)
  2. For each remaining part, use SearchSimplex to find the placement
     that maximizes surface contact with the already-placed assembly
  3. Merge the placed part into a growing super-part STL
  4. Repeat until all parts are placed

Key insight: solving pairs sequentially avoids the combinatorial explosion
of multi-part beam search.  Contact maximization naturally places parts
at their correct interfaces (shaft ends, socket bottoms, etc.).
"""

from __future__ import annotations

import json, math, time
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np


def _norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 1e-12 else [0,0,1]


def _export_part_stl(step_path: Path, stl_path: Path) -> bool:
    """Export a STEP part to STL."""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.StlAPI import StlAPI_Writer

        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
            return False
        reader.TransferRoots()
        shape = reader.OneShape()
        solids = []
        exp = TopExp_Explorer(shape, TopAbs_SOLID)
        while exp.More():
            solids.append(exp.Current())
            exp.Next()
        if solids:
            BRepMesh_IncrementalMesh(solids[0], 0.5, True, 0.5).Perform()
            StlAPI_Writer().Write(solids[0], str(stl_path))
            return True
    except Exception:
        pass
    return False


def _merge_stl_files(stl_paths: list[Path], output_path: Path) -> bool:
    """Merge multiple STL files into one by concatenating triangles."""
    try:
        from stl import mesh as stl_mesh
        meshes = [stl_mesh.Mesh.from_file(str(p)) for p in stl_paths]
        combined = stl_mesh.Mesh(np.concatenate([m.data for m in meshes]))
        combined.save(str(output_path))
        return True
    except Exception:
        return False


def _select_core_part(
    parts_features: dict[str, Any],
    matches: list[dict[str, Any]],
) -> str:
    """Select the central structural part: largest main cylinder + most connections."""
    scores = {}
    for name, feats in parts_features.items():
        bbox = feats.get("bbox", {})
        lo = bbox.get("min", [0,0,0])
        hi = bbox.get("max", [0,0,0])
        volume = 1.0
        for i in range(3):
            volume *= max(1.0, hi[i] - lo[i])
        
        # Main cylinder radius (largest)
        cyls = feats.get("cylinders", [])
        max_radius = max((c["radius"] for c in cyls), default=0)
        n_cyl = len(cyls)
        n_conn = sum(1 for m in matches if name in m["parts"])
        
        # Core = has a LARGE main cylinder (shaft), not many small bolt holes
        scores[name] = max_radius * 10 + n_conn * 5 + math.log(volume + 1)
    
    return max(scores, key=scores.get)


def _get_clearance_axis(
    part_features: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    """Get the main cylinder axis (origin, direction) for this part."""
    cyls = part_features.get("cylinders", [])
    if not cyls:
        return None
    c = max(cyls, key=lambda x: x["radius"])
    return (list(c["origin"]), list(c["axis"]))


def sequential_assemble(
    case_dir: str | Path,
    *,
    beam_width: int = 20,
    search_budget: int = 40,
    num_samples: int = 500,
) -> dict[str, Any]:
    """Run sequential greedy assembly for a known-group case.

    Returns:
        dict with placements, edges, collisions, etc.
    """
    case_dir = Path(case_dir)
    
    # Import project modules
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import extract_features
    from constraints import match_features
    from direct_assembly_graph import (
        canonical_pair, select_direct_connections, build_pair_candidates,
    )
    from match_scoring import score_matches
    
    # Step 1: Extract features and find edges
    step_files = sorted(
        p for p in case_dir.iterdir()
        if p.suffix.lower() in {".step", ".stp"}
        and not p.name.lower().startswith("assembly")
    )
    parts = [p.name for p in step_files]
    features = {p.name: extract_features(str(p)) for p in step_files}
    raw_matches = match_features(features)
    
    # Select edges (conservative spanning tree)
    scored = score_matches(raw_matches, features)
    pair_candidates = build_pair_candidates(scored, {})
    part_weights = {p.name: float(p.stat().st_size) for p in step_files}
    graph = select_direct_connections(
        parts, pair_candidates, conservative=True, part_weights=part_weights,
    )
    selected_edges = [row["parts"] for row in graph["selected"]]
    print(f"Selected edges: {selected_edges}")
    
    # Step 2: Select core part
    core = _select_core_part(features, raw_matches)
    print(f"Core part: {core}")
    
    # Step 3: Export STLs
    stl_dir = case_dir / "_seq_stl"
    stl_dir.mkdir(exist_ok=True)
    stl_cache = {}
    for p in step_files:
        stl_path = stl_dir / f"{p.stem}.stl"
        if not stl_path.exists():
            _export_part_stl(p, stl_path)
        if stl_path.exists():
            stl_cache[p.name] = stl_path
    
    # Step 4: Sequential placement
    from search_simplex import SearchSimplex
    
    placements = {core: {"translate": [0.0, 0.0, 0.0]}}
    placed_parts = [core]
    unplaced = [p for p in parts if p != core]
    
    # Build initial super-part STL from core
    super_stl = stl_dir / "_super.stl"
    import shutil
    shutil.copy(str(stl_cache[core]), str(super_stl))
    super_stl_parts = [core]
    
    while unplaced:
        # Find the unplaced part with the best edge connection to placed parts
        best_part = None
        best_edge_count = 0
        for up in unplaced:
            n = sum(1 for e in selected_edges if up in e and any(p in e for p in placed_parts))
            if n > best_edge_count:
                best_edge_count = n
                best_part = up
        
        if best_part is None:
            best_part = unplaced[0]  # Fallback
        
        print(f"\nPlacing {best_part} (connected to {best_edge_count} placed parts)...")
        tgt_stl = stl_cache[best_part]
        
        # Get joint axis: prefer from a CONNECTED placed part that has cylinders
        axis_info = None
        # First try: find a placed part connected to best_part that has cylinders
        for ep in selected_edges:
            if best_part not in ep:
                continue
            other = ep[0] if ep[1] == best_part else ep[1]
            if other in placed_parts:
                other_axis = _get_clearance_axis(features[other])
                if other_axis:
                    axis_info = other_axis
                    break
        # Fallback: use best_part's own cylinders
        if axis_info is None:
            axis_info = _get_clearance_axis(features[best_part])
        # Last resort
        if axis_info is None:
            axis_info = ([0,0,0], [0,0,1])
        
        origin, direction = axis_info
        print(f"  Joint axis: origin={[round(x,1) for x in origin]}, dir={[round(x,2) for x in direction]}")
        
        # Run SearchSimplex against the super-part
        t0 = time.time()
        ss = SearchSimplex(
            super_stl, tgt_stl,
            origin, direction,
            num_surface_samples=num_samples,
            budget=search_budget,
            contact_weight=50.0,  # Maximize contact area
            overlap_weight=5.0,
        )
        result = ss.search()
        elapsed = time.time() - t0
        
        offset = result["offset"]
        rotation = result["rotation_deg"]
        flip = result["flip"]
        overlap = result["overlap"]
        contact = result["contact"]
        gap = result.get("mean_gap", 999)
        
        print(f"  Result ({elapsed:.1f}s): offset={offset:.1f} rot={rotation:.1f}° "
              f"flip={flip} overlap={overlap:.3f} contact={contact:.3f} gap={gap:.1f}")
        
        # Compute placement from SearchSimplex result
        d = _norm(direction)
        # Offset along axis
        trans = [offset * d[i] for i in range(3)]
        
        # Rotation
        import math as _math
        angle = _math.radians(rotation)
        c = _math.cos(angle); s = _math.sin(angle)
        x, y, z = d
        rot_seq = []
        if abs(rotation) > 0.5:
            # Rodrigues → axis-angle
            trace = 2*c + (1-c)*(x*x + y*y + z*z)  # = 1 + 2c
            rot_angle_deg = _math.degrees(_math.acos(max(-1, min(1, (trace-1)/2))))
            if rot_angle_deg > 0.1:
                rx = -z*s; ry = 0.0; rz = x*s  # Simplified for axis-aligned
                rn = _math.sqrt(rx*rx + ry*ry + rz*rz)
                if rn > 1e-9:
                    rot_seq = [{'axis_angle': [rx/rn, ry/rn, rz/rn, rot_angle_deg]}]
        
        placement = {"translate": trans}
        if rot_seq:
            placement["rotate_sequence"] = rot_seq
        if flip:
            existing = list(placement.get("rotate_sequence", []))
            existing.append({'axis_angle': [d[0], d[1], d[2], 180.0]})
            placement["rotate_sequence"] = existing
        
        placements[best_part] = placement
        placed_parts.append(best_part)
        unplaced.remove(best_part)
        
        # Merge into super-part STL
        super_stl_parts.append(best_part)
        stl_list = [stl_cache[p] for p in super_stl_parts if p in stl_cache]
        _merge_stl_files(stl_list, super_stl)
    
    # Step 5: Build output
    components = []
    for i, part_name in enumerate(parts):
        plac = placements.get(part_name, {"translate": [0.0, 0.0, 0.0]})
        components.append({
            "id": f"comp_{i+1:02d}",
            "source": f"../{part_name}",
            "label": Path(part_name).stem,
            "role": "component",
            "placement": plac,
        })
    
    manifest = {
        "schema_version": "2.0.0",
        "assembly_name": case_dir.name,
        "global_units": "mm",
        "components": components,
    }
    
    out_dir = case_dir / "known_group_output"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "assembly_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    # Write assembly STEP
    from build_assembly import build_assembly
    build_assembly(
        str(out_dir / "assembly_manifest.json"),
        str(out_dir / "assembly.step"),
    )
    
    return {
        "placements": placements,
        "edges": selected_edges,
        "core_part": core,
        "placement_order": placed_parts,
    }
