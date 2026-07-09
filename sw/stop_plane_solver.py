"""
stop_plane_solver.py — Universal closed-form placement by stop-plane alignment.

For every assembly, the receiver part defines "stop planes" — geometric
surfaces where insert parts must sit.  This solver computes these planes
from geometry primitives (cylinder ends, pocket bottoms, planar faces)
and places inserts directly, without optimization or beam search.

Algorithm (generic):
  1. Identify receiver (highest-degree part in edge graph)
  2. For each insert part, find the constraint type with the receiver
  3. Derive a stop plane from receiver geometry
  4. Derive the insert's mating point from its geometry
  5. Place insert so its mating point lies on the stop plane
"""

from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any
from collections import defaultdict


def _norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 1e-12 else [0,0,1]

def _dot(a, b):
    return sum(a[i]*b[i] for i in range(3))

def _sub(a, b):
    return [a[i]-b[i] for i in range(3)]

def _add(a, b):
    return [a[i]+b[i] for i in range(3)]

def _scale(v, s):
    return [v[i]*s for i in range(3)]


def _stop_plane_from_cylinder(features, placement):
    """Return (origin, normal) of the shaft cylinder end face nearest to +axis."""
    cyls = features.get("cylinders", [])
    if not cyls:
        return None
    cyl = max(cyls, key=lambda c: c["radius"])
    axis = _norm(cyl["axis"])
    origin = cyl["origin"]

    # Compute cylinder extent along axis from bbox
    bbox = features.get("bbox", {})
    lo = bbox.get("min", [0,0,0])
    hi = bbox.get("max", [0,0,0])
    proj = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                p = _sub([x,y,z], origin)
                proj.append(_dot(p, axis))
    t_min, t_max = min(proj), max(proj)

    # Stop plane at the +axis end of the cylinder
    stop_origin = _add(origin, _scale(axis, t_max))
    return (stop_origin, axis)


def _stop_plane_from_pocket(pocket):
    """Return (origin, normal) of a pocket's bottom face."""
    center = pocket.get("center")
    direction = pocket.get("direction")
    if not center or not direction:
        return None
    return (list(center), list(direction))


def _stop_plane_from_planar(features, feat_idx):
    """Return (position, normal) of a planar face."""
    planes = features.get("planes", [])
    if not planes or feat_idx >= len(planes):
        return None
    p = planes[feat_idx]
    return (p["position"], p["normal"])


def _insert_mating_point(features, match):
    """Compute the insert part's mating point along the matched feature axis."""
    # For clearance: the bore center
    # For pocket: the pocket center
    # For planar: the face position
    ctype = match["type"]
    part_name = match["parts"][1]  # target/insert side

    if ctype == "clearance":
        cyls = features.get("cylinders", [])
        if cyls:
            cyl = max(cyls, key=lambda c: c["radius"])
            axis = _norm(cyl["axis"])
            origin = cyl["origin"]
            # The flange bore entrance is at the -axis end of its bbox
            bbox = features.get("bbox", {})
            lo, hi = bbox.get("min", [0,0,0]), bbox.get("max", [0,0,0])
            proj = []
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        p = _sub([x,y,z], origin)
                        proj.append(_dot(p, axis))
            return _add(origin, _scale(axis, min(proj)))
        return features.get("cylinders", [{}])[0].get("origin", [0,0,0])

    if ctype == "pocket_mate":
        pkt = match.get("pocket_a") or match.get("pocket_b") or {}
        return pkt.get("center", [0,0,0])

    if ctype in ("planar_mate", "planar_align"):
        planes = features.get("planes", [])
        idx = match.get("feat_b_idx", 0)
        if idx < len(planes):
            return planes[idx].get("position", [0,0,0])

    return [0,0,0]


def solve_stop_plane(
    case_dir: str | Path,
) -> dict[str, Any]:
    """Compute placements for all parts using stop-plane alignment."""
    case_dir = Path(case_dir)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from features import extract_features
    from constraints import match_features
    from direct_assembly_graph import (
        canonical_pair, select_direct_connections, build_pair_candidates,
    )
    from match_scoring import score_matches

    # Extract features and edges
    step_files = sorted(
        p for p in case_dir.iterdir()
        if p.suffix.lower() in {".step", ".stp"}
        and not p.name.lower().startswith("assembly")
    )
    parts = [p.name for p in step_files]
    features = {p.name: extract_features(str(p)) for p in step_files}
    raw_matches = match_features(features)

    scored = score_matches(raw_matches, features)
    pair_candidates = build_pair_candidates(scored, {})
    part_weights = {p.name: float(p.stat().st_size) for p in step_files}
    graph = select_direct_connections(
        parts, pair_candidates, conservative=True, part_weights=part_weights,
    )
    selected_edges = [row["parts"] for row in graph["selected"]]
    print(f"Edges: {selected_edges}")

    # Identify receiver (highest degree)
    degree = defaultdict(int)
    for a, b in selected_edges:
        degree[a] += 1; degree[b] += 1
    receiver = max(degree, key=degree.get)
    print(f"Receiver: {receiver} (degree={degree[receiver]})")

    # Group matches by insert part
    inserts = []
    for match in raw_matches:
        if receiver not in match["parts"]:
            continue
        pair = canonical_pair(match["parts"])
        if pair not in {canonical_pair(e) for e in selected_edges}:
            continue
        insert = match["parts"][0] if match["parts"][1] == receiver else match["parts"][1]
        inserts.append((insert, match))

    # Place receiver at origin
    placements = {receiver: {"translate": [0.0, 0.0, 0.0]}}
    used_ends: dict[str, list[str]] = defaultdict(list)  # axis_key → list of placed inserts

    for insert, match in inserts:
        print(f"\nPlacing {Path(insert).stem} (type={match['type']})...")

        # Get stop plane from receiver
        if match["type"] in ("coaxial", "clearance"):
            stop = _stop_plane_from_cylinder(features[receiver], placements[receiver])
            if not stop:
                continue
            stop_origin, stop_normal = stop

            # Mate point on insert
            mate_pt = _insert_mating_point(features[insert], match)

            # Axis alignment (same as coordinate_solver)
            from coordinate_solver import _global_vector
            ref_axis = _global_vector(
                features[receiver]["cylinders"][0]["axis"], placements[receiver]
            )
            tgt_cyls = features[insert].get("cylinders", [])
            if tgt_cyls:
                tgt_axis = max(tgt_cyls, key=lambda c: c["radius"])["axis"]
                # Align axes
                dot = abs(_dot(tgt_axis, ref_axis))
                if dot < 0.999:
                    from coordinate_solver import _axis_angle_to_rotation
                    rot_axis, rot_angle = _axis_angle_to_rotation(tgt_axis, ref_axis)
                    rot_seq = [{'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                                               math.degrees(rot_angle)]}] if rot_axis else []
                else:
                    rot_seq = []
            else:
                rot_seq = []

            # Find which stop end to use based on used_ends
            axis_key = tuple(round(v, 2) for v in ref_axis)
            ends_used = len(used_ends.get(axis_key, []))
            if ends_used == 0:
                # First insert: use +axis end
                target = stop_origin
            else:
                # Second+ insert: use -axis end
                bbox = features[receiver].get("bbox", {})
                lo, hi = bbox.get("min", [0,0,0]), bbox.get("max", [0,0,0])
                cyl = features[receiver]["cylinders"][0]
                origin = cyl["origin"]
                axis = _norm(cyl["axis"])
                proj = []
                for x in (lo[0], hi[0]):
                    for y in (lo[1], hi[1]):
                        for z in (lo[2], hi[2]):
                            p = _sub([x,y,z], origin)
                            proj.append(_dot(p, axis))
                t_min = min(proj)
                target = _add(origin, _scale(axis, t_min))

            # Compute translation: mate_pt → target
            # Apply rotation first
            from coordinate_solver import _apply_rotation_to_vector
            rotated_mate = _apply_rotation_to_vector(mate_pt, rot_seq)
            translate = _sub(target, rotated_mate)

            placement = {"translate": translate}
            if rot_seq:
                placement["rotate_sequence"] = rot_seq
            placements[insert] = placement
            used_ends[axis_key].append(insert)

        elif match["type"] == "pocket_mate":
            pkt_a = match.get("pocket_a", {})
            pkt_b = match.get("pocket_b", {})
            # Receiver pocket
            rpkt = pkt_a if match["parts"][0] == receiver else pkt_b
            ipkt = pkt_b if match["parts"][0] == receiver else pkt_a
            stop = _stop_plane_from_pocket(rpkt)
            mate = ipkt.get("center", [0,0,0])
            if stop and mate:
                translate = _sub(stop[0], mate)
                placements[insert] = {"translate": translate}

        elif match["type"] in ("planar_mate", "planar_align"):
            stop = _stop_plane_from_planar(
                features[receiver],
                match.get("feat_a_idx" if match["parts"][0] == receiver else "feat_b_idx", 0)
            )
            mate = _insert_mating_point(features[insert], match)
            if stop and mate:
                translate = _sub(stop[0], mate)
                placements[insert] = {"translate": translate}

        print(f"  → translate={[round(v,1) for v in placements[insert].get('translate',[0,0,0])]}")

    # Build output
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

    from build_assembly import build_assembly
    build_assembly(
        str(out_dir / "assembly_manifest.json"),
        str(out_dir / "assembly.step"),
    )

    return {"placements": placements, "edges": selected_edges, "receiver": receiver}
