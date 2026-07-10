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

def _vec_len(v):
    return math.sqrt(sum(x*x for x in v))

def _bbox_diag(features):
    bbox = features.get("bbox", {})
    lo = bbox.get("min", [0,0,0])
    hi = bbox.get("max", [0,0,0])
    return _vec_len(_sub(hi, lo))

def _is_shaft_like(features):
    """Return True if the part has a dominant cylinder typical of shafts."""
    cyls = features.get("cylinders", [])
    if not cyls:
        return False
    diag = _bbox_diag(features)
    if diag < 1.0:
        return False
    max_r = max(c["radius"] for c in cyls)
    # A shaft cylinder should be at least 5% of the bbox diagonal
    return max_r > 0.05 * diag and max_r > 10.0


def _stop_plane_from_cylinder(features, placement):
    """Return (origin, normal) of the stop plane for a cylinder feature.

    For shaft-like parts: finds the largest planar face perpendicular
    to the cylinder axis — this is the shaft end face / shoulder.
    For small holes: uses the nearest bbox face perpendicular to axis.
    """
    cyls = features.get("cylinders", [])
    if not cyls:
        return None
    cyl = max(cyls, key=lambda c: c["radius"])
    axis = _norm(cyl["axis"])
    origin = cyl["origin"]
    radius = cyl["radius"]

    # Find planar faces perpendicular to the cylinder axis
    planes = features.get("planes", [])
    perpendicular_faces = []
    for i, p in enumerate(planes):
        dot = abs(_dot(p["normal"], axis))
        if dot > 0.8:
            proj = _dot(p["position"], axis)
            area = p.get("area", 0) or 0
            perpendicular_faces.append({
                "index": i,
                "position": list(p["position"]),
                "normal": list(p["normal"]),
                "area": area,
                "proj": proj,
            })

    if perpendicular_faces:
        # For shaft-like: use the largest area perpendicular face
        # at each end of the shaft
        if _is_shaft_like(features):
            # Sort by projection along axis
            perpendicular_faces.sort(key=lambda f: f["proj"])
            # Find large faces at each end (area > 100 or largest in that half)
            mid_proj = (perpendicular_faces[0]["proj"] + perpendicular_faces[-1]["proj"]) / 2
            faces_minus = [f for f in perpendicular_faces if f["proj"] < mid_proj]
            faces_plus = [f for f in perpendicular_faces if f["proj"] >= mid_proj]

            # Pick the largest area face at each end
            face_minus = max(faces_minus, key=lambda f: f["area"]) if faces_minus else perpendicular_faces[0]
            face_plus = max(faces_plus, key=lambda f: f["area"]) if faces_plus else perpendicular_faces[-1]

            # Return both as a dict for the caller to pick
            return {
                "type": "shaft_faces",
                "face_minus": face_minus,  # stop face at -axis end
                "face_plus": face_plus,    # stop face at +axis end
                "axis": axis,
            }

        # For small holes: use the face whose normal best aligns with axis
        best = max(perpendicular_faces, key=lambda f: f["area"])
        return (best["position"], best["normal"])

    # Fallback: bbox-based (legacy)
    bbox = features.get("bbox", {})
    lo = bbox.get("min", [0,0,0])
    hi = bbox.get("max", [0,0,0])
    proj = []
    for x in (lo[0], hi[0]):
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                p = _sub([x,y,z], origin)
                proj.append(_dot(p, axis))
    t_max = max(proj)
    return (_add(origin, _scale(axis, t_max)), axis)


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


def _insert_mating_point(features, match, use_plus_end=False):
    """Compute the insert part's mating point — the point that contacts the stop plane.

    For clearance/coaxial: the bore entrance at the -axis end of the insert cylinder.
      use_plus_end=True: use the +axis end instead (for reversed flanges).
    For pocket_mate: the face of the insert that bottoms out in the receiver pocket.
    For planar: the face position.
    """
    ctype = match["type"]

    if ctype in ("clearance", "coaxial"):
        cyls = features.get("cylinders", [])
        if cyls:
            cyl = max(cyls, key=lambda c: c["radius"])
            axis = _norm(cyl["axis"])

            # Find the largest planar face perpendicular to bore axis
            # This is the flange's disc face that contacts the shaft shoulder
            planes = features.get("planes", [])
            disc_faces = []
            for p in planes:
                dot = abs(_dot(p["normal"], axis))
                if dot > 0.8:
                    area = p.get("area", 0) or 0
                    proj = _dot(p["position"], axis)
                    disc_faces.append({
                        "position": list(p["position"]),
                        "normal": list(p["normal"]),
                        "area": area,
                        "proj": proj,
                    })
            if disc_faces:
                # Pick the face based on use_plus_end
                disc_faces.sort(key=lambda f: f["proj"])
                if use_plus_end:
                    # Use the face at the +axis end (pipe side)
                    mate_face = disc_faces[-1]
                else:
                    # Use the face at the -axis end (disc side — contacts shaft)
                    mate_face = disc_faces[0]
                return mate_face["position"]

            # Fallback: bore entrance
            origin = cyl["origin"]
            bbox = features.get("bbox", {})
            lo, hi = bbox.get("min", [0,0,0]), bbox.get("max", [0,0,0])
            proj = []
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    for z in (lo[2], hi[2]):
                        p = _sub([x,y,z], origin)
                        proj.append(_dot(p, axis))
            if use_plus_end:
                return _add(origin, _scale(axis, max(proj)))
            else:
                return _add(origin, _scale(axis, min(proj)))
        return [0,0,0]

    if ctype == "pocket_mate":
        # The insert's mating point is its face that contacts the pocket bottom.
        # For a pocket on the insert, the bottom face is the deepest point in
        # the pocket direction.  We use the insert's bbox extreme in the
        # direction OPPOSITE to the pocket opening.
        ipkt = match.get("pocket_b") or match.get("pocket_a") or {}
        pkt_dir = ipkt.get("direction")
        if pkt_dir:
            pkt_dir = _norm(pkt_dir)
        else:
            pkt_dir = [0, 0, 1]
        bbox = features.get("bbox", {})
        lo = bbox.get("min", [0,0,0])
        hi = bbox.get("max", [0,0,0])
        # The face that goes into the pocket is the one farthest in the -pkt_dir direction
        proj = []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    proj.append(_dot([x,y,z], pkt_dir))
        t_min = min(proj)
        # The mating point is on the bbox face at t_min, at the center of that face
        bbox_center_opposite = [0.0, 0.0, 0.0]
        count = 0
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    if abs(_dot([x,y,z], pkt_dir) - t_min) < 0.01:
                        bbox_center_opposite[0] += x
                        bbox_center_opposite[1] += y
                        bbox_center_opposite[2] += z
                        count += 1
        if count > 0:
            bbox_center_opposite = [v/count for v in bbox_center_opposite]
        else:
            bbox_center_opposite = list(lo)
        return bbox_center_opposite

    if ctype in ("planar_mate", "planar_align"):
        planes = features.get("planes", [])
        idx = match.get("feat_b_idx", 0)
        face_normal = None
        if idx < len(planes):
            face_normal = planes[idx].get("normal", None)

        # Use bbox face corresponding to the matched planar face normal
        # rather than the face center (which may not represent the part body)
        bbox = features.get("bbox", {})
        lo = bbox.get("min", [0,0,0])
        hi = bbox.get("max", [0,0,0])

        if face_normal:
            fn = _norm(face_normal)
            # Find which bbox face best matches this normal direction
            bbox_faces = [
                (lo, [-1, 0, 0]),
                (hi, [1, 0, 0]),
                (lo, [0, -1, 0]),
                (hi, [0, 1, 0]),
                (lo, [0, 0, -1]),
                (hi, [0, 0, 1]),
            ]
            best_dot = -2.0
            best_corner = None
            for corner, normal in bbox_faces:
                d = _dot(fn, normal)
                if d > best_dot:
                    best_dot = d
                    best_corner = corner
            if best_corner and best_dot > 0.7:
                # Return the center of that bbox face
                face_center = list(best_corner)
                # Average with other corners that share this face position
                other_corners = []
                for corner, normal in bbox_faces:
                    if abs(_dot(fn, normal) - 1.0) < 0.01:
                        other_corners.append(list(corner))
                if other_corners:
                    avg = [0.0, 0.0, 0.0]
                    for c in other_corners:
                        avg[0] += c[0]; avg[1] += c[1]; avg[2] += c[2]
                    face_center = [v/len(other_corners) for v in avg]
                return face_center

        # Fallback to face position
        if idx < len(planes):
            return planes[idx].get("position", [0,0,0])

    return [0,0,0]


def _apply_contact_maximization(placements, features, receiver, case_dir, selected_edges, raw_matches):
    """Post-process: slide inserts along joint axis to maximize face contact.
    
    Uses JoinABLe SDF cost function. Skipped when mesh quality is insufficient
    (< 500 vertices for receiver). The stop-plane placement is geometrically
    optimal for simple shaft+flange cases.
    """
    import numpy as np
    try:
        from joinable_pose_solver import step_to_mesh, maximize_contact_along_axis
    except ImportError:
        return

    cyls = features[receiver].get("cylinders", [])
    if not cyls:
        return
    ref_cyl = max(cyls, key=lambda c: c["radius"])
    ref_axis = _norm(ref_cyl["axis"])
    ref_origin = ref_cyl["origin"]

    receiver_path = case_dir / receiver
    try:
        receiver_mesh = step_to_mesh(str(receiver_path))
    except Exception:
        return

    # Skip if mesh too coarse for SDF contact
    if len(receiver_mesh.vertices) < 500:
        return

    for insert_name, plac in placements.items():
        if insert_name == receiver:
            continue

        is_clearance = any(
            receiver in m["parts"] and insert_name in m["parts"] and m["type"] in ("coaxial", "clearance")
            for m in raw_matches
        )
        if not is_clearance:
            continue

        insert_path = case_dir / insert_name
        try:
            insert_mesh = step_to_mesh(str(insert_path))
        except Exception:
            continue

        translate = plac.get("translate", [0, 0, 0])
        rot_seq = plac.get("rotate_sequence", [])
        aff = np.eye(4)
        aff[:3, 3] = np.array(translate, dtype=float)
        for rot in reversed(rot_seq):
            aa = rot["axis_angle"]
            axis = np.array(aa[:3], dtype=float) / (np.linalg.norm(aa[:3]) + 1e-12)
            rad = np.deg2rad(aa[3])
            x, y, z = axis
            c = math.cos(rad); s = math.sin(rad); C = 1 - c
            R = np.array([
                [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
                [x*y*C+z*s, y*y*C+c, y*z*C-x*s],
                [x*z*C-y*s, y*z*C+x*s, z*z*C+c],
            ])
            aff[:3, :3] = R @ aff[:3, :3]

        # Search direction toward shaft center
        cyl_origin = np.array(ref_origin)
        cyl_axis = np.array(ref_axis)
        flange_proj = np.dot(np.array(translate), cyl_axis)
        shaft_center_proj = np.dot(cyl_origin, cyl_axis)
        toward_center = -cyl_axis if flange_proj > shaft_center_proj else cyl_axis

        best_offset, final_aff, ov, ct = maximize_contact_along_axis(
            insert_mesh, receiver_mesh,
            aff, ref_origin, toward_center,
            search_range=(0, 200),
            num_samples=4096, budget=20,
        )

        if abs(best_offset) > 0.5 and ov < 0.01:
            plac["translate"] = final_aff[:3, 3].tolist()
            print(f"  [max contact] {Path(insert_name).stem}: +{best_offset:.1f}mm"
                  f" (contact={ct:.3f})")


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

    # Group matches by insert part — pick best constraint type per insert
    # For shaft-like receivers: shaft constraints (clearance/coaxial) take priority
    # For non-shaft receivers: planar constraints take priority (large contact faces)
    if _is_shaft_like(features[receiver]):
        CONSTRAINT_PRIORITY = {"clearance": 5, "coaxial": 5, "pocket_mate": 4, "planar_mate": 3, "planar_align": 2}
    else:
        # planar_mate > planar_align because face-to-face contact is stronger evidence
        CONSTRAINT_PRIORITY = {"planar_mate": 5, "planar_align": 4, "pocket_mate": 3, "clearance": 2, "coaxial": 2}

    insert_best_match = {}
    for match in raw_matches:
        if receiver not in match["parts"]:
            continue
        pair = canonical_pair(match["parts"])
        if pair not in {canonical_pair(e) for e in selected_edges}:
            continue
        insert = match["parts"][0] if match["parts"][1] == receiver else match["parts"][1]
        prio = CONSTRAINT_PRIORITY.get(match["type"], 0)

        # For planar constraints, boost priority for larger faces (primary contact surfaces)
        if match["type"] in ("planar_mate", "planar_align"):
            recv_idx = match.get("feat_a_idx" if match["parts"][0] == receiver else "feat_b_idx", -1)
            recv_planes = features[receiver].get("planes", [])
            if 0 <= recv_idx < len(recv_planes):
                area = recv_planes[recv_idx].get("area", 0) or 0
                # Large faces (>2000 mm²) get a priority boost
                if area > 2000:
                    prio += 1

        if insert not in insert_best_match:
            insert_best_match[insert] = match
        else:
            old_prio = CONSTRAINT_PRIORITY.get(insert_best_match[insert]["type"], 0)
            old_recv_idx = insert_best_match[insert].get(
                "feat_a_idx" if insert_best_match[insert]["parts"][0] == receiver else "feat_b_idx", -1)
            old_planes = features[receiver].get("planes", [])
            if 0 <= old_recv_idx < len(old_planes):
                old_area = old_planes[old_recv_idx].get("area", 0) or 0
                if old_area > 2000:
                    old_prio += 1

            if prio > old_prio:
                insert_best_match[insert] = match
            elif prio == old_prio:
                # Tiebreak by receiver face area (larger = better)
                recv_idx_new = match.get("feat_a_idx" if match["parts"][0] == receiver else "feat_b_idx", -1)
                new_planes = features[receiver].get("planes", [])
                new_area = new_planes[recv_idx_new].get("area", 0) if 0 <= recv_idx_new < len(new_planes) else 0
                old_area2 = old_planes[old_recv_idx].get("area", 0) if 0 <= old_recv_idx < len(old_planes) else 0
                if (new_area or 0) > (old_area2 or 0):
                    insert_best_match[insert] = match

    inserts = list(insert_best_match.items())
    print(f"Inserts to place: {[(Path(k).stem, v['type']) for k,v in inserts]}")

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

            # Handle new shaft_faces format
            if isinstance(stop, dict) and stop.get("type") == "shaft_faces":
                face_minus = stop["face_minus"]
                face_plus = stop["face_plus"]
                shaft_axis = stop["axis"]
            else:
                # Legacy tuple format
                face_plus = {"position": stop[0], "normal": stop[1]}
                face_minus = face_plus
                shaft_axis = stop[1]

            # Axis alignment
            from coordinate_solver import _global_vector
            ref_axis = _global_vector(
                features[receiver]["cylinders"][0]["axis"], placements[receiver]
            )
            axis_key = tuple(round(v, 2) for v in ref_axis)
            ends_used = len(used_ends.get(axis_key, []))

            tgt_cyls = features[insert].get("cylinders", [])
            if tgt_cyls:
                tgt_cyl = max(tgt_cyls, key=lambda c: c["radius"])
                tgt_axis = tgt_cyl["axis"]
                tgt_origin = tgt_cyl["origin"]

                ibbox = features[insert].get("bbox", {})
                ilo = ibbox.get("min", [0,0,0])
                ihi = ibbox.get("max", [0,0,0])
                insert_center = [(ilo[i]+ihi[i])/2 for i in range(3)]
                body_dir_raw = _sub(insert_center, tgt_origin)
                body_dot_raw = _dot(body_dir_raw, tgt_axis)

                if ends_used == 0:
                    need_reverse = body_dot_raw < 0
                else:
                    need_reverse = body_dot_raw > 0

                if need_reverse:
                    target_align_axis = _scale(ref_axis, -1.0)
                else:
                    target_align_axis = ref_axis

                if abs(_dot(tgt_axis, target_align_axis)) < 0.999:
                    from coordinate_solver import _axis_angle_to_rotation
                    rot_axis, rot_angle = _axis_angle_to_rotation(tgt_axis, target_align_axis)
                    rot_seq = [{'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                                               math.degrees(rot_angle)]}] if rot_axis else []
                else:
                    rot_seq = []

                if need_reverse:
                    print(f"  [reverse align] bore→-ref (body_dot_raw={body_dot_raw:.2f})")
            else:
                rot_seq = []
                need_reverse = False

            # Mate point: flange disc face (face-to-face contact with shaft end)
            mate_pt = _insert_mating_point(features[insert], match, use_plus_end=False)

            # Target: shaft end face
            if ends_used == 0:
                target = face_plus["position"]
            else:
                target = face_minus["position"]

            # Compute translation
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

            r_dir = rpkt.get("direction", [0,0,1])
            i_dir = ipkt.get("direction", r_dir)

            # Stop plane: receiver pocket bottom (center is the cluster center,
            # but the actual pocket bottom is offset from center in -direction)
            r_center = rpkt.get("center", [0,0,0])
            r_size = rpkt.get("size", [0,0,0])
            # Pocket bottom is at the deepest point along -direction
            r_proj = _dot(r_center, _norm(r_dir))
            # The bottom face is at center - (half the pocket depth) in direction
            # Pocket size component along direction:
            depth = abs(_dot(r_size, _norm(r_dir))) if any(r_size) else 0.0
            pocket_bottom = _sub(r_center, _scale(_norm(r_dir), depth * 0.5))
            stop = (pocket_bottom, _norm(r_dir))

            # Insert mating point: bbox face in -direction (the face that bottoms out)
            mate = _insert_mating_point(features[insert], match)

            # Direction alignment rotation
            rot_seq = []
            r_dir_n = _norm(r_dir)
            i_dir_n = _norm(i_dir)
            dot_d = _dot(i_dir_n, r_dir_n)
            if dot_d < 0.999:
                from coordinate_solver import _axis_angle_to_rotation
                rot_axis, rot_angle = _axis_angle_to_rotation(i_dir_n, r_dir_n)
                if rot_axis:
                    rot_seq = [{'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                                               math.degrees(rot_angle)]}]

            from coordinate_solver import _apply_rotation_to_vector
            rotated_mate = _apply_rotation_to_vector(mate, rot_seq)
            translate = _sub(stop[0], rotated_mate)

            placement = {"translate": translate}
            if rot_seq:
                placement["rotate_sequence"] = rot_seq
            placements[insert] = placement

        elif match["type"] in ("planar_mate", "planar_align"):
            # Use the planar face as stop plane
            receiver_feat_idx = match.get("feat_a_idx" if match["parts"][0] == receiver else "feat_b_idx", 0)
            stop = _stop_plane_from_planar(features[receiver], receiver_feat_idx)

            # For insert, find the matching planar face
            insert_feat_idx = match.get("feat_b_idx" if match["parts"][0] == receiver else "feat_a_idx", 0)
            insert_planes = features[insert].get("planes", [])
            if insert_feat_idx < len(insert_planes):
                mate = insert_planes[insert_feat_idx].get("position", [0,0,0])
            else:
                mate = _insert_mating_point(features[insert], match)

            # For planar_mate: normals opposite (face-to-face contact)
            # For planar_align: normals same direction
            if stop and mate:
                # Align normals
                r_normal = stop[1]
                i_normal = insert_planes[insert_feat_idx].get("normal", [0,0,1]) if insert_feat_idx < len(insert_planes) else [0,0,1]
                rot_seq = []
                if match["type"] == "planar_mate":
                    # Mate: normals should be anti-parallel
                    target_normal = _scale(r_normal, -1)
                else:
                    # Align: normals should be parallel
                    target_normal = r_normal
                dot_n = _dot(i_normal, target_normal)
                if dot_n < 0.999:
                    from coordinate_solver import _axis_angle_to_rotation
                    rot_axis, rot_angle = _axis_angle_to_rotation(i_normal, target_normal)
                    if rot_axis:
                        rot_seq = [{'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                                                   math.degrees(rot_angle)]}]

                from coordinate_solver import _apply_rotation_to_vector
                rotated_mate = _apply_rotation_to_vector(mate, rot_seq)
                translate = _sub(stop[0], rotated_mate)

                placement = {"translate": translate}
                if rot_seq:
                    placement["rotate_sequence"] = rot_seq
                placements[insert] = placement

        print(f"  → translate={[round(v,1) for v in placements[insert].get('translate',[0,0,0])]}")

    # ── Contact maximization: slide inserts along axis for max face contact ──
    _apply_contact_maximization(placements, features, receiver, case_dir, selected_edges, raw_matches)

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
