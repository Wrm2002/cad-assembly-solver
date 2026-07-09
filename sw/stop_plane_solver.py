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

    # ── Detect face-to-face stacks: inserts that have planar_mate with each other ──
    # These should be placed on the SAME shaft end, stacked face-to-face.
    insert_names = [k for k, v in inserts]
    flange_mates = {}  # insert_name → mate_insert_name
    for match in raw_matches:
        if match["type"] != "planar_mate":
            continue
        p0, p1 = match["parts"]
        if p0 in insert_names and p1 in insert_names:
            flange_mates[p0] = p1
            flange_mates[p1] = p0
            print(f"  [stack] {Path(p0).stem} <-> {Path(p1).stem} planar_mate detected")

    # Build stack groups
    visited = set()
    stacks = []
    for ins in insert_names:
        if ins in visited:
            continue
        stack = [ins]
        visited.add(ins)
        # Follow mates
        cur = ins
        while cur in flange_mates:
            nxt = flange_mates[cur]
            if nxt not in visited:
                stack.append(nxt)
                visited.add(nxt)
                cur = nxt
            else:
                break
        if len(stack) > 1:
            stacks.append(stack)
            print(f"  Stack group: {[Path(s).stem for s in stack]}")

    # Place receiver at origin
    placements = {receiver: {"translate": [0.0, 0.0, 0.0]}}
    used_ends: dict[str, list[str]] = defaultdict(list)  # axis_key → list of placed inserts
    placed_in_stack: set = set()  # track which inserts are handled by stack logic

    for insert, match in inserts:
        # Skip if already placed as part of a stack
        if insert in placed_in_stack:
            continue

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

            # ── Stack mate: if this flange has a planar_mate with another insert,
            #     place the mate on the SAME shaft end, face-to-face ──
            if insert in flange_mates:
                mate_name = flange_mates[insert]
                if mate_name not in placements:
                    print(f"\nPlacing {Path(mate_name).stem} (stacked on {Path(insert).stem})...")
                    mate_match = insert_best_match.get(mate_name)
                    if mate_match:
                        # Same axis alignment as the first flange (no need_reverse for stacked)
                        mate_tgt_cyls = features[mate_name].get("cylinders", [])
                        if mate_tgt_cyls:
                            mate_tgt_cyl = max(mate_tgt_cyls, key=lambda c: c["radius"])
                            mate_tgt_axis = mate_tgt_cyl["axis"]
                            mate_tgt_origin = mate_tgt_cyl["origin"]

                            # Body direction check for mate
                            mibbox = features[mate_name].get("bbox", {})
                            milo = mibbox.get("min", [0,0,0])
                            mihi = mibbox.get("max", [0,0,0])
                            mate_center = [(milo[i]+mihi[i])/2 for i in range(3)]
                            mate_body_raw = _sub(mate_center, mate_tgt_origin)
                            mate_body_dot = _dot(mate_body_raw, mate_tgt_axis)

                            # For a stacked flange (same end as first), body should
                            # also point away from shaft. Use same reverse logic.
                            if ends_used == 0:
                                mate_reverse = mate_body_dot < 0
                            else:
                                mate_reverse = mate_body_dot > 0

                            if mate_reverse:
                                mate_target_axis = _scale(ref_axis, -1.0)
                            else:
                                mate_target_axis = ref_axis

                            if abs(_dot(mate_tgt_axis, mate_target_axis)) < 0.999:
                                from coordinate_solver import _axis_angle_to_rotation
                                m_rot_axis, m_rot_angle = _axis_angle_to_rotation(mate_tgt_axis, mate_target_axis)
                                mate_rot_seq = [{'axis_angle': [m_rot_axis[0], m_rot_axis[1], m_rot_axis[2],
                                                               math.degrees(m_rot_angle)]}] if m_rot_axis else []
                            else:
                                mate_rot_seq = []

                            if mate_reverse:
                                print(f"  [reverse align] stacked mate bore→-ref (body_dot={mate_body_dot:.2f})")
                        else:
                            mate_rot_seq = []

                        # Mate point: mate flange's disc face
                        mate_pt2 = _insert_mating_point(features[mate_name], mate_match, use_plus_end=False)

                        # Target: the placed flange's disc face (the face on the pipe side)
                        # The placed flange's pipe-side face is where the second flange mates
                        # Use the placed flange's +axis face (pipe side = opposite of the disc face)
                        placed_disc_face = _insert_mating_point(features[insert], match, use_plus_end=True)
                        from coordinate_solver import _apply_rotation_to_vector
                        placed_disc_rotated = _apply_rotation_to_vector(placed_disc_face, rot_seq)
                        placed_disc_world = _add(translate, placed_disc_rotated)

                        # Target is the placed flange's pipe-side face
                        target2 = placed_disc_world

                        mate_pt2_rotated = _apply_rotation_to_vector(mate_pt2, mate_rot_seq)
                        translate2 = _sub(target2, mate_pt2_rotated)

                        placement2 = {"translate": translate2}
                        if mate_rot_seq:
                            placement2["rotate_sequence"] = mate_rot_seq
                        placements[mate_name] = placement2
                        placed_in_stack.add(mate_name)
                        used_ends[axis_key].append(mate_name)
                        print(f"  -> translate={[round(v,1) for v in translate2]}")

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
