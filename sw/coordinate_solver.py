"""
coordinate_solver.py — Phase 2: global-frame constraint solving.

Takes match pairs from Phase 1 (feature indices) + part features,
establishes a global coordinate frame at origin (0,0,0),
and computes each part's placement transform via BFS graph propagation.

Architecture:
  Phase 1 (constraints.py):       match_features()          — "who pairs with whom"
  Phase 2 (this file):            solve_in_global_frame()   — "how to place"
  Phase 3 (refinement.py):        refine_placements()       — "fine-tune"
"""

import math

from constraints import (
    COAXIAL, CLEARANCE, PLANAR_MATE, PLANAR_ALIGN,
    _vec_dot, _vec_cross, _vec_norm, _vec_sub,
)


# ── axis-angle rotation helper ────────────────────────────────────

def _apply_rotation_to_vector(vec, rotate_seq):
    """Apply a rotation sequence to a free vector (direction or translation)."""
    if not rotate_seq or not any(abs(v) > 1e-12 for v in vec):
        return list(vec)
    try:
        from build_assembly import transform_point
        temp = {'rotate_sequence': list(rotate_seq), 'translate': [0.0, 0.0, 0.0]}
        rx, ry, rz = transform_point(temp, vec[0], vec[1], vec[2])
        return [rx, ry, rz]
    except Exception:
        return list(vec)


def _global_vector(vec, placement):
    """Transform a local direction by an already placed part's rotation."""
    return _apply_rotation_to_vector(vec, placement.get('rotate_sequence', []))


def _global_point(point, placement):
    """Transform a local feature point into the current global frame."""
    rotated = _global_vector(point, placement)
    translation = placement.get('translate', [0.0, 0.0, 0.0])
    return [rotated[i] + float(translation[i]) for i in range(3)]


def _axis_angle_to_rotation(from_dir, to_dir):
    """Return (axis, angle_radians) to rotate from_dir to to_dir."""
    src = _vec_norm(from_dir)
    tgt = _vec_norm(to_dir)
    dot = _vec_dot(src, tgt)

    if abs(dot - 1.0) < 1e-9:
        return None, 0.0
    if abs(dot + 1.0) < 1e-9:
        # 180° flip — pick perpendicular axis
        if abs(tgt[0]) < 0.9:
            perp = _vec_norm(_vec_cross(tgt, (1, 0, 0)))
        else:
            perp = _vec_norm(_vec_cross(tgt, (0, 1, 0)))
        return perp, math.pi

    axis = _vec_norm(_vec_cross(src, tgt))
    angle = math.acos(max(-1.0, min(1.0, dot)))
    return axis, angle


# ── feature resolution from match indices ─────────────────────────

def _resolve_match_features(match, ref_part, target_part, parts_features):
    """
    Given a match pair and which part is ref / target, return
    (ref_feature_dict, target_feature_dict).

    Match format:
      parts[0] = pi,  parts[1] = pj
      feat_a_idx → parts_features[pi][key]
      feat_b_idx → parts_features[pj][key]
      For CLEARANCE: parts[0]=shaft, parts[1]=bore.
    """
    pi, pj = match['parts']
    ctype = match['type']

    if ctype == CLEARANCE:
        # parts[0] = shaft, parts[1] = bore
        feat_key = 'cylinders'
        shaft_part, bore_part = pi, pj
        if ref_part == shaft_part:
            # ref is shaft → target is bore
            ref_feat = parts_features[shaft_part][feat_key][match['feat_a_idx']]
            target_feat = parts_features[bore_part][feat_key][match['feat_b_idx']]
            ref_role = 'shaft'
            target_role = 'bore'
        else:
            # ref is bore → target is shaft
            ref_feat = parts_features[bore_part][feat_key][match['feat_b_idx']]
            target_feat = parts_features[shaft_part][feat_key][match['feat_a_idx']]
            ref_role = 'bore'
            target_role = 'shaft'
        return ref_feat, target_feat, ref_role, target_role

    # COAXIAL, PLANAR_MATE, PLANAR_ALIGN: symmetric semantics
    feat_key = 'cylinders' if ctype == COAXIAL else 'planes'

    if ref_part == pi:
        ref_feat = parts_features[pi][feat_key][match['feat_a_idx']]
        target_feat = parts_features[pj][feat_key][match['feat_b_idx']]
    else:
        ref_feat = parts_features[pj][feat_key][match['feat_b_idx']]
        target_feat = parts_features[pi][feat_key][match['feat_a_idx']]

    return ref_feat, target_feat, None, None


# ── axial contact helper ──────────────────────────────────────────

def _part_bbox_interval_along_axis(part_features, placement, axis):
    """Return [t_min, t_max] interval of part bbox projected onto *axis*."""
    bbox = part_features.get("bbox")
    if not bbox:
        return None
    low, high = bbox["min"], bbox["max"]
    values = []
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            for z in (low[2], high[2]):
                point = _global_point([x, y, z], placement)
                values.append(_vec_dot(point, axis))
    return [min(values), max(values)]


def _apply_axial_contact(
    placement, ref_part, target_part, parts_features,
    ref_placement, shared_axis,
):
    """Slide *placement* along *shared_axis* so target just touches ref.

    After coaxial alignment the axial position is arbitrary (set by cylinder
    origins).  This function uses bounding-box projections to slide the
    target until its bbox is adjacent to the reference bbox along the shared
    axis — emulating a planar-mate / shoulder-stop constraint.
    """
    ref_feats = parts_features.get(ref_part)
    tgt_feats = parts_features.get(target_part)
    if not ref_feats or not tgt_feats:
        return placement

    # Build a temporary placement for the target to compute its global bbox
    tgt_placement = dict(placement)
    # The placement currently has rotation but the translate was set for
    # cylinder-origin alignment.  We need to compute the bbox of the target
    # in this position.
    ref_interval = _part_bbox_interval_along_axis(ref_feats, ref_placement, shared_axis)
    tgt_interval = _part_bbox_interval_along_axis(tgt_feats, tgt_placement, shared_axis)

    if not ref_interval or not tgt_interval:
        return placement

    ref_min, ref_max = ref_interval
    tgt_min, tgt_max = tgt_interval

    # Compute how far to slide the target so its bbox touches the ref bbox
    # Try both sides: slide toward ref_min or toward ref_max
    # Choose the direction that requires less movement
    
    # Current overlap/gap
    gap_to_min = tgt_max - ref_min  # if positive: overlap, if negative: gap (tgt before ref)
    gap_to_max = ref_max - tgt_min  # if positive: overlap, if negative: gap (ref before tgt)

    # We want the target to touch the nearest end of the ref
    # If target is currently to the left of ref (tgt_max < ref_min):
    #   → slide right by (ref_min - tgt_max)
    # If target is to the right of ref (tgt_min > ref_max):
    #   → slide left by (tgt_min - ref_max)
    # If target overlaps ref:
    #   → slide to exit overlap, toward the closer side
    
    if tgt_max <= ref_min:
        # Target is completely to the left → slide right to touch
        slide = ref_min - tgt_max
    elif tgt_min >= ref_max:
        # Target is completely to the right → slide left to touch
        slide = ref_max - tgt_min
    else:
        # Overlap: exit toward the closer edge
        dist_to_min = abs(tgt_min - ref_min)
        dist_to_max = abs(tgt_max - ref_max)
        if dist_to_min < dist_to_max:
            slide = ref_min - tgt_min
        else:
            slide = ref_max - tgt_max

    # Apply the slide along the shared axis
    axis_unit = _vec_norm(shared_axis)
    current_translate = list(placement.get('translate', [0.0, 0.0, 0.0]))
    new_placement = dict(placement)
    new_placement['translate'] = [
        current_translate[i] + slide * axis_unit[i] for i in range(3)
    ]
    return new_placement

def _compute_placement_from_match(match, ref_part, target_part,
                                   parts_features, ref_placement):
    """
    Compute placement for target_part that satisfies one match edge
    by aligning target's matched feature to ref's matched feature.

    All computation is in the global frame: ref_part is already placed,
    target_part needs its transform.
    """
    ctype = match['type']

    if ctype == COAXIAL:
        ref_feat, target_feat, _, _ = _resolve_match_features(
            match, ref_part, target_part, parts_features)
        from_dir = target_feat['axis']   # rotate target's axis
        to_dir = _global_vector(ref_feat['axis'], ref_placement)
        dot_check = abs(_vec_dot(from_dir, to_dir))

        rot_axis, rot_angle = None, 0.0
        if dot_check < 0.999:
            rot_axis, rot_angle = _axis_angle_to_rotation(from_dir, to_dir)

        plac = {}
        rotate_seq = []
        if rot_axis:
            rotate_seq = [
                {'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                               math.degrees(rot_angle)]}
            ]
            plac['rotate_sequence'] = rotate_seq

        # ── Coaxial origin alignment ──
        # After rotation, the cylinder origins should coincide.
        ref_origin = _global_point(ref_feat['origin'], ref_placement)
        tgt_origin = target_feat['origin']
        rotated_origin = _apply_rotation_to_vector(tgt_origin, rotate_seq)
        plac['translate'] = [
            ref_origin[0] - rotated_origin[0],
            ref_origin[1] - rotated_origin[1],
            ref_origin[2] - rotated_origin[2],
        ]
        return plac

    if ctype == CLEARANCE:
        ref_feat, target_feat, ref_role, target_role = _resolve_match_features(
            match, ref_part, target_part, parts_features)

        # Use target part's MAIN cylinder axis, not necessarily the matched one
        # (the matched cylinder might be a small inner bore with opposite direction)
        target_cyls = parts_features.get(target_part, {}).get('cylinders', [])
        if target_cyls:
            target_main = max(target_cyls, key=lambda c: c['radius'])
            from_dir = target_main['axis']
        else:
            from_dir = target_feat['axis']
        to_dir = _global_vector(ref_feat['axis'], ref_placement)

        # For clearance, parallelism is enough (anti-parallel is fine)
        dot_check = abs(_vec_dot(from_dir, to_dir))

        rot_axis, rot_angle = None, 0.0
        if dot_check < 0.999:
            rot_axis, rot_angle = _axis_angle_to_rotation(from_dir, to_dir)

        plac = {}
        if rot_axis:
            plac['rotate_sequence'] = [
                {'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                               math.degrees(rot_angle)]}
            ]
        ref_origin = _global_point(ref_feat['origin'], ref_placement)
        rotated_origin = _apply_rotation_to_vector(
            target_feat['origin'], plac.get('rotate_sequence', [])
        )
        plac['translate'] = [
            ref_origin[i] - rotated_origin[i] for i in range(3)
        ]
        # ── Axial contact positioning ──
        # After coaxial alignment, slide the target part along the shared axis
        # so the two bounding boxes touch (no gap, no overlap).  This replaces
        # the missing planar-mate constraint when the shaft shoulder and flange
        # face are not parallel in their local coordinate frames.
        plac = _apply_axial_contact(
            plac, ref_part, target_part, parts_features,
            ref_placement, to_dir,
        )
        return plac

    if ctype == PLANAR_MATE:
        ref_feat, target_feat, _, _ = _resolve_match_features(
            match, ref_part, target_part, parts_features)
        from_dir = target_feat['normal']
        # Mate: target normal should face OPPOSITE of ref normal
        global_ref_normal = _global_vector(ref_feat['normal'], ref_placement)
        to_dir = tuple(-value for value in global_ref_normal)

        rot_axis, rot_angle = None, 0.0
        dot = _vec_dot(from_dir, to_dir)
        if dot < 0.999:
            rot_axis, rot_angle = _axis_angle_to_rotation(from_dir, to_dir)

        plac = {}
        if rot_axis:
            plac['rotate_sequence'] = [
                {'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                               math.degrees(rot_angle)]}
            ]
        # Compute signed translation directly from feature positions:
        # move target_feat onto the plane defined by ref_feat (mate: normals opposite)
        ref_pos = _global_point(ref_feat['position'], ref_placement)
        tgt_pos = target_feat['position']
        ref_n = global_ref_normal
        rotated_tgt = _apply_rotation_to_vector(
            tgt_pos, plac.get('rotate_sequence', [])
        )
        plac['translate'] = [
            ref_pos[i] - rotated_tgt[i] for i in range(3)
        ]
        return plac

    if ctype == PLANAR_ALIGN:
        ref_feat, target_feat, _, _ = _resolve_match_features(
            match, ref_part, target_part, parts_features)
        from_dir = target_feat['normal']
        global_ref_normal = _global_vector(ref_feat['normal'], ref_placement)
        to_dir = global_ref_normal

        rot_axis, rot_angle = None, 0.0
        dot = _vec_dot(from_dir, to_dir)
        if dot < 0.999:
            rot_axis, rot_angle = _axis_angle_to_rotation(from_dir, to_dir)

        plac = {}
        if rot_axis:
            plac['rotate_sequence'] = [
                {'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                               math.degrees(rot_angle)]}
            ]
        # Compute signed translation directly from feature positions:
        # move target_feat onto the plane defined by ref_feat (align: normals same)
        ref_pos = _global_point(ref_feat['position'], ref_placement)
        tgt_pos = target_feat['position']
        rotated_tgt = _apply_rotation_to_vector(
            tgt_pos, plac.get('rotate_sequence', [])
        )
        plac['translate'] = [
            ref_pos[i] - rotated_tgt[i] for i in range(3)
        ]
        return plac

    if ctype == 'pocket_mate':
        # Skip pocket mate rotation for parts with unreliable mesh cylinders
        # (tessellated sheet metal). Parts with few/zero cylinders are OK —
        # their pocket geometry is structural (e.g. key, keyway).
        from constraints import _classify_part
        tfeats = parts_features.get(target_part, {})
        n_cyl = len(tfeats.get('cylinders', []))
        if n_cyl > 50 and _classify_part(tfeats) == 'planar':
            return None  # mesh artifact part, skip

        # Pocket mate: align the pocket bottom direction and wall normal
        # to establish rotational constraint. No translation computed here —
        # planar/coaxial constraints handle positioning.
        dir_a = match.get('_dir_a')
        dir_b = match.get('_dir_b')
        wall_a = match.get('_wall_a')
        wall_b = match.get('_wall_b')

        if not dir_a or not dir_b:
            return None

        plac = {}
        rotate_seq = []

        # Align target pocket direction to ref pocket direction (mate = opposite)
        from_dir = dir_b if match['parts'][1] == target_part else dir_a
        to_dir = dir_a if match['parts'][1] == target_part else dir_b
        # For a mate, the pocket directions should be opposite (one faces out, one faces in)
        # Use negative for mate semantics
        to_dir_mate = (-to_dir[0], -to_dir[1], -to_dir[2])

        dot = _vec_dot(from_dir, to_dir_mate)
        if dot < 0.999:
            rot_axis, rot_angle = _axis_angle_to_rotation(from_dir, to_dir_mate)
            if rot_axis:
                rotate_seq.append({
                    'axis_angle': [rot_axis[0], rot_axis[1], rot_axis[2],
                                   math.degrees(rot_angle)]
                })

        # Wall normal alignment (secondary rotation, around the now-aligned direction)
        if wall_a and wall_b and rotate_seq:
            wall_from = wall_b if match['parts'][1] == target_part else wall_a
            wall_to = wall_a if match['parts'][1] == target_part else wall_b
            # Apply first rotation to wall_from
            rotated_wall = _apply_rotation_to_vector(list(wall_from), rotate_seq)
            dot_w = _vec_dot(rotated_wall, wall_to)
            if abs(dot_w) < 0.999:
                # The rotation axis is the already-aligned pocket direction
                aligned_dir = _apply_rotation_to_vector(list(from_dir), rotate_seq)
                rot_axis2, rot_angle2 = _axis_angle_to_rotation(
                    rotated_wall, list(wall_to))
                if rot_axis2 and abs(rot_angle2) > 0.001:
                    rotate_seq.append({
                        'axis_angle': [rot_axis2[0], rot_axis2[1], rot_axis2[2],
                                       math.degrees(rot_angle2)]
                    })

        if rotate_seq:
            plac['rotate_sequence'] = rotate_seq

        # Pocket mate translation: align pocket centers after rotation.
        # Determine which pocket belongs to ref and which to target
        pkt_a = match.get('pocket_a', {})
        pkt_b = match.get('pocket_b', {})
        if match['parts'][0] == ref_part:
            pkt_ref = pkt_a
            pkt_tgt = pkt_b
        else:
            pkt_ref = pkt_b
            pkt_tgt = pkt_a

        center_ref = pkt_ref.get('center')
        center_tgt = pkt_tgt.get('center')

        if center_ref and center_tgt:
            # Rotate target pocket center, then compute translation
            rotated_center = _apply_rotation_to_vector(list(center_tgt), rotate_seq)
            plac['translate'] = [
                center_ref[0] - rotated_center[0],
                center_ref[1] - rotated_center[1],
                center_ref[2] - rotated_center[2],
            ]
        else:
            plac['translate'] = [0.0, 0.0, 0.0]
        return plac

    return None


# ── combined placement from multiple matches ──────────────────────

def _compute_combined_placement(matches_for_pair, ref_part, target_part,
                                 parts_features, ref_placement):
    """
    Merge multiple matches between the same (ref, target) pair into one placement.
    - Rotation: taken from the first coaxial or clearance match
    - Translation: taken from the first planar match (mate or align)
    Falls back to single-match behavior if only one match type exists.
    """
    rotation_plac = None
    translation_plac = None

    for match in matches_for_pair:
        ctype = match['type']
        plac = _compute_placement_from_match(
            match, ref_part, target_part, parts_features, ref_placement
        )
        if not plac:
            continue

        if ctype in (COAXIAL, CLEARANCE, 'pocket_mate') and rotation_plac is None:
            rotation_plac = plac
        elif ctype in (PLANAR_MATE, PLANAR_ALIGN) and translation_plac is None:
            translation_plac = plac

    if rotation_plac is None and translation_plac is None:
        return None

    merged = {'translate': [0.0, 0.0, 0.0]}
    rotate_seq = []

    if rotation_plac:
        rotate_seq.extend(rotation_plac.get('rotate_sequence', []))
        # Coaxial/Clearance provides both rotation and translation.
        # The coaxial origin alignment is the authoritative positioning.
        merged['translate'] = list(rotation_plac.get('translate', [0.0, 0.0, 0.0]))

    if translation_plac:
        # Planar provides face-to-face distance.
        # When coaxial rotation exists, the coaxial origin alignment already
        # positions the parts correctly (they share a common CAD origin).
        # Planar translation is only used when there is NO coaxial constraint.
        if not rotation_plac:
            merged['translate'] = list(translation_plac.get('translate', [0.0, 0.0, 0.0]))
            for r in translation_plac.get('rotate_sequence', []):
                if r not in rotate_seq:
                    rotate_seq.append(r)

    if rotate_seq:
        merged['rotate_sequence'] = rotate_seq

    return merged


# ── main solver: BFS propagation in global frame ──────────────────

def solve_in_global_frame(parts_features, matches):
    """
    Phase 2 — solve placements in a global coordinate frame.

    Global frame origin: (0, 0, 0).
    Algorithm:
      1. Build adjacency graph from match pairs
      2. Pick reference part (most matches + largest cylinder)
         → identity placement (sits at origin)
      3. BFS: for each unvisited neighbor, collect ALL matches between
         (ref, target) and merge rotation (coaxial/clearance) with
         translation (planar) into a single placement.

    Returns {part_name: placement_dict} with rotate_sequence and translate.
    """
    part_names = list(parts_features.keys())
    if len(part_names) == 1:
        return {part_names[0]: {'translate': [0.0, 0.0, 0.0]}}

    # Build adjacency graph
    adj = {p: [] for p in part_names}
    for m in matches:
        a, b = m['parts']
        adj[a].append((b, m))
        adj[b].append((a, m))

    # Pick reference: most coaxial/clearance edges (not planar-only),
    # tiebreak by largest main cylinder
    def part_weight(p):
        feats = parts_features[p]
        main_r = max((c['radius'] for c in feats.get('cylinders', [])), default=0)
        axial_edges = sum(1 for (n, m) in adj[p] if m['type'] in (COAXIAL, CLEARANCE))
        return (axial_edges, main_r)

    ref = max(part_names, key=part_weight)

    # BFS propagation from reference
    solved = {ref: {'translate': [0.0, 0.0, 0.0]}}  # reference = identity at origin
    queue = [ref]
    visited = {ref}

    while queue:
        current = queue.pop(0)

        # Group matches by neighbor → collect all between same (ref, target) pair
        neighbor_matches = {}
        for neighbor, match in adj[current]:
            if neighbor in visited:
                continue
            if neighbor not in neighbor_matches:
                neighbor_matches[neighbor] = []
            neighbor_matches[neighbor].append(match)

        for neighbor, match_list in neighbor_matches.items():
            placement = _compute_combined_placement(
                match_list, current, neighbor, parts_features, solved[current]
            )

            if placement:
                solved[neighbor] = placement
                visited.add(neighbor)
                queue.append(neighbor)

    # Any unsolved parts → identity
    for p in part_names:
        if p not in solved:
            solved[p] = {'translate': [0.0, 0.0, 0.0]}

    # ── Multi-constraint relaxation ──
    # DISABLED: single-pass BFS with coaxial-only translation is
    # more reliable than averaging across multiple constraints.
    # Re-enable when iterative constraint solving is needed.
    # for _ in range(3):
    #     for part in part_names:
    #         ...

    return solved


# ── convert solved placements to manifest component list ──────────

def placements_to_manifest(parts_features, solved_placements):
    """
    Convert solved placements dict to manifest component list.
    Translations are rounded to 6 decimal places to eliminate
    floating-point noise (e.g., 1.11e-15 → 0.0).
    """
    components = []
    for i, (name, placement) in enumerate(solved_placements.items()):
        # Round translation values
        trans = placement.get('translate', [0.0, 0.0, 0.0])
        # Round to eliminate floating-point noise (< 1e-9 → 0.0, else 3 decimals)
        clean_trans = [0.0 if abs(v) < 1e-9 else round(v, 3) for v in trans]
        comp = {
            'id': f"comp_{i+1:02d}",
            'source': name,
            'label': name.replace('.step', '').replace('.stp', ''),
            'role': 'component',
            'placement': {
                'translate': clean_trans,
            },
        }
        if placement.get('rotate_sequence'):
            comp['placement']['rotate_sequence'] = placement['rotate_sequence']
        components.append(comp)
    return components


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    from features import extract_features
    from constraints import match_features

    folder = sys.argv[1] if len(sys.argv) > 1 else '1'
    folder = os.path.abspath(folder)

    step_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.step', '.stp'))
        and not f.lower().startswith('assembly')
    ])

    print(f"Folder: {folder}")
    print(f"Parts: {len(step_files)}")

    parts = {}
    for f in step_files:
        fp = os.path.join(folder, f)
        parts[f] = extract_features(fp)

    # Phase 1
    matches = match_features(parts)
    print(f"\nPhase 1 — matches: {len(matches)}")

    # Phase 2
    solved = solve_in_global_frame(parts, matches)
    print(f"\nPhase 2 — global-frame placements:")
    for name, p in solved.items():
        rot = p.get('rotate_sequence', [])
        trans = p.get('translate', [0, 0, 0])
        print(f"  {name}: translate={trans}, rotations={len(rot)}")
