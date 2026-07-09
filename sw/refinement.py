"""
refinement.py — Phase 3: strategy-based refinement in global frame.

After Phase 2 places all parts via constraint solving, this module applies
specialized refinement strategies to fine-tune placements. Each strategy
declares when it can apply and what it does.

Architecture:
  Phase 1 (constraints.py):        match_features()         — feature pairing
  Phase 2 (coordinate_solver.py):  solve_in_global_frame()  — initial placement
  Phase 3 (this file):             refine_placements()      — fine-tune

Strategy priority: specialized heuristics first, generic constraint solving
as fallback for unmatched cases.
"""

import copy
import math
import os

from constraints import COAXIAL, CLEARANCE, PLANAR_MATE, PLANAR_ALIGN
# Import part classifier to skip cylinder-centric strategies on planar parts
try:
    from constraints import _classify_part
except ImportError:
    def _classify_part(feats):
        return 'cylindrical'  # fallback


# ── strategy base class ───────────────────────────────────────────

class RefinementStrategy:
    """A named, self-describing placement refinement."""
    name: str = "base"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        """Return True if this strategy should run on this component."""
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        """Modify comp['placement'] in-place. Return True if anything changed."""
        return False


# ── helper: apply rotation to direction ───────────────────────────

def _apply_rotation_to_dir(direction, rotate_seq):
    """Apply a rotation sequence to a direction vector via OCCT."""
    if not rotate_seq:
        return tuple(direction)
    from build_assembly import transform_point
    temp = {'rotate_sequence': list(rotate_seq), 'translate': [0.0, 0.0, 0.0]}
    rx, ry, rz = transform_point(temp, direction[0], direction[1], direction[2])
    return (rx, ry, rz)


# ── helper: collision detection ──────────────────────────────────

def _infer_push_direction(src_a, src_b, matches, parts_features,
                           ox, oy, oz):
    """
    When two parts have coincident bbox centers (nested/embedded),
    infer the push-apart direction from their match features.
    Priority: cylinder axis (coaxial/clearance) > plane normal (planar)
    > bbox overlap axis.
    """
    from constraints import COAXIAL, CLEARANCE, PLANAR_MATE, PLANAR_ALIGN

    # Find all matches between these two parts
    for m in matches:
        if {m['parts'][0], m['parts'][1]} != {src_a, src_b}:
            continue

        ctype = m['type']
        if ctype in (COAXIAL, CLEARANCE):
            # Use the cylinder axis — for nested parts this is the natural push dir
            pi, pj = m['parts']
            # Which part's cylinder to use? Either — they're coaxial.
            feat_key = 'cylinders'
            cyl = parts_features[pi][feat_key][m['feat_a_idx']]
            axis = cyl['axis']
            mag = math.sqrt(sum(x * x for x in axis))
            if mag > 1e-12:
                return (axis[0] / mag, axis[1] / mag, axis[2] / mag)

        if ctype in (PLANAR_MATE, PLANAR_ALIGN):
            # Use the plane normal
            pi = m['parts'][0]
            feat_key = 'planes'
            plane = parts_features[pi][feat_key][m['feat_a_idx']]
            normal = plane['normal']
            mag = math.sqrt(sum(x * x for x in normal))
            if mag > 1e-12:
                return (normal[0] / mag, normal[1] / mag, normal[2] / mag)

    # No match features → fallback to largest bbox overlap axis
    if oz >= ox and oz >= oy:
        return (0.0, 0.0, 1.0)
    elif ox >= oy:
        return (1.0, 0.0, 0.0)
    else:
        return (0.0, 1.0, 0.0)


def detect_collisions(components, matches, parts_features, folder):
    """
    Run BEFORE refinement: detect all inter-part collisions (OCCT Boolean Common).
    Returns list of collision dicts:
    {
        'pair': (source_a, source_b),
        'penetration_mm': float,     # max penetration depth along collision direction
        'direction': (dx, dy, dz),   # push-apart direction (unit vector)
        'bbox_overlap_z': float,     # bbox Z overlap amount (fallback)
    }
    """
    MIN_PENETRATION_MM = 0.01     # ignore overlaps smaller than 10 microns (noise)
    MIN_BBOX_OVERLAP_MM = 0.001   # bboxes must overlap by at least 1 micron
    VOLUME_EPS = 1e-3             # Boolean result must have >= 1 micron extent per axis

    collisions = []
    try:
        from build_assembly import load_step, build_transform
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box

        # Build transformed shapes + bbox centers (cheap proxies for part centers)
        shapes = {}
        centers = {}
        for comp in components:
            src = comp['source']
            fp = os.path.join(folder, src) if not os.path.isabs(src) else src
            shape = load_step(fp)
            placement = comp.get('placement', {})
            trsf = build_transform(placement)
            if trsf.Form() != 0:
                shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
            shapes[src] = shape

            bb = Bnd_Box(); bb.SetGap(0.0); brepbndlib.Add(shape, bb)
            xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
            centers[src] = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)

        # Pairwise check
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                a, b = components[i], components[j]
                sa, sb = shapes[a['source']], shapes[b['source']]
                ca, cb = centers[a['source']], centers[b['source']]

                # Fast bbox check first (with noise threshold)
                bb_a = Bnd_Box(); bb_a.SetGap(0.0); brepbndlib.Add(sa, bb_a)
                bb_b = Bnd_Box(); bb_b.SetGap(0.0); brepbndlib.Add(sb, bb_b)
                xmin_a, ymin_a, zmin_a, xmax_a, ymax_a, zmax_a = bb_a.Get()
                xmin_b, ymin_b, zmin_b, xmax_b, ymax_b, zmax_b = bb_b.Get()

                ox = max(0.0, min(xmax_a, xmax_b) - max(xmin_a, xmin_b))
                oy = max(0.0, min(ymax_a, ymax_b) - max(ymin_a, ymin_b))
                oz = max(0.0, min(zmax_a, zmax_b) - max(zmin_a, zmin_b))

                if ox < MIN_BBOX_OVERLAP_MM or oy < MIN_BBOX_OVERLAP_MM or oz < MIN_BBOX_OVERLAP_MM:
                    continue

                # ── Coaxial/clearance pairs: XY overlap is expected ──
                # For parts that share an axis, XY bbox overlap is normal.
                # Only Z (axial) penetration is a real concern.
                is_axial_pair = any(
                    m['type'] in (COAXIAL, CLEARANCE)
                    and {m['parts'][0], m['parts'][1]} == {a['source'], b['source']}
                    for m in matches
                )

                if is_axial_pair:
                    # For axial pairs: only Z overlap matters
                    if oz < MIN_PENETRATION_MM:
                        continue  # no axial penetration → skip
                    # Report with Z-only penetration, axis direction
                    penetration = oz
                    direction = (0.0, 0.0, 1.0)
                    collisions.append({
                        'pair': (a['source'], b['source']),
                        'penetration_mm': round(penetration, 4),
                        'direction': direction,
                        'bbox_overlap_z': round(oz, 4),
                        'from_boolean': False,
                        'is_axial': True,
                    })
                    continue

                # ── Planar-planar pairs (sheet metal enclosures): skip ──
                # Planar-dominant parts often have one part nested inside
                # another by design. Bbox overlap is the correct assembly
                # position, not a collision to resolve.
                # The planar_mate distance already sets correct face separation.
                ti = _classify_part(parts_features.get(a['source'], {}))
                tj = _classify_part(parts_features.get(b['source'], {}))
                if ti == 'planar' and tj == 'planar':
                    continue

                # Precise Boolean intersection
                intersection_shape = None
                try:
                    common = BRepAlgoAPI_Common(sa, sb)
                    if common.IsDone():
                        result = common.Shape()
                        bb_r = Bnd_Box(); bb_r.SetGap(0.0); brepbndlib.Add(result, bb_r)
                        rxmin, rymin, rzmin, rxmax, rymax, rzmax = bb_r.Get()
                        rdx = rxmax - rxmin
                        rdy = rymax - rymin
                        rdz = rzmax - rzmin
                        if rdx >= VOLUME_EPS and rdy >= VOLUME_EPS and rdz >= VOLUME_EPS:
                            intersection_shape = result
                except Exception:
                    pass

                if intersection_shape is not None:
                    # ── Geometry-based direction (not bbox axis) ──
                    dx_c = ca[0] - cb[0]
                    dy_c = ca[1] - cb[1]
                    dz_c = ca[2] - cb[2]
                    mag = math.sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c)

                    if mag < 1e-12:
                        # Centers coincide (nested/embedded parts).
                        # Use match features to infer the correct push direction.
                        direction = _infer_push_direction(
                            a['source'], b['source'], matches, parts_features,
                            ox, oy, oz)
                    else:
                        direction = (dx_c / mag, dy_c / mag, dz_c / mag)

                    # Compute penetration depth along this direction:
                    # project the intersection bbox extents onto the direction
                    proj_extent = abs(rdx * direction[0]) + abs(rdy * direction[1]) + abs(rdz * direction[2])
                    penetration = max(proj_extent, max(ox, oy, oz) * 0.5)  # at least half bbox
                else:
                    # ── Boolean failed → bbox fallback ──
                    extents = [(ox, (1.0, 0.0, 0.0)),
                               (oy, (0.0, 1.0, 0.0)),
                               (oz, (0.0, 0.0, 1.0))]
                    penetration, direction = max(extents, key=lambda e: e[0])

                if penetration < MIN_PENETRATION_MM:
                    continue

                collisions.append({
                    'pair': (a['source'], b['source']),
                    'penetration_mm': round(penetration, 4),
                    'direction': tuple(round(d, 6) for d in direction),
                    'bbox_overlap_z': round(oz, 4),
                    'from_boolean': intersection_shape is not None,
                })

    except ImportError:
        pass  # OCCT not available

    return collisions


# ── helper: OCCT face positioning ─────────────────────────────────

def _refine_face_z(source_path, placement, folder, features=None):
    """
    Use OCCT shape bbox after rotation to position the mating face at z=0.
    Returns updated placement (new dict, does not mutate input).
    """
    try:
        from build_assembly import load_step, build_transform
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box

        full_path = os.path.join(folder, source_path) if not os.path.isabs(source_path) else source_path
        shape = load_step(full_path)

        rot_only = copy.deepcopy(placement)
        rot_only['translate'] = [0.0, 0.0, 0.0]
        trsf = build_transform(rot_only)
        shape_rot = BRepBuilderAPI_Transform(shape, trsf, True).Shape()

        bb = Bnd_Box()
        bb.SetGap(0.0)
        brepbndlib.Add(shape_rot, bb)
        _, _, zmin, _, _, zmax = bb.Get()

        # Determine which end is the mating face:
        # closer to main cylinder origin after rotation → that's the face
        if features:
            cyls = features.get('cylinders', [])
            if cyls:
                main = max(cyls, key=lambda c: c['radius'])
                bore_origin = main['origin']
                from build_assembly import transform_point
                rx, ry, rz = transform_point(rot_only, bore_origin[0], bore_origin[1], bore_origin[2])
                face_z = zmin if abs(rz - zmin) < abs(rz - zmax) else zmax
            else:
                face_z = zmin if abs(zmin) < abs(zmax) else zmax
        else:
            face_z = zmin if abs(zmin) < abs(zmax) else zmax

        new_plac = copy.deepcopy(placement)
        new_plac['translate'] = [0.0, 0.0, round(-face_z, 6)]
        return new_plac
    except Exception:
        return placement


# ── Strategy 1: Unify main axis to global Z ───────────────────────

class UnifyMainAxisStrategy(RefinementStrategy):
    """
    For every part with cylinders: rotate its main cylinder axis to
    global Z, so all parts share a common up direction.
    """
    name = "unify_main_axis"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        if not feats.get('cylinders'):
            return False
        # Skip planar-dominant parts: their "main cylinder" is a mesh artifact
        if _classify_part(feats) == 'planar':
            return False
        return True

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        cyls = feats.get('cylinders', [])
        main = max(cyls, key=lambda c: c['radius'])

        post_dir = _apply_rotation_to_dir(
            main['axis'],
            comp['placement'].get('rotate_sequence', [])
        )
        to_axis = (0.0, 0.0, 1.0)
        dot = abs(sum(post_dir[i] * to_axis[i] for i in range(3)))
        if dot < 0.999:
            seq = comp['placement'].get('rotate_sequence', [])
            seq.append({'axis_to': {'from': list(post_dir), 'to': list(to_axis)}})
            comp['placement']['rotate_sequence'] = seq
            return True
        return False


# ── Strategy 2: Planar distance as Z offset ───────────────────────

class PlanarOffsetStrategy(RefinementStrategy):
    """
    For coaxial + planar pairs: compute the actual face Z from OCCT bbox
    after all rotations, and align both faces to the same Z.
    No guessing — pure geometry computation.

    For non-coaxial planar pairs: the Phase 2 planar translation is correct
    in the local frame, so no further adjustment needed.
    """
    name = "planar_offset"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        if not feats.get('cylinders'):
            return False
        # Skip planar-dominant parts
        if _classify_part(feats) == 'planar':
            return False
        for m in matches:
            if m['type'] in (PLANAR_MATE, 'planar_align') and m['parts'][1] == src:
                ref_feats = parts_features.get(m['parts'][0], {})
                if ref_feats.get('cylinders'):
                    return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']

        for m in matches:
            if m['type'] not in (PLANAR_MATE, 'planar_align'):
                continue
            if m['parts'][1] != src:
                continue
            ref_src = m['parts'][0]
            ref_comp = next((c for c in components if c['source'] == ref_src), None)
            if not ref_comp:
                continue

            # Compute both faces in GLOBAL coordinates (includes rotation + translation)
            ref_face_z = _compute_face_z_global(ref_src, ref_comp['placement'], folder,
                                                 parts_features.get(ref_src, {}))
            tgt_face_z = _compute_face_z_global(src, comp['placement'], folder,
                                                 parts_features.get(src, {}))

            # In global coords, adjust target's translate Z so faces align
            old_t = comp['placement'].get('translate', [0.0, 0.0, 0.0])
            new_z = old_t[2] + (ref_face_z - tgt_face_z)
            comp['placement']['translate'] = [old_t[0], old_t[1], round(new_z, 6)]
            return True
        return False


def _compute_face_z_global(source_path, placement, folder, features):
    """Compute the mating face Z position in the GLOBAL coordinate system."""
    from build_assembly import load_step, build_transform, transform_point
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box

    full_path = os.path.join(folder, source_path) if not os.path.isabs(source_path) else source_path
    shape = load_step(full_path)
    trsf = build_transform(placement)
    shape_xform = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    bb = Bnd_Box(); bb.SetGap(0.0)
    brepbndlib.Add(shape_xform, bb)
    _, _, zmin, _, _, zmax = bb.Get()

    cyls = features.get('cylinders', [])
    if cyls:
        main = max(cyls, key=lambda c: c['radius'])
        rx, ry, rz = transform_point(placement,
                                      main['origin'][0], main['origin'][1], main['origin'][2])
        return zmin if abs(rz - zmin) < abs(rz - zmax) else zmax
    else:
        return zmin if abs(zmin) < abs(zmax) else zmax


# ── Strategy 4: Coaxial X/Y translation ──────────────────────────

class CoaxialTranslationStrategy(RefinementStrategy):
    """
    After UnifyMainAxis aligns all axes to Z, coaxial pairs should
    have coincident X,Y centers. Use cylinder origin positions to
    compute the X/Y offset that aligns them.

    Runs before face/planar Z positioning because X/Y alignment
    may change which part of the bbox is the mating face.
    """
    name = "coaxial_translation"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        if not feats.get('cylinders'):
            return False
        # Skip planar-dominant parts
        if _classify_part(feats) == 'planar':
            return False
        # Apply to any non-reference part that has a coaxial match
        for m in matches:
            if m['type'] == COAXIAL and m['parts'][1] == src:
                # Check if reference part has cylinders (axes aligned to Z)
                ref_feats = parts_features.get(m['parts'][0], {})
                if ref_feats.get('cylinders'):
                    return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        for m in matches:
            if m['type'] != COAXIAL or m['parts'][1] != src:
                continue
            ref_src = m['parts'][0]

            # Get cylinder origins from features (local frames)
            ref_cyls = parts_features[ref_src]['cylinders']
            tgt_cyls = parts_features[src]['cylinders']
            ref_cyl = ref_cyls[m['feat_a_idx']]
            tgt_cyl = tgt_cyls[m['feat_b_idx']]

            ref_comp = next((c for c in components if c['source'] == ref_src), None)
            if not ref_comp:
                continue

            # Rotate both cylinder origins by their current rotations
            # (translations stripped — we only care about orientation)
            from build_assembly import transform_point

            def _rotated_origin(placement, origin):
                rot_only = {'rotate_sequence': list(
                    placement.get('rotate_sequence', [])),
                    'translate': [0.0, 0.0, 0.0]}
                rx, ry, rz = transform_point(rot_only,
                                              origin[0], origin[1], origin[2])
                return (rx, ry, rz)

            ref_rot_origin = _rotated_origin(ref_comp['placement'], ref_cyl['origin'])
            tgt_rot_origin = _rotated_origin(comp['placement'], tgt_cyl['origin'])

            # X,Y offset needed to make cylinder centers coincident in XY
            dx = ref_rot_origin[0] - tgt_rot_origin[0]
            dy = ref_rot_origin[1] - tgt_rot_origin[1]

            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                return False

            old_t = comp['placement'].get('translate', [0.0, 0.0, 0.0])
            comp['placement']['translate'] = [
                old_t[0] + dx,
                old_t[1] + dy,
                old_t[2],
            ]
            return True
        return False


# ── Strategy 5: OCCT face positioning ─────────────────────────────

class FacePositioningStrategy(RefinementStrategy):
    """
    ONLY for the reference part (the one with cylinders that sits at origin
    after Phase 2). Adjusts its Z so the mating face is at z=0.
    All other parts keep their Phase 2 relative translations — their Z
    offset from the reference is already correct from constraint solving.
    """
    name = "face_positioning"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        if not feats.get('cylinders'):
            return False
        # Skip planar-dominant parts — their cylinders are mesh artifacts
        if _classify_part(feats) == 'planar':
            return False
        # Only the reference part (identity placement from Phase 2)
        t = comp['placement'].get('translate', [1, 1, 1])
        r = comp['placement'].get('rotate_sequence', None)
        is_ref = (abs(t[0]) < 1e-9 and abs(t[1]) < 1e-9 and abs(t[2]) < 1e-9
                  and (r is None or len(r) == 0))
        return is_ref

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})

        # Compute reference face Z in global coords and set translate so face is at z=0
        face_z = _compute_face_z_global(src, comp['placement'], folder, feats)
        old_t = comp['placement'].get('translate', [0.0, 0.0, 0.0])
        new_z = old_t[2] - face_z  # move face to z=0
        comp['placement']['translate'] = [old_t[0], old_t[1], round(new_z, 6)]
        return abs(new_z - old_t[2]) > 1e-9


# ── Strategy 3: Coaxial pair flip ─────────────────────────────────

class CoaxialFlipStrategy(RefinementStrategy):
    """
    For coaxial pairs without a planar mate on the second part:
    flip the second part 180° so it faces the opposite direction.
    """
    name = "coaxial_flip"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        # Check if this part is the 'pb' (second part) of any coaxial match
        for m in matches:
            if m['type'] == COAXIAL and m['parts'][1] == src:
                # Skip if there's already a planar constraint (mate or align)
                # that correctly positions the mating faces.
                # Without a planar constraint, the coaxial alignment may put
                # faces on opposite sides, requiring a 180° flip to mate.
                has_planar = any(
                    mm['type'] in (PLANAR_MATE, PLANAR_ALIGN) and mm['parts'][1] == src
                    for mm in matches
                )
                if not has_planar:
                    return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        # Determine the current cylinder axis direction (after all prior rotations)
        feats = parts_features.get(src, {})
        cyls = feats.get('cylinders', [])
        if not cyls:
            return False
        main = max(cyls, key=lambda c: c['radius'])
        from build_assembly import transform_point
        # Apply existing rotations to get current axis direction in global frame
        current_placement = comp['placement']
        rot_only = {'rotate_sequence': list(current_placement.get('rotate_sequence', [])),
                    'translate': [0.0, 0.0, 0.0]}
        ax = main['axis']
        rx, ry, rz = transform_point(rot_only, ax[0], ax[1], ax[2])
        # Pick a flip axis perpendicular to the cylinder axis
        # (flipping around the cylinder axis would leave it unchanged)
        if abs(rx) > 0.9:
            # Axis is near X — flip around Y
            flip_axis = (0.0, 1.0, 0.0)
        elif abs(ry) > 0.9:
            flip_axis = (1.0, 0.0, 0.0)
        else:
            flip_axis = (1.0, 0.0, 0.0)
        
        seq = comp['placement'].get('rotate_sequence', [])
        seq.append({'axis_angle': [flip_axis[0], flip_axis[1], flip_axis[2], 180.0]})
        comp['placement']['rotate_sequence'] = seq
        return True


# ── Strategy 4: Axial centering for clearance fits ────────────────

class AxialCenteringStrategy(RefinementStrategy):
    """
    For clearance fits where the shaft body is smaller than the bore body:
    center the shaft axially inside the bore.
    """
    name = "axial_centering"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        # Check if this part is a shaft in a clearance match
        for m in matches:
            if m['type'] != CLEARANCE:
                continue
            shaft_name = m['parts'][0]  # parts[0] = shaft
            if shaft_name != src:
                continue
            bore_name = m['parts'][1]
            shaft_max_r = max(
                (c['radius'] for c in parts_features[shaft_name].get('cylinders', [])),
                default=0
            )
            bore_max_r = max(
                (c['radius'] for c in parts_features[bore_name].get('cylinders', [])),
                default=0
            )
            if shaft_max_r < bore_max_r:
                return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']

        # Collect all bore partners
        bore_names = []
        for m in matches:
            if m['type'] == CLEARANCE and m['parts'][0] == src:
                bore_names.append(m['parts'][1])

        if not bore_names:
            return False

        try:
            from build_assembly import load_step, build_transform
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.BRepBndLib import brepbndlib
            from OCC.Core.Bnd import Bnd_Box

            def _get_zmid(source_name, placement):
                fp = os.path.join(folder, source_name)
                shape = load_step(fp)
                rot_only = copy.deepcopy(placement)
                rot_only['translate'] = [0.0, 0.0, 0.0]
                trsf = build_transform(rot_only)
                shape_rot = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
                bb = Bnd_Box(); bb.SetGap(0.0)
                brepbndlib.Add(shape_rot, bb)
                _, _, zmin, _, _, zmax = bb.Get()
                return (zmin + zmax) / 2.0, zmax - zmin

            shaft_mid, shaft_h = _get_zmid(src, comp['placement'])
            bore_zmin, bore_zmax = float('inf'), float('-inf')
            for bn in bore_names:
                bc = next((c for c in components if c['source'] == bn), None)
                if not bc:
                    continue
                bmid, bh = _get_zmid(bn, bc['placement'])
                bore_zmin = min(bore_zmin, bmid - bh / 2)
                bore_zmax = max(bore_zmax, bmid + bh / 2)

            if bore_zmin == float('inf'):
                return False

            bore_center = (bore_zmin + bore_zmax) / 2.0
            new_tz = bore_center - shaft_mid
            comp['placement']['translate'] = [0.0, 0.0, round(new_tz, 6)]
            return True
        except Exception:
            return False


# ── Strategy 5: Kabsch bolt-pattern rigid transform ───────────────

def _apply_body_flip(comp, bolt_pos_ref, bolt_pos_target, parts_features, ref_part, target_part):
    """
    After Kabsch alignment, apply a 180° flip around a symmetry axis
    of the bolt pattern to ensure flange bodies extend in opposite
    directions (preventing overlap).
    
    Only applied when the two parts have different main bore axis
    directions in their local coordinate systems (e.g., X vs Y).
    When they share the same axis direction, Phase 2 planar constraints
    already handle the body orientation correctly.
    """
    import numpy as np
    from build_assembly import transform_point
    
    if len(bolt_pos_ref) < 3 or len(bolt_pos_target) < 3:
        return
    
    ref_feats = parts_features.get(ref_part, {})
    tgt_feats = parts_features.get(target_part, {})
    ref_cyls = ref_feats.get('cylinders', [])
    tgt_cyls = tgt_feats.get('cylinders', [])
    if not ref_cyls or not tgt_cyls:
        return
    
    ref_main = max(ref_cyls, key=lambda c: c['radius'])
    tgt_main = max(tgt_cyls, key=lambda c: c['radius'])
    ref_axis = ref_main['axis']
    tgt_axis = tgt_main['axis']
    
    # Normalize axes
    ref_mag = math.sqrt(sum(x*x for x in ref_axis))
    tgt_mag = math.sqrt(sum(x*x for x in tgt_axis))
    if ref_mag < 1e-12 or tgt_mag < 1e-12:
        return
    ref_dir = [x/ref_mag for x in ref_axis]
    tgt_dir = [x/tgt_mag for x in tgt_axis]
    
    # If both parts have nearly parallel bore axes in local coords,
    # the Phase 2 planar constraints already handle orientation.
    dot_axes = abs(sum(ref_dir[i] * tgt_dir[i] for i in range(3)))
    if dot_axes > 0.95:
        # Same axis direction — skip body flip (already correct)
        return
    
    # Determine the shared bore axis direction in global coords
    # (use reference part's axis, which is at identity)
    rot_only = {'rotate_sequence': [], 'translate': [0.0, 0.0, 0.0]}
    bx, by, bz = transform_point(rot_only, ref_axis[0], ref_axis[1], ref_axis[2])
    bore_mag = math.sqrt(bx*bx + by*by + bz*bz)
    if bore_mag < 1e-12:
        return
    bore_dir = (bx/bore_mag, by/bore_mag, bz/bore_mag)
    
    # Compute bolt hole positions in global coords (already aligned by Kabsch)
    current_placement = comp['placement']
    rot_only_tgt = {'rotate_sequence': list(current_placement.get('rotate_sequence', [])),
                    'translate': [0.0, 0.0, 0.0]}
    trans_tgt = current_placement.get('translate', [0.0, 0.0, 0.0])
    
    global_positions = []
    for pos in bolt_pos_target:
        rx, ry, rz = transform_point(rot_only_tgt, pos[0], pos[1], pos[2])
        global_positions.append((rx + trans_tgt[0], ry + trans_tgt[1], rz + trans_tgt[2]))
    
    if len(global_positions) < 3:
        return
    
    # Project bolt positions onto the plane perpendicular to bore_dir
    # and compute their angles around the bore axis
    def proj_perp(pos):
        axial = sum(pos[i] * bore_dir[i] for i in range(3))
        return [pos[i] - axial * bore_dir[i] for i in range(3)]
    
    # Build a reference frame in the perp plane
    if abs(bore_dir[0]) < 0.9:
        ref_v = (1.0, 0.0, 0.0)
    else:
        ref_v = (0.0, 1.0, 0.0)
    dot_rv = sum(ref_v[i] * bore_dir[i] for i in range(3))
    u = [ref_v[i] - dot_rv * bore_dir[i] for i in range(3)]
    u_mag = math.sqrt(sum(x*x for x in u))
    if u_mag < 1e-12:
        return
    u = [x / u_mag for x in u]
    v = [bore_dir[1]*u[2] - bore_dir[2]*u[1],
         bore_dir[2]*u[0] - bore_dir[0]*u[2],
         bore_dir[0]*u[1] - bore_dir[1]*u[0]]
    
    # Compute angles of bolt positions in the perp plane
    angles = []
    for pos in global_positions:
        rp = proj_perp(pos)
        rmag = math.sqrt(sum(x*x for x in rp))
        if rmag < 1e-12:
            continue
        rp_u = sum(rp[i]*u[i] for i in range(3)) / rmag
        rp_v = sum(rp[i]*v[i] for i in range(3)) / rmag
        angles.append(math.atan2(rp_v, rp_u))
    angles.sort()
    
    if len(angles) < 3:
        return
    
    # Find a symmetry axis: midpoint between two adjacent bolt holes
    # For a regular N-gon, the symmetry axis is at angle (angles[0] + angles[1]) / 2
    # plus multiples of pi/N
    mid_angle = (angles[0] + angles[1]) / 2.0
    
    # Flip axis direction in the perp plane
    flip_u = math.cos(mid_angle)
    flip_v = math.sin(mid_angle)
    
    # The flip axis in 3D: combination of u and v
    flip_axis_3d = [flip_u * u[i] + flip_v * v[i] for i in range(3)]
    flip_mag = math.sqrt(sum(x*x for x in flip_axis_3d))
    if flip_mag < 1e-12:
        return
    flip_axis_3d = [x / flip_mag for x in flip_axis_3d]
    
    # Add the 180° flip to the rotate_sequence
    seq = current_placement.get('rotate_sequence', [])
    seq.append({'axis_angle': [flip_axis_3d[0], flip_axis_3d[1], flip_axis_3d[2], 180.0]})
    comp['placement']['rotate_sequence'] = seq


def _resolve_collision(comp, ref_part, parts_features, folder):
    """
    Check for collision between the placed component and the reference part.
    If overlap is detected, translate the component along its body direction
    to separate the bodies.
    """
    try:
        from build_assembly import load_step, build_transform, transform_point
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCC.Core.BRepBndLib import brepbndlib
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.gp import gp_Trsf, gp_Vec
    except ImportError:
        return
    
    src = comp['source']
    target_path = os.path.join(folder, src) if not os.path.isabs(src) else src
    ref_path = os.path.join(folder, ref_part) if not os.path.isabs(ref_part) else ref_part
    
    if not os.path.exists(target_path) or not os.path.exists(ref_path):
        return
    
    shape_tgt = load_step(target_path)
    shape_ref = load_step(ref_path)
    
    placement = comp.get('placement', {})
    trsf_tgt = build_transform(placement)
    if trsf_tgt.Form() != 0:
        shape_tgt = BRepBuilderAPI_Transform(shape_tgt, trsf_tgt, True).Shape()
    
    # Check bounding box overlap
    bb_tgt = Bnd_Box(); bb_tgt.SetGap(0.0); brepbndlib.Add(shape_tgt, bb_tgt)
    bb_ref = Bnd_Box(); bb_ref.SetGap(0.0); brepbndlib.Add(shape_ref, bb_ref)
    tx1,ty1,tz1,tx2,ty2,tz2 = bb_tgt.Get()
    rx1,ry1,rz1,rx2,ry2,rz2 = bb_ref.Get()
    
    # Compute overlap in each axis
    ox = min(tx2, rx2) - max(tx1, rx1)
    oy = min(ty2, ry2) - max(ty1, ry1)
    oz = min(tz2, rz2) - max(tz1, rz1)
    
    if ox < 0.01 and oy < 0.01 and oz < 0.01:
        return  # No bbox overlap at all
    
    # Find the primary overlap axis and the separation direction
    # Use the target's body direction (its main bore axis after placement)
    tgt_feats = parts_features.get(src, {})
    tgt_cyls = tgt_feats.get('cylinders', [])
    if not tgt_cyls:
        return
    tgt_main = max(tgt_cyls, key=lambda c: c['radius'])
    tgt_axis = tgt_main['axis']
    
    rot_only = {'rotate_sequence': list(placement.get('rotate_sequence', [])), 'translate': [0.0, 0.0, 0.0]}
    bx, by, bz = transform_point(rot_only, tgt_axis[0], tgt_axis[1], tgt_axis[2])
    
    # Determine which direction along body_dir moves target AWAY from reference
    # Try both signs with increasing offset until collision is resolved
    for test_sign in [-1.0, 1.0]:
        for step in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            test_dx = test_sign * step
            t = gp_Trsf()
            t.SetTranslation(gp_Vec(test_dx * bx, test_dx * by, test_dx * bz))
            shape_test = BRepBuilderAPI_Transform(shape_tgt, t, True).Shape()
            try:
                common_test = BRepAlgoAPI_Common(shape_test, shape_ref)
                if common_test.IsDone():
                    r_test = common_test.Shape()
                    bb_test = Bnd_Box()
                    bb_test.SetGap(0.0)
                    brepbndlib.Add(r_test, bb_test)
                    trx1, try1, trz1, trx2, try2, trz2 = bb_test.Get()
                    trdx = trx2 - trx1
                    trdy = try2 - try1
                    trdz = trz2 - trz1
                    if trdx < 0.01 and trdy < 0.01 and trdz < 0.01:
                        # Found a collision-free translation
                        old_trans = list(placement.get('translate', [0.0, 0.0, 0.0]))
                        new_trans = [
                            old_trans[0] + test_dx * bx,
                            old_trans[1] + test_dx * by,
                            old_trans[2] + test_dx * bz,
                        ]
                        comp['placement']['translate'] = new_trans
                        return
            except Exception:
                continue


class KabschBoltPatternStrategy(RefinementStrategy):
    """
    For flange pairs with 3+ bolt holes on each part: compute the full 4×4
    rigid transform (rotation + translation) that maps bolt hole positions
    from the target part onto the reference part using the Kabsch algorithm.
    
    Bolt holes are identified directly from cylinder features (radius 2-10mm,
    parallel to the main bore axis, arranged in a circular pattern).
    Correspondence is established by sorting holes by angular position
    around their respective bore axes.
    
    This replaces the fragile multi-step coaxial→flip→bolt_rotate pipeline
    with a single optimal point-set registration, which works regardless
    of whether the parts share a common local coordinate system.
    """
    name = "kabsch_bolt_pattern"

    def _get_bolt_positions(self, feats):
        """Extract bolt hole origin positions from a part's cylinder features.
        Bolt holes: radius 2-10mm, axis parallel to the largest cylinder's axis,
        at a meaningful radial distance (> 5mm) from that axis."""
        cyls = feats.get('cylinders', [])
        if len(cyls) < 3:
            return []
        main = max(cyls, key=lambda c: c['radius'])
        main_r = main['radius']
        main_axis = main['axis']
        main_mag = math.sqrt(sum(x*x for x in main_axis))
        if main_mag < 1e-12:
            return []
        bore_dir = [x/main_mag for x in main_axis]
        
        positions = []
        for cyl in cyls:
            r = cyl.get('radius', 0)
            if r < 2 or r > 10:
                continue
            if r >= main_r * 0.5:
                continue
            origin = cyl.get('origin')
            if not origin:
                continue
            cyl_axis = cyl.get('axis', [0,0,1])
            cyl_mag = math.sqrt(sum(x*x for x in cyl_axis))
            if cyl_mag < 1e-12:
                continue
            cyl_dir = [x/cyl_mag for x in cyl_axis]
            dot = abs(sum(a*b for a,b in zip(bore_dir, cyl_dir)))
            if dot < 0.99:
                continue
            axial = sum(origin[i]*bore_dir[i] for i in range(3))
            radial_vec = [origin[i]-axial*bore_dir[i] for i in range(3)]
            radial_dist = math.sqrt(sum(x*x for x in radial_vec))
            if radial_dist < 5.0:
                continue
            positions.append(tuple(origin))
        return positions

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        if not feats.get('cylinders'):
            return False
        bolt_pos = self._get_bolt_positions(feats)
        if len(bolt_pos) < 3:
            return False
        # Check if there's another part with 3+ bolt holes and a coaxial match
        for m in matches:
            if m['type'] != COAXIAL:
                continue
            if m['parts'][1] != src:
                continue
            ref = m['parts'][0]
            ref_feats = parts_features.get(ref, {})
            ref_bolt_pos = self._get_bolt_positions(ref_feats)
            if len(ref_bolt_pos) >= 3:
                return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        bolt_pos_target = self._get_bolt_positions(feats)
        
        # Save Phase 2 translation before overwriting.
        # Kabsch rotation is correct for bolt alignment, but the axial
        # (face-to-face) translation is better from Phase 2 constraints.
        phase2_translate = list(comp['placement'].get('translate', [0.0, 0.0, 0.0]))
        
        # Find the reference part via a coaxial match
        ref_part = None
        for m in matches:
            if m['type'] != COAXIAL:
                continue
            if m['parts'][1] != src:
                continue
            ref = m['parts'][0]
            ref_feats = parts_features.get(ref, {})
            ref_bolt_pos = self._get_bolt_positions(ref_feats)
            if len(ref_bolt_pos) >= 3:
                ref_part = ref
                bolt_pos_ref = ref_bolt_pos
                break
        
        if ref_part is None or len(bolt_pos_target) < 3 or len(bolt_pos_ref) < 3:
            return False
        
        import numpy as np
        n = min(len(bolt_pos_ref), len(bolt_pos_target))
        P_full = np.array(bolt_pos_ref[:n])
        Q_full = np.array(bolt_pos_target[:n])
        
        # Try all cyclic shifts to find the best correspondence
        best_R, best_t, best_err = None, None, float('inf')
        for shift in range(n):
            Q_shifted = np.roll(Q_full, shift, axis=0)
            cP = np.mean(P_full, axis=0)
            cQ = np.mean(Q_shifted, axis=0)
            Pc = P_full - cP
            Qc = Q_shifted - cQ
            H = Pc.T @ Qc
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            t = cQ - R @ cP
            P_t = (R @ P_full.T).T + t
            err = np.sqrt(np.mean(np.sum((P_t - Q_shifted)**2, axis=1)))
            if err < best_err:
                best_err = err
                best_R = R
                best_t = t
        
        if best_R is None:
            return False
        
        # Kabsch found the transform mapping reference-bolt-positions (P)
        # to target-bolt-positions (Q):  R*P + t ≈ Q
        # We need the placement transform for the target part that brings
        # its local coordinates into the global frame where the reference sits.
        # The reference part is at identity, so we need T such that:
        #   T(target_bolt_local) = ref_bolt_local
        # Since R*ref_bolt + t ≈ target_bolt:
        #   target_bolt = R*ref_bolt + t
        #   T = R^{-1} (inverse rotation) with translation -R^T*t
        best_R_inv = best_R.T
        best_t_inv = -best_R_inv @ best_t
        
        # Convert the INVERSE rotation matrix to axis-angle
        trace = np.trace(best_R_inv)
        theta = math.acos(max(-1.0, min(1.0, (trace - 1) / 2)))
        if theta < 1e-9:
            rotate_seq = []
        elif abs(theta - math.pi) < 1e-9:
            xx = (best_R_inv[0,0] + 1) / 2; yy = (best_R_inv[1,1] + 1) / 2; zz = (best_R_inv[2,2] + 1) / 2
            xy = (best_R_inv[0,1] + best_R_inv[1,0]) / 4; xz = (best_R_inv[0,2] + best_R_inv[2,0]) / 4; yz = (best_R_inv[1,2] + best_R_inv[2,1]) / 4
            if xx > yy and xx > zz:
                ax = math.sqrt(max(0, xx)); s = 0.5/ax if ax>1e-12 else 0; ay=xy*s; az=xz*s
            elif yy > zz:
                ay = math.sqrt(max(0, yy)); s = 0.5/ay if ay>1e-12 else 0; ax=xy*s; az=yz*s
            else:
                az = math.sqrt(max(0, zz)); s = 0.5/az if az>1e-12 else 0; ax=xz*s; ay=yz*s
            rotate_seq = [{'axis_angle': [ax, ay, az, math.degrees(theta)]}]
        else:
            axis_x = best_R_inv[2,1] - best_R_inv[1,2]; axis_y = best_R_inv[0,2] - best_R_inv[2,0]; axis_z = best_R_inv[1,0] - best_R_inv[0,1]
            axis_mag = math.sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z)
            if axis_mag > 1e-12:
                axis = [axis_x/axis_mag, axis_y/axis_mag, axis_z/axis_mag]
            else:
                axis = [0.0, 0.0, 1.0]
            rotate_seq = [{'axis_angle': [axis[0], axis[1], axis[2], math.degrees(theta)]}]
        
        comp['placement'] = {
            'translate': phase2_translate,
            'rotate_sequence': rotate_seq,
        }
        
        # ── Flip body direction to prevent overlap ──
        # Kabsch aligns bolt hole centers, but flange faces should touch
        # with bodies extending in opposite directions.  Apply a 180°
        # flip around a symmetry axis of the bolt pattern in the plane
        # perpendicular to the shared bore axis.  This reverses the body
        # direction while preserving the (Y,Z) bolt pattern.
        _apply_body_flip(comp, bolt_pos_ref, bolt_pos_target, parts_features, ref_part, src)
        
        # ── Anti-collision: separate flanges if bodies overlap ──
        # After body flip, identical flanges overlap by the combined ring
        # thickness (typically 20mm). Translate target along its body
        # direction by this amount.
        try:
            from build_assembly import load_step, build_transform, transform_point
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.BRepBndLib import brepbndlib
            from OCC.Core.Bnd import Bnd_Box
            
            # Load both shapes with current placements
            tgt_path = os.path.join(folder, src) if not os.path.isabs(src) else src
            ref_path = os.path.join(folder, ref_part) if not os.path.isabs(ref_part) else ref_part
            if os.path.exists(tgt_path) and os.path.exists(ref_path):
                shape_tgt = load_step(tgt_path)
                shape_ref = load_step(ref_path)
                trsf_tgt = build_transform(comp.get('placement', {}))
                if trsf_tgt.Form() != 0:
                    shape_tgt = BRepBuilderAPI_Transform(shape_tgt, trsf_tgt, True).Shape()
                
                bb_tgt = Bnd_Box(); bb_tgt.SetGap(0.0); brepbndlib.Add(shape_tgt, bb_tgt)
                bb_ref = Bnd_Box(); bb_ref.SetGap(0.0); brepbndlib.Add(shape_ref, bb_ref)
                tx1,ty1,tz1,tx2,ty2,tz2 = bb_tgt.Get()
                rx1,ry1,rz1,rx2,ry2,rz2 = bb_ref.Get()
                
                # Overlap along each axis
                ox = min(tx2, rx2) - max(tx1, rx1)
                oy = min(ty2, ry2) - max(ty1, ry1)
                oz = min(tz2, rz2) - max(tz1, rz1)
                
                # Project overlap onto body direction to get separation distance
                tgt_feats = parts_features.get(src, {})
                tgt_cyls = tgt_feats.get('cylinders', [])
                if tgt_cyls:
                    tgt_main = max(tgt_cyls, key=lambda c: c['radius'])
                    tgt_axis = tgt_main['axis']
                    rot_only = {'rotate_sequence': list(comp['placement'].get('rotate_sequence', [])), 'translate': [0,0,0]}
                    bx, by, bz = transform_point(rot_only, tgt_axis[0], tgt_axis[1], tgt_axis[2])
                    # The overlap along the body direction is approximately:
                    # max(ox*|bx|, oy*|by|, oz*|bz|)
                    sep = max(ox * abs(bx), oy * abs(by), oz * abs(bz))
                    if sep > 0.01:
                        old_t = comp['placement'].get('translate', [0,0,0])
                        new_t = [
                            old_t[0] + sep * bx,
                            old_t[1] + sep * by,
                            old_t[2] + sep * bz,
                        ]
                        # Round to eliminate floating-point noise
                        new_t = [0.0 if abs(v) < 1e-9 else round(v, 3) for v in new_t]
                        comp['placement']['translate'] = new_t
        except Exception:
            pass
        
        return True


# ── Strategy 6: Bolt/keyway angular alignment ─────────────────────

class BoltKeywayAlignStrategy(RefinementStrategy):
    """
    For coaxial flange pairs with bolt holes: compute the optimal
    Z-axis rotation to align bolt patterns and keyways.
    """
    name = "bolt_keyway_align"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        for m in matches:
            if m['type'] != COAXIAL:
                continue
            if m['parts'][1] != src:
                continue
            pa, pb = m['parts']
            feats_a = parts_features.get(pa, {})
            feats_b = parts_features.get(pb, {})
            if not feats_a.get('cylinders') or not feats_b.get('cylinders'):
                continue
            if len(feats_a['cylinders']) < 3 or len(feats_b['cylinders']) < 3:
                continue
            return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        for m in matches:
            if m['type'] != COAXIAL or m['parts'][1] != src:
                continue
            pa, pb = m['parts']
            ref_placement = next((c['placement'] for c in components if c['source'] == pa), {})
            offset_deg, score, axis_dir = _compute_bolt_offset_and_keyway(
                parts_features, pa, pb,
                ref_placement,
                comp['placement'],
                folder
            )
            if abs(offset_deg) > 1e-6 and score < float('inf') and axis_dir is not None:
                if 'rotate_sequence' not in comp['placement']:
                    comp['placement']['rotate_sequence'] = []
                comp['placement']['rotate_sequence'].append(
                    {'axis_angle': [axis_dir[0], axis_dir[1], axis_dir[2], offset_deg]}
                )
                return True
        return False


# ── Strategy 6: Boreless part follow ──────────────────────────────

class BorelessFollowStrategy(RefinementStrategy):
    """
    For parts without cylinders: copy the placement of the geometrically
    closest part that does have cylinders.
    """
    name = "boreless_follow"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})
        if feats.get('cylinders'):
            return False
        # Apply if no rotation yet (needs orientation from parent)
        r = comp['placement'].get('rotate_sequence', None)
        return (r is None or len(r) == 0)

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        feats = parts_features.get(src, {})

        # Check if this part has any matches at all
        has_own_matches = any(
            m['parts'][0] == src or m['parts'][1] == src
            for m in matches
        )

        bbox = feats.get('bbox')
        if not bbox:
            return False

        center = tuple((bbox['min'][i] + bbox['max'][i]) / 2 for i in range(3))

        best_src = None
        best_dist = float('inf')
        for other in components:
            osrc = other['source']
            ofeats = parts_features.get(osrc, {})
            if not ofeats.get('cylinders'):
                continue
            obbox = ofeats.get('bbox')
            if not obbox:
                continue
            ocenter = tuple((obbox['min'][i] + obbox['max'][i]) / 2 for i in range(3))
            dist = sum((a - b) ** 2 for a, b in zip(center, ocenter)) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_src = osrc

        if best_src:
            for other in components:
                if other['source'] == best_src:
                    if has_own_matches:
                        # Has own constraints: copy rotation, keep translation
                        old_t = comp['placement'].get('translate', [0.0, 0.0, 0.0])
                        parent_r = other['placement'].get('rotate_sequence', [])
                        if parent_r:
                            comp['placement']['rotate_sequence'] = list(parent_r)
                        comp['placement']['translate'] = list(old_t)
                    else:
                        # No own constraints: copy full placement
                        comp['placement'] = copy.deepcopy(other['placement'])
                    return True
        return False


# ── Strategy 7: Global-frame planar fallback ─────────────────────

class PlanarFallbackStrategy(RefinementStrategy):
    """
    Last resort: for planar-matched parts that no other strategy handled
    AND Phase 2 didn't place (still at identity), realign faces using
    global bbox geometry. This catches cases where Phase 2 deferred
    planar translation but PlanarOffsetStrategy skipped (e.g. no cylinders).
    """
    name = "planar_fallback"

    def can_apply(self, comp, matches, parts_features, components, folder, collisions=None):
        # Only apply if part is still untouched (no translation, no rotation)
        t = comp['placement'].get('translate', [1, 1, 1])
        r = comp['placement'].get('rotate_sequence', None)
        if any(abs(v) > 1e-9 for v in t) or (r is not None and len(r) > 0):
            return False
        # Must have a planar match as parts[1]
        src = comp['source']
        # Skip planar-dominant parts — their Phase 2 identity placement
        # is the correct assembly position (shared coordinate system).
        feats = parts_features.get(src, {})
        if _classify_part(feats) == 'planar':
            return False
        for m in matches:
            if m['type'] in (PLANAR_MATE, 'planar_align') and m['parts'][1] == src:
                return True
        return False

    def apply(self, comp, matches, parts_features, components, folder, collisions=None):
        src = comp['source']
        for m in matches:
            if m['type'] not in (PLANAR_MATE, 'planar_align'):
                continue
            if m['parts'][1] != src:
                continue
            ref_src = m['parts'][0]
            ref_comp = next((c for c in components if c['source'] == ref_src), None)
            if not ref_comp:
                continue

            # Compute both faces in global coordinates
            ref_face_z = _compute_face_z_global(ref_src, ref_comp['placement'], folder,
                                                 parts_features.get(ref_src, {}))
            tgt_face_z = _compute_face_z_global(src, comp['placement'], folder,
                                                 parts_features.get(src, {}))
            # Align target face to reference face
            old_t = comp['placement'].get('translate', [0.0, 0.0, 0.0])
            new_z = old_t[2] + (ref_face_z - tgt_face_z)
            comp['placement']['translate'] = [old_t[0], old_t[1], round(new_z, 6)]
            return True
        return False


# ── bolt/keyway computation (extracted from compute_manifest) ─────

def _compute_bolt_offset_and_keyway(parts_features, pa, pb,
                                     placement_a, placement_b, folder):
    """
    Compute optimal rotation around the shared bore axis to align bolt holes
    and keyways between two coaxial parts. Returns (degrees, score, axis_dir).
    """
    from analyze_geometry import detect_bolt_holes, detect_bore_keyway
    from build_assembly import transform_point

    feats_a = parts_features[pa]
    feats_b = parts_features[pb]

    main_a = max(feats_a['cylinders'], key=lambda c: c['radius']) if feats_a['cylinders'] else None
    main_b = max(feats_b['cylinders'], key=lambda c: c['radius']) if feats_b['cylinders'] else None
    if not main_a or not main_b:
        return 0.0, float('inf'), None

    bh_a = detect_bolt_holes(feats_a['cylinders'], main_a)
    bh_b = detect_bolt_holes(feats_b['cylinders'], main_b)

    if not bh_a or not bh_b or len(bh_a.get('angles', [])) < 3 or len(bh_b.get('angles', [])) < 3:
        return 0.0, float('inf'), None

    kw_a = detect_bore_keyway(feats_a['filepath'], main_a)
    kw_b = detect_bore_keyway(feats_b['filepath'], main_b)

    def _compute_angles_around_axis(positions, bore_dir_local, rotate_seq):
        """Project positions onto plane perpendicular to the bore axis
        (after applying rotate_seq), then compute angles relative to a
        globally-fixed reference direction (world Z projected onto the
        perp plane). Returns sorted list of angles in radians.
        
        The reference direction is the same for all parts regardless of
        bore orientation, ensuring consistent angular measurements."""
        if not positions:
            return []
        temp = {'rotate_sequence': list(rotate_seq), 'translate': [0.0, 0.0, 0.0]}
        # Transform local bore axis to get global direction
        bx, by, bz = transform_point(temp, bore_dir_local[0], bore_dir_local[1], bore_dir_local[2])
        bore_mag = math.sqrt(bx*bx + by*by + bz*bz)
        if bore_mag < 1e-12:
            return []
        bd = [bx/bore_mag, by/bore_mag, bz/bore_mag]
        
        # Build a globally-consistent reference direction in the perp plane:
        # project world Z onto the plane perpendicular to bd.
        # If bd is nearly parallel to Z, use world Y instead.
        world_ref = (0.0, 0.0, 1.0)
        dot_wr = sum(world_ref[i] * bd[i] for i in range(3))
        if abs(dot_wr) > 0.99:
            world_ref = (0.0, 1.0, 0.0)
            dot_wr = sum(world_ref[i] * bd[i] for i in range(3))
        ref_dir = [world_ref[i] - dot_wr * bd[i] for i in range(3)]
        ref_mag = math.sqrt(sum(x * x for x in ref_dir))
        if ref_mag < 1e-12:
            return []
        ref_dir = [x / ref_mag for x in ref_dir]
        
        # Compute cross product for signed angle: bd × ref_dir (right-handed)
        cross_ref = [bd[1] * ref_dir[2] - bd[2] * ref_dir[1],
                     bd[2] * ref_dir[0] - bd[0] * ref_dir[2],
                     bd[0] * ref_dir[1] - bd[1] * ref_dir[0]]
        
        angles = []
        for pos in positions:
            px, py, pz = transform_point(temp, pos[0], pos[1], pos[2])
            # Project onto plane perpendicular to bd
            axial = px*bd[0] + py*bd[1] + pz*bd[2]
            rx = px - axial * bd[0]
            ry = py - axial * bd[1]
            rz = pz - axial * bd[2]
            rmag = math.sqrt(rx*rx + ry*ry + rz*rz)
            if rmag < 1e-12:
                continue
            rv = (rx/rmag, ry/rmag, rz/rmag)
            # cos = dot with ref_dir, sin = dot with cross_ref
            cos_a = rv[0]*ref_dir[0] + rv[1]*ref_dir[1] + rv[2]*ref_dir[2]
            sin_a = rv[0]*cross_ref[0] + rv[1]*cross_ref[1] + rv[2]*cross_ref[2]
            angles.append(math.atan2(sin_a, cos_a))
        angles.sort()
        return angles

    def bolt_positions(geom_features, main_bore):
        """Extract bolt hole origin positions (in local coords) that are
        parallel to the main bore and at a meaningful radial distance."""
        all_cyls = geom_features['cylinders']
        bore_axis = main_bore['axis']
        bore_mag = math.sqrt(sum(x * x for x in bore_axis))
        if bore_mag < 1e-12:
            return []
        bore_dir = [x / bore_mag for x in bore_axis]
        main_r = main_bore['radius']
        positions = []
        for cyl in all_cyls:
            r = cyl.get('radius', 0)
            if r < 2 or r > 20:
                continue
            if r >= main_r * 0.8:
                continue
            origin = cyl.get('origin')
            if not origin:
                continue
            cyl_axis = cyl.get('axis', [0, 0, 1])
            cyl_mag = math.sqrt(sum(x * x for x in cyl_axis))
            if cyl_mag < 1e-12:
                continue
            cyl_dir = [x / cyl_mag for x in cyl_axis]
            dot = abs(sum(a * b for a, b in zip(bore_dir, cyl_dir)))
            if dot < 0.99:
                continue
            axial = sum(origin[i] * bore_dir[i] for i in range(3))
            radial_vec = [origin[i] - axial * bore_dir[i] for i in range(3)]
            radial_dist = math.sqrt(sum(x * x for x in radial_vec))
            if radial_dist < 5.0:
                continue
            positions.append(tuple(origin))
        return positions

    pos_a = bolt_positions(feats_a, main_a)
    pos_b = bolt_positions(feats_b, main_b)
    if len(pos_a) < 3 or len(pos_b) < 3:
        return 0.0, float('inf'), None

    seq_a = placement_a.get('rotate_sequence', [])
    seq_b = placement_b.get('rotate_sequence', [])

    # Determine the shared bore axis direction in global coords
    # (both parts share the same axis after Phase 2 alignment)
    ref_axis_dir = main_a['axis']  # local axis of reference part
    temp_a = {'rotate_sequence': list(seq_a), 'translate': [0.0, 0.0, 0.0]}
    gx, gy, gz = transform_point(temp_a, ref_axis_dir[0], ref_axis_dir[1], ref_axis_dir[2])
    gmag = math.sqrt(gx*gx + gy*gy + gz*gz)
    if gmag < 1e-12:
        return 0.0, float('inf'), None
    global_axis = (gx/gmag, gy/gmag, gz/gmag)

    ang_a = _compute_angles_around_axis(pos_a, main_a['axis'], seq_a)
    ang_b = _compute_angles_around_axis(pos_b, main_b['axis'], seq_b)
    if len(ang_a) < 3 or len(ang_b) < 3:
        return 0.0, float('inf'), None

    n = len(ang_a)
    base_offset = ang_a[0] - ang_b[0]
    snap_step = 2 * math.pi / n

    best_offset = base_offset
    best_score = float('inf')

    for k in range(n):
        test_offset = base_offset + k * snap_step
        score = 0.0  # bolt-only alignment: all positions equally valid without keyway
        if score < best_score:
            best_score = score
            best_offset = test_offset

    return math.degrees(best_offset), best_score, global_axis


# ── main refinement pipeline ──────────────────────────────────────

# Registered strategies in application order.
# Each is tried: if can_apply → apply.
_STRATEGIES = [
    KabschBoltPatternStrategy(),
    UnifyMainAxisStrategy(),
    CoaxialTranslationStrategy(),
    FacePositioningStrategy(),
    PlanarOffsetStrategy(),
    CoaxialFlipStrategy(),
    AxialCenteringStrategy(),
    BoltKeywayAlignStrategy(),
    BorelessFollowStrategy(),
    PlanarFallbackStrategy(),
]


def refine_placements(components, matches, parts_features, folder):
    """
    Phase 3 — collision-aware refinement.

    1. Detect collisions between all pairs after Phase 2 placement
    2. Run each registered strategy, passing collision info for context
    3. Strategies use collision data + surrounding features to adjust
       translations and rotations appropriately

    Returns components (mutated in-place for convenience).
    """
    # ── Refinement: run CoaxialFlipStrategy + BoltKeywayAlignStrategy ──
    # CoaxialFlip ensures correct face orientation for coaxial pairs.
    # BoltKeywayAlign fixes rotational alignment using bolt hole patterns.
    # Other strategies are disabled to avoid spurious rotations/translations
    # on CAD-exported parts that share a common coordinate system.
    _ENABLED_STRATEGIES = {'kabsch_bolt_pattern', 'coaxial_flip'}
    for comp in components:
        for strategy in _STRATEGIES:
            if strategy.name not in _ENABLED_STRATEGIES:
                continue
            try:
                if strategy.can_apply(comp, matches, parts_features, components, folder,
                                      collisions=None):
                    strategy.apply(comp, matches, parts_features, components, folder,
                                   collisions=None)
            except Exception:
                pass

    # Clean up internal metadata from placements
    for comp in components:
        comp['placement'].pop('_planar_distance', None)

    return components

