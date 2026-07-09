"""
constraints.py — Phase 1: feature-space matching + pocket detection.
Detects which geometric features pair across parts — purely descriptive,
working in each part's LOCAL coordinate system. No coordinate transforms.

Output: lightweight match pairs with feature indices.

Architecture:
  Phase 1 (this file):   match_features()          — "who pairs with whom"
  Phase 2 (coordinate_solver.py): solve_in_global_frame() — "how to place"
  Phase 3 (refinement.py):        refine_placements()     — "fine-tune"
"""

import math
import itertools

from pocket import detect_pockets, match_pockets


# ── constraint types ──────────────────────────────────────────────

COAXIAL      = "coaxial"       # two cylinders share the same axis
PLANAR_MATE  = "planar_mate"   # two planes touch with opposite normals
PLANAR_ALIGN = "planar_align"  # two planes are coplanar (same normal, close)
CLEARANCE    = "clearance"     # cylinder inside larger cylinder (shaft in bore)
POCKET_MATE  = "pocket_mate"   # pocket/slot matched to insert part


# ── vector helpers ────────────────────────────────────────────────

def _vec_dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def _vec_cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def _vec_norm(a):
    m = math.sqrt(sum(x*x for x in a))
    return tuple(x/m for x in a) if m > 1e-30 else (0, 0, 1)

def _vec_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))

def _radial_dist_from_axis(point, axis_origin, axis_dir):
    """Distance from a point to a line."""
    op = _vec_sub(point, axis_origin)
    dot_op = sum(op[i] * axis_dir[i] for i in range(3))
    proj = tuple(dot_op * axis_dir[i] for i in range(3))
    perp = _vec_sub(op, proj)
    return math.sqrt(sum(x*x for x in perp))


# ── matching thresholds ───────────────────────────────────────────

RADIUS_TOL     = 2.0      # coaxial if radius diff < this
CLEARANCE_MIN  = 0.1      # clearance if radius diff > this
MIN_CYL_RADIUS = 2.0      # include bolt-hole-sized cylinders (down from 10.0)
MIN_PLANE_AREA = 50.0   # lowered from 500 to catch small parts like keys
MIN_PLANE_AREA_RATIO = 0.2  # ignore planar pairs where areas differ by >5x

# ── part type classification ──────────────────────────────────────

def _classify_part(feats):
    """
    Classify a part's cylinder reliability for matching purposes.
    
    Returns 'planar' if the part's cylinders are likely mesh artifacts
    (tessellated sheet metal) and should NOT be used for coaxial/clearance
    matching. Returns 'cylindrical' if cylinders are reliable structural features.
    
    Heuristic:
      - 0 cylinders → planar (no cylinders to match anyway)
      - Few cylinders (< 10) → cylindrical (likely structural, even if many planes)
      - Many cylinders (> 50) AND (planes dominate OR most cyls are tiny) → planar
      - Otherwise → cylindrical
    """
    n_cyl = len(feats.get('cylinders', []))
    n_plane = len(feats.get('planes', []))
    
    if n_cyl == 0:
        return 'planar'
    
    # Few cylinders are almost certainly structural (shaft, bolt holes)
    if n_cyl < 10:
        return 'cylindrical'
    
    # Many cylinders: check for tessellation artifact patterns
    cyls = feats.get('cylinders', [])
    
    # Criterion 1: planes vastly outnumber cylinders
    if n_plane / max(n_cyl, 1) > 3:
        return 'planar'
    
    # Criterion 2: most cylinders are tiny (mesh triangulation noise)
    small_cyl_ratio = sum(1 for c in cyls if c['radius'] < 15) / max(n_cyl, 1)
    if small_cyl_ratio > 0.7:
        return 'planar'
    
    return 'cylindrical'


# ── feature-space matching ────────────────────────────────────────

def match_features(parts_features, thresholds=None):
    """
    Phase 1 — feature-space matching + pocket detection.
    Returns match pairs including standard geometric matches
    and pocket/slot matches for small-feature pairs.
    """
    matches = _match_geometric(parts_features, thresholds)

    # Pocket/slot detection — catches small orthogonal faces
    thresholds = thresholds or {}
    maximum_pocket_planes = int(
        thresholds.get("maximum_pocket_detection_planes", 800)
    )
    pockets_by_part = {}
    for name, feats in parts_features.items():
        pocket_features = feats
        planes = list(feats.get("planes") or [])
        if len(planes) > maximum_pocket_planes:
            pocket_features = dict(feats)
            pocket_features["planes"] = sorted(
                planes,
                key=lambda row: float(row.get("area", 0.0)),
                reverse=True,
            )[:maximum_pocket_planes]
        pockets_by_part[name] = detect_pockets(name, pocket_features)
    pocket_matches = match_pockets(parts_features, pockets_by_part)
    matches.extend(pocket_matches)

    preserve_planar = bool(
        (thresholds or {}).get("preserve_planar_face_hypotheses", False)
    )
    preserve_cylindrical = bool(
        (thresholds or {}).get(
            "preserve_cylindrical_face_hypotheses",
            False,
        )
    )
    return _deduplicate_matches(
        matches,
        parts_features,
        preserve_planar_face_hypotheses=preserve_planar,
        preserve_cylindrical_face_hypotheses=preserve_cylindrical,
    )


def _match_geometric(parts_features, thresholds=None):
    """Standard geometric matching: coaxial, clearance, planar."""
    thresholds = thresholds or {}
    radius_tolerance = float(thresholds.get("radius_tolerance_mm", RADIUS_TOL))
    clearance_minimum = float(thresholds.get("clearance_minimum_mm", CLEARANCE_MIN))
    minimum_cylinder_radius = float(
        thresholds.get("minimum_cylinder_radius_mm", MIN_CYL_RADIUS)
    )
    minimum_plane_area = float(
        thresholds.get("minimum_plane_area_mm2", MIN_PLANE_AREA)
    )
    minimum_plane_area_ratio = float(
        thresholds.get("minimum_plane_area_ratio", MIN_PLANE_AREA_RATIO)
    )
    minimum_local_plane_area = float(
        thresholds.get(
            "minimum_local_plane_area_mm2", minimum_plane_area
        )
    )
    local_component_diagonal = float(
        thresholds.get("local_component_diagonal_mm", 0.0)
    )
    maximum_local_planes = int(
        thresholds.get("maximum_local_planar_faces_per_part", 40)
    )
    matches = []
    part_names = list(parts_features.keys())

    # Classify each part
    part_types = {p: _classify_part(parts_features[p]) for p in part_names}

    for i, j in itertools.combinations(range(len(part_names)), 2):
        pi, pj = part_names[i], part_names[j]
        fi, fj = parts_features[pi], parts_features[pj]
        ti, tj = part_types[pi], part_types[pj]

        # ── cylinder-cylinder matches ──
        # Skip cylinder matching when BOTH parts have unreliable cylinders
        # (tessellated sheet metal). If only one part is planar, its few
        # cylinders might still be structural — keep them.
        both_unreliable = (ti == 'planar' and tj == 'planar')
        cyl_radius_min = minimum_cylinder_radius
        if both_unreliable:
            # Both parts are tessellated sheet metal — cylinders are mesh noise.
            # Only keep very large structural cylinders (e.g. housing shell).
            cyl_radius_min = max(minimum_cylinder_radius, 50.0)

        cyls_i = [(idx, c) for idx, c in enumerate(fi.get('cylinders', []))
                   if c['radius'] >= cyl_radius_min]
        cyls_j = [(idx, c) for idx, c in enumerate(fj.get('cylinders', []))
                   if c['radius'] >= cyl_radius_min]

        for idx_i, ci in cyls_i:
            for idx_j, cj in cyls_j:
                dr = ci['radius'] - cj['radius']

                polarity_i = ci.get('surface_polarity', 'unknown')
                polarity_j = cj.get('surface_polarity', 'unknown')
                opposite_known_polarity = {
                    polarity_i,
                    polarity_j,
                } == {'convex', 'concave'}
                coaxial_radius_match = abs(dr) < radius_tolerance and not (
                    opposite_known_polarity
                    and abs(dr) > clearance_minimum
                )
                if coaxial_radius_match:
                    sum_r = ci['radius'] + cj['radius']
                    matches.append({
                        'type': COAXIAL,
                        'parts': (pi, pj),
                        'feat_a_idx': idx_i,
                        'feat_b_idx': idx_j,
                        'radius_match': abs(dr),
                        '_sort_key': -sum_r,
                        '_radius_a': ci['radius'],
                    })
                # A small, positive engineering clearance is also valid
                # coaxial evidence. Emit both hypotheses so scoring/search can
                # distinguish an equal-radius alignment from shaft-in-bore.
                if dr < -clearance_minimum:
                    inner_polarity = ci.get('surface_polarity', 'unknown')
                    outer_polarity = cj.get('surface_polarity', 'unknown')
                    polarity_known = (
                        inner_polarity in {'convex', 'concave'}
                        and outer_polarity in {'convex', 'concave'}
                    )
                    if not polarity_known or (
                        inner_polarity == 'convex'
                        and outer_polarity == 'concave'
                    ):
                        matches.append({
                            'type': CLEARANCE,
                            'parts': (pi, pj),
                            'feat_a_idx': idx_i,
                            'feat_b_idx': idx_j,
                            'gap': -dr,
                            '_sort_key': -dr,
                            '_radius_a': ci['radius'],
                        })
                elif dr > clearance_minimum:
                    inner_polarity = cj.get('surface_polarity', 'unknown')
                    outer_polarity = ci.get('surface_polarity', 'unknown')
                    polarity_known = (
                        inner_polarity in {'convex', 'concave'}
                        and outer_polarity in {'convex', 'concave'}
                    )
                    if not polarity_known or (
                        inner_polarity == 'convex'
                        and outer_polarity == 'concave'
                    ):
                        matches.append({
                            'type': CLEARANCE,
                            'parts': (pj, pi),
                            'feat_a_idx': idx_j,
                            'feat_b_idx': idx_i,
                            'gap': dr,
                            '_sort_key': dr,
                            '_radius_a': cj['radius'],
                        })

        # ── plane-plane matches ──
        # A small key, tab, retainer, or insert often mates through faces much
        # smaller than the main structural faces of the larger part.  Lower
        # the area floor only for pairs containing a bounded small component;
        # this localizes interfaces without globally enumerating every tiny
        # face on large CAD.
        def bbox_diagonal(feature):
            bbox = feature.get("bbox", {})
            minimum = bbox.get("min", [0.0, 0.0, 0.0])
            maximum = bbox.get("max", [0.0, 0.0, 0.0])
            return math.sqrt(
                sum(
                    (float(maximum[index]) - float(minimum[index])) ** 2
                    for index in range(3)
                )
            )

        localized_pair = (
            local_component_diagonal > 0.0
            and min(bbox_diagonal(fi), bbox_diagonal(fj))
            <= local_component_diagonal
        )
        pair_plane_area = (
            min(minimum_plane_area, minimum_local_plane_area)
            if localized_pair
            else minimum_plane_area
        )
        planes_i = [(idx, p) for idx, p in enumerate(fi.get('planes', []))
                     if p.get('area', 0) >= pair_plane_area]
        planes_j = [(idx, p) for idx, p in enumerate(fj.get('planes', []))
                     if p.get('area', 0) >= pair_plane_area]
        if localized_pair:
            planes_i = sorted(
                planes_i,
                key=lambda row: float(row[1].get("area", 0.0)),
                reverse=True,
            )[:maximum_local_planes]
            planes_j = sorted(
                planes_j,
                key=lambda row: float(row[1].get("area", 0.0)),
                reverse=True,
            )[:maximum_local_planes]

        for idx_i, pa in planes_i:
            for idx_j, pb in planes_j:
                na, nb = pa['normal'], pb['normal']
                dot = _vec_dot(na, nb)
                # Skip if areas differ drastically — likely spurious
                # (e.g. small bevel face matching main structural face)
                area_a = pa.get('area', 0)
                area_b = pb.get('area', 0)
                if area_a > 0 and area_b > 0:
                    area_ratio = min(area_a, area_b) / max(area_a, area_b)
                    if area_ratio < minimum_plane_area_ratio:
                        continue
                dist = _vec_dot(_vec_sub(pb['position'], pa['position']), na)
                # Local face normals encode the intended assembly orientation.
                #   dot < -0.95  →  anti-parallel local normals  →  faces face each
                #                   other  →  planar_mate (true mate)
                #   dot >  0.95  →  parallel local normals       →  faces face same
                #                   direction  →  planar_align (true align)
                # Generating both types for the wrong polarity only adds noise
                # that the solver cannot disambiguate without collision awareness.
                if dot < -0.95:
                    matches.append({
                        'type': PLANAR_MATE,
                        'parts': (pi, pj),
                        'feat_a_idx': idx_i,
                        'feat_b_idx': idx_j,
                        'distance': dist,
                        'candidate_origin': (
                            'localized_small_component_planar'
                            if localized_pair
                            else 'global_structural_planar'
                        ),
                        '_sort_key': abs(dist),
                        '_normal_hash': _bucket_normal(na),
                    })
                if dot > 0.95:
                    matches.append({
                        'type': PLANAR_ALIGN,
                        'parts': (pi, pj),
                        'feat_a_idx': idx_i,
                        'feat_b_idx': idx_j,
                        'distance': dist,
                        'candidate_origin': (
                            'localized_small_component_planar'
                            if localized_pair
                            else 'global_structural_planar'
                        ),
                        '_sort_key': abs(dist),
                        '_normal_hash': _bucket_normal(na),
                    })

    # ── Planar matches: dedup by normal bucket to preserve
    # multiple orientation constraints (needed for enclosures).
    return _deduplicate_matches(
        matches,
        parts_features,
        preserve_planar_face_hypotheses=bool(
            thresholds.get("preserve_planar_face_hypotheses", False)
        ),
        preserve_cylindrical_face_hypotheses=bool(
            thresholds.get(
                "preserve_cylindrical_face_hypotheses",
                False,
            )
        ),
    )


def _deduplicate_matches(
    matches,
    parts_features=None,
    *,
    preserve_planar_face_hypotheses=False,
    preserve_cylindrical_face_hypotheses=False,
):
    """
    Deduplicate matches:
      - Coaxial / Clearance: compact mode keeps one best per type and radius
        bucket; pose mode can preserve every distinct feature pair so repeated
        equal-radius holes remain separate placement hypotheses.
      - Planar mate / align: one best per NORMAL BUCKET per part pair.
        This preserves multiple planar constraints at different orientations
        (e.g. +X wall, -X wall, +Y floor), which is essential for
        positioning planar-dominant parts (sheet metal enclosures).
    """
    groups = {}
    for m in matches:
        t = m['type']
        if t in (PLANAR_MATE, PLANAR_ALIGN):
            nh = m.get('_normal_hash', '??')
            if preserve_planar_face_hypotheses:
                # Pose search needs parallel end faces as distinct hypotheses.
                # Local-frame plane offsets cannot identify the mating face.
                a, b = m["parts"]
                ia, ib = m["feat_a_idx"], m["feat_b_idx"]
                ordered_features = (ia, ib) if a <= b else (ib, ia)
                key = (
                    min(a, b),
                    max(a, b),
                    t,
                    nh,
                    ordered_features,
                )
            else:
                # Compact indexing mode: one candidate per orientation.
                key = (min(m['parts']), max(m['parts']), t, nh)
        elif t in (COAXIAL, CLEARANCE):
            # Bucket by the matched cylinder radius (rounded) so bolt holes
            # (r≈4mm) and main bores (r≈60mm) are preserved as separate constraints
            if preserve_cylindrical_face_hypotheses:
                # Repeated equal-radius holes are distinct pose hypotheses.
                # Pose reconstruction must retain feature identity so two
                # dowels can occupy two different holes.
                a, b = m["parts"]
                ia, ib = m["feat_a_idx"], m["feat_b_idx"]
                ordered_features = (ia, ib) if a <= b else (ib, ia)
                key = (
                    min(a, b),
                    max(a, b),
                    t,
                    ordered_features,
                )
            else:
                r_bucket = round(
                    m.get('_radius_a', m.get('radius_match', 0))
                )
                key = (
                    min(m['parts']),
                    max(m['parts']),
                    t,
                    r_bucket,
                )
        else:
            key = (min(m['parts']), max(m['parts']), t)

        if key not in groups:
            groups[key] = []
        groups[key].append(m)

    result = []
    # Metadata keys to preserve (not stripped even though they start with '_')
    _KEEP_META = {'_dir_a', '_dir_b', '_wall_a', '_wall_b', '_size_a', '_size_b', '_radius_a'}
    for key, group in groups.items():
        if group[0].get('_sort_key') is not None:
            best = min(group, key=lambda m: m.get('_sort_key', 0))
        else:
            best = group[0]

        clean = {k: v for k, v in best.items()
                 if not k.startswith('_') or k in _KEEP_META}
        result.append(clean)

    return result


def _bucket_normal(normal):
    """Bucket a unit normal to its nearest principal axis direction."""
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if ax >= ay and ax >= az:
        return '+X' if normal[0] > 0 else '-X'
    elif ay >= az:
        return '+Y' if normal[1] > 0 else '-Y'
    else:
        return '+Z' if normal[2] > 0 else '-Z'


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, os
    from features import extract_features

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

    matches = match_features(parts)
    print(f"\nFeature matches found: {len(matches)}")
    # Show part types
    for f in step_files:
        pt = _classify_part(parts[f])
        print(f"  {f[:45]} type={pt}")
    for m in matches[:10]:
        a, b = m['parts']
        extra = ""
        if m['type'] == COAXIAL:
            extra = f" radius_match={m['radius_match']:.2f}"
        elif m['type'] == CLEARANCE:
            extra = f" gap={m['gap']:.2f}"
        elif m['type'] in (PLANAR_MATE, PLANAR_ALIGN):
            extra = f" distance={m['distance']:.2f}"
        print(f"  {m['type']:12s}  {a} <-> {b}  feat({m['feat_a_idx']},{m['feat_b_idx']}){extra}")
    if len(matches) > 10:
        print(f"  ... and {len(matches)-10} more")
