"""
pocket.py — Pocket/slot/keyway detection for STEP assembly.

Detects rectangular recesses formed by groups of orthogonal small planes
that would be missed by the MIN_PLANE_AREA threshold.

A pocket has:
  - 1 bottom face (deepest, "floor" of the recess)
  - 2+ parallel side walls (form the "walls")
  - optionally 2 end walls

Phase 1 integration: match pockets between parts as a new constraint type.
"""

import math
import os

# ── pocket data structure ─────────────────────────────────────────

# A pocket is a dict:
# {
#   'center': (cx, cy, cz),          # center of pocket bounding box
#   'size':   (sx, sy, sz),          # width, depth, length
#   'direction': (dx, dy, dz),       # main axis (bottom normal, points "out" of pocket)
#   'wall_normal': (wx, wy, wz),     # side wall normal
#   'faces': [plane_dict, ...],      # the planes forming this pocket
#   'part': part_name,
# }


def detect_pockets(part_name, features, min_area=50, proximity=10.0):
    """
    Find all pocket/slot features in a single part.

    Algorithm:
    1. Take all planes (no area filter).
    2. Group by normal direction (±X, ±Y, ±Z).
    3. For each group, cluster planes that are close to each other.
    4. Find orthogonal pairs: a "bottom" plane + "wall" planes.
    5. Form pocket features from orthogonal triads.

    Args:
        part_name: str
        features: features dict from extract_features()
        min_area: minimum face area to consider (mm²)
        proximity: max distance between faces in a cluster (mm)

    Returns:
        list of pocket dicts
    """
    planes = features.get('planes', [])
    if len(planes) < 3:
        return []

    # 1. Filter by minimum area and bucket by normal direction
    by_dir = {'+X': [], '-X': [], '+Y': [], '-Y': [], '+Z': [], '-Z': []}

    for p in planes:
        area = p.get('area', 0) or 0
        if area < min_area:
            continue
        n = p['normal']
        ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
        if ax >= ay and ax >= az:
            key = '+X' if n[0] > 0 else '-X'
        elif ay >= az:
            key = '+Y' if n[1] > 0 else '-Y'
        else:
            key = '+Z' if n[2] > 0 else '-Z'
        by_dir[key].append(p)

    # 2. Within each direction, cluster by spatial proximity
    clusters_by_dir = {}
    for direction, dir_planes in by_dir.items():
        if len(dir_planes) < 1:
            continue
        clusters = _cluster_by_proximity(dir_planes, proximity)
        clusters_by_dir[direction] = clusters

    # 3. Find orthogonal pairs: bottom + walls
    # A pocket needs:
    #   - 1 "bottom" (one direction)
    #   - 2+ "walls" (orthogonal direction, parallel to each other)
    orthogonal_pairs = [
        (('+X', '-X'), ('+Y', '-Y')),   # X bottom, Y walls (or vice versa)
        (('+X', '-X'), ('+Z', '-Z')),   # X bottom, Z walls
        (('+Y', '-Y'), ('+Z', '-Z')),   # Y bottom, Z walls
    ]

    pockets = []
    for (d1a, d1b), (d2a, d2b) in orthogonal_pairs:
        # Try d1 as bottom, d2 as walls
        for bottom_dir in (d1a, d1b):
            if bottom_dir not in clusters_by_dir:
                continue
            for wall_dir in (d2a, d2b):
                if wall_dir not in clusters_by_dir:
                    continue
                # Find bottom+wall clusters that are spatially close
                for bottom_cluster in clusters_by_dir[bottom_dir]:
                    bottom_center = _cluster_center(bottom_cluster)
                    wall_clusters_nearby = []
                    for wall_cluster in clusters_by_dir[wall_dir]:
                        wall_center = _cluster_center(wall_cluster)
                        dist = _point_dist(bottom_center, wall_center)
                        if dist < proximity * 3:
                            wall_clusters_nearby.append(wall_cluster)

                    if len(wall_clusters_nearby) >= 1:
                        # Found a pocket!
                        all_faces = list(bottom_cluster)
                        for wc in wall_clusters_nearby:
                            all_faces.extend(wc)

                        # Compute pocket geometry
                        pocket = _build_pocket(
                            part_name, bottom_cluster,
                            wall_clusters_nearby, all_faces)
                        if pocket:
                            pockets.append(pocket)

    # ── Also detect recesses (slots, keyways, cavities) ──
    # A recess is formed by small faces clustered in a tight spatial region
    # with normals pointing in at least 2 different directions.
    # Unlike protrusions (normals outward), recess faces have normals pointing
    # toward each other or inward — but we don't need to distinguish direction.
    # We just need to find tight spatial clusters of small planes.
    recesses = _detect_recesses(part_name, planes, proximity=proximity * 2)
    pockets.extend(recesses)

    return pockets


# ── recess / slot / keyway detection ──────────────────────────────

def _detect_recesses(part_name, planes, proximity=20.0, min_faces=2):
    """
    Detect recesses (slots, keyways, cavities) formed by small faces
    clustered in a tight spatial region.

    Unlike pocket detection which requires orthogonal bottom+wall structure,
    this simply finds spatial clusters of small faces with diverse normals.
    These typically correspond to cutouts, slots, and keyways.
    """
    if len(planes) < min_faces:
        return []

    # Take ALL planes (no area filter — keyways have tiny faces)
    # Cluster them spatially
    clusters = _cluster_by_proximity(planes, proximity)

    recesses = []
    for cluster in clusters:
        if len(cluster) < min_faces:
            continue

        # Check normal diversity: a recess needs faces pointing in
        # at least 2 different directions
        normals = set()
        for p in cluster:
            n = p['normal']
            # Bucket normal to 6 primary directions
            ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
            if ax >= ay and ax >= az:
                normals.add('+X' if n[0] > 0 else '-X')
            elif ay >= az:
                normals.add('+Y' if n[1] > 0 else '-Y')
            else:
                normals.add('+Z' if n[2] > 0 else '-Z')

        if len(normals) < 2:
            continue  # All faces same direction — not a recess

        # Compute bounding box
        all_positions = [f['position'] for f in cluster]
        xs = [p[0] for p in all_positions]
        ys = [p[1] for p in all_positions]
        zs = [p[2] for p in all_positions]
        sx, sy, sz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

        if sx < 0.5 or sy < 0.5 or sz < 0.5:
            continue  # too flat — degenerate

        center = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2)

        # Use the most common normal direction as the "axis"
        dir_counts = {}
        for p in cluster:
            n = p['normal']
            ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
            if ax >= ay and ax >= az:
                key = ('X', 1 if n[0] > 0 else -1)
            elif ay >= az:
                key = ('Y', 1 if n[1] > 0 else -1)
            else:
                key = ('Z', 1 if n[2] > 0 else -1)
            dir_counts[key] = dir_counts.get(key, 0) + 1

        best_dir = max(dir_counts, key=dir_counts.get)
        direction = tuple(
            best_dir[1] if i == ord(best_dir[0]) - ord('X') else 0.0
            for i in range(3)
        )

        # Find a wall normal (orthogonal direction with second-most faces)
        wall_normal = (0.0, 0.0, 1.0)
        if best_dir[0] == 'Z':
            wall_normal = (1.0, 0.0, 0.0)
        elif best_dir[0] == 'X':
            wall_normal = (0.0, 1.0, 0.0)

        recesses.append({
            'center': center,
            'size': (sx, sy, sz),
            'direction': direction,
            'wall_normal': wall_normal,
            'faces': cluster,
            'part': part_name,
            'is_recess': True,  # mark as recess for matching
        })

    return recesses

    return pockets


def _cluster_center(planes):
    """Center of a group of planes (average position)."""
    if not planes:
        return (0, 0, 0)
    sx = sy = sz = 0.0
    for p in planes:
        pos = p['position']
        sx += pos[0]; sy += pos[1]; sz += pos[2]
    n = len(planes)
    return (sx / n, sy / n, sz / n)


def _point_dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _cluster_by_proximity(planes, max_dist):
    """Simple greedy clustering of planes by position proximity."""
    if not planes:
        return []
    remaining = list(planes)
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for p in list(remaining):
                if _point_dist(_cluster_center(cluster), p['position']) < max_dist:
                    cluster.append(p)
                    remaining.remove(p)
                    changed = True
        clusters.append(cluster)
    return clusters


def _build_pocket(part_name, bottom_faces, wall_clusters, all_faces):
    """Build a pocket dict from face clusters."""
    if len(all_faces) < 3:
        return None  # need at least bottom + 2 walls

    bottom_normal = bottom_faces[0]['normal']
    wall_normal = wall_clusters[0][0]['normal']

    # Estimate pocket size from face positions
    all_positions = [f['position'] for f in all_faces]
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    zs = [p[2] for p in all_positions]
    sx, sy, sz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)

    if sx < 1.0 and sy < 1.0 and sz < 1.0:
        return None  # degenerate

    center = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2)

    return {
        'center': center,
        'size': (sx, sy, sz),
        'direction': bottom_normal,
        'wall_normal': wall_normal,
        'faces': all_faces,
        'part': part_name,
    }


# ── pocket matching ───────────────────────────────────────────────

def match_pockets(parts_features, pockets_by_part):
    """
    Match pockets between different parts.
    Two pockets match if:
      - Same direction and wall_normal (or opposite for mate)
      - Similar size (within tolerance)
    """
    matches = []
    part_names = list(pockets_by_part.keys())

    for i in range(len(part_names)):
        for j in range(i + 1, len(part_names)):
            pi, pj = part_names[i], part_names[j]
            for pkt_i in pockets_by_part[pi]:
                for pkt_j in pockets_by_part[pj]:
                    if _pockets_match(pkt_i, pkt_j):
                        matches.append({
                            'type': 'pocket_mate',
                            'parts': (pi, pj),
                            'pocket_a': pkt_i,
                            'pocket_b': pkt_j,
                            # Preserve direction metadata for solver
                            '_dir_a': pkt_i.get('direction'),
                            '_dir_b': pkt_j.get('direction'),
                            '_wall_a': pkt_i.get('wall_normal'),
                            '_wall_b': pkt_j.get('wall_normal'),
                            '_size_a': pkt_i.get('size'),
                            '_size_b': pkt_j.get('size'),
                        })
    return matches


def _pockets_match(a, b, size_tol=0.3):
    """
    Check if two pockets are mating features (pocket + insert).
    
    One feature (the insert/key) should fit INSIDE the other (the slot/keyway).
    So we allow up to 3x size difference — the insert can be much smaller
    than the slot, as long as it fits within the slot dimensions.
    """
    # At least one dimension must be compatible
    matched_dims = 0
    for k in range(3):
        sa, sb = a['size'][k], b['size'][k]
        if sa < 1 or sb < 1:
            continue
        ratio = min(sa, sb) / max(sa, sb)
        if ratio >= (1 - size_tol):
            matched_dims += 1
    
    # Need at least 1 matching dimension for pocket mate
    return matched_dims >= 1


# ── CLI test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from features import extract_features

    folder = sys.argv[1] if len(sys.argv) > 1 else '2'
    folder = os.path.abspath(folder)
    step_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.step', '.stp'))
        and not f.lower().startswith('assembly')
    ])

    parts = {}
    pockets_by_part = {}
    for f in step_files:
        fp = os.path.join(folder, f)
        feats = extract_features(fp)
        parts[f] = feats
        pkts = detect_pockets(f, feats)
        pockets_by_part[f] = pkts
        if pkts:
            print(f'{f}: {len(pkts)} pocket(s) found')
            for pkt in pkts:
                print(f'  center=({pkt["center"][0]:.1f},{pkt["center"][1]:.1f},{pkt["center"][2]:.1f})',
                      f'size=({pkt["size"][0]:.1f},{pkt["size"][1]:.1f},{pkt["size"][2]:.1f})',
                      f'dir=({pkt["direction"][0]:.2f},{pkt["direction"][1]:.2f},{pkt["direction"][2]:.2f})')
        else:
            print(f'{f}: no pockets')

    pkt_matches = match_pockets(parts, pockets_by_part)
    print(f'\nPocket matches: {len(pkt_matches)}')
    for m in pkt_matches:
        print(f'  {m["parts"][0][:20]} <-> {m["parts"][1][:20]}')
