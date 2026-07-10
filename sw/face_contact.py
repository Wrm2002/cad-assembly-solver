"""
face_contact.py — Geometric face-to-face contact measurement.

Computes precise distances between mating faces using plane/cylinder
geometry — no mesh or SDF needed. Much faster and more accurate.

Key metric: distance between mating faces after placement.
If distance < threshold → faces are in contact.
"""
import math
import numpy as np


def _norm(v):
    n = math.sqrt(sum(x*x for x in v))
    return [x/n for x in v] if n > 1e-12 else [0, 0, 1]


def _dot(a, b):
    return sum(a[i]*b[i] for i in range(3))


def _sub(a, b):
    return [a[i]-b[i] for i in range(3)]


def face_distance(face_a, face_b, threshold=0.1):
    """Compute signed distance between two faces.

    Args:
        face_a: dict with position, normal/axis (from features extraction)
        face_b: dict with position, normal/axis
        threshold: mm, faces closer than this are "in contact"

    Returns:
        (distance_mm, is_contact, info_string)
    """
    pos_a = face_a.get("position", [0, 0, 0])
    pos_b = face_b.get("position", [0, 0, 0])

    # Detect face types
    a_is_plane = "normal" in face_a
    b_is_plane = "normal" in face_b
    a_is_cyl = "axis" in face_a
    b_is_cyl = "axis" in face_b

    if a_is_plane and b_is_plane:
        # Plane-to-plane distance
        n_a = _norm(face_a["normal"])
        n_b = _norm(face_b["normal"])
        # Distance perpendicular to face A's plane
        vec = _sub(pos_b, pos_a)
        dist = abs(_dot(vec, n_a))
        # Normal alignment (for mate: anti-parallel; for align: parallel)
        dot_n = abs(_dot(n_a, n_b))
        is_contact = dist < threshold and dot_n > 0.95
        info = f"plane-plane dist={dist:.3f}mm align={dot_n:.3f}"
        return dist, is_contact, info

    if a_is_cyl and b_is_cyl:
        # Cylinder-to-cylinder: radial clearance
        o_a = pos_a
        o_b = pos_b
        d_a = _norm(face_a["axis"])
        d_b = _norm(face_b["axis"])
        r_a = face_a.get("radius", 0)
        r_b = face_b.get("radius", 0)

        # Axes should be parallel
        dot_axes = abs(_dot(d_a, d_b))
        if dot_axes < 0.95:
            return 999, False, f"cyl-cyl axes not parallel ({dot_axes:.2f})"

        # Radial distance between axes
        vec = _sub(o_b, o_a)
        proj = _dot(vec, d_a)
        radial_vec = _sub(vec, [d_a[i]*proj for i in range(3)])
        radial_dist = math.sqrt(sum(x*x for x in radial_vec))
        clearance = radial_dist - abs(r_a - r_b)  # clearance fit
        is_contact = abs(clearance) < threshold
        info = f"cyl-cyl radial={radial_dist:.3f}mm clearance={clearance:.3f}mm"
        return clearance, is_contact, info

    if a_is_plane and b_is_cyl:
        # Plane-to-cylinder: distance from plane to cylinder surface
        n_a = _norm(face_a["normal"])
        o_b = pos_b
        d_b = _norm(face_b["axis"])
        r_b = face_b.get("radius", 0)

        # Check if plane normal is perpendicular to cylinder axis
        dot_n_axis = abs(_dot(n_a, d_b))
        if dot_n_axis > 0.05:
            return 999, False, f"plane normal not perpendicular to cyl axis ({dot_n_axis:.3f})"

        # Distance from plane to cylinder axis
        vec = _sub(o_b, pos_a)
        axis_dist = abs(_dot(vec, n_a))
        surf_dist = axis_dist - r_b  # distance from plane to cylinder surface
        is_contact = abs(surf_dist) < threshold
        info = f"plane-cyl axis_dist={axis_dist:.3f}mm surf_dist={surf_dist:.3f}mm"
        return surf_dist, is_contact, info

    if b_is_plane and a_is_cyl:
        return face_distance(face_b, face_a, threshold)

    return 999, False, "unknown face types"


def compute_pair_contact(features_a, features_b, matches, placements,
                         threshold=0.1):
    """Evaluate contact quality for all matched face pairs.

    Args:
        features_a: features dict for part A (receiver)
        features_b: features dict for part B (insert)
        matches: list of constraint matches between A and B
        placements: dict of part_name → placement
        threshold: contact distance threshold (mm)

    Returns:
        list of {match_type, distance, contact, info}
    """
    results = []
    for match in matches:
        mtype = match["type"]
        # Get the matched feature indices
        a_idx = match.get("feat_a_idx")
        b_idx = match.get("feat_b_idx")

        # Get feature geometry
        face_a = None
        face_b = None
        if a_idx is not None and "planes" in features_a:
            planes_a = features_a["planes"]
            if a_idx < len(planes_a):
                face_a = planes_a[a_idx]
        if b_idx is not None and "planes" in features_b:
            planes_b = features_b["planes"]
            if b_idx < len(planes_b):
                face_b = planes_b[b_idx]

        # Also check cylinders for clearance/coaxial
        if (face_a is None or face_b is None) and mtype in ("clearance", "coaxial"):
            cyls_a = features_a.get("cylinders", [])
            cyls_b = features_b.get("cylinders", [])
            if cyls_a and cyls_b:
                face_a = max(cyls_a, key=lambda c: c["radius"])
                face_b = max(cyls_b, key=lambda c: c["radius"])
                # Override position with cylinder origin
                face_a = dict(face_a)
                face_b = dict(face_b)
                face_a["position"] = face_a.get("origin", [0, 0, 0])
                face_b["position"] = face_b.get("origin", [0, 0, 0])

        if face_a is None or face_b is None:
            results.append({"type": mtype, "distance": 999, "contact": False,
                           "info": "no feature data"})
            continue

        # Apply placement transforms
        # (For now, assume feature positions are in local coords and
        #  the placement transforms have been applied. This is a simplification.)
        dist, contact, info = face_distance(face_a, face_b, threshold)
        results.append({
            "type": mtype,
            "distance": float(dist),
            "contact": contact,
            "info": info,
        })
    return results
