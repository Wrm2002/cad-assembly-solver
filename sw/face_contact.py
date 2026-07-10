"""
face_contact.py — Geometric face-to-face contact measurement.

Computes precise distances between mating faces using plane/cylinder
geometry — no mesh or SDF needed. Applies placement transforms to
measure distances in world coordinates.
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

def _scale(v, s):
    return [v[i]*s for i in range(3)]


def placement_to_4x4(plac):
    """Convert placement dict to 4x4 transform matrix."""
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
    return aff


def _transform_point(pt, aff):
    p = np.array([pt[0], pt[1], pt[2], 1.0])
    r = aff @ p
    return [r[0], r[1], r[2]]


def _transform_direction(d, aff):
    v = np.array([d[0], d[1], d[2], 0.0])
    r = aff @ v
    return [r[0], r[1], r[2]]


def face_distance(face_a, face_b, threshold=0.1):
    """Compute signed distance between two faces in world coordinates."""
    pos_a = face_a.get("position", [0, 0, 0])
    pos_b = face_b.get("position", [0, 0, 0])
    a_is_plane = "normal" in face_a
    b_is_plane = "normal" in face_b
    a_is_cyl = "axis" in face_a
    b_is_cyl = "axis" in face_b

    if a_is_plane and b_is_plane:
        n_a = _norm(face_a["normal"])
        n_b = _norm(face_b["normal"])
        vec = _sub(pos_b, pos_a)
        dist = abs(_dot(vec, n_a))
        dot_n = abs(_dot(n_a, n_b))
        is_contact = dist < threshold and dot_n > 0.95
        return dist, is_contact, f"plane-plane dist={dist:.3f}mm align={dot_n:.3f}"

    if a_is_cyl and b_is_cyl:
        o_a, o_b = pos_a, pos_b
        d_a = _norm(face_a["axis"])
        d_b = _norm(face_b["axis"])
        r_a = face_a.get("radius", 0)
        r_b = face_b.get("radius", 0)
        dot_axes = abs(_dot(d_a, d_b))
        if dot_axes < 0.95:
            return 999, False, f"cyl-cyl axes not parallel ({dot_axes:.2f})"
        vec = _sub(o_b, o_a)
        proj = _dot(vec, d_a)
        radial_vec = _sub(vec, _scale(d_a, proj))
        radial_dist = math.sqrt(sum(x*x for x in radial_vec))
        clearance = radial_dist - abs(r_a - r_b)
        is_contact = abs(clearance) < threshold
        return clearance, is_contact, f"cyl-cyl radial={radial_dist:.3f}mm clearance={clearance:.3f}mm"

    return 999, False, "unknown face types"


def compute_pair_contact(features_a, features_b, matches, plac_a, plac_b,
                         name_a="", name_b="", threshold=0.1):
    """Evaluate contact quality for all matched face pairs in world coords."""
    aff_a = placement_to_4x4(plac_a)
    aff_b = placement_to_4x4(plac_b)

    results = []
    for match in matches:
        mtype = match["type"]
        parts = match["parts"]

        # Map match indices: feat_a_idx ↔ parts[0], feat_b_idx ↔ parts[1]
        # Match parts use short names; compare by containment
        p0_matches_a = parts[0] in name_a or name_a in parts[0]
        p0_matches_b = parts[0] in name_b or name_b in parts[0]
        if p0_matches_a and (parts[1] in name_b or name_b in parts[1]):
            a_idx, b_idx = match.get("feat_a_idx"), match.get("feat_b_idx")
        elif p0_matches_b and (parts[1] in name_a or name_a in parts[1]):
            a_idx, b_idx = match.get("feat_b_idx"), match.get("feat_a_idx")
        else:
            a_idx, b_idx = match.get("feat_a_idx"), match.get("feat_b_idx")

        face_a = None
        face_b = None

        # Try planes
        if a_idx is not None and "planes" in features_a:
            planes_a = features_a["planes"]
            if a_idx < len(planes_a):
                face_a = dict(planes_a[a_idx])
                face_a["position"] = _transform_point(face_a["position"], aff_a)
                if "normal" in face_a:
                    face_a["normal"] = _transform_direction(face_a["normal"], aff_a)
                if "axis" in face_a:
                    face_a["axis"] = _transform_direction(face_a["axis"], aff_a)

        if b_idx is not None and "planes" in features_b:
            planes_b = features_b["planes"]
            if b_idx < len(planes_b):
                face_b = dict(planes_b[b_idx])
                face_b["position"] = _transform_point(face_b["position"], aff_b)
                if "normal" in face_b:
                    face_b["normal"] = _transform_direction(face_b["normal"], aff_b)
                if "axis" in face_b:
                    face_b["axis"] = _transform_direction(face_b["axis"], aff_b)

        # For clearance/coaxial, use cylinders
        if (face_a is None or face_b is None) and mtype in ("clearance", "coaxial"):
            cyls_a = features_a.get("cylinders", [])
            cyls_b = features_b.get("cylinders", [])
            if cyls_a and cyls_b:
                ca = max(cyls_a, key=lambda c: c["radius"])
                cb = max(cyls_b, key=lambda c: c["radius"])
                face_a = {"axis": ca["axis"], "radius": ca["radius"],
                          "position": _transform_point(ca["origin"], aff_a)}
                face_b = {"axis": cb["axis"], "radius": cb["radius"],
                          "position": _transform_point(cb["origin"], aff_b)}
                face_a["axis"] = _transform_direction(face_a["axis"], aff_a)
                face_b["axis"] = _transform_direction(face_b["axis"], aff_b)

        if face_a is None or face_b is None:
            results.append({"type": mtype, "distance": 999, "contact": False,
                           "info": "no feature data"})
            continue

        dist, contact, info = face_distance(face_a, face_b, threshold)
        results.append({"type": mtype, "distance": float(dist), "contact": contact,
                       "info": info})
    return results
