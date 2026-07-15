"""
features.py — Universal geometry feature detection.
Extracts cylinders, planes, cones, torii, spheres and bounding boxes
from STEP files via mixed text-regex + OCCT face classification.

Zero assembly logic — pure geometry extraction.

Usage:
    from features import extract_features
    feats = extract_features("path/to/part.step")
    # keys: cylinders, planes, cones, torii, spheres, bbox, filepath
"""

import math
import os

# ── noise cleanup ─────────────────────────────────────────────────
SNAP_EPS = 1e-9

def _snap_scalar(v):
    return 0.0 if abs(v) < SNAP_EPS else v

def _snap_point(pt):
    return tuple(_snap_scalar(x) for x in pt)

def _snap_direction(d):
    cleaned = [_snap_scalar(x) for x in d]
    mag = math.sqrt(sum(x * x for x in cleaned))
    if mag < SNAP_EPS:
        return (0.0, 0.0, 1.0)
    return tuple(x / mag for x in cleaned)


# ── STEP text parsing (fast path for CYLINDRICAL_SURFACE) ─────────
def _parse_step_text(path):
    with open(path, 'rb') as handle:
        data = handle.read()
    for encoding in ('utf-8', 'gb18030', 'latin-1'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('latin-1', errors='replace')

def _extract_cartesian_points(text):
    import re
    pts = {}
    for m in re.finditer(
        r'#(\d+) = CARTESIAN_POINT\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)\s*\)', text
    ):
        eid, coords = m.groups()
        raw = tuple(float(x.strip()) for x in coords.split(','))
        if len(raw) == 3:
            pts[int(eid)] = _snap_point(raw)
        else:
            pts[int(eid)] = raw
    return pts

def _extract_directions(text):
    import re
    dirs = {}
    for m in re.finditer(
        r'#(\d+) = DIRECTION\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)\s*\)', text
    ):
        eid, coords = m.groups()
        raw = tuple(float(x.strip()) for x in coords.split(','))
        dirs[int(eid)] = _snap_direction(raw)
    return dirs

def _extract_cylinders(text):
    import re
    cylinders = []
    for m in re.finditer(
        r'#(\d+) = CYLINDRICAL_SURFACE\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*([\d.]+)\s*\)', text
    ):
        cylinders.append({
            'surface_id': int(m.group(1)),
            'axis_placement_id': int(m.group(2)),
            'radius': float(m.group(3))
        })
    return cylinders

def _extract_axis2_placements(text):
    import re
    placements = {}
    for m in re.finditer(
        r'#(\d+) = AXIS2_PLACEMENT_3D\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)', text
    ):
        placements[int(m.group(1))] = {
            'cart_id': int(m.group(2)),
            'axis_id': int(m.group(3)),
            'ref_id': int(m.group(4))
        }
    return placements


# ── OCCT universal face detection ─────────────────────────────────
def _detect_all_faces_occt(filepath):
    """
    OCCT-based face classification for ALL surface types.

    Returns dict with keys:
        planes, cylinders, cones, torii, spheres,
        surfaces_of_revolution, surfaces_of_extrusion,
        bspline_surfaces, bezier_surfaces, other_surfaces
    plus a unified bbox computed from the shape.
    Each list entry has the same shape as the text-parsed cylinders/planes.
    """
    try:
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import (
            TopAbs_FACE,
            TopAbs_REVERSED,
            TopAbs_SHELL,
            TopAbs_SOLID,
        )
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder,
            GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus,
            GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
            GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
            GeomAbs_OtherSurface)
        try:
            from OCC.Core.TopoDS import topods_Face
        except ImportError:
            # pythonocc-core 7.9+ exposes the down-cast as ``Face``.
            from OCC.Core.TopoDS import Face as topods_Face
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.BRepTools import breptools
        from OCC.Core.GProp import GProp_GProps
        from OCC.Core.Bnd import Bnd_Box, Bnd_OBB
        from OCC.Core.BRepBndLib import brepbndlib
    except ImportError:
        return None

    reader = STEPControl_Reader()
    if reader.ReadFile(filepath) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    shape = reader.OneShape()

    def _subshape_count(kind):
        count = 0
        explorer = TopExp_Explorer(shape, kind)
        while explorer.More():
            count += 1
            explorer.Next()
        return count

    topology = {
        'solid_count': _subshape_count(TopAbs_SOLID),
        'shell_count': _subshape_count(TopAbs_SHELL),
    }

    # Overall bbox from OCCT (more reliable than text points)
    bbox = Bnd_Box()
    bbox.SetGap(0.0)
    brepbndlib.Add(shape, bbox)
    x1, y1, z1, x2, y2, z2 = bbox.Get()
    occt_bbox = {
        'min': (_snap_scalar(x1), _snap_scalar(y1), _snap_scalar(z1)),
        'max': (_snap_scalar(x2), _snap_scalar(y2), _snap_scalar(z2)),
    }
    occt_obb = None
    try:
        obb = Bnd_OBB()
        brepbndlib.AddOBB(shape, obb, False, False, False)
        if not obb.IsVoid():
            center = obb.Center()
            directions = (
                obb.XDirection(), obb.YDirection(), obb.ZDirection()
            )
            occt_obb = {
                'center': [center.X(), center.Y(), center.Z()],
                'axes': [
                    list(_snap_direction((axis.X(), axis.Y(), axis.Z())))
                    for axis in directions
                ],
                'dimensions': [
                    2.0 * float(obb.XHSize()),
                    2.0 * float(obb.YHSize()),
                    2.0 * float(obb.ZHSize()),
                ],
                'is_axis_aligned': bool(obb.IsAABox()),
                'method': 'occt_brepbndlib_addobb',
            }
    except Exception:
        # OBB is a candidate-frame aid.  Failure must not remove the existing
        # high-recall analytic features or silently fabricate an orientation.
        occt_obb = None

    # Classify every face
    result = {
        'planes': [],
        'cylinders': [],
        'cones': [],
        'torii': [],
        'spheres': [],
        'surfaces_of_revolution': [],
        'surfaces_of_extrusion': [],
        'bspline_surfaces': [],
        'bezier_surfaces': [],
        'other_surfaces': [],
        '_bbox': occt_bbox,
        '_obb': occt_obb,
        '_topology': topology,
    }

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        adaptor = BRepAdaptor_Surface(face)
        stype = adaptor.GetType()

        # Compute face area
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()

        if stype == GeomAbs_Plane:
            pln = adaptor.Plane()
            normal = pln.Axis().Direction()
            if face.Orientation() == TopAbs_REVERSED:
                normal.Reverse()
            origin = pln.Location()
            centroid = props.CentreOfMass()
            plane_row = {
                'normal': _snap_direction((normal.X(), normal.Y(), normal.Z())),
                'position': _snap_point((origin.X(), origin.Y(), origin.Z())),
                'centroid': list(_snap_point((
                    centroid.X(), centroid.Y(), centroid.Z()
                ))),
                'area': area,
                'surface_orientation': (
                    'reversed'
                    if face.Orientation() == TopAbs_REVERSED
                    else 'forward'
                ),
            }
            # A planar face area does not determine its aspect ratio.  Preserve
            # the actual trimmed UV footprint so downstream interface recall
            # can compare thin-part dimensions without inventing a square.
            # UV bounds are constant-time and reuse this traversal; unlike a
            # second vertex walk over a very large STEP they add negligible
            # overhead.  Older/degenerate bindings simply keep the legacy row.
            try:
                u_min, u_max, v_min, v_max = breptools.UVBounds(face)
                footprint_dimensions = [
                    abs(float(u_max) - float(u_min)),
                    abs(float(v_max) - float(v_min)),
                ]
                x_direction = pln.XAxis().Direction()
                y_direction = pln.YAxis().Direction()
                if (
                    all(math.isfinite(value) for value in footprint_dimensions)
                    and min(footprint_dimensions) > 1e-9
                ):
                    plane_row['footprint_axes'] = [
                        list(_snap_direction((
                            x_direction.X(),
                            x_direction.Y(),
                            x_direction.Z(),
                        ))),
                        list(_snap_direction((
                            y_direction.X(),
                            y_direction.Y(),
                            y_direction.Z(),
                        ))),
                    ]
                    plane_row['footprint_dimensions'] = footprint_dimensions
            except Exception:
                pass
            result['planes'].append(plane_row)

        elif stype == GeomAbs_Cylinder:
            cyl = adaptor.Cylinder()
            axis = cyl.Axis()
            origin = axis.Location()
            direction = axis.Direction()
            surface_polarity = "unknown"
            normal_radial_dot = None
            try:
                first_u = float(adaptor.FirstUParameter())
                last_u = float(adaptor.LastUParameter())
                first_v = float(adaptor.FirstVParameter())
                last_v = float(adaptor.LastVParameter())
                u = (
                    0.5 * (first_u + last_u)
                    if math.isfinite(first_u) and math.isfinite(last_u)
                    else 0.0
                )
                v = (
                    0.5 * (first_v + last_v)
                    if math.isfinite(first_v) and math.isfinite(last_v)
                    else 0.0
                )
                point = adaptor.Value(u, v)
                properties = BRepLProp_SLProps(adaptor, u, v, 1, 1e-7)
                normal = properties.Normal()
                if face.Orientation() == TopAbs_REVERSED:
                    normal.Reverse()

                offset = (
                    point.X() - origin.X(),
                    point.Y() - origin.Y(),
                    point.Z() - origin.Z(),
                )
                axial = (
                    offset[0] * direction.X()
                    + offset[1] * direction.Y()
                    + offset[2] * direction.Z()
                )
                radial = (
                    offset[0] - axial * direction.X(),
                    offset[1] - axial * direction.Y(),
                    offset[2] - axial * direction.Z(),
                )
                radial_norm = math.sqrt(sum(value * value for value in radial))
                if radial_norm > 1e-9:
                    normal_radial_dot = (
                        normal.X() * radial[0]
                        + normal.Y() * radial[1]
                        + normal.Z() * radial[2]
                    ) / radial_norm
                    surface_polarity = (
                        "convex" if normal_radial_dot > 0.0 else "concave"
                    )
            except Exception:
                # Polarity is useful but optional. A STEP surface with an
                # undefined normal must not make the entire feature extraction
                # fail or silently remove a high-recall candidate.
                pass
            result['cylinders'].append({
                'radius': cyl.Radius(),
                'origin': list(_snap_point((origin.X(), origin.Y(), origin.Z()))),
                'axis': list(_snap_direction((direction.X(), direction.Y(), direction.Z()))),
                'area': area,
                'surface_polarity': surface_polarity,
                'normal_radial_dot': normal_radial_dot,
            })

        elif stype == GeomAbs_Cone:
            cone = adaptor.Cone()
            axis = cone.Axis()
            origin = axis.Location()
            direction = axis.Direction()
            result['cones'].append({
                'radius': cone.RefRadius(),
                'semi_angle_deg': math.degrees(cone.SemiAngle()),
                'origin': list(_snap_point((origin.X(), origin.Y(), origin.Z()))),
                'axis': list(_snap_direction((direction.X(), direction.Y(), direction.Z()))),
                'area': area,
            })

        elif stype == GeomAbs_Sphere:
            sphere = adaptor.Sphere()
            center = sphere.Location()
            result['spheres'].append({
                'radius': sphere.Radius(),
                'center': list(_snap_point((center.X(), center.Y(), center.Z()))),
                'area': area,
            })

        elif stype == GeomAbs_Torus:
            torus = adaptor.Torus()
            axis = torus.Axis()
            origin = axis.Location()
            direction = axis.Direction()
            result['torii'].append({
                'major_radius': torus.MajorRadius(),
                'minor_radius': torus.MinorRadius(),
                'origin': list(_snap_point((origin.X(), origin.Y(), origin.Z()))),
                'axis': list(_snap_direction((direction.X(), direction.Y(), direction.Z()))),
                'area': area,
            })

        elif stype == GeomAbs_SurfaceOfRevolution:
            result['surfaces_of_revolution'].append({'area': area, 'type': 'revolution'})

        elif stype == GeomAbs_SurfaceOfExtrusion:
            result['surfaces_of_extrusion'].append({'area': area, 'type': 'extrusion'})

        elif stype == GeomAbs_BSplineSurface:
            result['bspline_surfaces'].append({'area': area, 'type': 'bspline'})

        elif stype == GeomAbs_BezierSurface:
            result['bezier_surfaces'].append({'area': area, 'type': 'bezier'})

        else:
            result['other_surfaces'].append({'area': area, 'type': 'other', 'geom_type': stype})

        exp.Next()

    return result


# ── main extraction ───────────────────────────────────────────────
def extract_features(filepath, use_occt=True):
    """
    Extract all geometric features from a STEP file.

    Strategy:
      1. Text regex for CYLINDRICAL_SURFACE (fast, no OCCT load needed)
      2. OCCT face traversal for ALL surface types (authoritative)
      3. Merge: OCCT cylinders supplement text cylinders (dedup by radius+origin+axis)
      4. OCCT planes replace text-plane detection
      5. OCCT cones, torii, spheres added as new feature types
      6. OCCT bbox preferred when available (covers tessellated files)

    Args:
        filepath: path to STEP file
        use_occt: if False, skip OCCT (text-only, faster but incomplete)

    Returns dict with keys:
        cylinders, planes, cones, torii, spheres, bbox, filepath,
        plus occt_stats for diagnostics.
    """
    # ── Fast path: text regex for cylinders + bbox ──
    text = _parse_step_text(filepath)
    pts = _extract_cartesian_points(text)
    dirs = _extract_directions(text)
    cyls_raw = _extract_cylinders(text)
    placements = _extract_axis2_placements(text)

    text_cylinders = []
    for cyl in cyls_raw:
        pid = cyl['axis_placement_id']
        if pid not in placements:
            continue
        pl = placements[pid]
        origin = pts.get(pl['cart_id'])
        axis = dirs.get(pl['axis_id'])
        if not origin or not axis:
            continue
        text_cylinders.append({
            'radius': cyl['radius'],
            'origin': list(origin),
            'axis': list(axis),
        })

    # Text-based bbox (fallback)
    xs, ys, zs = [], [], []
    for pt in pts.values():
        if len(pt) == 3:
            xs.append(pt[0]); ys.append(pt[1]); zs.append(pt[2])
    text_bbox = None
    if xs:
        text_bbox = {
            'min': (_snap_scalar(min(xs)), _snap_scalar(min(ys)), _snap_scalar(min(zs))),
            'max': (_snap_scalar(max(xs)), _snap_scalar(max(ys)), _snap_scalar(max(zs))),
        }

    # ── OCCT universal detection ──
    occt = None
    if use_occt:
        occt = _detect_all_faces_occt(filepath)

    if occt is None:
        # OCCT unavailable or failed — fall back to text-only
        return {
            'filepath': filepath,
            'cylinders': text_cylinders,
            'planes': [],
            'cones': [],
            'torii': [],
            'spheres': [],
            'bbox': text_bbox,
            'obb': None,
            'occt_stats': {'used': False},
        }

    # ── Merge cylinders: text + OCCT, dedup by (radius, snapped origin, snapped axis) ──
    def _cyl_key(c):
        return (round(c['radius'], 4),
                tuple(round(v, 4) for v in c['origin']),
                tuple(round(v, 4) for v in c['axis']))

    # The text parser preserves stable STEP entity ordering, while OCCT adds
    # authoritative face area and solid-normal polarity. Enrich matching text
    # records instead of discarding the OCCT fields as duplicates.
    occt_by_key = {}
    for cylinder in occt['cylinders']:
        occt_by_key.setdefault(_cyl_key(cylinder), []).append(cylinder)

    merged_cylinders = []
    for cylinder in text_cylinders:
        enriched = dict(cylinder)
        matches = occt_by_key.get(_cyl_key(cylinder), [])
        if matches:
            authoritative = matches.pop(0)
            for key in (
                'area',
                'surface_polarity',
                'normal_radial_dot',
            ):
                if authoritative.get(key) is not None:
                    enriched[key] = authoritative[key]
        merged_cylinders.append(enriched)

    occt_cyl_only = 0
    for remaining in occt_by_key.values():
        for c in remaining:
            merged_cylinders.append(c)
            occt_cyl_only += 1

    # Use OCCT bbox when text bbox is missing
    bbox = text_bbox if text_bbox is not None else occt['_bbox']

    result = {
        'filepath': filepath,
        'cylinders': merged_cylinders,
        'planes': occt['planes'],
        'cones': occt['cones'],
        'torii': occt['torii'],
        'spheres': occt['spheres'],
        'bbox': bbox,
        'obb': occt.get('_obb'),
        'topology': occt.get('_topology'),
        'occt_stats': {
            'used': True,
            'text_cylinders': len(text_cylinders),
            'occt_cylinders': len(occt['cylinders']),
            'occt_cyl_only': occt_cyl_only,
            'occt_planes': len(occt['planes']),
            'occt_cones': len(occt['cones']),
            'occt_torii': len(occt['torii']),
            'occt_spheres': len(occt['spheres']),
            'occt_revolution': len(occt['surfaces_of_revolution']),
            'occt_extrusion': len(occt['surfaces_of_extrusion']),
            'occt_bspline': len(occt['bspline_surfaces']),
            'occt_bezier': len(occt['bezier_surfaces']),
            'occt_other': len(occt['other_surfaces']),
            'occt_obb_available': occt.get('_obb') is not None,
        },
    }
    return result


def get_main_axis(features):
    """Return the axis direction of the largest-radius cylinder, or Z if none."""
    if not features.get('cylinders'):
        return (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)
    main = max(features['cylinders'], key=lambda c: c['radius'])
    return tuple(main['axis']), tuple(main['origin'])


# ── CLI test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, json
    fp = sys.argv[1] if len(sys.argv) > 1 else '2/flange_a_pipe_X_fan15deg.step'
    feats = extract_features(fp)
    out = {k: v for k, v in feats.items() if k not in ('filepath', 'occt_stats')}
    print(f"File: {fp}")
    print(f"  Cylinders: {len(feats['cylinders'])}")
    for c in sorted(feats['cylinders'], key=lambda c: c['radius']):
        print(f"    r={c['radius']:6.1f}  axis=({c['axis'][0]:.3f},{c['axis'][1]:.3f},{c['axis'][2]:.3f})  origin=({c['origin'][0]:.1f},{c['origin'][1]:.1f},{c['origin'][2]:.1f})")
    for label, key in [('Planes', 'planes'), ('Cones', 'cones'), ('Torii', 'torii'), ('Spheres', 'spheres')]:
        items = feats.get(key, [])
        if items:
            print(f"  {label}: {len(items)}")
            for item in items[:3]:
                if key == 'planes':
                    print(f"    n=({item['normal'][0]:.3f},{item['normal'][1]:.3f},{item['normal'][2]:.3f}) pos=({item['position'][0]:.1f},{item['position'][1]:.1f},{item['position'][2]:.1f}) area={item['area']:.0f}")
                elif key == 'cones':
                    print(f"    r={item['radius']:.1f} semi_angle={item['semi_angle_deg']:.1f}deg area={item['area']:.0f}")
                elif key == 'torii':
                    print(f"    R={item['major_radius']:.1f} r={item['minor_radius']:.1f} area={item['area']:.0f}")
                elif key == 'spheres':
                    print(f"    r={item['radius']:.1f} area={item['area']:.0f}")
            if len(items) > 3:
                print(f"    ... and {len(items)-3} more")
    if feats.get('occt_stats'):
        s = feats['occt_stats']
        print(f"  OCCT stats: txt-cyl={s['text_cylinders']} occ-cyl={s['occt_cylinders']} "
              f"cyl-only-occt={s['occt_cyl_only']} planes={s['occt_planes']} "
              f"cones={s['occt_cones']} torii={s['occt_torii']} spheres={s['occt_spheres']}")
    if feats.get('bbox'):
        b = feats['bbox']
        dx = b['max'][0] - b['min'][0]
        dy = b['max'][1] - b['min'][1]
        dz = b['max'][2] - b['min'][2]
        print(f"  BBox: ({b['min'][0]:.1f},{b['min'][1]:.1f},{b['min'][2]:.1f}) → ({b['max'][0]:.1f},{b['max'][1]:.1f},{b['max'][2]:.1f})  size=({dx:.0f},{dy:.0f},{dz:.0f})mm")
