"""
analyze_geometry.py — Bolt hole pattern detection and bore keyway detection.
Used by refinement.BoltKeywayAlignStrategy to compute rotational alignment
between coaxial flange pairs.
"""
import math
import os


def detect_bolt_holes(cylinders, main_bore):
    """
    Identify bolt holes from a list of cylinder dicts.
    
    Bolt holes are:
      - Radius between 2mm and main_bore.radius * 0.8
      - Axis parallel to the main bore axis (dot > 0.99)
      - At a meaningful radial distance from the bore axis (> 2mm)
    
    Returns:
      dict with 'angles' key: list of angular positions (radians) of
      bolt holes around the bore axis, sorted. Or empty dict if
      fewer than 3 bolt holes found.
    """
    if not cylinders or not main_bore:
        return {}
    
    main_r = main_bore.get('radius', 0)
    if main_r <= 0:
        return {}
    
    bore_axis = main_bore.get('axis', [0, 0, 1])
    bore_mag = math.sqrt(sum(x * x for x in bore_axis))
    if bore_mag < 1e-12:
        return {}
    bore_dir = [x / bore_mag for x in bore_axis]
    
    # Build reference frame for computing angles around bore axis
    # Pick two perpendicular directions u, v orthogonal to bore_dir
    if abs(bore_dir[0]) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)
    dot_ref = sum(ref[i] * bore_dir[i] for i in range(3))
    u = [ref[i] - dot_ref * bore_dir[i] for i in range(3)]
    u_mag = math.sqrt(sum(x * x for x in u))
    if u_mag < 1e-12:
        return {}
    u = [x / u_mag for x in u]
    v = [
        bore_dir[1] * u[2] - bore_dir[2] * u[1],
        bore_dir[2] * u[0] - bore_dir[0] * u[2],
        bore_dir[0] * u[1] - bore_dir[1] * u[0],
    ]
    
    positions = []
    for cyl in cylinders:
        r = cyl.get('radius', 0)
        # Skip the main bore itself and tiny/noise cylinders
        if r < 2.0 or r > main_r * 0.8:
            continue
        
        cyl_axis = cyl.get('axis', [0, 0, 1])
        cyl_mag = math.sqrt(sum(x * x for x in cyl_axis))
        if cyl_mag < 1e-12:
            continue
        cyl_dir = [x / cyl_mag for x in cyl_axis]
        
        # Must be parallel to bore axis
        dot = abs(sum(a * b for a, b in zip(bore_dir, cyl_dir)))
        if dot < 0.99:
            continue
        
        origin = cyl.get('origin')
        if not origin:
            continue
        
        # Project origin onto plane perpendicular to bore axis
        axial = sum(origin[i] * bore_dir[i] for i in range(3))
        radial_vec = [origin[i] - axial * bore_dir[i] for i in range(3)]
        radial_dist = math.sqrt(sum(x * x for x in radial_vec))
        
        if radial_dist < 2.0:
            continue  # on the axis, not a bolt hole
        
        positions.append(tuple(origin))
    
    if len(positions) < 3:
        return {}
    
    # Compute angles around the bore axis
    angles = []
    for pos in positions:
        radial_vec = [pos[i] - sum(pos[j] * bore_dir[j] for j in range(3)) * bore_dir[i] for i in range(3)]
        rmag = math.sqrt(sum(x * x for x in radial_vec))
        if rmag < 1e-12:
            continue
        rv = [x / rmag for x in radial_vec]
        # Project onto u,v frame
        x_u = sum(rv[i] * u[i] for i in range(3))
        x_v = sum(rv[i] * v[i] for i in range(3))
        angle = math.atan2(x_v, x_u)
        angles.append(angle)
    
    angles.sort()
    return {'angles': angles}


def detect_bore_keyway(filepath, main_bore):
    """
    Detect keyway features in a bore.
    
    Uses plane faces that are parallel to the bore axis and positioned
    near the bore surface to identify keyway walls.
    
    Returns:
      List of angular positions (radians) of keyway corners,
      or None if no keyway detected.
    """
    if not filepath or not os.path.exists(filepath):
        return None
    if not main_bore:
        return None
    
    # Try to load features if not already available
    # The keyway is detected from plane faces close to the bore surface
    try:
        from features import extract_features
        feats = extract_features(filepath)
    except Exception:
        return None
    
    planes = feats.get('planes', [])
    if not planes:
        return None
    
    bore_axis = main_bore.get('axis', [0, 0, 1])
    bore_mag = math.sqrt(sum(x * x for x in bore_axis))
    if bore_mag < 1e-12:
        return None
    bore_dir = [x / bore_mag for x in bore_axis]
    bore_r = main_bore.get('radius', 0)
    bore_origin = main_bore.get('origin', [0, 0, 0])
    
    # Look for small planes that are:
    #   - Parallel to bore axis (normal perpendicular to bore_dir)
    #   - Close to the bore cylindrical surface (distance from axis ≈ bore_r)
    #   - Small area (keyway walls are narrow)
    keyway_angles = []
    
    for p in planes:
        normal = p.get('normal', [0, 0, 1])
        area = p.get('area', 0)
        # Keyway walls are small
        if area > 500 or area < 5:
            continue
        
        # Normal should be perpendicular to bore axis (parallel face)
        dot_axis = abs(sum(normal[i] * bore_dir[i] for i in range(3)))
        if dot_axis > 0.1:
            continue  # not parallel to bore axis
        
        pos = p.get('position', [0, 0, 0])
        # Distance from bore axis
        op = [pos[i] - bore_origin[i] for i in range(3)]
        axial_proj = sum(op[i] * bore_dir[i] for i in range(3))
        radial_vec = [op[i] - axial_proj * bore_dir[i] for i in range(3)]
        radial_dist = math.sqrt(sum(x * x for x in radial_vec))
        
        # Keyway walls are at or slightly beyond the bore radius
        if radial_dist < bore_r * 0.6 or radial_dist > bore_r * 1.3:
            continue
        
        # Compute angular position
        rmag = math.sqrt(sum(x * x for x in radial_vec))
        if rmag < 1e-12:
            continue
        
        # Build reference frame
        if abs(bore_dir[0]) < 0.9:
            ref = (1.0, 0.0, 0.0)
        else:
            ref = (0.0, 1.0, 0.0)
        dot_ref = sum(ref[i] * bore_dir[i] for i in range(3))
        u = [ref[i] - dot_ref * bore_dir[i] for i in range(3)]
        u_mag = math.sqrt(sum(x * x for x in u))
        if u_mag < 1e-12:
            continue
        u = [x / u_mag for x in u]
        v = [
            bore_dir[1] * u[2] - bore_dir[2] * u[1],
            bore_dir[2] * u[0] - bore_dir[0] * u[2],
            bore_dir[0] * u[1] - bore_dir[1] * u[0],
        ]
        
        rv = [radial_vec[i] / rmag for i in range(3)]
        x_u = sum(rv[i] * u[i] for i in range(3))
        x_v = sum(rv[i] * v[i] for i in range(3))
        angle = math.atan2(x_v, x_u)
        keyway_angles.append(angle)
    
    if len(keyway_angles) < 2:
        return None
    
    return sorted(keyway_angles)
