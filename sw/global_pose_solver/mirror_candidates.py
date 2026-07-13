"""Mirror candidate generator: create opposite-end variants of pairwise placements.

For parts connected via cylindrical joints (shaft-hub), the simplex search
often converges to the nearest shaft end.  This module generates mirrored
variants by reflecting across the joint axis midpoint to explore both ends.

Mirroring is done by:
  1. Extracting the joint axis from the pair's B-Rep graph
  2. Finding the two end faces (planes perpendicular to the axis)
  3. Reflecting any candidate across the midpoint plane of the axis
  4. Flipping the rotation 180° around the axis to maintain contact orientation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def get_axis_from_graph(graph_path: str | Path) -> dict[str, Any] | None:
    """Extract main cylinder axis and end faces from a B-Rep graph JSON.

    Returns dict with:
      axis_origin, axis_direction, end_faces (list of plane centroids),
      minus_end, plus_end (centroids of the two extreme perpendicular planes)
    """
    with open(graph_path) as f:
        g = json.load(f)

    # Find the largest cylinder
    cylinders = []
    for n in g["nodes"]:
        if (n.get("surface_type") in ("cylinder", "CylinderSurfaceType")
                and n.get("axis_direction") and n.get("axis_origin")):
            r = n.get("radius", 0) or 0
            cylinders.append((r, n))

    if not cylinders:
        return None

    # Use largest-radius cylinder (main shaft/hub surface)
    _, main_cyl = max(cylinders, key=lambda x: x[0])
    axis_origin = np.array(main_cyl["axis_origin"], dtype=float)
    axis_dir = np.array(main_cyl["axis_direction"], dtype=float)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    # Find planes perpendicular to this axis (end faces)
    perp_planes = []
    for n in g["nodes"]:
        if (n.get("surface_type") in ("plane", "PlaneSurfaceType")
                and n.get("normal") and n.get("centroid")):
            normal = np.array(n["normal"], dtype=float)
            normal = normal / np.linalg.norm(normal)
            dot = abs(np.dot(normal, axis_dir))
            if dot > 0.95:  # nearly parallel = perpendicular face to axis
                centroid = np.array(n["centroid"], dtype=float)
                proj = np.dot(centroid - axis_origin, axis_dir)
                perp_planes.append((proj, centroid))

    if len(perp_planes) < 2:
        return None

    perp_planes.sort(key=lambda x: x[0])
    minus_proj, minus_centroid = perp_planes[0]
    plus_proj, plus_centroid = perp_planes[-1]

    return {
        "axis_origin": axis_origin.tolist(),
        "axis_direction": axis_dir.tolist(),
        "minus_end_centroid": minus_centroid.tolist(),
        "plus_end_centroid": plus_centroid.tolist(),
        "minus_proj": float(minus_proj),
        "plus_proj": float(plus_proj),
        "midpoint_proj": float((minus_proj + plus_proj) / 2),
    }


def mirror_placement(placement: dict, axis_info: dict) -> dict | None:
    """Create a mirrored version of a placement on the opposite shaft end.

    Given a placement of part B relative to part A (shaft), reflect it across
    the shaft midpoint and rotate 180° around the axis.

    Args:
        placement: dict with 'translate' and 'rotate_sequence'.
        axis_info: dict from get_axis_from_graph.

    Returns:
        New placement dict (mirrored), or None if mirroring doesn't help.
    """
    axis_origin = np.array(axis_info["axis_origin"])
    axis_dir = np.array(axis_info["axis_direction"])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    minus = np.array(axis_info["minus_end_centroid"])
    plus = np.array(axis_info["plus_end_centroid"])

    # Current position
    T = np.eye(4)
    for rs in reversed(placement.get("rotate_sequence", [])):
        aa = rs["axis_angle"]
        R = Rotation.from_rotvec(np.array(aa[:3]) * aa[3] * np.pi / 180).as_matrix()
        T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = np.array(placement.get("translate", [0, 0, 0]))

    current_pos = T[:3, 3]
    current_proj = np.dot(current_pos - axis_origin, axis_dir)
    midpoint = (np.dot(minus - axis_origin, axis_dir) + np.dot(plus - axis_origin, axis_dir)) / 2

    # If already on the plus side, no need to mirror
    if current_proj > midpoint:
        return None

    # Reflect: move to opposite end
    # The distance from current to minus end, mirrored to plus end
    dist_from_minus = current_proj - np.dot(minus - axis_origin, axis_dir)
    target_proj = np.dot(plus - axis_origin, axis_dir) - dist_from_minus
    shift = (target_proj - current_proj) * axis_dir

    T_new = T.copy()
    T_new[:3, 3] = T[:3, 3] + shift

    # Rotate 180° around axis so contact face still mates
    R_flip = Rotation.from_rotvec(axis_dir * np.pi).as_matrix()
    T_new[:3, :3] = R_flip @ T_new[:3, :3]

    # Convert back to placement dict format
    rotvec = Rotation.from_matrix(T_new[:3, :3]).as_rotvec()
    angle_deg = float(np.linalg.norm(rotvec))
    axis_unit = (rotvec / angle_deg).tolist() if angle_deg > 1e-12 else [0, 0, 1]
    return {
        "translate": T_new[:3, 3].tolist(),
        "rotate_sequence": [{"axis_angle": axis_unit + [angle_deg * 180 / np.pi]}],
    }
