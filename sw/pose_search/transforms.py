"""Rigid-transform helpers shared by JoinABLe-style pose search."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def unit(vector: np.ndarray | list[float]) -> np.ndarray:
    result = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(result))
    if length <= 1e-12:
        raise ValueError("joint axis direction must be non-zero")
    return result / length


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a proper 3x3 rotation mapping *source* onto *target*."""

    source = unit(source)
    target = unit(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-12:
        return np.eye(3)
    if dot <= -1.0 + 1e-12:
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(source[0])) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(source, seed))
        return Rotation.from_rotvec(math.pi * axis).as_matrix()
    axis = unit(np.cross(source, target))
    return Rotation.from_rotvec(math.acos(dot) * axis).as_matrix()


def axis_alignment_matrix(
    moving_origin: np.ndarray | list[float],
    moving_direction: np.ndarray | list[float],
    fixed_origin: np.ndarray | list[float],
    fixed_direction: np.ndarray | list[float],
) -> np.ndarray:
    """Align a moving joint axis with a fixed joint axis.

    The moving axis origin is mapped exactly to the fixed axis origin.  The
    returned matrix is a proper rigid transform (determinant +1).
    """

    moving_origin = np.asarray(moving_origin, dtype=float)
    fixed_origin = np.asarray(fixed_origin, dtype=float)
    rotation = rotation_from_vectors(
        np.asarray(moving_direction, dtype=float),
        np.asarray(fixed_direction, dtype=float),
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = fixed_origin - rotation @ moving_origin
    return result


def rotation_about_axis_matrix(
    origin: np.ndarray | list[float],
    direction: np.ndarray | list[float],
    degrees: float,
) -> np.ndarray:
    origin = np.asarray(origin, dtype=float)
    rotation = Rotation.from_rotvec(
        math.radians(float(degrees)) * unit(direction)
    ).as_matrix()
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = origin - rotation @ origin
    return result


def translation_along_axis_matrix(
    direction: np.ndarray | list[float], offset: float
) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = unit(direction) * float(offset)
    return result


def joint_parameter_matrix(
    moving_origin: np.ndarray | list[float],
    moving_direction: np.ndarray | list[float],
    fixed_origin: np.ndarray | list[float],
    fixed_direction: np.ndarray | list[float],
    *,
    offset: float,
    rotation_degrees: float,
    axis_flip: bool,
) -> np.ndarray:
    """Materialize JoinABLe's axis/offset/rotation/flip parameterization.

    The released JoinABLe code implements ``flip`` as a reflection, whose
    determinant is -1.  A reflected CAD solid is not a physically valid rigid
    pose.  Production search therefore resolves the same axis-sign ambiguity
    by aligning to the opposite fixed direction; every emitted transform has
    determinant +1.
    """

    fixed_direction = unit(fixed_direction)
    target_direction = -fixed_direction if axis_flip else fixed_direction
    align = axis_alignment_matrix(
        moving_origin,
        moving_direction,
        fixed_origin,
        target_direction,
    )
    rotate = rotation_about_axis_matrix(
        fixed_origin, target_direction, rotation_degrees
    )
    translate = translation_along_axis_matrix(target_direction, offset)
    return translate @ rotate @ align


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.ones((len(points), 4), dtype=float)
    homogeneous[:, :3] = points
    return (np.asarray(matrix, dtype=float) @ homogeneous.T).T[:, :3]


def matrix_to_placement(matrix: np.ndarray) -> dict[str, Any]:
    """Convert a rigid 4x4 matrix to the project's placement schema."""

    matrix = np.asarray(matrix, dtype=float)
    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(matrix).all() or abs(determinant - 1.0) > 1e-5:
        raise ValueError("matrix is not a proper finite rigid transform")
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    placement: dict[str, Any] = {
        "translate": [float(value) for value in matrix[:3, 3]],
    }
    if angle > 1e-10:
        axis = rotvec / angle
        placement["rotate_sequence"] = [{
            "axis_angle": [
                float(axis[0]),
                float(axis[1]),
                float(axis[2]),
                math.degrees(angle),
            ]
        }]
    return placement


def placement_to_matrix(placement: dict[str, Any]) -> np.ndarray:
    """Convert the project's rotate-then-translate placement to a matrix."""

    rotation = np.eye(3)
    specs = list(placement.get("rotate_sequence") or [])
    if not specs and placement.get("rotate_axis_angle"):
        specs = [{"axis_angle": placement["rotate_axis_angle"]}]
    for spec in specs:
        if "axis_angle" in spec:
            values = spec["axis_angle"]
            current = Rotation.from_rotvec(
                math.radians(float(values[3])) * unit(values[:3])
            ).as_matrix()
        elif "axis_to" in spec:
            values = spec["axis_to"]
            current = rotation_from_vectors(
                np.asarray(values["from"], dtype=float),
                np.asarray(values.get("to", [0.0, 0.0, 1.0]), dtype=float),
            )
        else:
            continue
        rotation = current @ rotation
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(
        placement.get("translate", [0.0, 0.0, 0.0]), dtype=float
    )
    return result
