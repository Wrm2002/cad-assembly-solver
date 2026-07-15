"""Geometry-only OBB orientation proposals for insertable CAD components.

The module enumerates proper signed axis permutations.  OBB dimensions only
order the bounded frontier; they never establish contact, collision freedom,
functional validity, or an automatic acceptance decision.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


def _obb_arrays(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axes = np.asarray(obb.get("axes"), dtype=float)
    dimensions = np.asarray(obb.get("dimensions"), dtype=float)
    if axes.shape != (3, 3) or dimensions.shape != (3,):
        raise ValueError("OBB requires three axes and three dimensions")
    if not np.all(np.isfinite(axes)) or not np.all(np.isfinite(dimensions)):
        raise ValueError("OBB contains non-finite values")
    if np.any(dimensions <= 0.0):
        raise ValueError("OBB dimensions must be positive")
    axes = np.asarray([row / np.linalg.norm(row) for row in axes], dtype=float)
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5):
        raise ValueError("OBB axes must be orthonormal")
    return axes, dimensions


def _matrix_axis_angle(matrix: np.ndarray) -> list[float] | None:
    cosine = max(-1.0, min(1.0, (float(np.trace(matrix)) - 1.0) * 0.5))
    angle = math.acos(cosine)
    if angle <= 1e-9:
        return None
    if abs(math.pi - angle) <= 1e-6:
        diagonal = np.maximum(0.0, (np.diag(matrix) + 1.0) * 0.5)
        axis = np.sqrt(diagonal)
        if axis[0] > 1e-6:
            axis[1] = math.copysign(axis[1], matrix[0, 1] + matrix[1, 0])
            axis[2] = math.copysign(axis[2], matrix[0, 2] + matrix[2, 0])
        elif axis[1] > 1e-6:
            axis[2] = math.copysign(axis[2], matrix[1, 2] + matrix[2, 1])
    else:
        axis = np.asarray([
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]) / (2.0 * math.sin(angle))
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        return None
    axis /= norm
    return [float(axis[0]), float(axis[1]), float(axis[2]), math.degrees(angle)]


def enumerate_axis_role_frames(
    fixed_obb: dict[str, Any],
    moving_obb: dict[str, Any],
    *,
    maximum: int = 24,
) -> list[dict[str, Any]]:
    """Enumerate a diverse, bounded frontier of proper OBB rotations.

    ``axis_mapping[i]`` is the fixed OBB axis receiving moving OBB axis ``i``.
    Every returned matrix maps source coordinates of the moving part into the
    fixed part's coordinate frame.
    """

    fixed_axes, fixed_dimensions = _obb_arrays(fixed_obb)
    moving_axes, moving_dimensions = _obb_arrays(moving_obb)
    fixed_basis = fixed_axes.T
    moving_basis = moving_axes.T
    fixed_smallest = int(np.argmin(fixed_dimensions))
    moving_longest = int(np.argmax(moving_dimensions))
    fixed_longest = int(np.argmax(fixed_dimensions))
    rows = []
    for mapping in itertools.permutations(range(3)):
        moving_for_fixed_smallest = mapping.index(fixed_smallest)
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            target_basis = np.zeros((3, 3), dtype=float)
            for moving_axis, fixed_axis in enumerate(mapping):
                target_basis[:, moving_axis] = (
                    signs[moving_axis] * fixed_basis[:, fixed_axis]
                )
            rotation = target_basis @ moving_basis.T
            determinant = float(np.linalg.det(rotation))
            if determinant < 1.0 - 1e-6:
                continue
            normal_ratio = (
                moving_dimensions[moving_for_fixed_smallest]
                / fixed_dimensions[fixed_smallest]
            )
            normal_compatibility = math.exp(-abs(math.log(normal_ratio)))
            long_axis_support = 1.0 if mapping[moving_longest] == fixed_longest else 0.0
            ordered_support = sum(
                (
                    moving_dimensions[left] - moving_dimensions[right]
                ) * (
                    fixed_dimensions[mapping[left]]
                    - fixed_dimensions[mapping[right]]
                ) >= 0.0
                for left, right in itertools.combinations(range(3), 2)
            ) / 3.0
            score = (
                0.50 * normal_compatibility
                + 0.30 * long_axis_support
                + 0.20 * ordered_support
            )
            rows.append({
                "axis_mapping": list(mapping),
                "axis_signs": [int(value) for value in signs],
                "rotation_matrix": rotation.tolist(),
                "rotation_axis_angle": _matrix_axis_angle(rotation),
                "determinant": determinant,
                "fixed_smallest_axis": fixed_smallest,
                "moving_axis_for_fixed_smallest": moving_for_fixed_smallest,
                "moving_longest_axis": moving_longest,
                "moving_longest_target_axis": mapping[moving_longest],
                "dimension_order_score": round(float(score), 9),
                "proposal_only": True,
                "review_required": True,
                "can_auto_accept": False,
            })
    rows.sort(key=lambda row: (
        -float(row["dimension_order_score"]),
        row["axis_mapping"],
        row["axis_signs"],
    ))

    # Before repeated sign variants consume the budget, preserve at least one
    # candidate for every possible moving axis assigned to the carrier's thin
    # or height axis. Interface evidence decides which role is physically real.
    selected = []
    covered = set()
    for row in rows:
        role = int(row["moving_axis_for_fixed_smallest"])
        if role not in covered:
            selected.append(row)
            covered.add(role)
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= max(1, int(maximum)):
            break
    return selected[: max(1, int(maximum))]

