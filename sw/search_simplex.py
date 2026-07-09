"""
search_simplex.py — Nelder-Mead pose optimization inspired by JoinABLe.

Given a predicted joint axis (origin, direction) for a pair of parts,
optimize [offset, rotation_deg, flip] to minimize overlap and maximize
surface contact.  Uses STL mesh sampling + nearest-face distance as a
lightweight SDF proxy — no trimesh required.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, Bounds


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _rotation_matrix_around_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotating around *axis* by *angle_deg*."""
    a = _norm(np.asarray(axis, dtype=float))
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = a
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
    ])


def _flip_matrix(origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Reflection matrix across a plane at *origin* normal to *direction*."""
    d = _norm(np.asarray(direction, dtype=float))
    o = np.asarray(origin, dtype=float)
    R = np.eye(3) - 2.0 * np.outer(d, d)
    t = 2.0 * np.dot(o, d) * d
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _transform_mesh(
    vertices: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    offset: float,
    rotation_deg: float,
    flip: bool,
) -> np.ndarray:
    """Apply joint parameters to mesh vertices.
    
    Transform chain: align → rotate → offset → (optional flip)
    Returns transformed vertices.
    """
    d = _norm(np.asarray(direction, dtype=float))
    o = np.asarray(origin, dtype=float)
    
    # Rotation around joint axis
    R = _rotation_matrix_around_axis(d, rotation_deg)
    
    # Offset along axis
    T = offset * d
    
    # Apply to vertices (N×3)
    v = vertices.copy()
    
    # 1. Rotate around axis
    # Center at origin for rotation
    v_centered = v - o
    v_rotated = v_centered @ R.T
    v = v_rotated + o
    
    # 2. Translate along axis
    v = v + T
    
    # 3. Flip if needed
    if flip:
        M = _flip_matrix(o, d)
        v_homo = np.hstack([v, np.ones((len(v), 1))])
        v = (v_homo @ M.T)[:, :3]
    
    return v


def _mesh_distance_kdtree(
    points: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
) -> np.ndarray:
    """Fast approximate signed distance using KD-tree of face centers.
    
    Returns signed distances.  Positive = outside, negative = inside.
    """
    from scipy.spatial import cKDTree
    
    face_centers = np.mean(mesh_vertices[mesh_faces], axis=1)
    face_normals = np.cross(
        mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]],
        mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 0]],
    )
    n_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
    n_len[n_len < 1e-12] = 1.0
    face_normals = face_normals / n_len
    
    tree = cKDTree(face_centers)
    k = min(3, len(face_centers))
    dists, idxs = tree.query(points, k=k)
    
    if k == 1:
        idxs = idxs[:, None]
    
    result = np.full(len(points), np.inf)
    for i in range(len(points)):
        best_dist = np.inf
        for j in range(k):
            fi = idxs[i, j]
            signed = np.dot(points[i] - face_centers[fi], face_normals[fi])
            if abs(signed) < abs(best_dist):
                best_dist = signed
        result[i] = best_dist
    
    return result


def load_stl_mesh(stl_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load STL file, return (vertices, faces)."""
    from stl import mesh as stl_mesh
    
    m = stl_mesh.Mesh.from_file(str(stl_path))
    # STL stores triangles as (N, 3, 3) → extract unique vertices and faces
    vectors = m.vectors  # (N, 3, 3)
    
    # Flatten to get all vertices
    all_verts = vectors.reshape(-1, 3)
    
    # Remove duplicate vertices (simple approach: round and unique)
    rounded = np.round(all_verts, decimals=4)
    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    vertices = all_verts[unique_idx]
    faces = inverse.reshape(-1, 3)
    
    return vertices, faces


class SearchSimplex:
    """Nelder-Mead pose optimization for a part pair with a joint axis."""
    
    def __init__(
        self,
        body_one_stl: str | Path,   # Reference part (static)
        body_two_stl: str | Path,   # Target part (to be moved)
        joint_origin: list[float],   # Point on joint axis
        joint_direction: list[float],  # Joint axis direction
        num_surface_samples: int = 2000,
        budget: int = 80,
        contact_weight: float = 10.0,
        overlap_weight: float = 1.0,
    ):
        self.verts1, self.faces1 = load_stl_mesh(body_one_stl)
        self.verts2, self.faces2 = load_stl_mesh(body_two_stl)
        self.origin = np.asarray(joint_origin, dtype=float)
        self.direction = _norm(np.asarray(joint_direction, dtype=float))
        self.num_samples = num_surface_samples
        self.budget = budget
        self.contact_weight = contact_weight
        self.overlap_weight = overlap_weight
        
        # Sample surface points on body 1 (reference)
        try:
            self._sample_surface_points()
        except Exception:
            self.sample_points = self.verts1[:min(len(self.verts1), self.num_samples)]
        
        # Compute offset limit from bounding boxes
        try:
            self._compute_offset_limit()
        except Exception:
            self.offset_limit = 100.0
        
        # Final safety: ensure critical attributes exist
        if not hasattr(self, 'offset_limit'):
            self.offset_limit = 100.0
        if not hasattr(self, 'sample_points'):
            self.sample_points = self.verts1[:min(len(self.verts1), 500)]
    
    def _sample_surface_points(self):
        """Randomly sample points on the surface of body 1."""
        n_faces = len(self.faces1)
        # Random face indices
        face_idx = np.random.randint(0, n_faces, self.num_samples)
        # Random barycentric coordinates
        r1 = np.random.random(self.num_samples)
        r2 = np.random.random(self.num_samples)
        # Ensure r1 + r2 <= 1
        mask = r1 + r2 > 1.0
        r1[mask] = 1.0 - r1[mask]
        r2[mask] = 1.0 - r2[mask]
        r3 = 1.0 - r1 - r2
        
        # Get triangle vertices
        v0 = self.verts1[self.faces1[face_idx, 0]]
        v1 = self.verts1[self.faces1[face_idx, 1]]
        v2 = self.verts1[self.faces1[face_idx, 2]]
        
        # Interpolate
        self.sample_points = (
            r1[:, None] * v0 +
            r2[:, None] * v1 +
            r3[:, None] * v2
        )
    
    def _compute_offset_limit(self):
        """Compute search range from bounding box extents along joint axis."""
        # Project bbox corners onto joint axis
        def _bbox_range(verts):
            proj = verts @ self.direction
            return proj.min(), proj.max()
        
        r1_min, r1_max = _bbox_range(self.verts1)
        r2_min, r2_max = _bbox_range(self.verts2)
        
        len1 = abs(r1_max - r1_min)
        len2 = abs(r2_max - r2_min)
        self._kd_cache: dict | None = None
    
    def cost_function(self, x: np.ndarray) -> float:
        offset = x[0] * self.offset_limit * 1500.0
        rotation = x[1] * 1500.0 if len(x) >= 2 else 0.0
        flip = bool(x[2] > 0.5) if len(x) >= 3 else False
        
        transformed = _transform_mesh(
            self.verts2, self.origin, self.direction,
            offset, rotation, flip,
        )
        
        distances = _mesh_distance_kdtree(
            self.sample_points, transformed, self.faces2,
        )
        
        overlap_mask = distances < -0.5
        overlap_ratio = overlap_mask.sum() / self.num_samples
        contact_mask = (np.abs(distances) < 1.0) & (~overlap_mask)
        contact_ratio = contact_mask.sum() / self.num_samples
        
        # Mean absolute distance (gap penalty when no contact)
        mean_gap = float(np.mean(np.abs(distances)))
        
        if overlap_ratio > 0.01:
            cost = self.overlap_weight * overlap_ratio + 0.1 * mean_gap
        elif contact_ratio > 0.001:
            cost = -self.contact_weight * contact_ratio
        else:
            # No overlap, no contact → penalize distance
            cost = 0.5 + 0.01 * mean_gap
        
        return float(cost)
    
    def search(self) -> dict[str, Any]:
        best_result = {'evaluation': float('inf'), 'offset': 0.0, 'rotation_deg': 0.0, 'flip': False}
        
        for try_flip in [False, True]:
            x0 = np.array([0.0, 0.0])
            bounds = Bounds([-1.0, 0.0], [1.0, 1.0])
            
            result = minimize(
                lambda x: self._cost_with_flip(x, try_flip),
                x0, method='Nelder-Mead', bounds=bounds,
                options={'maxiter': self.budget, 'xatol': 1e-6, 'fatol': 1e-6},
            )
            
            offset = result.x[0] * self.offset_limit * 1500.0
            rotation = result.x[1] * 1500.0 if len(result.x) >= 2 else 0.0
            
            if result.fun < best_result['evaluation']:
                transformed = _transform_mesh(
                    self.verts2, self.origin, self.direction,
                    offset, rotation, try_flip,
                )
                distances = _mesh_distance_kdtree(
                    self.sample_points, transformed, self.faces2,
                )
                overlap = (distances < -0.5).sum() / self.num_samples
                contact = (np.abs(distances) < 1.0).sum() / self.num_samples
                mean_gap = float(np.mean(np.abs(distances)))
                
                best_result = {
                    'evaluation': float(result.fun),
                    'offset': float(offset), 'rotation_deg': float(rotation),
                    'flip': try_flip, 'overlap': float(overlap),
                    'contact': float(contact), 'mean_gap': float(mean_gap),
                }
        
        return best_result
    
    def _cost_with_flip(self, x: np.ndarray, flip: bool) -> float:
        offset = x[0] * self.offset_limit * 1500.0
        rotation = x[1] * 1500.0 if len(x) >= 2 else 0.0
        
        transformed = _transform_mesh(
            self.verts2, self.origin, self.direction,
            offset, rotation, flip,
        )
        distances = _mesh_distance_kdtree(
            self.sample_points, transformed, self.faces2,
        )
        overlap_mask = distances < -0.5
        overlap_ratio = overlap_mask.sum() / self.num_samples
        contact_mask = (np.abs(distances) < 1.0) & (~overlap_mask)
        contact_ratio = contact_mask.sum() / self.num_samples
        mean_gap = float(np.mean(np.abs(distances)))
        
        if overlap_ratio > 0.01:
            return float(self.overlap_weight * overlap_ratio + 0.1 * mean_gap)
        elif contact_ratio > 0.001:
            return float(-self.contact_weight * contact_ratio)
        return float(0.5 + 0.01 * mean_gap)


def searchsimplex_for_pair(
    ref_stl: str | Path,
    tgt_stl: str | Path,
    joint_origin: list[float],
    joint_direction: list[float],
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to run SearchSimplex on a part pair."""
    searcher = SearchSimplex(
        ref_stl, tgt_stl, joint_origin, joint_direction, **kwargs
    )
    return searcher.search()
