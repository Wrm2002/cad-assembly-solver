"""Global pose graph optimizer using scipy.least_squares.

Takes a list of edges (pairwise relative pose constraints) and finds
globally consistent SE(3) poses for all parts.

No hard-coded logic — purely data-driven from the edge list.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .se3_manifold import se3_exp, relative_error


def optimize_poses(
    part_ids: list[str],
    edges: list[dict[str, Any]],
    anchor_id: str | None = None,
    loss: str = "soft_l1",
    f_scale: float = 0.1,
    max_nfev: int = 1000,
    xtol: float = 1e-8,
    verbose: bool = False,
) -> dict[str, Any]:
    """Optimize SE(3) poses for all parts given pairwise constraints.

    Args:
        part_ids:  Ordered list of part identifiers, e.g. ['shaft','flange_a','flange_b','key'].
        edges:     List of constraint edges, each dict with:
                     'src':     part_id (source)
                     'dst':     part_id (target)
                     'T_rel':   4×4 numpy array (measured relative transform, src→dst)
                     'weight':  float (confidence, typically JoinABLe score or probability)
        anchor_id: Part to fix at identity.  Defaults to part_ids[0].
        loss:      scipy loss function ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan').
        f_scale:   Loss scale parameter (only for soft_l1/huber/cauchy).
        max_nfev:  Max function evaluations.
        xtol:      Solver tolerance.

    Returns:
        dict with:
          'status':          'converged' | 'max_iter' | 'failed'
          'part_poses':      {part_id: 4×4 numpy array}
          'global_residual': float (RMSE of all edges)
          'edge_residuals':  list of per-edge residual norms (6-DOF 2-norm)
          'optimizer':       scipy OptimizeResult
          'runtime_ms':      float
    """
    t0 = time.time()

    if anchor_id is None:
        anchor_id = part_ids[0]
    if anchor_id not in part_ids:
        raise ValueError(f"anchor_id '{anchor_id}' not in part_ids")

    n_parts = len(part_ids)
    id_to_idx = {pid: i for i, pid in enumerate(part_ids)}
    anchor_idx = id_to_idx[anchor_id]

    # ── Residual function ──────────────────────────────────────
    # State vector: se(3) tangent vectors for all parts EXCEPT anchor.
    # Each non-anchor part contributes 6 params: [ρ, ω].
    # The state is arranged as [part_0_xi, part_1_xi, ...] skipping anchor.

    def state_to_poses(xi_vec: np.ndarray) -> dict[int, np.ndarray]:
        """Convert flat se(3) state vector to dict of 4×4 poses."""
        xi_vec = xi_vec.reshape(-1, 6)
        T = {}
        T[anchor_idx] = np.eye(4)
        for i in range(n_parts):
            if i == anchor_idx:
                continue
            xi_idx = i if i < anchor_idx else i - 1
            T[i] = se3_exp(xi_vec[xi_idx])
        return T

    def residual_fun(xi_vec: np.ndarray) -> np.ndarray:
        T = state_to_poses(xi_vec)
        residuals = []
        for e in edges:
            si = id_to_idx[e["src"]]
            ti = id_to_idx[e["dst"]]
            err = relative_error(e["T_rel"], T[si], T[ti])
            w = e.get("weight", 1.0)
            residuals.extend((w * err).tolist())
        return np.array(residuals, dtype=float)

    # ── Initial guess ──────────────────────────────────────────
    # Use the first edge to initialize poses relative to anchor.
    # Parts not reachable from anchor via edges get identity.
    x0 = np.zeros((n_parts - 1) * 6, dtype=float)

    # Simple chain initialization: walk edges from anchor
    initialized = {anchor_idx}
    parent_pose = {anchor_idx: np.eye(4)}
    changed = True
    while changed:
        changed = False
        for e in edges:
            si = id_to_idx[e["src"]]
            ti = id_to_idx[e["dst"]]
            # Try si→ti
            if si in initialized and ti not in initialized:
                parent_pose[ti] = parent_pose[si] @ e["T_rel"]
                initialized.add(ti)
                changed = True
            # Try ti→si (reverse)
            if ti in initialized and si not in initialized:
                T_rel_inv = np.eye(4)
                T_rel_inv[:3, :3] = e["T_rel"][:3, :3].T
                T_rel_inv[:3, 3] = -e["T_rel"][:3, :3].T @ e["T_rel"][:3, 3]
                parent_pose[si] = parent_pose[ti] @ T_rel_inv
                initialized.add(si)
                changed = True

    # Set x0 from initialized poses
    for i in range(n_parts):
        if i == anchor_idx:
            continue
        if i in parent_pose:
            from .se3_manifold import se3_log
            xi = se3_log(parent_pose[i])
            xi_idx = i if i < anchor_idx else i - 1
            x0[xi_idx * 6: (xi_idx + 1) * 6] = xi

    # ── Optimize ───────────────────────────────────────────────
    result = least_squares(
        residual_fun,
        x0,
        method="trf",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        xtol=xtol,
        verbose=2 if verbose else 0,
    )

    # ── Build output ───────────────────────────────────────────
    T_final = state_to_poses(result.x)
    part_poses = {part_ids[i]: T_final[i] for i in range(n_parts)}

    # Per-edge residuals
    edge_residuals = []
    for e in edges:
        si = id_to_idx[e["src"]]
        ti = id_to_idx[e["dst"]]
        err = relative_error(e["T_rel"], T_final[si], T_final[ti])
        edge_residuals.append({
            "src": e["src"],
            "dst": e["dst"],
            "residual_norm": float(np.linalg.norm(err)),
            "residual_rho": float(np.linalg.norm(err[:3])),
            "residual_omega": float(np.linalg.norm(err[3:])),
        })

    global_residual = float(np.sqrt(np.mean([er["residual_norm"] ** 2
                                              for er in edge_residuals])))

    status_map = {0: "running", 1: "converged", 2: "converged", 3: "max_iter", 4: "failed"}
    status = status_map.get(result.status, f"unknown_{result.status}")

    return {
        "status": status,
        "part_poses": part_poses,
        "global_residual": global_residual,
        "edge_residuals": edge_residuals,
        "optimizer": result,
        "runtime_ms": (time.time() - t0) * 1000,
    }
