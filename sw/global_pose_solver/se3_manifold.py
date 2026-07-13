"""SE(3) manifold operations for pose graph optimization.

Provides:
  se3_log(T)     – SE(3) → se(3)  logarithm map (4×4 → 6-vector)
  se3_exp(xi)    – se(3) → SE(3)  exponential map (6-vector → 4×4)
  rel_err(Tm, Ti, Tj) – log(Tm⁻¹ · Ti⁻¹ · Tj), the 6-DOF error between
                          measured relative pose Tm and current poses Ti,Tj

All functions are pure numpy/scipy (Rotation only), no scipy.linalg dependencies.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=float)


def se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) → se(3) logarithm.  Input: 4×4, output: 6-vector [ρ, ω].

    Uses analytical formula:
      ω = log(R) via Rotation.as_rotvec()
      θ = ||ω||
      If θ ≈ 0: ρ = t
      Else:      ρ = V⁻¹(ω) · t
    where V⁻¹ = I - ½[ω_hat] + (1/θ² - (1+cosθ)/(2θ·sinθ))[ω_hat]²
    """
    R = np.asarray(T[:3, :3], dtype=float)
    t = np.asarray(T[:3, 3], dtype=float)

    omega = Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(omega)

    if theta < 1e-12:
        rho = t.copy()
    else:
        omega_hat = omega / theta
        W = skew(omega_hat)
        W2 = W @ W
        # V⁻¹(θ) from Barfoot "State Estimation for Robotics" §7.1.4
        # V⁻¹ = I - θ/2·[ω_hat] + (1 - θ/(2·tan(θ/2)))·[ω_hat]²
        half_theta = theta / 2.0
        tan_half = np.tan(half_theta)
        if abs(tan_half) > 1e-12:
            coeff = 1.0 - half_theta / tan_half
        else:
            # tan(θ/2) ≈ θ/2 for small θ, use series
            coeff = theta * theta / 12.0  # 1 - (θ/2)/(θ/2 + θ³/24) ≈ θ²/12
        V_inv = np.eye(3) - half_theta * W + coeff * W2
        rho = V_inv @ t

    return np.concatenate([rho, omega])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """se(3) → SE(3) exponential.  Input: 6-vector [ρ, ω], output: 4×4."""
    xi = np.asarray(xi, dtype=float)
    rho = xi[:3]
    omega = xi[3:]
    theta = np.linalg.norm(omega)

    if theta < 1e-12:
        R = np.eye(3)
        t = rho
    else:
        omega_hat = omega / theta
        R = Rotation.from_rotvec(omega).as_matrix()
        omega_skew = skew(omega_hat)
        omega_skew2 = omega_skew @ omega_skew
        V = (np.eye(3)
             + (1.0 - np.cos(theta)) / theta * omega_skew
             + (theta - np.sin(theta)) / theta * omega_skew2)
        t = V @ rho

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) matrix efficiently."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def relative_error(T_measured: np.ndarray, T_i: np.ndarray, T_j: np.ndarray) -> np.ndarray:
    """Compute 6-DOF error between measured and current relative pose.

    error = log(T_measured⁻¹ · T_i⁻¹ · T_j)

    Args:
        T_measured: 4×4 measured relative transform (T_AB from JoinABLe)
        T_i: current global pose of part A (4×4)
        T_j: current global pose of part B (4×4)

    Returns:
        6-vector [ρ_err, ω_err] in se(3)
    """
    T_measured_inv = se3_inv(T_measured)
    T_rel_current = se3_inv(T_i) @ T_j
    T_error = T_measured_inv @ T_rel_current
    return se3_log(T_error)


# ── Quick self-test ──
if __name__ == "__main__":
    # Test round-trip: exp(log(T)) == T
    np.random.seed(42)
    for _ in range(100):
        # Random SE(3)
        R = Rotation.random().as_matrix()
        t = np.random.randn(3) * 10
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        xi = se3_log(T)
        T2 = se3_exp(xi)
        err = np.linalg.norm(T - T2)
        if err > 1e-10:
            print(f"Round-trip error: {err:.2e}")
    print("SE(3) manifold self-test passed.")
