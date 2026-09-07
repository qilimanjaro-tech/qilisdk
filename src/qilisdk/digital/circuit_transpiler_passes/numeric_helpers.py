# Copyright 2025 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np
from loguru import logger

# ======================= numeric helpers =======================

_EPS = 1e-12
_SIG_DECIMALS = 12


def _round_float(x: float, d: int = _SIG_DECIMALS) -> float:
    return 0.0 if abs(x) < _EPS else round(x, d)


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the (-pi, pi] range.

    Args:
        angle (float): Angle value in radians.
    Returns:
        float: Angle mapped into the open-closed interval (-pi, pi].
    """
    logger.trace("[NumericHelpers] Wrapping angle {}", angle)
    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if angle <= -math.pi:
        angle = math.pi
    return angle


def _zyz_from_unitary(unitary: np.ndarray) -> tuple[float, float, float]:
    """Recover ZYZ Euler angles from a 2x2 unitary.

    Args:
        unitary (np.ndarray): 2x2 unitary matrix.
    Returns:
        tuple[float, float, float]: Tuple containing theta, phi and gamma angles.
    Raises:
        ValueError: If matrix is not 2x2.
    """
    logger.trace("[NumericHelpers] Recovering ZYZ Euler angles from unitary")
    if unitary.shape != (2, 2):
        raise ValueError("Expected 2x2 unitary for ZYZ decomposition.")
    det = np.linalg.det(unitary)
    if abs(det) < _EPS:
        raise ValueError("Matrix is singular.")
    # remove phase to a U3 rotation
    phase = np.angle(unitary[0, 0])
    unitary /= np.exp(1j * phase, dtype=complex)

    a00, a01 = unitary[0, 0], unitary[0, 1]
    a10, a11 = unitary[1, 0], unitary[1, 1]
    theta = 2.0 * math.atan2(np.abs(a01), np.abs(a00))
    s = math.sin(theta / 2.0)

    if s < _EPS:
        lam = _wrap_angle(np.angle(a11))
        return (0.0, 0.0, lam)

    phi = _wrap_angle(np.angle(a10))
    lam = _wrap_angle(np.angle(-a01))
    return (theta, phi, lam)


def _u3_and_phase_from_unitary(unitary: np.ndarray) -> tuple[float, float, float, float]:
    """Decompose a 2x2 unitary as ``exp(i*alpha) * U3(theta, phi, gamma)``.

    Unlike :func:`_zyz_from_unitary`, the discarded global phase is returned instead of being dropped, so the
    factorisation is exact. This matters whenever the gate is later placed under a control, where a global phase
    becomes an observable relative phase.

    Args:
        unitary (np.ndarray): 2x2 unitary matrix.
    Returns:
        tuple[float, float, float, float]: Tuple containing theta, phi, gamma and the residual phase alpha.
    Raises:
        ValueError: If matrix is not 2x2 or is singular.
    """
    logger.trace("[NumericHelpers] Recovering U3 angles and global phase from unitary")
    if unitary.shape != (2, 2):
        raise ValueError("Expected 2x2 unitary for U3 decomposition.")
    if abs(np.linalg.det(unitary)) < _EPS:
        raise ValueError("Matrix is singular.")

    a00, a01 = unitary[0, 0], unitary[0, 1]
    a10, a11 = unitary[1, 0], unitary[1, 1]

    # theta == pi: the diagonal vanishes, so alpha can be absorbed into phi and gamma.
    if abs(a00) < _EPS:
        return (math.pi, _wrap_angle(float(np.angle(a10))), _wrap_angle(float(np.angle(-a01))), 0.0)

    # U3 has a real non-negative top-left entry, so alpha is entirely fixed by it.
    alpha = _wrap_angle(float(np.angle(a00)))
    dephase = np.exp(-1j * alpha)

    # theta == 0: the anti-diagonal vanishes and only phi + gamma is determined.
    if abs(a01) < _EPS:
        return (0.0, 0.0, _wrap_angle(float(np.angle(a11 * dephase))), alpha)

    theta = 2.0 * math.atan2(float(np.abs(a01)), float(np.abs(a00)))
    phi = _wrap_angle(float(np.angle(a10 * dephase)))
    gamma = _wrap_angle(float(np.angle(-a01 * dephase)))
    return (theta, phi, gamma, alpha)


def _unitary_sqrt_2x2(unitary: np.ndarray) -> np.ndarray:
    """Compute a unitary square root of a 2x2 unitary.

    The closed form below is used instead of an eigendecomposition because `numpy.linalg.eig` returns an
    ill-conditioned (non-orthogonal) eigenvector basis for degenerate spectra, which yields a non-unitary "square
    root". Factoring the input as ``exp(i*delta) * A`` with ``A`` in SU(2) and applying the Cayley-Hamilton identity
    ``(A + I)^2 = (tr(A) + 2) * A`` keeps the result exactly unitary for every input.

    Args:
        unitary (np.ndarray): 2x2 unitary matrix.
    Returns:
        np.ndarray: Unitary matrix V such that V · V equals U.
    Raises:
        ValueError: If matrix is not 2x2 or is not unitary.
    """
    logger.trace("[NumericHelpers] Computing square root of 2x2 unitary")
    if unitary.shape != (2, 2):
        raise ValueError("Expected 2x2 unitary for square root.")
    if not np.allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-9):
        raise ValueError("Expected a unitary matrix for square root.")

    # Factor out the phase that makes the remainder special unitary, so that Cayley-Hamilton applies.
    delta = 0.5 * float(np.angle(np.linalg.det(unitary)))
    special = unitary * np.exp(-1j * delta)

    # When trace == -2 the remainder is -I, whose square roots are degenerate; diag(i, -i) is one of them.
    trace = float(np.real(np.trace(special)))
    root = np.diag([1j, -1j]).astype(complex) if trace + 2.0 < _EPS else (special + np.eye(2)) / math.sqrt(trace + 2.0)

    return root * np.exp(0.5j * delta)


def _is_close_mod_2pi(a: float, b: float, eps: float = _EPS) -> bool:
    return abs(_wrap_angle(a - b)) < eps


def _mat_RZ(phi: float) -> np.ndarray:
    logger.trace("[NumericHelpers] Building RZ matrix for phi {}", phi)
    return np.array([[np.exp(-0.5j * phi), 0.0], [0.0, np.exp(0.5j * phi)]], dtype=complex)


def _mat_RY(theta: float) -> np.ndarray:
    logger.trace("[NumericHelpers] Building RY matrix for theta {}", theta)
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _mat_RX(theta: float) -> np.ndarray:
    logger.trace("[NumericHelpers] Building RX matrix for theta {}", theta)
    c, s = math.cos(theta / 2.0), -1j * math.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=complex)


def _mat_U3(theta: float, phi: float, lam: float) -> np.ndarray:
    logger.trace("[NumericHelpers] Building U3 matrix for theta {}, phi {}, lam {}", theta, phi, lam)
    # Convention: U3(θ, φ, λ) = RZ(φ) · RY(θ) · RZ(λ)
    return _mat_RZ(phi) @ _mat_RY(theta) @ _mat_RZ(lam) * np.exp(0.5j * (phi + lam), dtype=complex)
