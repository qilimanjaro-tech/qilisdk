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

# ======================= numeric helpers =======================

_EPS = 1e-12
_SIG_DECIMALS = 12


def _wrap_angle(a: float) -> float:
    """Wrap an angle to the (-pi, pi] range.

    Args:
        a (float): Angle value in radians.
    Returns:
        float: Angle mapped into the open-closed interval (-pi, pi].
    """
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    if a <= -math.pi:
        a = math.pi
    return a


def _zyz_from_unitary(U: np.ndarray) -> tuple[float, float, float]:
    """Recover ZYZ Euler angles from a 2x2 unitary.

    Args:
        U (np.ndarray): 2x2 unitary matrix.
    Returns:
        tuple[float, float, float]: Tuple containing theta, phi and gamma angles.
    Raises:
        ValueError: If matrix is not 2x2.
    """
    if U.shape != (2, 2):
        raise ValueError("Expected 2x2 unitary for ZYZ decomposition.")
    det = np.linalg.det(U)
    if abs(det) < _EPS:
        raise ValueError("Matrix is singular.")
    # remove phase to a U3 rotation
    phase = np.angle(U[0, 0])
    U /= np.exp(1j * phase, dtype=complex)

    a00, a01 = U[0, 0], U[0, 1]
    a10, a11 = U[1, 0], U[1, 1]
    theta = 2.0 * math.atan2(np.abs(a01), np.abs(a00))
    s = math.sin(theta / 2.0)

    if s < _EPS:
        lam = _wrap_angle(np.angle(a11))
        return (0.0, 0.0, lam)

    phi = _wrap_angle(np.angle(a10))
    lam = _wrap_angle(np.angle(-a01))
    return (theta, phi, lam)


def _unitary_sqrt_2x2(U: np.ndarray) -> np.ndarray:
    """Compute the principal square root of a 2x2 unitary.

    Args:
        U (np.ndarray): 2x2 unitary matrix.
    Returns:
        np.ndarray: Matrix V such that V · V equals U.
    """
    w, V = np.linalg.eig(U)
    ph = np.angle(w)
    sqrt_w = np.exp(0.5j * ph)
    return V @ np.diag(sqrt_w) @ np.linalg.inv(V)
