# Copyright 2026 Qilimanjaro Quantum Tech
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

import numpy as np


def _check_probability(p: float, name: str = "p") -> float:
    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be in [0, 1].")
    return p


def _identity() -> np.ndarray:
    return np.array([[1.0, 0.0],
                     [0.0, 1.0]], dtype=complex)


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0],
                     [1.0, 0.0]], dtype=complex)


def _sigma_y() -> np.ndarray:
    return np.array([[0.0, -1.0j],
                     [1.0j, 0.0]], dtype=complex)


def _sigma_z() -> np.ndarray:
    return np.array([[1.0, 0.0],
                     [0.0, -1.0]], dtype=complex)


def _sigma_minus() -> np.ndarray:
    return np.array([[0.0, 1.0],
                     [0.0, 0.0]], dtype=complex)


def _sigma_plus() -> np.ndarray:
    return np.array([[0.0, 0.0],
                     [1.0, 0.0]], dtype=complex)
