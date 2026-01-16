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

from qilisdk.core import QTensor

from .noise import Noise
from .protocols import SupportsLindblad, SupportsTimeDerivedKraus
from .representations import KrausChannel, LindbladGenerator
from .utils import _sigma_minus


class AmplitudeDamping(Noise, SupportsTimeDerivedKraus, SupportsLindblad):
    """Amplitude damping noise model for energy relaxation."""

    def __init__(self, *, T1: float) -> None:
        """Args:
            T1 (float): Relaxation time constant (must be > 0).

        Raises:
            ValueError: If T1 is not positive.
        """
        if T1 <= 0:
            raise ValueError("T1 must be > 0.")
        self._T1 = float(T1)

    @property
    def T1(self) -> float:
        """Return the relaxation time constant.

        Returns:
            float: The T1 value.
        """
        return self._T1

    def as_lindblad(self) -> LindbladGenerator:
        gamma = 1.0 / self._T1
        L = np.sqrt(gamma) * _sigma_minus()
        return LindbladGenerator([QTensor(L)])

    def as_kraus(self, *, duration: float) -> KrausChannel:
        if duration < 0:
            raise ValueError("duration must be >= 0.")
        gamma = 1.0 - float(np.exp(-duration / self._T1))
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
        return KrausChannel(operators=[QTensor(K0), QTensor(K1)])
