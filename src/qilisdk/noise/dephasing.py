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
from .utils import _identity, _sigma_z


class Dephasing(Noise, SupportsTimeDerivedKraus, SupportsLindblad):
    """Pure dephasing (Tphi) noise model for single qubits.

    This model supports both Lindblad and time-derived Kraus forms, with
    coherences decaying as exp(-t / Tphi).
    """

    def __init__(self, *, Tphi: float) -> None:
        """Args:
            Tphi (float): Dephasing time constant (must be > 0).

        Raises:
            ValueError: If Tphi is not positive.
        """
        if Tphi <= 0:
            raise ValueError("Tphi must be > 0.")
        self._Tphi = float(Tphi)

    @property
    def Tphi(self) -> float:
        """Return the dephasing time constant.

        Returns:
            The Tphi value.
        """
        return self._Tphi

    def as_lindblad(self) -> LindbladGenerator:
        gamma = 1.0 / self._Tphi
        L = np.sqrt(gamma / 2.0) * _sigma_z()
        return LindbladGenerator([QTensor(L)])

    def as_kraus(self, *, duration: float) -> KrausChannel:
        if duration < 0:
            raise ValueError("duration must be >= 0.")
        gamma = float(np.exp(-duration / self._Tphi))
        K0 = np.sqrt((1.0 + gamma) / 2.0) * _identity()
        K1 = np.sqrt((1.0 - gamma) / 2.0) * _sigma_z()
        return KrausChannel([QTensor(K0), QTensor(K1)])
