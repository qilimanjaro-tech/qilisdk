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
from .protocols import AttachmentScope, SupportsLindblad, SupportsTimeDerivedKraus
from .representations import KrausChannel, LindbladGenerator
from .utils import _sigma_minus


class AmplitudeDamping(Noise, SupportsTimeDerivedKraus, SupportsLindblad):
    """Amplitude damping noise model for energy relaxation."""

    def __init__(self, *, t1: float) -> None:
        """Args:
            t1 (float): Relaxation time constant (must be > 0).

        Raises:
            ValueError: If t1 is not positive.
        """
        if t1 <= 0:
            raise ValueError("t1 must be > 0.")
        self._t1 = float(t1)

    @property
    def t1(self) -> float:
        """Return the relaxation time constant.

        Returns:
            float: The t1 value.
        """
        return self._t1

    def as_lindblad(self) -> LindbladGenerator:
        """
        Return the Lindblad representation for this noise type.

        Returns:
            LindbladGenerator: The Lindblad representation.
        """
        gamma = 1.0 / self._t1
        L = np.sqrt(gamma) * _sigma_minus()
        return LindbladGenerator([QTensor(L)])

    def as_kraus(self, *, duration: float) -> KrausChannel:
        """
        Return the time-derived Kraus representation for this noise type.

        Args:
            duration (float): The time duration over which the noise acts.

        Raises:
            ValueError: If duration is negative.

        Returns:
            KrausChannel: The Kraus representation.
        """
        if duration < 0:
            raise ValueError("duration must be >= 0.")
        gamma = 1.0 - float(np.exp(-duration / self._t1))
        K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
        K1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
        return KrausChannel(operators=[QTensor(K0), QTensor(K1)])

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """Return the attachment scopes supported by this perturbation type.

        Returns:
            The set of scopes where this perturbation can be attached.
        """
        return frozenset({AttachmentScope.GLOBAL, AttachmentScope.PER_QUBIT})
