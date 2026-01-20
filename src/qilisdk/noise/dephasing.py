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
from .protocols import AttachmentScope, HasAllowedScopes, SupportsStaticLindblad, SupportsTimeDerivedKraus
from .representations import KrausChannel, LindbladGenerator
from .utils import _identity, _sigma_z


class Dephasing(Noise, SupportsTimeDerivedKraus, SupportsStaticLindblad, HasAllowedScopes):
    """Pure dephasing (Tphi) noise model for single qubits.

    This model supports both Lindblad and time-derived Kraus forms, with
    coherences decaying as exp(-t / Tphi).
    """

    def __init__(self, *, t_phi: float) -> None:
        """Args:
            t_phi (float): Dephasing time constant (must be > 0).

        Raises:
            ValueError: If t_phi is not positive.
        """
        if t_phi <= 0:
            raise ValueError("t_phi must be > 0.")
        self._t_phi = float(t_phi)

    @property
    def t_phi(self) -> float:
        """Return the dephasing time constant.

        Returns:
            The t_phi value.
        """
        return self._t_phi

    def as_lindblad(self) -> LindbladGenerator:
        """
        Return the Lindblad representation for this noise type.

        Returns:
            LindbladGenerator: The Lindblad representation.
        """
        gamma = 1.0 / self._t_phi
        rate = gamma / 2.0
        L = QTensor(_sigma_z())
        return LindbladGenerator(jump_operators=[L], rates=[rate])

    def as_kraus_from_duration(self, *, duration: float) -> KrausChannel:
        """
        Return the Kraus representation for this noise type over a given duration.

        Args:
            duration (float): Duration over which to apply the noise.

        Returns:
            KrausChannel: The Kraus representation.

        Raises:
            ValueError: If duration is negative.
        """
        if duration < 0:
            raise ValueError("duration must be >= 0.")
        gamma = float(np.exp(-duration / self._t_phi))
        K0 = np.sqrt((1.0 + gamma) / 2.0) * _identity()
        K1 = np.sqrt((1.0 - gamma) / 2.0) * _sigma_z()
        return KrausChannel([QTensor(K0), QTensor(K1)])

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """Return the attachment scopes supported by this perturbation type.

        Returns:
            The set of scopes where this perturbation can be attached.
        """
        return frozenset({AttachmentScope.GLOBAL, AttachmentScope.PER_QUBIT})
