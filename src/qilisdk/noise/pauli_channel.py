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

import numpy as np

from qilisdk.core import QTensor

from .noise import Noise
from .protocols import AttachmentScope, HasAllowedScopes, SupportsStaticKraus, SupportsTimeDerivedLindblad
from .representations import KrausChannel, LindbladGenerator
from .utils import _check_probability, _identity, _sigma_x, _sigma_y, _sigma_z


class PauliChannel(Noise, SupportsStaticKraus, SupportsTimeDerivedLindblad, HasAllowedScopes):
    """General single-qubit Pauli channel.

    Channel:
        E(rho) = pI * rho + pX * X rho X + pY * Y rho Y + pZ * Z rho Z
        where pI = 1 - (pX + pY + pZ).
    """

    def __init__(self, *, pX: float = 0.0, pY: float = 0.0, pZ: float = 0.0) -> None:
        """Args:
            pX (float): Probability of X error.
            pY (float): Probability of Y error.
            pZ (float): Probability of Z error.

        Raises:
            ValueError: If any probability is outside [0, 1] or pX + pY + pZ > 1.
        """
        self._pX = _check_probability(pX, "pX")
        self._pY = _check_probability(pY, "pY")
        self._pZ = _check_probability(pZ, "pZ")
        if self._pX + self._pY + self._pZ > 1.0 + 1e-12:
            raise ValueError("pX + pY + pZ must be <= 1.")

    @property
    def pX(self) -> float:
        """Return the probability of an X error.

        Returns:
            float: Probability for the X Pauli error.
        """
        return self._pX

    @property
    def pY(self) -> float:
        """Return the probability of a Y error.

        Returns:
            float: Probability for the Y Pauli error.
        """
        return self._pY

    @property
    def pZ(self) -> float:
        """Return the probability of a Z error.

        Returns:
            float: Probability for the Z Pauli error.
        """
        return self._pZ

    def as_kraus(self) -> KrausChannel:
        """
        Return the Kraus representation for this noise type.

        Returns:
            KrausChannel: The Kraus representation.
        """
        pI = max(0.0, 1.0 - (self._pX + self._pY + self._pZ))
        operators = []
        if pI > 0:
            operators.append(np.sqrt(pI) * _identity())
        if self._pX > 0:
            operators.append(np.sqrt(self._pX) * _sigma_x())
        if self._pY > 0:
            operators.append(np.sqrt(self._pY) * _sigma_y())
        if self._pZ > 0:
            operators.append(np.sqrt(self._pZ) * _sigma_z())
        return KrausChannel([QTensor(operator) for operator in operators])

    def as_lindblad_from_duration(self, *, duration: float) -> LindbladGenerator:
        """Return the Lindblad generator for this noise type.

        Args:
            duration (float): Duration over which the noise acts.

        Returns:
            LindbladGenerator: The Lindblad generator representation.

        Raises:
            ValueError: If duration is not positive.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive.")
        rates = []
        if self._pX > 0:
            rates.append(self._pX / duration)
        if self._pY > 0:
            rates.append(self._pY / duration)
        if self._pZ > 0:
            rates.append(self._pZ / duration)
        operators = []
        if self._pX > 0:
            operators.append(QTensor(_sigma_x()))
        if self._pY > 0:
            operators.append(QTensor(_sigma_y()))
        if self._pZ > 0:
            operators.append(QTensor(_sigma_z()))
        return LindbladGenerator(operators, rates)

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """Return the attachment scopes supported by this perturbation type.

        Returns:
            The set of scopes where this perturbation can be attached.
        """
        return frozenset(
            {
                AttachmentScope.GLOBAL,
                AttachmentScope.PER_QUBIT,
                AttachmentScope.PER_GATE_TYPE,
                AttachmentScope.PER_GATE_TYPE_PER_QUBIT,
            }
        )
