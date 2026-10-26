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

from __future__ import annotations

import numpy as np
from loguru import logger

from qilisdk.core import QTensor

from .noise import Noise
from .protocols import AttachmentScope, HasAllowedScopes, SupportsStaticKraus, SupportsTimeDerivedLindblad
from .representations import KrausChannel, LindbladGenerator
from .utils import _check_probability, _identity, _sigma_x, _sigma_y, _sigma_z

_RELATIVE_RATE_TOLERANCE = 1e-10
_MIN_BLOCH_SCALING = 1e-12


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
        self._lindblad_warning_logged = False

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
        logger.debug(
            "[PauliChannel] Building Kraus representation with pX={}, pY={}, pZ={}", self._pX, self._pY, self._pZ
        )
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
        """Return the Lindblad generator whose evolution over ``duration`` reproduces this channel exactly.

        The channel scales the Bloch vector components by lambda_x = 1 - 2 (pY + pZ), lambda_y = 1 - 2 (pX + pZ)
        and lambda_z = 1 - 2 (pX + pY). Pauli jump operators with rates gamma_X, gamma_Y, gamma_Z scale them by
        exp(-2 (gamma_Y + gamma_Z) t), exp(-2 (gamma_X + gamma_Z) t) and exp(-2 (gamma_X + gamma_Y) t). Matching
        the two gives gamma_X = (a_y + a_z - a_x) / 2 (and cyclically), where a_i = -ln(lambda_i) / (2 t).

        Some Pauli channels cannot be reproduced by any Lindbladian, so they are approximated with a warning
        (logged once per channel instance):

        - If pX + pY, pX + pZ or pY + pZ exceeds 0.5, the channel inverts a Bloch component. That component's
          scaling is clamped to a tiny positive value, i.e. the component is fully decohered instead of inverted.
        - If the matching rates include a negative one (e.g. pX = pY = 0.2), it is clamped to zero, giving the
          closest Markovian approximation.

        A pair sum of exactly 0.5 (e.g. Depolarizing(0.75) or BitFlip(0.5)) fully decoheres a component, which is
        reproduced up to the same tiny scaling without a warning.

        Args:
            duration (float): Duration over which the noise acts.

        Returns:
            LindbladGenerator: Pauli jump operators with the matching rates.

        Raises:
            ValueError: If the duration is not positive.
        """
        if duration <= 0:
            raise ValueError("Duration must be positive.")
        logger.debug("[PauliChannel] Building Lindblad representation from duration {}", duration)
        pair_sums = np.array([self._pY + self._pZ, self._pX + self._pZ, self._pX + self._pY])
        inverts_component = bool(np.any(pair_sums > 0.5 + 1e-12))
        decays = -np.log1p(-np.minimum(2.0 * pair_sums, 1.0 - _MIN_BLOCH_SCALING)) / (2.0 * duration)
        rates = decays.sum() / 2.0 - decays
        tolerance = _RELATIVE_RATE_TOLERANCE * decays.max()
        has_negative_rate = bool(np.any(rates < -tolerance))
        if (inverts_component or has_negative_rate) and not self._lindblad_warning_logged:
            logger.warning(
                "[PauliChannel] Channel with pX={}, pY={}, pZ={} is not reachable by a Lindbladian, approximating it "
                "by fully decohering inverted Bloch components and clamping negative rates to zero.",
                self._pX,
                self._pY,
                self._pZ,
            )
            self._lindblad_warning_logged = True
        operators = []
        kept_rates = []
        for pauli, rate in zip((_sigma_x, _sigma_y, _sigma_z), rates):
            if rate > tolerance:
                operators.append(QTensor(pauli()))
                kept_rates.append(float(rate))
        return LindbladGenerator(operators, kept_rates)

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        return frozenset(
            {
                AttachmentScope.GLOBAL,
                AttachmentScope.PER_QUBIT,
                AttachmentScope.PER_GATE_TYPE,
                AttachmentScope.PER_GATE_TYPE_PER_QUBIT,
            }
        )

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}(pX={self._pX}, pY={self._pY}, pZ={self._pZ})"
