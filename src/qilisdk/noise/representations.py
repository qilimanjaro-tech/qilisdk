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
from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from .noise import Noise
from .protocols import AttachmentScope

if TYPE_CHECKING:
    from qilisdk.analog import Hamiltonian
    from qilisdk.core import QTensor


class KrausChannel(Noise):
    """Kraus operator representation of a quantum channel."""

    def __init__(self, operators: list[QTensor]) -> None:
        """Args:
        operators (list[QTensor]): Kraus operators defining the channel."""
        self._operators: list[QTensor] = operators

    @property
    def operators(self) -> list[QTensor]:
        """Return the Kraus operators defining the channel.

        Returns:
            list[QTensor]: The Kraus operators for this channel.
        """
        return self._operators

    def as_kraus(self) -> Self:
        return self

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """
        Return the allowed attachment scopes for this noise representation.

        Returns:
            frozenset[AttachmentScope]: Allowed attachment scopes.
        """
        return frozenset(
            {
                AttachmentScope.GLOBAL,
                AttachmentScope.PER_QUBIT,
                AttachmentScope.PER_GATE_TYPE,
                AttachmentScope.PER_GATE_TYPE_PER_QUBIT,
            }
        )


class LindbladGenerator(Noise):
    """Lindblad generator representation for Markovian noise."""

    def __init__(
        self,
        jump_operators: list[QTensor],
        rates: list[float] | None = None,
        hamiltonian: Hamiltonian | None = None,
    ) -> None:
        """
        Args:
            jump_operators (list[QTensor]): Jump operators defining dissipation.
            rates (list[float] | None): Optional rates for each jump operator.
            hamiltonian (Hamiltonian | None): Optional Hamiltonian term for coherent evolution.

        Raises:
            ValueError: If rates are provided and their length does not match jump_operators.
        """
        if rates is not None and len(rates) != len(jump_operators):
            raise ValueError("Length of rates must match length of jump_operators.")
        self._jump_operators = jump_operators
        self._rates = rates
        self._hamiltonian = hamiltonian

    @property
    def jump_operators(self) -> list[QTensor]:
        """Return the jump operators defining dissipation.

        Returns:
            list[QTensor]: Jump operators for this generator.
        """
        return self._jump_operators

    @property
    def jump_operators_with_rates(self) -> list[QTensor]:
        """Return the jump operators defining dissipation, scaled by their rates.

        Raises:
            ValueError: If the rate list is provided but its length does not match jump_operators.

        Returns:
            list[QTensor]: Jump operators for this generator.
        """
        if self._rates is None:
            return self._jump_operators
        if len(self._rates) != len(self._jump_operators):
            raise ValueError("Length of rates must match length of jump_operators.")
        return [self._jump_operators[i] * np.sqrt(self._rates[i]) for i in range(len(self._jump_operators))]

    @property
    def rates(self) -> list[float] | None:
        """Return the rates for each jump operator, if provided.

        Returns:
            list[float] | None: Rates for each jump operator.
        """
        return self._rates

    @property
    def hamiltonian(self) -> Hamiltonian | None:
        """Return the optional coherent Hamiltonian term.

        Returns:
            QTensor | None: The Hamiltonian term if provided.
        """
        return self._hamiltonian

    def as_lindblad(self) -> LindbladGenerator:
        """Return this instance as a Lindblad generator representation.

        Returns:
            The current LindbladGenerator instance.
        """
        return self

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        """
        Return the allowed attachment scopes for this noise representation.

        Returns:
            frozenset[AttachmentScope]: Allowed attachment scopes.
        """
        return frozenset({AttachmentScope.GLOBAL, AttachmentScope.PER_QUBIT})
