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

from .noise import Noise

if TYPE_CHECKING:
    from qilisdk.core import QTensor


class KrausChannel(Noise):
    """Kraus operator representation of a quantum channel."""

    def __init__(self, operators: list[QTensor]) -> None:
        """Args:
            operators (list[QTensor]): Kraus operators defining the channel."""
        self.operators: list[QTensor] = operators

    def as_kraus(self) -> Self:
        return self


class LindbladGenerator(Noise):
    """Lindblad generator representation for Markovian noise."""

    def __init__(
        self,
        jump_operators: list[QTensor],
        rates: list[float] | None = None,
        hamiltonian: QTensor | None = None,
    ) -> None:
        """Args:
            jump_operators (list[QTensor]): Jump operators defining dissipation.
            rates (list[float] | None): Optional rates for each jump operator.
            hamiltonian (QTensor | None): Optional Hamiltonian term for coherent evolution."""
        self.jump_operators = jump_operators
        self.rates = rates
        self.hamiltonian = hamiltonian

    def as_lindbland(self) -> Self:
        """Return this instance as a Lindblad generator representation.

        Returns:
            The current LindbladGenerator instance.
        """
        return self
