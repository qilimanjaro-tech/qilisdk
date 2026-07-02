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

from numbers import Real
from typing import TYPE_CHECKING, Callable, Self, Union, cast

import numpy as np

from .noise import Noise
from .protocols import AttachmentScope

if TYPE_CHECKING:
    from qilisdk.analog import Hamiltonian
    from qilisdk.core import QTensor

# A rate is either a constant value or a callable evaluated at the simulation time ``t``.
Rate = Union[float, Callable[[float], float]]


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

    def __repr__(self) -> str:
        return f"KrausChannel(operators={self._operators})"


class LindbladGenerator(Noise):
    """Lindblad generator representation for Markovian noise."""

    def __init__(
        self,
        jump_operators: list[QTensor],
        rates: list[Rate] | None = None,
        hamiltonian: Hamiltonian | None = None,
    ) -> None:
        """
        Args:
            jump_operators (list[QTensor]): Jump operators defining dissipation.
            rates (list[float | Callable[[float], float]] | None): Optional rates for each jump
                operator. Each rate is either a constant value or a callable ``rate(t)`` that is
                evaluated at the simulation time ``t`` of an analog evolution, allowing the
                dissipation strength to vary in time. Time-dependent (callable) rates are only
                supported by the analog evolution of :class:`~qilisdk.backends.QiliSim` (dense
                methods); see :attr:`is_time_dependent`.
            hamiltonian (Hamiltonian | None): Optional Hamiltonian term for coherent evolution.

        Raises:
            ValueError: If rates are provided and their length does not match jump_operators, or if
                any rate is neither a real number nor a callable.
        """
        if rates is not None:
            if len(rates) != len(jump_operators):
                raise ValueError("Length of rates must match length of jump_operators.")
            for rate in rates:
                if not (isinstance(rate, Real) or callable(rate)):
                    raise ValueError(
                        f"Each rate must be a real number or a callable rate(t) -> float, got {type(rate).__name__}."
                    )
        self._jump_operators = jump_operators
        self._rates = rates
        self._hamiltonian = hamiltonian

    @property
    def is_time_dependent(self) -> bool:
        """Whether any rate is a callable evaluated at the simulation time.

        Returns:
            bool: ``True`` if at least one rate is a callable ``rate(t)``, ``False`` otherwise.
        """
        if self._rates is None:
            return False
        return any(callable(rate) for rate in self._rates)

    @property
    def jump_operators(self) -> list[QTensor]:
        """Return the jump operators defining dissipation.

        Returns:
            list[QTensor]: Jump operators for this generator.
        """
        return self._jump_operators

    @property
    def jump_operators_with_rates(self) -> list[QTensor]:
        """Return the jump operators defining dissipation, scaled by their (constant) rates.

        Raises:
            ValueError: If the rate list is provided but its length does not match jump_operators.
            ValueError: If any rate is time-dependent (a callable). Time-dependent rates cannot be
                statically folded into the operators; the analog evolution of
                :class:`~qilisdk.backends.QiliSim` consumes :attr:`jump_operators` and :attr:`rates`
                directly instead.

        Returns:
            list[QTensor]: Jump operators for this generator, each scaled by ``sqrt(rate)``.
        """
        if self._rates is None:
            return self._jump_operators
        if len(self._rates) != len(self._jump_operators):
            raise ValueError("Length of rates must match length of jump_operators.")
        if self.is_time_dependent:
            raise ValueError(
                "Cannot statically scale jump operators by time-dependent (callable) rates. "
                "Time-dependent Lindblad rates are only supported by QiliSim's analog evolution, "
                "which evaluates rate(t) at each time step; this backend does not support them."
            )
        # All rates are constant here (time-dependent rates are rejected above).
        return [
            self._jump_operators[i] * np.sqrt(cast("float", self._rates[i])) for i in range(len(self._jump_operators))
        ]

    @property
    def rates(self) -> list[Rate] | None:
        """Return the rates for each jump operator, if provided.

        Returns:
            list[float | Callable[[float], float]] | None: Rates for each jump operator. Each rate
            is either a constant value or a callable ``rate(t)``.
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

    def __repr__(self) -> str:
        return f"LindbladGenerator(jump_operators={self._jump_operators}, rates={self._rates}, hamiltonian={self._hamiltonian})"
