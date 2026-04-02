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
"""Readout method definitions for quantum circuit execution.

This module provides the :class:`ReadoutMethod` base class and its concrete
subclasses -- :class:`SamplingReadout`, :class:`ExpectationReadout`, and
:class:`StateTomographyReadout` -- that specify how results are extracted from
a quantum backend after execution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from qilisdk.analog import Hamiltonian  # noqa: TC001
from qilisdk.core import QTensor


class ReadoutMethod(BaseModel):
    """Base type for readout configurations.

    :class:`ReadoutMethod` is not meant to be instantiated directly.  Use one
    of the concrete subclasses (:class:`SamplingReadout`,
    :class:`ExpectationReadout`, :class:`StateTomographyReadout`) or the
    convenience factory methods defined on this class:

    * :meth:`sampling` -- creates a :class:`SamplingReadout`
    * :meth:`expectation` -- creates an :class:`ExpectationReadout`
    * :meth:`state_tomography` -- creates a :class:`StateTomographyReadout`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def sampling(cls, nshots: int) -> SamplingReadout:
        """Create a :class:`SamplingReadout`.

        Args:
            nshots (int): Number of measurement shots.

        Returns:
            SamplingReadout: A new sampling readout configuration.

        Raises:
            TypeError: If called on a subclass rather than :class:`ReadoutMethod` directly.
        """
        if cls is not ReadoutMethod:
            raise TypeError("factory methods are only available on ReadoutMethod, not on subclasses")
        return SamplingReadout(nshots=nshots)

    @classmethod
    def expectation(cls, observables: list, nshots: int = 0) -> ExpectationReadout:
        """Create an :class:`ExpectationReadout`.

        Args:
            observables (list): Observables whose expectation values will be evaluated.
            nshots (int): Number of measurement shots. Defaults to ``0`` (exact evaluation).

        Returns:
            ExpectationReadout: A new expectation-value readout configuration.

        Raises:
            TypeError: If called on a subclass rather than :class:`ReadoutMethod` directly.
        """
        if cls is not ReadoutMethod:
            raise TypeError("factory methods are only available on ReadoutMethod, not on subclasses")
        return ExpectationReadout(observables=observables, nshots=nshots)

    @classmethod
    def state_tomography(cls, method: Literal["exact"] = "exact") -> StateTomographyReadout:
        """Create a :class:`StateTomographyReadout`.

        Args:
            method (Literal["exact"]): Tomography method. Currently only ``"exact"`` is supported.

        Returns:
            StateTomographyReadout: A new state-tomography readout configuration.

        Raises:
            TypeError: If called on a subclass rather than :class:`ReadoutMethod` directly.
        """
        if cls is not ReadoutMethod:
            raise TypeError("factory methods are only available on ReadoutMethod, not on subclasses")
        return StateTomographyReadout(state_tomography_method=method)

    def is_sampling_readout(self) -> bool:
        """Check whether this readout is a :class:`SamplingReadout`.

        Returns:
            bool: ``True`` if this instance is a :class:`SamplingReadout`,
            ``False`` otherwise.
        """
        return isinstance(self, SamplingReadout)

    def is_expectation_readout(self) -> bool:
        """Check whether this readout is an :class:`ExpectationReadout`.

        Returns:
            bool: ``True`` if this instance is an :class:`ExpectationReadout`,
            ``False`` otherwise.
        """
        return isinstance(self, ExpectationReadout)

    def is_state_tomography_readout(self) -> bool:
        """Check whether this readout is a :class:`StateTomographyReadout`.

        Returns:
            bool: ``True`` if this instance is a
            :class:`StateTomographyReadout`, ``False`` otherwise.
        """
        return isinstance(self, StateTomographyReadout)


class SamplingReadout(ReadoutMethod):
    """Sampling readout configuration.

    Instructs the backend to perform repeated measurement shots and return
    bitstring counts.  Can also be created via
    :meth:`ReadoutMethod.sample`.

    Args:
        nshots (int): Number of measurement shots (must be >= 0).  Accepted
            as a keyword argument or as the first positional argument.

    Examples:
        >>> SamplingReadout(nshots=1000)
    """

    nshots: int = Field(ge=0)


class ExpectationReadout(ReadoutMethod):
    """Expectation-value readout configuration.

    Instructs the backend to compute expectation values
    ``<observable>`` for one or more observables.  Can also be created via
    :meth:`ReadoutMethod.expectation_values`.

    Args:
        observables (list[Hamiltonian | QTensor]): Observables whose
            expectation values are requested.  Accepted as a keyword
            argument or as the first positional argument.
        nshots (int): Number of measurement shots.  Use ``0`` (the default)
            for exact state-vector evaluation.  Accepted as a keyword
            argument or as the second positional argument.

    Attributes:
        qtensor_observables (list[QTensor]): The ``observables`` converted
            to :class:`~qilisdk.core.QTensor` form.  Populated
            automatically by a model validator; not intended to be set
            manually.

    Examples:
        >>> ExpectationReadout(observables=[hamiltonian], nshots=0)
    """

    nshots: int = Field(default=0, ge=0)
    observables: list[Hamiltonian | QTensor]
    qtensor_observables: list[QTensor] = Field(default_factory=list, init=False)

    _scaled_nqubits: int | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_qtensor_observables(self) -> ExpectationReadout:
        self.qtensor_observables = [(o if isinstance(o, QTensor) else o.to_qtensor()) for o in self.observables]
        self._scaled_nqubits = None
        return self

    def expand_observables(self, nqubits: int) -> None:
        """Scale each observable to match a given number of qubits.

        The conversion is cached: calling this method again with the same
        ``nqubits`` value is a no-op.

        Args:
            nqubits (int): Target qubit count to scale the observables to.
        """
        if self._scaled_nqubits == nqubits:
            return
        self.qtensor_observables = [
            (o.expand(nqubits) if isinstance(o, QTensor) else o.to_qtensor(nqubits)) for o in self.observables
        ]
        self._scaled_nqubits = nqubits


class StateTomographyReadout(ReadoutMethod):
    """State-tomography readout configuration.

    Instructs the backend to return the full quantum state (ket or density
    matrix) after execution.  Can also be created via
    :meth:`ReadoutMethod.state_tomography`.

    Args:
        state_tomography_method (Literal["exact"]): Tomography method
            identifier.  Currently only ``"exact"`` is supported.  Accepted
            as a keyword argument or as the first positional argument.

    Examples:
        >>> StateTomographyReadout(state_tomography_method="exact")
    """

    state_tomography_method: Literal["exact"] = Field(default="exact")
