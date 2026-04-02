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

from qilisdk.analog import Hamiltonian
from qilisdk.core import QTensor


class ReadoutMethod:
    """Base type for readout configurations.

    :class:`ReadoutMethod` is not meant to be instantiated directly.  Use one
    of the concrete subclasses (:class:`SamplingReadout`,
    :class:`ExpectationReadout`, :class:`StateTomographyReadout`) or the
    convenience factory methods defined on this class:

    * :meth:`sampling` -- creates a :class:`SamplingReadout`
    * :meth:`expectation` -- creates an :class:`ExpectationReadout`
    * :meth:`state_tomography` -- creates a :class:`StateTomographyReadout`
    """

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

    def __init__(self, nshots: int) -> None:
        """

        Args:
            nshots (int): The number of shots to use during sampling. Needs to be a positive integer.

        Raises:
            ValueError: If the number of shots is not a positive integer.
        """
        if nshots <= 0 or not isinstance(nshots, int):
            raise ValueError("The number of shots has to be a positive integer")
        self._nshots: int = nshots

    @property
    def nshots(self) -> int:
        return self._nshots


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

    def __init__(self, observables: list[Hamiltonian | QTensor], nshots: int = 0) -> None:
        if nshots < 0 or not isinstance(nshots, int):
            raise ValueError("The number of shots has to be a positive integer")
        if any(not isinstance(o, (Hamiltonian, QTensor)) for o in observables):
            raise ValueError("Invalid Observable: All observables need to be QTensors or a Hamiltonian.")
        self._nshots: int = nshots
        self._observables: list[Hamiltonian | QTensor] = observables
        self._qtensor_observables: list[QTensor] = [
            (o if isinstance(o, QTensor) else o.to_qtensor()) for o in self.observables
        ]
        self._scaled_nqubits: int | None = None

    @property
    def nshots(self) -> int:
        return self._nshots

    @property
    def observables(self) -> list[Hamiltonian | QTensor]:
        return self._observables

    @property
    def qtensor_observables(self) -> list[QTensor]:
        return self._qtensor_observables

    def expand_observables(self, nqubits: int) -> None:
        """Scale each observable to match a given number of qubits.

        The conversion is cached: calling this method again with the same
        ``nqubits`` value is a no-op.

        Args:
            nqubits (int): Target qubit count to scale the observables to.
        """
        if self._scaled_nqubits == nqubits:
            return
        self._qtensor_observables = [
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

    def __init__(self, method: Literal["exact"] = "exact") -> None:
        self._method: Literal["exact"] = method

    @property
    def method(self) -> Literal["exact"]:
        return self._method
