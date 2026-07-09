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
"""Low-level readout method classes for quantum backend execution.

This module defines the :class:`ReadoutMethod` base class and its concrete subclasses -
:class:`SamplingReadout`, :class:`ExpectationReadout`, and :class:`StateTomographyReadout` - that
describe *how* results are extracted from the quantum backend after execution.

**The recommended way to compose readout for a functional is** :class:`~qilisdk.readout.readout_spec.Readout`.
The classes in this module are typically constructed internally by :class:`~qilisdk.readout.readout_spec.Readout`
and do not need to be instantiated directly in user code.
"""

from __future__ import annotations

from typing import Literal

from loguru import logger

from qilisdk.analog import Hamiltonian
from qilisdk.core import QTensor
from qilisdk.yaml import yaml


@yaml.register_class
class ReadoutMethod:
    """Base type for readout configurations.

    :class:`ReadoutMethod` is not meant to be instantiated directly.  Use
    :class:`~qilisdk.readout.Readout` to compose a readout specification and pass it to
    :meth:`~qilisdk.backends.Backend.execute`.
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


@yaml.register_class
class SamplingReadout(ReadoutMethod):
    """Sampling readout configuration.

    Instructs the backend to perform repeated measurement shots and return bitstring counts.
    Typically constructed via :meth:`Readout.with_sampling <qilisdk.readout.Readout.with_sampling>`.

    Args:
        nshots (int): Number of measurement shots.  Must be a positive integer.

    Examples:
        >>> from qilisdk.readout import Readout
        >>> spec = Readout().with_sampling(nshots=1000)
    """

    def __init__(self, nshots: int, expand_samples: bool = True) -> None:
        """

        Args:
            nshots (int): The number of shots to use during sampling. Needs to be a positive integer.
            expand_samples (bool): Whether to display partial samples as "00_0" instead of "000" for better readability.

        Raises:
            ValueError: If the number of shots is not a positive integer.
        """
        if nshots <= 0 or not isinstance(nshots, int):
            raise ValueError("The number of shots has to be a positive integer")
        logger.trace("[Readout] Constructing sampling readout with {} shots", nshots)
        self._nshots: int = nshots
        self._expand_samples: bool = expand_samples

    @property
    def nshots(self) -> int:
        return self._nshots

    @property
    def expand_samples(self) -> bool:
        return self._expand_samples


@yaml.register_class
class ExpectationReadout(ReadoutMethod):
    """Expectation-value readout configuration.

    Instructs the backend to compute ``<psi|O|psi>`` for one or more observables.
    Typically constructed via
    :meth:`Readout.with_expectation <qilisdk.readout.Readout.with_expectation>`.

    Args:
        observables (list[Hamiltonian | QTensor]): Observables whose expectation values are requested.
        nshots (int): Number of measurement shots.  Use ``0`` (the default) for exact state-vector
            evaluation.

    Examples:
        >>> from qilisdk.analog import Z
        >>> from qilisdk.readout import Readout
        >>> spec = Readout().with_expectation(observables=[Z(0)], nshots=0)
    """

    def __init__(self, observables: list[Hamiltonian | QTensor], nshots: int = 0) -> None:
        if nshots < 0 or not isinstance(nshots, int):
            raise ValueError("The number of shots has to be a positive integer")
        if any(not isinstance(o, (Hamiltonian, QTensor)) for o in observables):
            raise ValueError("Invalid Observable: All observables need to be QTensors or a Hamiltonian.")
        logger.trace("[Readout] Constructing expectation readout with {} observables, {} shots", len(observables), nshots)
        self._nshots: int = nshots
        self._observables: list[Hamiltonian | QTensor] = observables
        self._scaled_nqubits: int | None = None
        self._expanded_observables: list[Hamiltonian | QTensor] | None = None

    @property
    def nshots(self) -> int:
        """
        The number of shots to use when estimating the expectation values.
        """
        return self._nshots

    @property
    def observables(self) -> list[Hamiltonian | QTensor]:
        """
        The observables whose expectation values are requested, in their original form as provided by the user.
        """
        return self._observables

    def expanded_observables(self, nqubits: int) -> list[Hamiltonian | QTensor]:
        """Scale each observable to match a given number of qubits.

        The conversion is cached: calling this method again with the same
        ``nqubits`` value is a no-op.

        Note that only QTensors are expanded, Hamiltonians are returned as-is.

        Args:
            nqubits (int): Target qubit count to scale the observables to.

        Returns:
            list[Hamiltonian | QTensor]: The scaled observables as QTensors.
        """
        if self._scaled_nqubits == nqubits and self._expanded_observables is not None:
            return self._expanded_observables
        self._expanded_observables = [(o.expand(nqubits) if isinstance(o, QTensor) else o) for o in self.observables]
        self._scaled_nqubits = nqubits
        return self._expanded_observables


@yaml.register_class
class StateTomographyReadout(ReadoutMethod):
    """State-tomography readout configuration.

    Instructs the backend to return the full quantum state (ket or density matrix) after execution.
    Typically constructed via
    :meth:`Readout.with_state_tomography <qilisdk.readout.Readout.with_state_tomography>`.

    Args:
        method (Literal): Tomography method identifier.  Currently only 'exact' is supported.

    Examples:
        >>> from qilisdk.readout import Readout
        >>> spec = Readout().with_state_tomography()
    """

    def __init__(self, method: Literal["exact"] = "exact") -> None:
        logger.trace("[Readout] Constructing state-tomography readout with method {}", method)
        self._method: Literal["exact"] = method

    @property
    def method(self) -> Literal["exact"]:
        return self._method
