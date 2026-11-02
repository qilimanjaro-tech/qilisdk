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

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.core.qtensor import QTensor, expect_val, ket
from qilisdk.cost_functions.cost_function import CostFunction
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core.types import Number
    from qilisdk.functionals.functional_result import FunctionalResult  # type: ignore[type-arg]


class ObservableCostFunction(CostFunction):
    """Compute costs by taking the expectation value of a quantum observable.

    The observable can be supplied as a :class:`~qilisdk.core.qtensor.QTensor`,
    a :class:`~qilisdk.analog.hamiltonian.Hamiltonian`, or a
    :class:`~qilisdk.analog.hamiltonian.PauliOperator`. A ``PauliOperator`` is
    stored as a one-term ``Hamiltonian``, and the observable is resolved against
    the width of the evaluated state at compute time, so ``Z(0)`` measures qubit
    0 of a state of any size. A ``QTensor`` observable acts on the leading qubits
    and is padded with identities on the rest.

    When a ``FunctionalResult`` (from a ``DigitalPropagation`` or
    ``AnalogEvolution``) is passed to :meth:`compute_cost`, the expectation
    value is computed either exactly from the final state or estimated from
    sampled probabilities.

    Example:
        .. code-block:: python

            from qilisdk.analog.hamiltonian import Z
            from qilisdk.cost_functions import ObservableCostFunction

            cost_fn = ObservableCostFunction(Z(0))
    """

    def __init__(self, observable: QTensor | Hamiltonian | PauliOperator) -> None:
        """Initialise an ``ObservableCostFunction``.

        Args:
            observable (QTensor | Hamiltonian | PauliOperator): Quantum
                observable whose expectation value defines the cost.

        Raises:
            ValueError: If the provided observable type is unsupported.
        """
        super().__init__()
        self._observable: QTensor | Hamiltonian
        if isinstance(observable, (QTensor, Hamiltonian)):
            self._observable = observable
        elif isinstance(observable, PauliOperator):
            self._observable = observable.to_hamiltonian()
        else:
            raise ValueError(
                f"Observable needs to be of type QTensor, Hamiltonian, or PauliOperator but {type(observable)} was provided"
            )

    @property
    def observable(self) -> QTensor | Hamiltonian:
        """Return the observable, with a ``PauliOperator`` converted to a ``Hamiltonian``.

        Returns:
            QTensor | Hamiltonian: The observable whose expectation value defines the cost.
        """
        return self._observable

    def _observable_expanded(self, nqubits: int) -> QTensor | Hamiltonian:
        """Resolve the observable against a state of ``nqubits`` qubits.

        A ``QTensor`` is padded with identities on the trailing qubits, while a
        ``Hamiltonian`` is returned as-is since it is evaluated term by term.

        Args:
            nqubits (int): Number of qubits of the state being evaluated.

        Returns:
            QTensor | Hamiltonian: An observable acting on ``nqubits`` qubits.

        Raises:
            ValueError: If the observable acts on more qubits than the state has.
        """
        if self._observable.nqubits > nqubits:
            raise ValueError(
                f"The result has {nqubits} qubits but the observable acts on {self._observable.nqubits} qubits"
            )
        if isinstance(self._observable, QTensor) and self._observable.nqubits < nqubits:
            return self._observable.expand(nqubits)
        return self._observable

    def compute_cost(self, results: FunctionalResult) -> Number:
        """Compute the cost from a ``FunctionalResult``.

        Uses the final state if available (exact expectation value via
        ``StateTomography``), otherwise falls back to sampling-based estimation.

        Args:
            results (FunctionalResult): The result from executing a functional.

        Returns:
            Number: The expectation value of the observable.

        Raises:
            ValueError: If ``results`` contains neither a ``StateTomography``
                nor a ``Sampling`` readout.
        """
        if results.has_state():
            logger.trace("[ObservableCostFunction] Computing expectation value from final state")
            return self._compute_from_state(results)
        if results.has_samples():
            logger.trace("[ObservableCostFunction] Computing expectation value from samples")
            return self._compute_from_samples(results)
        raise ValueError("ObservableCostFunction requires either a StateTomography or Sampling readout in the results.")

    def _compute_from_state(self, results: FunctionalResult) -> Number:
        """Compute the exact expectation value from the final quantum state.

        Args:
            results (FunctionalResult): A result whose ``final_state`` is
                available.

        Returns:
            Number: The expectation value ``<psi|O|psi>`` (or ``Tr(rho O)``
            for mixed states).
        """
        final_state = results.get_state()
        observable = self._observable_expanded(final_state.nqubits)
        total_cost = complex(np.real_if_close(expect_val(observable, final_state), tol=get_settings().atol))
        if abs(total_cost.imag) < get_settings().atol:
            return total_cost.real
        return total_cost

    def _compute_from_samples(self, results: FunctionalResult) -> Number:
        """Estimate the expectation value from sampled probability distributions.

        Each bitstring sample is converted to a computational-basis ket and the
        observable is evaluated against it; the total is a probability-weighted
        sum of those evaluations.

        Args:
            results (FunctionalResult): A result whose ``probabilities``
                are available.

        Returns:
            Number: The probability-weighted expectation value.

        Raises:
            ValueError: If the samples have fewer qubits than the observable
                acts on.
        """
        total_cost = complex(0.0)
        samples = results.get_samples()
        nshots = sum(samples.values())
        observable = self._observable_expanded(len(next(iter(samples))))
        logger.trace("[ObservableCostFunction] Estimating expectation value from {} distinct samples", len(samples))
        for sample, prob in samples.items():
            state = ket(*[int(bit) for bit in sample])
            evaluate_results = complex(np.real_if_close(expect_val(observable, state), tol=get_settings().atol))
            total_cost += evaluate_results * (prob / nshots)

        if abs(total_cost.imag) < get_settings().atol:
            return total_cost.real
        return total_cost

    def __repr__(self) -> str:
        """Return a string representation of this cost function.

        Returns:
            str: A string of the form ``ObservableCostFunction(observable=...)``.
        """
        return f"ObservableCostFunction(observable={self._observable})"
