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

from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.core.qtensor import QTensor, expect_val, ket, tensor_prod
from qilisdk.cost_functions.cost_function import CostFunction
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core.types import Number
    from qilisdk.functionals.functional_result import FunctionalResult  # type: ignore[type-arg]


class ObservableCostFunction(CostFunction):
    """Compute costs by taking the expectation value of a quantum observable.

    The observable can be supplied as a :class:`~qilisdk.core.qtensor.QTensor`,
    a :class:`~qilisdk.analog.hamiltonian.Hamiltonian`, or a
    :class:`~qilisdk.analog.hamiltonian.PauliOperator`. It is stored internally
    as a ``QTensor``.

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
        if isinstance(observable, QTensor):
            self._observable = observable
        elif isinstance(observable, Hamiltonian):
            self._observable = QTensor(observable.to_matrix())
        elif isinstance(observable, PauliOperator):
            self._observable = QTensor(observable.matrix)
        else:
            raise ValueError(
                f"Observable needs to be of type QTensor, Hamiltonian, or PauliOperator but {type(observable)} was provided"
            )

    @property
    def observable(self) -> QTensor:
        """Return the observable in ``QTensor`` form.

        Returns:
            QTensor: The matrix representation of the observable.
        """
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
            return self._compute_from_state(results)
        if results.has_samples():
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
        final_state = results.state
        total_cost = complex(np.real_if_close(expect_val(self._observable, final_state), tol=get_settings().atol))
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
            ValueError: If the number of qubits in a sample does not match the
                number of qubits of the observable.
        """
        total_cost = complex(0.0)
        nqubits = self._observable.nqubits
        probabilities = results.probabilities
        for sample, prob in probabilities.items():
            state = tensor_prod([ket(int(i)) for i in sample])
            if nqubits != state.nqubits:
                raise ValueError(
                    f"The samples provided have {state.nqubits} qubits but the observable has {nqubits} qubits"
                )
            evaluate_results = complex(np.real_if_close(expect_val(self._observable, state), tol=get_settings().atol))
            total_cost += evaluate_results * prob

        if abs(total_cost.imag) < get_settings().atol:
            return total_cost.real
        return total_cost

    def __repr__(self) -> str:
        """Return a string representation of this cost function.

        Returns:
            str: A string of the form ``ObservableCostFunction(observable=...)``.
        """
        return f"ObservableCostFunction(observable={self._observable})"
