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
    from qilisdk.functionals.functional_result import FunctionalResult


class ObservableCostFunction(CostFunction):
    """
    Compute costs by taking expectation values of observables.

    Example:
        .. code-block:: python

            from qilisdk.analog.hamiltonian import Z
            from qilisdk.cost_functions import ObservableCostFunction

            cost_fn = ObservableCostFunction(Z(0))
    """

    def __init__(self, observable: QTensor | Hamiltonian | PauliOperator) -> None:
        """
        Args:
            observable (QTensor | Hamiltonian | PauliOperator): Quantum observable whose expectation value defines the cost.

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
        """Return the observable in ``QTensor`` form."""
        return self._observable

    def compute_cost(self, results: FunctionalResult) -> Number:
        """
        Compute the cost from a functional result.

        Uses the final state if available (exact expectation value),
        otherwise falls back to sampling-based estimation.

        Returns:
            Number: the expectation value of the observable.

        Raises:
            ValueError: If the results contain neither a StateTomography nor a Sampling readout.
        """
        if results.has_final_state():
            return self._compute_from_state(results)
        if results.has_samples():
            return self._compute_from_samples(results)
        raise ValueError("ObservableCostFunction requires either a StateTomography or Sampling readout in the results.")

    def _compute_from_state(self, results: FunctionalResult) -> Number:
        final_state = results.final_state
        total_cost = complex(np.real_if_close(expect_val(self._observable, final_state), tol=get_settings().atol))
        if abs(total_cost.imag) < get_settings().atol:
            return total_cost.real
        return total_cost

    def _compute_from_samples(self, results: FunctionalResult) -> Number:
        total_cost = complex(0.0)
        nqubits = self._observable.nqubits
        probabilities = results.final_probabilities
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
        return f"ObservableCostFunction(observable={self._observable})"
