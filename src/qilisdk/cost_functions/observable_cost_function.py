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
from qilisdk.common.quantum_objects import QuantumObject, expect_val, ket, tensor_prod
from qilisdk.cost_functions.cost_function import CostFunction

if TYPE_CHECKING:
    from qilisdk.common.variables import Number
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


class ObservableCostFunction(CostFunction):
    def __init__(self, observable: QuantumObject | Hamiltonian | PauliOperator) -> None:
        super().__init__()
        if isinstance(observable, QuantumObject):
            self._observable = observable
        elif isinstance(observable, Hamiltonian):
            self._observable = QuantumObject(observable.to_matrix())
        elif isinstance(observable, PauliOperator):
            self._observable = QuantumObject(observable.matrix)
        else:
            raise ValueError(
                f"Observable needs to be of type QuantumObject, Hamiltonian, or PauliOperator but {type(observable)} was provided"
            )

    @property
    def observable(self) -> QuantumObject:
        return self._observable

    def _compute_cost_time_evolution(self, results: TimeEvolutionResult) -> Number:
        if results.final_state is None:
            raise ValueError(
                "can't compute cost using Observables from time evolution results when the state is not provided."
            )
        total_cost = complex(np.real_if_close(expect_val(self._observable, results.final_state)))
        if total_cost.imag == 0:
            return total_cost.real
        return total_cost

    def _compute_cost_sampling(self, results: SamplingResult) -> Number:
        total_cost = complex(0.0)
        nqubits = self._observable.nqubits
        for sample, prob in results.get_probabilities():
            state = tensor_prod([ket(int(i)) for i in sample])
            if nqubits != state.nqubits:
                raise ValueError(
                    f"The samples provided have {state.nqubits} qubits but the observable has {nqubits} qubits"
                )
            evaluate_results = complex(np.real_if_close(expect_val(self._observable, state)))
            total_cost += evaluate_results * prob

        if total_cost.imag == 0:
            return total_cost.real
        return total_cost
