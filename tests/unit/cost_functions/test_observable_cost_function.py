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
import numpy as np
import pytest

from qilisdk.analog.hamiltonian import Z
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.cost_functions.observable_cost_function import ObservableCostFunction
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


def test_compute_cost_time_evolution():
    n = 2

    H = sum(Z(i) for i in range(n))

    ocf = ObservableCostFunction(H)

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=tensor_prod([ket(1), ket(1)]),
        intermediate_states=None,
    )
    cost = ocf.compute_cost(te_results)

    assert cost == -2

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=tensor_prod([ket(1), ket(1)]).to_density_matrix(),
        intermediate_states=None,
    )
    cost = ocf.compute_cost(te_results)

    assert cost == -2

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=None,
        intermediate_states=None,
    )

    with pytest.raises(
        ValueError,
        match=r"can't compute cost using Observables from time evolution results when the state is not provided.",
    ):
        cost = ocf.compute_cost(te_results)

    with pytest.raises(
        ValueError,
        match=r"Observable needs to be of type QTensor, Hamiltonian, or PauliOperator but <class 'qilisdk.functionals.time_evolution_result.TimeEvolutionResult'> was provided",
    ):
        ObservableCostFunction(te_results)


def test_compute_cost_sampling():
    n = 2

    H = sum(Z(i) for i in range(n))

    ocf = ObservableCostFunction(H)

    te_results = SamplingResult(nshots=100, samples={"11": 100})
    cost = ocf.compute_cost(te_results)

    assert cost == -2

    te_results = SamplingResult(nshots=100, samples={"0": 100})

    with pytest.raises(ValueError, match=r"The samples provided have 1 qubits but the observable has 2 qubits"):
        cost = ocf.compute_cost(te_results)

    te_results = SamplingResult(nshots=100, samples={"11": 50, "00": 50})
    cost = ocf.compute_cost(te_results)

    assert cost == 0
