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

from qilisdk.common import Model
from qilisdk.common.model import ObjectiveSense
from qilisdk.common.qtensor import QTensor, bra, ket, tensor_prod
from qilisdk.common.variables import EQ, BinaryVariable
from qilisdk.cost_functions import ModelCostFunction
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


def test_compute_cost_time_evolution():
    n = 2
    b = [BinaryVariable(f"b({i})") for i in range(n)]

    model = Model("test")

    model.set_objective(term=sum(b), label="obj", sense=ObjectiveSense.MAXIMIZE)

    model.add_constraint("b0", term=EQ(b[0], 0), lagrange_multiplier=10)

    mcf = ModelCostFunction(model)

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=tensor_prod([ket(0), ket(1)]),
        intermediate_states=None,
    )
    cost = mcf.compute_cost(te_results)

    assert cost == -1
    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=tensor_prod([ket(0), ket(1)]).to_density_matrix(),
        intermediate_states=None,
    )
    cost = mcf.compute_cost(te_results)

    assert cost == -1

    mcf = ModelCostFunction(model)

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=tensor_prod([bra(0), bra(1)]),
        intermediate_states=None,
    )
    cost = mcf.compute_cost(te_results)
    assert cost == -1

    mcf = ModelCostFunction(model.to_qubo())

    cost = mcf.compute_cost(te_results)
    assert cost == -1

    mcf = ModelCostFunction(model)

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=QTensor(np.array([[1, 1], [0, 0]])),
        intermediate_states=None,
    )
    with pytest.raises(ValueError, match=r"The final state is invalid."):
        cost = mcf.compute_cost(te_results)

    te_results = TimeEvolutionResult(
        final_expected_values=np.array([[-0.9, 0]]),
        expected_values=None,
        final_state=None,
        intermediate_states=None,
    )
    with pytest.raises(
        ValueError, match=r"can't compute cost using Models from time evolution results when the state is not provided."
    ):
        cost = mcf.compute_cost(te_results)


def test_compute_cost_sampling():
    n = 2
    b = [BinaryVariable(f"b({i})") for i in range(n)]

    model = Model("test")

    model.set_objective(term=sum(b), label="obj", sense=ObjectiveSense.MAXIMIZE)

    model.add_constraint("b0", term=EQ(b[0], 0), lagrange_multiplier=10)

    mcf = ModelCostFunction(model)

    te_results = SamplingResult(nshots=100, samples={"01": 100})
    cost = mcf.compute_cost(te_results)

    assert cost == -1

    te_results = SamplingResult(nshots=100, samples={"0": 100})

    with pytest.raises(ValueError, match=r"Mapping samples to the model's variables is ambiguous."):
        cost = mcf.compute_cost(te_results)

    te_results = SamplingResult(nshots=100, samples={"01": 50, "10": 50})
    cost = mcf.compute_cost(te_results)

    assert cost == -1 * 0.5 + 9 * 0.5
