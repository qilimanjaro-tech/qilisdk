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

from qilisdk.core.qtensor import QTensor
from qilisdk.cost_functions import ModelCostFunction
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


def test_time_evolution_results_initialization():
    ter = TimeEvolutionResult()

    assert len(ter.final_expected_values) == 0
    assert isinstance(ter.final_expected_values, np.ndarray)
    assert isinstance(ter.expected_values, np.ndarray)
    assert len(ter.expected_values) == 0
    assert ter.final_state is None
    assert len(ter.intermediate_states) == 0

    ter = TimeEvolutionResult(
        final_expected_values=np.array([0, 0, 0]), expected_values=np.array([[0, 0, 0], [1, 0, 1]])
    )

    assert list(ter.final_expected_values) == [0, 0, 0]
    expected_list = [[0, 0, 0], [1, 0, 1]]
    for i, l in enumerate(list(ter.expected_values)):
        assert list(l) == expected_list[i]
    assert ter.final_state is None
    assert len(ter.intermediate_states) == 0

    ter = TimeEvolutionResult(
        final_expected_values=np.array([0, 0, 0]),
        expected_values=np.array([[0, 0, 0], [1, 0, 1]]),
        final_state=QTensor(np.array([[0], [1]])),
    )

    expected_list = [[0], [1]]
    for i, l in enumerate(list(ter.final_state.dense())):
        assert list(l) == expected_list[i]


def test_time_evolution_results_output():
    result = TimeEvolutionResult(
        final_expected_values=np.array([0.5, -0.5]),
        expected_values=np.array([[0.0, 1.0], [0.5, -0.5], [1.0, 0.0]]),
        final_state=QTensor(np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])),
        intermediate_states=[
            QTensor(np.array([[1], [0]])),
            QTensor(np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])),
            QTensor(np.array([[0], [1]])),
        ],
    )
    output = repr(result)
    assert "TimeEvolutionResult" in output
    assert "final_expected_values=" in output
    assert "expected_values=" in output
    assert "final_state=" in output
    assert "intermediate_states=" in output


def test_time_evolution_results_compute_cost():
    results = TimeEvolutionResult(
        final_expected_values=np.array([1.0, 2.0, 3.0]),
        expected_values=np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
    )

    class MockCostFunction(ModelCostFunction):
        def __init__(self): ...

    with pytest.raises(NotImplementedError):
        results.compute_cost(MockCostFunction())
