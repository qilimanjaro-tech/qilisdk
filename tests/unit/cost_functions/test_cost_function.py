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

from qilisdk.cost_functions.cost_function import CostFunction
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult


def test_cost_function_init():
    cost_function = CostFunction()
    assert isinstance(cost_function, CostFunction)


def test_compute_cost():
    cost_function = CostFunction()
    with pytest.raises(NotImplementedError):
        cost_function.compute_cost(None)


def test_compute_cost_sampling():
    cost_function = CostFunction()
    with pytest.raises(NotImplementedError):
        cost_function.compute_cost(SamplingResult(1, {"0": 1}))


def test_compute_cost_time_evolution():
    cost_function = CostFunction()
    with pytest.raises(NotImplementedError):
        cost_function.compute_cost(
            TimeEvolutionResult(
                final_expected_values=np.array([]),
                expected_values=None,
                final_state=None,
                intermediate_states=None,
            )
        )
