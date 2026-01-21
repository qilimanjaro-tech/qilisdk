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

import pytest

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import EQ, BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.functionals.sampling_result import SamplingResult


def test_sample_results_initialization():
    with pytest.raises(ValueError, match=r"The samples dictionary is empty."):
        SamplingResult(1, {})
    with pytest.raises(ValueError, match=r"Not all bitstring keys have the same length."):
        SamplingResult(1, {"01": 2, "000": 1})

    sr = SamplingResult(1, {"00": 1})
    assert sr.nqubits == 2
    assert sr.nshots == 1
    assert sr.probabilities == {"00": 1}
    assert sr.samples == {"00": 1}
    assert sr.get_probability("00") == 1
    assert sr.get_probabilities() == [("00", 1)]


def test_sample_results_probabilities():
    sr = SamplingResult(100, {"000": 10, "010": 20, "101": 40, "001": 30})
    assert sr.get_probabilities() == [("101", 0.4), ("001", 0.3), ("010", 0.2), ("000", 0.1)]
    assert sr.get_probabilities(2) == [("101", 0.4), ("001", 0.3)]


def test_sample_results_compute_cost():
    sr = SamplingResult(100, {"000": 10, "010": 20, "101": 40, "001": 30})

    model = Model("test")
    b = [BinaryVariable(f"b({i})") for i in range(sr.nqubits)]
    model.set_objective(sum(b), sense=ObjectiveSense.MAXIMIZE)
    model.add_constraint("second_b_bad", EQ(b[1], 0), lagrange_multiplier=10)
    mcf = ModelCostFunction(model)

    assert mcf.compute_cost(sr) == (-2 * 0.4 + -1 * 0.3 + 9 * 0.2 + 0 * 0.1)
