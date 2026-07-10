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

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import EQ, BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout.readout_result import ReadoutCompositeResults, SamplingReadoutResult


def _make_sampling_result(nshots: int, samples: dict[str, int]) -> FunctionalResult:
    """Helper to create a FunctionalResult with sampling readout results."""
    readout_result = SamplingReadoutResult.from_samples(samples=samples)
    return FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))


def test_sample_results_initialization():
    sr = _make_sampling_result(1, {"00": 1})
    assert sr.get_probabilities() == {"00": 1.0}
    assert sr.get_samples() == {"00": 1}


def test_sample_results_probabilities():
    sr = _make_sampling_result(100, {"000": 10, "010": 20, "101": 40, "001": 30})
    probs = sr.get_probabilities()
    assert np.isclose(probs["101"], 0.4)
    assert np.isclose(probs["001"], 0.3)
    assert np.isclose(probs["010"], 0.2)
    assert np.isclose(probs["000"], 0.1)


def test_sample_results_compute_cost():
    sr = _make_sampling_result(100, {"000": 10, "010": 20, "101": 40, "001": 30})

    model = Model("test")
    nqubits = 3
    b = [BinaryVariable(f"b({i})") for i in range(nqubits)]
    model.set_objective(sum(b), sense=ObjectiveSense.MAXIMIZE)
    model.add_constraint("second_b_bad", EQ(b[1], 0), lagrange_multiplier=10)
    mcf = ModelCostFunction(model)

    assert mcf.compute_cost(sr) == (-2 * 0.4 + -1 * 0.3 + 9 * 0.2 + 0 * 0.1)


def test_sample_results_printable_representation():
    sr = _make_sampling_result(10, {"00": 3, "01": 2, "10": 4, "11": 1})
    output = str(sr)
    assert "Functional Results" in output
    assert "Execution Time" not in output


def test_execution_time_defaults_to_none():
    sr = _make_sampling_result(10, {"00": 10})
    assert sr.execution_time is None


def test_execution_time_can_be_set_and_shown_in_repr():
    sr = _make_sampling_result(10, {"00": 3, "01": 2, "10": 4, "11": 1})
    sr.execution_time = 1.5
    assert sr.execution_time == 1.5
    output = str(sr)
    assert "Execution Time" in output
    assert "1.5" in output


def test_sample_results_with_intermediate_results():
    readout_result = SamplingReadoutResult.from_samples(samples={"00": 5, "01": 5})
    intermediate_results = [
        ReadoutCompositeResults(sampling=SamplingReadoutResult.from_samples(samples={"00": 1, "01": 1}))
    ]
    sr = FunctionalResult(
        readout_results=ReadoutCompositeResults(sampling=readout_result),
        intermediate_results=intermediate_results,
    )
    assert sr.get_probabilities() == {"00": 0.5, "01": 0.5}
    assert sr.get_samples() == {"00": 5, "01": 5}
    assert len(sr.intermediate_results) == 1
    assert sr.intermediate_results[0].get_samples() == {"00": 1, "01": 1}
    assert sr.get_intermediate_samples() == [{"00": 1, "01": 1}]
    assert sr.get_intermediate_probabilities() == [{"00": 0.5, "01": 0.5}]


def test_no_intermediate_results_by_default():
    sr = _make_sampling_result(10, {"00": 10})
    with pytest.raises(ValueError, match="intermediate samples"):
        _ = sr.get_intermediate_samples()
