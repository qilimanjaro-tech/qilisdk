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

from qilisdk.core.qtensor import ket
from qilisdk.cost_functions.cost_function import CostFunction
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout import SamplingReadout, StateTomographyReadout
from qilisdk.readout.readout_result import (
    SamplingReadoutResult,
    StateTomographyReadoutResult,
)


def test_cost_function_cannot_be_instantiated():
    with pytest.raises(TypeError):
        CostFunction()


def test_compute_cost_with_sampling_result():
    # CostFunction is abstract, so we create a minimal subclass
    class DummyCostFunction(CostFunction):
        def compute_cost(self, results):
            return 42.0

    cost_function = DummyCostFunction()

    readout = SamplingReadout(nshots=1)
    readout_result = SamplingReadoutResult(readout=readout, samples={"0": 1})
    result = FunctionalResult(readout_results=[readout_result])

    assert cost_function.compute_cost(result) == 42.0


def test_compute_cost_with_state_tomography_result():
    class DummyCostFunction(CostFunction):
        def compute_cost(self, results):
            return 99.0

    cost_function = DummyCostFunction()

    readout = StateTomographyReadout()
    readout_result = StateTomographyReadoutResult(readout=readout, final_state=ket(0))
    result = FunctionalResult(readout_results=[readout_result])

    assert cost_function.compute_cost(result) == 99.0
