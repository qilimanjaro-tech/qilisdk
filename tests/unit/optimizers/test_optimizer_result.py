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

from qilisdk.optimizers.optimizer_result import OptimizerIntermediateResult, OptimizerResult


def test_intermediate_result():
    res = OptimizerIntermediateResult(parameters=[0.1, 0.2], cost=0.5)
    assert res.parameters == [0.1, 0.2]
    assert np.isclose(res.cost, 0.5)
    assert repr(res) == "OptimizerIntermediateResult(cost=0.5, parameters=[0.1, 0.2])"


def test_optimizer_result():
    intermediate1 = OptimizerIntermediateResult(parameters=[0.1, 0.2], cost=0.5)
    intermediate2 = OptimizerIntermediateResult(parameters=[0.3, 0.4], cost=0.2)
    res = OptimizerResult(
        optimal_parameters=[0.3, 0.4],
        optimal_cost=0.2,
        intermediate_results=[intermediate1, intermediate2],
    )
    assert res.optimal_parameters == [0.3, 0.4]
    assert np.isclose(res.optimal_cost, 0.2)
    assert res.intermediate_results == [intermediate1, intermediate2]
    as_str = repr(res)
    assert "OptimizerResult" in as_str
    assert "optimal_cost=0.2" in as_str
    assert "optimal_parameters=[0.3, 0.4]" in as_str
    assert "intermediate_results" in as_str
