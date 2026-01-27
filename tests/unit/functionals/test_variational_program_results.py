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


from qilisdk.functionals import VariationalProgramResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.optimizers.optimizer_result import OptimizerResult


def test_variational_program_results_initialization():
    optimizer_result = OptimizerResult(
        optimal_cost=1.5,
        optimal_parameters=[0.1, 0.2, 0.3],
    )
    result = TimeEvolutionResult(
        final_expected_values=None,
        expected_values=None,
        final_state=None,
        intermediate_states=None,
    )
    var_res = VariationalProgramResult(optimizer_result, result)

    assert var_res.optimal_cost == 1.5
    assert var_res.optimal_parameters == [0.1, 0.2, 0.3]
    assert var_res.optimal_execution_results == result
    assert var_res.intermediate_results == []

    as_str = repr(var_res)
    assert "VariationalProgramResult" in as_str
    assert "Optimal Cost=1.5" in as_str
    assert "Optimal Parameters=[0.1, 0.2, 0.3]" in as_str
    assert "Intermediate Results=[]" in as_str
