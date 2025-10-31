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

from qilisdk.core.qtensor import QTensor
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
    for i, l in enumerate(list(ter.final_state.dense)):
        assert list(l) == expected_list[i]
