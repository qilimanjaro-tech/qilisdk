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
from qilisdk.core.qtensor import QTensor
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout import ExpectationReadout, StateTomographyReadout
from qilisdk.readout.readout_result import ExpectationReadoutResult, StateTomographyReadoutResult


def test_functional_result_with_state_tomography():
    final_state = QTensor(np.array([[0], [1]]))
    readout = StateTomographyReadout()
    readout_result = StateTomographyReadoutResult(readout=readout, state=final_state)
    result = FunctionalResult(readout_results=[readout_result])

    assert result.has_state()
    expected_list = [[0], [1]]
    for i, row in enumerate(list(result.state.dense())):
        assert list(row) == expected_list[i]


def test_functional_result_with_expectation_values():

    state = QTensor(np.array([[1], [0]]))
    readout = ExpectationReadout(observables=[Z(0)])
    readout_result = ExpectationReadoutResult(readout=readout, state=state)
    result = FunctionalResult(readout_results=[readout_result])

    assert result.has_expectation_values()
    assert len(result.expected_values) == 1
    assert np.isclose(result.expected_values[0], 1.0, atol=1e-6)


def test_functional_result_with_intermediate_results():
    state1 = QTensor(np.array([[1], [0]]))
    state2 = QTensor(np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]]))
    state3 = QTensor(np.array([[0], [1]]))

    readout = StateTomographyReadout()
    inter1 = [StateTomographyReadoutResult(readout=readout, state=state1)]
    inter2 = [StateTomographyReadoutResult(readout=readout, state=state2)]
    final = [StateTomographyReadoutResult(readout=readout, state=state3)]

    result = FunctionalResult(readout_results=final, intermediate_results=[inter1, inter2])

    assert len(result) == 3
    assert result.has_state()
    assert result.state == state3

    states = result.intermediate_states
    assert len(states) == 3
    assert states[0] == state1
    assert states[1] == state2
    assert states[2] == state3


def test_functional_result_no_intermediate_raises():
    state = QTensor(np.array([[1], [0]]))
    readout = StateTomographyReadout()
    final = [StateTomographyReadoutResult(readout=readout, state=state)]
    result = FunctionalResult(readout_results=final)

    with pytest.raises(
        ValueError, match=r"Can't find intermediate states because intermediate Results were not stored."
    ):
        _ = result.intermediate_states


def test_functional_result_output():
    state = QTensor(np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]]))
    readout = StateTomographyReadout()
    readout_result = StateTomographyReadoutResult(readout=readout, state=state)
    result = FunctionalResult(readout_results=[readout_result])

    output = repr(result)
    assert "Functional Results" in output
