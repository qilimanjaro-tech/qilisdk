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

from qilisdk.analog.hamiltonian import PauliZ, Z
from qilisdk.core.qtensor import QTensor, ket, tensor_prod
from qilisdk.cost_functions.observable_cost_function import ObservableCostFunction
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout.readout_result import (
    ExpectationReadoutResult,
    ReadoutCompositeResults,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
)


def test_init_observable_cost_function():
    n = 2

    # from hamiltonian
    H = sum(Z(i) for i in range(n))
    ocf = ObservableCostFunction(H)
    assert ocf.observable == H

    # from qtensor
    qtensor = tensor_prod([ket(0), ket(1)])
    ocf = ObservableCostFunction(qtensor)
    assert ocf.observable == qtensor

    # from pauli operator
    pauli = PauliZ(0)
    ocf = ObservableCostFunction(pauli)
    assert ocf.observable == Z(0)


def test_compute_cost_state_tomography():
    n = 2

    H = sum(Z(i) for i in range(n))

    ocf = ObservableCostFunction(H)

    # With ket state
    readout_result = StateTomographyReadoutResult(state=tensor_prod([ket(1), ket(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == -2

    # With density matrix state
    readout_result = StateTomographyReadoutResult(state=tensor_prod([ket(1), ket(1)]).to_density_matrix())
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == -2

    # Without state or samples -- should raise
    exp_result = ExpectationReadoutResult(expectation_values=[0.0])
    no_state_result = FunctionalResult(readout_results=ReadoutCompositeResults(expectation_values=exp_result))
    with pytest.raises(
        ValueError,
        match=r"ObservableCostFunction requires either a StateTomography or Sampling readout in the results.",
    ):
        _ = ocf.compute_cost(no_state_result)

    with pytest.raises(
        ValueError,
        match=r"Observable needs to be of type QTensor, Hamiltonian, or PauliOperator but <class 'qilisdk.functionals.functional_result.FunctionalResult'> was provided",
    ):
        ObservableCostFunction(result)


def test_compute_cost_sampling():
    n = 2

    H = sum(Z(i) for i in range(n))

    ocf = ObservableCostFunction(H)

    readout_result = SamplingReadoutResult.from_samples(samples={"11": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == -2

    readout_result = SamplingReadoutResult.from_samples(samples={"0": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))

    with pytest.raises(ValueError, match=r"The result has 1 qubits but the observable acts on 2 qubits"):
        _ = ocf.compute_cost(result)

    readout_result = SamplingReadoutResult.from_samples(samples={"11": 50, "00": 50})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == 0


def _ghz_result(n: int) -> FunctionalResult:
    v = np.zeros((2**n, 1), dtype=complex)
    v[0, 0] = v[-1, 0] = 1 / np.sqrt(2)
    state = StateTomographyReadoutResult(state=QTensor(v))
    return FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=state))


@pytest.mark.parametrize(
    ("observable", "expected"),
    [
        (Z(0), 0.0),
        (Z(2), 0.0),
        (PauliZ(1), 0.0),
        (Z(0) * Z(2), 1.0),
        (QTensor(Z(0).to_matrix()), 0.0),
        (QTensor((Z(0) * Z(1)).to_matrix()), 1.0),
    ],
)
def test_observable_narrower_than_state(observable, expected):
    assert np.isclose(ObservableCostFunction(observable).compute_cost(_ghz_result(3)), expected)


def test_pauli_operator_keeps_qubit_index():
    state = StateTomographyReadoutResult(state=tensor_prod([ket(0), ket(0), ket(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=state))
    assert np.isclose(ObservableCostFunction(PauliZ(0)).compute_cost(result), 1)
    assert np.isclose(ObservableCostFunction(PauliZ(2)).compute_cost(result), -1)


def test_observable_narrower_than_samples():
    readout_result = SamplingReadoutResult.from_samples(samples={"001": 75, "100": 25})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    assert np.isclose(ObservableCostFunction(Z(2)).compute_cost(result), -0.5)
    assert np.isclose(ObservableCostFunction(QTensor(Z(0).to_matrix())).compute_cost(result), 0.5)


def test_observable_wider_than_state():
    ocf = ObservableCostFunction(Z(3))
    result = _ghz_result(3)
    with pytest.raises(ValueError, match=r"The result has 3 qubits but the observable acts on 4 qubits"):
        ocf.compute_cost(result)


def test_imag_state_tomography_result():
    n = 2
    ob = QTensor(1j * np.eye(2**n))

    ocf = ObservableCostFunction(ob)

    readout_result = StateTomographyReadoutResult(state=tensor_prod([ket(1), ket(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == 1j


def test_imag_sampling_result():
    n = 2
    ob = QTensor(1j * np.eye(2**n))

    ocf = ObservableCostFunction(ob)

    readout_result = SamplingReadoutResult.from_samples(samples={"11": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == 1j


def test_repr():
    ob = QTensor(np.eye(4))
    ocf = ObservableCostFunction(ob)
    repr_str = str(ocf)
    assert "ObservableCostFunction" in repr_str
    assert "observable=" in repr_str
