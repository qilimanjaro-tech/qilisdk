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
from qilisdk.readout import ExpectationReadout, SamplingReadout, StateTomographyReadout
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
    assert isinstance(ocf.observable, QTensor)


def test_compute_cost_state_tomography():
    n = 2

    H = sum(Z(i) for i in range(n))

    ocf = ObservableCostFunction(H)

    # With ket state
    readout = StateTomographyReadout()
    readout_result = StateTomographyReadoutResult(readout=readout, state=tensor_prod([ket(1), ket(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == -2

    # With density matrix state
    readout_result = StateTomographyReadoutResult(
        readout=readout, state=tensor_prod([ket(1), ket(1)]).to_density_matrix()
    )
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == -2

    # Without state or samples -- should raise
    exp_readout = ExpectationReadout(observables=[H])
    exp_result = ExpectationReadoutResult(readout=exp_readout, expected_values=[0.0])
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

    readout = SamplingReadout(nshots=100)
    readout_result = SamplingReadoutResult.from_samples(readout=readout, samples={"11": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == -2

    readout_result = SamplingReadoutResult.from_samples(readout=SamplingReadout(nshots=100), samples={"0": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))

    with pytest.raises(ValueError, match=r"The samples provided have 1 qubits but the observable has 2 qubits"):
        _ = ocf.compute_cost(result)

    readout_result = SamplingReadoutResult.from_samples(
        readout=SamplingReadout(nshots=100), samples={"11": 50, "00": 50}
    )
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == 0


def test_imag_state_tomography_result():
    n = 2
    ob = QTensor(1j * np.eye(2**n))

    ocf = ObservableCostFunction(ob)

    readout = StateTomographyReadout()
    readout_result = StateTomographyReadoutResult(readout=readout, state=tensor_prod([ket(1), ket(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == 1j


def test_imag_sampling_result():
    n = 2
    ob = QTensor(1j * np.eye(2**n))

    ocf = ObservableCostFunction(ob)

    readout = SamplingReadout(nshots=100)
    readout_result = SamplingReadoutResult.from_samples(readout=readout, samples={"11": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = ocf.compute_cost(result)

    assert cost == 1j


def test_repr():
    ob = QTensor(np.eye(4))
    ocf = ObservableCostFunction(ob)
    repr_str = str(ocf)
    assert "ObservableCostFunction" in repr_str
    assert "observable=" in repr_str
