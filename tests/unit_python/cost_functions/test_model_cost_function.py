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
from unittest.mock import MagicMock

import numpy as np
import pytest

from qilisdk.core import Model
from qilisdk.core.model import QUBO, ObjectiveSense
from qilisdk.core.qtensor import QTensor, bra, ket, tensor_prod
from qilisdk.core.variables import EQ, BinaryVariable
from qilisdk.cost_functions import ModelCostFunction
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.readout import ExpectationReadout, SamplingReadout, StateTomographyReadout
from qilisdk.readout.readout_result import (
    ExpectationReadoutResult,
    ReadoutCompositeResults,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
)


def test_compute_cost_state_tomography():
    n = 2
    b = [BinaryVariable(f"b({i})") for i in range(n)]

    model = Model("test")

    model.set_objective(term=sum(b), label="obj", sense=ObjectiveSense.MAXIMIZE)

    model.add_constraint("b0", term=EQ(b[0], 0), lagrange_multiplier=10)

    mcf = ModelCostFunction(model)

    # ket state
    readout = StateTomographyReadout()
    readout_result = StateTomographyReadoutResult(readout=readout, state=tensor_prod([ket(0), ket(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=None, expectation_values=None, state_tomography=readout_result))
    cost = mcf.compute_cost(result)

    assert cost == -1

    # density matrix state
    readout_result = StateTomographyReadoutResult(
        readout=readout, state=tensor_prod([ket(0), ket(1)]).to_density_matrix()
    )
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=None, expectation_values=None, state_tomography=readout_result))
    cost = mcf.compute_cost(result)

    assert cost == -1

    mcf = ModelCostFunction(model)

    # bra state
    readout_result = StateTomographyReadoutResult(readout=readout, state=tensor_prod([bra(0), bra(1)]))
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=None, expectation_values=None, state_tomography=readout_result))
    cost = mcf.compute_cost(result)
    assert cost == -1

    mcf = ModelCostFunction(model.to_qubo())

    cost = mcf.compute_cost(result)
    assert cost == -1

    mcf = ModelCostFunction(model)

    # no state and no samples -- should raise

    exp_readout = ExpectationReadout(observables=[QTensor(np.eye(2**n))])
    exp_result = ExpectationReadoutResult(readout=exp_readout, expected_values=[0.0])
    no_state_result = FunctionalResult(readout_results=ReadoutCompositeResults(expectation_values=exp_result))
    with pytest.raises(
        ValueError, match=r"ModelCostFunction requires either a StateTomography or Sampling readout in the results."
    ):
        _ = mcf.compute_cost(no_state_result)


def test_compute_cost_sampling():
    n = 2
    b = [BinaryVariable(f"b({i})") for i in range(n)]

    model = Model("test")

    model.set_objective(term=sum(b), label="obj", sense=ObjectiveSense.MAXIMIZE)

    model.add_constraint("b0", term=EQ(b[0], 0), lagrange_multiplier=10)

    mcf = ModelCostFunction(model)

    readout = SamplingReadout(nshots=100)
    readout_result = SamplingReadoutResult.from_samples(readout=readout, samples={"01": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = mcf.compute_cost(result)

    assert cost == -1

    readout_result = SamplingReadoutResult.from_samples(readout=SamplingReadout(nshots=100), samples={"0": 100})
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))

    with pytest.raises(ValueError, match=r"Mapping samples to the model's variables is ambiguous."):
        _ = mcf.compute_cost(result)

    readout_result = SamplingReadoutResult.from_samples(
        readout=SamplingReadout(nshots=100), samples={"01": 50, "10": 50}
    )
    result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = mcf.compute_cost(result)

    assert np.isclose(cost, -1 * 0.5 + 9 * 0.5)


def test_complex_return_values():
    return_val = complex(1, 2)
    model = MagicMock()
    model.variables = MagicMock(return_value=[MagicMock()])  # 1 variable for 1-qubit samples
    eval_results = MagicMock()
    eval_results.values = MagicMock(return_value=[return_val])
    model.evaluate = MagicMock(return_value=eval_results)
    mcf = ModelCostFunction(model)

    # sampling
    readout = SamplingReadout(nshots=100)
    readout_result = SamplingReadoutResult.from_samples(readout=readout, samples={"0": 100})
    sampling_result = FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result))
    cost = mcf._compute_from_samples(sampling_result)
    assert cost == return_val

    # state tomography
    model = MagicMock()
    model.variables = MagicMock(return_value=[MagicMock()])
    rho = QTensor(np.array([[1, 0], [0, 0]]))
    eval_results = MagicMock()
    eval_results.values = MagicMock(return_value=[return_val])
    model.evaluate = MagicMock(return_value=eval_results)
    mcf = ModelCostFunction(model)
    st_readout = StateTomographyReadout()
    st_readout_result = StateTomographyReadoutResult(readout=st_readout, state=rho)
    state_result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=st_readout_result))
    cost = mcf._compute_from_state(state_result)
    assert cost == return_val

    # state tomography (ket)
    model = MagicMock()
    rho = QTensor(np.array([1, 0]).reshape((2, 1)))
    eval_results = MagicMock()
    eval_results.values = MagicMock(return_value=[return_val])
    model.evaluate = MagicMock(return_value=eval_results)
    mcf = ModelCostFunction(model)
    st_readout_result = StateTomographyReadoutResult(readout=st_readout, state=rho)
    state_result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=st_readout_result))
    cost = mcf._compute_from_state(state_result)
    assert cost == return_val

    # state tomography (qubo)
    fake_ham = MagicMock()
    fake_ham.to_matrix = MagicMock(return_value=np.array([[return_val, 0], [0, 1]]))
    model = MagicMock(spec=QUBO)
    model.to_hamiltonian = MagicMock(return_value=fake_ham)
    rho = QTensor(np.array([[1, 0], [0, 0]]))
    eval_results = MagicMock()
    eval_results.values = MagicMock(return_value=[return_val])
    model.evaluate = MagicMock(return_value=eval_results)
    mcf = ModelCostFunction(model)
    st_readout_result = StateTomographyReadoutResult(readout=st_readout, state=rho)
    state_result = FunctionalResult(readout_results=ReadoutCompositeResults(state_tomography=st_readout_result))
    cost = mcf._compute_from_state(state_result)
    assert cost == return_val


def test_repr():
    model = Model("test")
    mcf = ModelCostFunction(model)
    repr_str = str(mcf)
    assert "ModelCostFunction" in repr_str
    assert "model=" in repr_str
