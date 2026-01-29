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

from qilisdk.backends.backend import Backend
from qilisdk.core import LT, Parameter
from qilisdk.digital import Circuit
from qilisdk.digital.gates import RZ, H
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.variational_program import VariationalProgram


def test_backend_initialization():
    backend = Backend()
    assert isinstance(backend, Backend)


def test_backend_execute():
    backend = Backend()
    circuit = Circuit(1)
    with pytest.raises(NotImplementedError, match="Backend does not support"):
        backend.execute(circuit)

    sampling = Sampling(circuit=circuit)
    with pytest.raises(NotImplementedError, match="has no Sampling"):
        backend._execute_sampling(sampling)

    time_evo = MagicMock()
    with pytest.raises(NotImplementedError, match="has no TimeEvolution"):
        backend._execute_time_evolution(time_evo)


class MockSampleResult:
    def __init__(self, functional):
        self.samples = {"0": functional.get_parameter_values()[0]}


class MockCostFunction:
    def compute_cost(self, sample_result):
        # bit of a hack, but basically we use it to simulate returning an almost real value
        if abs(sample_result.samples["0"] - 0.5) < 1e-8:
            return 0.5 + 0j
        return sample_result.samples["0"]


class MockOptimizerResult:
    def __init__(self, optimal_parameters, optimal_value):
        self.optimal_parameters = optimal_parameters
        self.optimal_value = optimal_value


class MockOptimizer:
    def __init__(self, has_parameter_constraints=False):
        self.has_parameter_constraints = has_parameter_constraints

    def optimize(self, cost_function, bounds, init_parameters, store_intermediate_results=False):
        if self.has_parameter_constraints:
            assert cost_function([3.0]) > 0
        else:
            with pytest.raises(ValueError, match="Unsupported result type"):
                cost_function([1])
            assert np.isclose(cost_function([0.5]), 0.5)
            assert np.isclose(cost_function([0.7]), 0.7)
        return MockOptimizerResult(optimal_parameters=[0.5], optimal_value=0.1)


def test_backend_variational_program(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.0, bounds=(0.0, np.pi))
    circ.add(RZ(0, phi=param))
    func = Sampling(circuit=circ)
    var_prog = VariationalProgram(functional=func, optimizer=MockOptimizer(), cost_function=MockCostFunction())

    monkeypatch.setattr(backend, "_execute_sampling", lambda f, noise_model=None: MockSampleResult(f))

    res = backend._execute_variational_program(
        functional=var_prog,
    )
    assert np.isclose(res.optimal_parameters[0], 0.5)


def test_non_parameterized_functional():
    backend = Backend()

    circ = Circuit(1)
    circ.add(H(0))
    func = Sampling(circuit=circ)
    var_prog = VariationalProgram(functional=func, optimizer=MagicMock(), cost_function=MagicMock())

    with pytest.raises(ValueError, match="Functional provided is not parameterized"):
        backend._execute_variational_program(
            functional=var_prog,
        )


def test_variational_program_parameter_constraints(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.0, bounds=(0.0, np.pi))
    term = param * 2 + 1
    circ.add(RZ(0, phi=param))
    circ._parameter_constraints.append(LT(term, 0))
    func = Sampling(circuit=circ)
    var_prog = VariationalProgram(
        functional=func, optimizer=MockOptimizer(has_parameter_constraints=True), cost_function=MockCostFunction()
    )

    monkeypatch.setattr(backend, "_execute_sampling", lambda f, noise_model=None: MockSampleResult(f))

    with pytest.raises(ValueError, match="Optimizer Failed at finding"):
        backend._execute_variational_program(
            functional=var_prog,
        )
