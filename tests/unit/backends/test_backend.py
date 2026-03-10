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

from qilisdk.analog import Schedule, Z
from qilisdk.backends.backend import Backend
from qilisdk.core import LT, Parameter, QTensor, ket
from qilisdk.digital import Circuit
from qilisdk.digital.gates import RZ, H
from qilisdk.functionals import QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
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

    schedule = Schedule(
        dt=1,
        hamiltonians={"h": Z(0)},
        coefficients={"h": {(0, 1): 1.0}},
    )
    quantum_reservoir = QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=ReservoirLayer(
            evolution_dynamics=schedule,
            observables=[QTensor(np.eye(2, dtype=np.complex128))],
        ),
        input_per_layer=[{}],
    )
    with pytest.raises(NotImplementedError, match="has no TimeEvolution"):
        backend._execute_quantum_reservoir(quantum_reservoir)


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

    with pytest.raises(ValueError, match=r"Functional provided does not contain trainable parameters."):
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


class _MockReservoirBackend(Backend):
    def _execute_time_evolution(self, functional):
        return TimeEvolutionResult(final_state=functional.initial_state)


def test_quantum_reservoir_invalidates_circuit_cache_on_parameter_updates(monkeypatch):
    backend = _MockReservoirBackend()

    schedule = Schedule(
        dt=1,
        hamiltonians={"h": Z(0)},
        coefficients={"h": {(0, 1): 1.0}},
    )
    input_param = ReservoirInput("u", 0.0)
    pre_processing = Circuit(1)
    pre_processing.add(RZ(0, phi=input_param))

    reservoir_layer = ReservoirLayer(
        evolution_dynamics=schedule,
        observables=[QTensor(np.eye(2, dtype=np.complex128))],
        input_encoding=pre_processing,
    )
    functional = QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=reservoir_layer,
        input_per_layer=[{"u": 0.1}, {"u": 0.2}],
    )

    tracked_input_encoding = reservoir_layer.input_encoding
    assert tracked_input_encoding is not None

    original_to_qtensor = Circuit.to_qtensor
    to_qtensor_calls = 0
    signatures: list[tuple[float, ...]] = []

    def _counting_to_qtensor(self):
        nonlocal to_qtensor_calls
        if self is tracked_input_encoding:
            to_qtensor_calls += 1
            signatures.append(tuple(self.get_parameter_values()))
        return original_to_qtensor(self)

    monkeypatch.setattr(Circuit, "to_qtensor", _counting_to_qtensor)

    backend._execute_quantum_reservoir(functional)

    assert to_qtensor_calls == 2
    assert signatures == [(0.1,), (0.2,)]


class _MockScaledStateReservoirBackend(Backend):
    def _execute_time_evolution(self, functional):
        return TimeEvolutionResult(final_state=functional.initial_state * 2.0)


def test_quantum_reservoir_raises_if_state_repair_is_too_large():
    backend = _MockScaledStateReservoirBackend()

    schedule = Schedule(
        dt=1,
        hamiltonians={"h": Z(0)},
        coefficients={"h": {(0, 1): 1.0}},
    )
    functional = QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=ReservoirLayer(
            evolution_dynamics=schedule,
            observables=[QTensor(np.eye(2, dtype=np.complex128))],
        ),
        input_per_layer=[{}],
    )

    with pytest.raises(ValueError, match="large correction"):
        backend._execute_quantum_reservoir(functional)
