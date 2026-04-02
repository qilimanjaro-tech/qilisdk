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
from qilisdk.backends.backend_config import AnalogMethod
from qilisdk.core import LT, Parameter, ket
from qilisdk.digital import Circuit
from qilisdk.digital.gates import RZ, H
from qilisdk.functionals import (
    AnalogEvolution,
    DigitalPropagation,
    FunctionalResult,
    QuantumReservoir,
    ReservoirInput,
    ReservoirLayer,
)
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.noise import NoiseModel
from qilisdk.readout import ReadoutSpec, SamplingReadout, StateTomographyReadout
from qilisdk.readout.readout_result import ReadoutCompositeResults, SamplingReadoutResult, StateTomographyReadoutResult


def test_backend_initialization():
    backend = Backend()
    assert isinstance(backend, Backend)


def test_backend_execute():
    backend = Backend()
    functional = DigitalPropagation(Circuit(1))
    readout = ReadoutSpec().with_sampling(SamplingReadout(nshots=10))

    with pytest.raises(NotImplementedError, match="has no DigitalPropagation"):
        backend.execute(functional, readout)


def test_backend_execute_empty_readout():
    backend = Backend()
    functional = DigitalPropagation(Circuit(1))

    with pytest.raises(ValueError, match="At least one readout method must be provided"):
        backend.execute(functional, ReadoutSpec())


def test_backend_execute_duplicate_readout():
    with pytest.raises(ValueError, match="Sampling readout already set"):
        ReadoutSpec().with_sampling(SamplingReadout(nshots=10)).with_sampling(SamplingReadout(nshots=20))


def _make_mock_result(functional):
    """Create a FunctionalResult with sampling results from functional parameters."""
    param_val = functional.get_parameter_values()[0] if functional.get_parameter_values() else 0.0
    samples = {"0": int(param_val * 100) if param_val > 0 else 100}
    readout_result = SamplingReadoutResult.from_samples(samples=samples)
    return FunctionalResult(readout_results=ReadoutCompositeResults(sampling=readout_result, expectation_values=None, state_tomography=None))


class MockCostFunction:
    def compute_cost(self, result):
        samples = result.samples
        if abs(samples.get("0", 0) - 50) < 1:
            return 0.5 + 0j
        return samples.get("0", 0) / 100.0


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
            assert np.isclose(cost_function([0.5]), 0.5)
            assert np.isclose(cost_function([0.7]), 0.7)
        return MockOptimizerResult(optimal_parameters=[0.5], optimal_value=0.1)


def test_backend_variational_program(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.0, bounds=(0.0, np.pi))
    circ.add(RZ(0, phi=param))
    func = DigitalPropagation(circuit=circ)
    readout = [SamplingReadout(nshots=100)]
    var_prog = VariationalProgram(functional=func, optimizer=MockOptimizer(), cost_function=MockCostFunction())

    monkeypatch.setattr(
        backend,
        "_execute_digital_propagation",
        lambda f, ro: _make_mock_result(f),
    )

    res = backend._execute_variational_program(functional=var_prog, readout=readout)
    assert np.isclose(res.optimal_parameters[0], 0.5)


def test_non_parameterized_functional():
    backend = Backend()

    circ = Circuit(1)
    circ.add(H(0))
    func = DigitalPropagation(circuit=circ)
    readout = [SamplingReadout(nshots=10)]
    var_prog = VariationalProgram(functional=func, optimizer=MagicMock(), cost_function=MagicMock())

    with pytest.raises(ValueError, match=r"Functional provided does not contain trainable parameters."):
        backend._execute_variational_program(functional=var_prog, readout=readout)


def test_variational_program_parameter_constraints(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.0, bounds=(0.0, np.pi))
    term = param * 2 + 1
    circ.add(RZ(0, phi=param))
    circ._parameter_constraints.append(LT(term, 0))
    func = DigitalPropagation(circuit=circ)
    readout = [SamplingReadout(nshots=100)]
    var_prog = VariationalProgram(
        functional=func, optimizer=MockOptimizer(has_parameter_constraints=True), cost_function=MockCostFunction()
    )

    monkeypatch.setattr(
        backend,
        "_execute_digital_propagation",
        lambda f, ro: _make_mock_result(f),
    )

    with pytest.raises(ValueError, match="Optimizer Failed at finding"):
        backend._execute_variational_program(functional=var_prog, readout=readout)


def test_quantum_reservoir_invalidates_circuit_cache_on_parameter_updates(monkeypatch):
    class _MockReservoirBackend(Backend):
        def _execute_analog_evolution(self, functional, readout):
            state = functional.initial_state
            return FunctionalResult(
                readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=state))
            )

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

    readout = [StateTomographyReadout()]
    backend._execute_quantum_reservoir(functional, readout)

    assert to_qtensor_calls == 2
    assert signatures == [(0.1,), (0.2,)]


def test_print_backend():
    backend = Backend()
    as_str = str(backend)
    assert "Backend" in as_str


def test_backend_execute_unsupported_functional():
    backend = Backend()

    class UnknownFunctional:
        pass

    with pytest.raises(NotImplementedError, match="does not support"):
        backend.execute(UnknownFunctional(), [SamplingReadout(nshots=10)])


def test_backend_analog_evolution_not_implemented():
    backend = Backend()
    schedule = Schedule(dt=1, hamiltonians={"h": Z(0)}, coefficients={"h": {(0, 1): 1.0}})
    functional = AnalogEvolution(schedule=schedule, initial_state=ket(0))
    with pytest.raises(NotImplementedError, match="has no AnalogEvolution"):
        backend.execute(functional, [StateTomographyReadout()])


def test_quantum_reservoir_with_noise_model_warns(monkeypatch):
    class _MockBackend(Backend):
        def _execute_analog_evolution(self, functional, readout):
            state = functional.initial_state
            return FunctionalResult(
                readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=state))
            )

    backend = _MockBackend(noise_model=NoiseModel())

    schedule = Schedule(dt=1, hamiltonians={"h": Z(0)}, coefficients={"h": {(0, 1): 1.0}})
    reservoir_layer = ReservoirLayer(evolution_dynamics=schedule)
    functional = QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=reservoir_layer,
        input_per_layer=[{}],
    )

    result = backend._execute_quantum_reservoir(functional, [StateTomographyReadout()])
    assert result is not None


def test_quantum_reservoir_uses_circuit_cache(monkeypatch):
    class _MockBackend(Backend):
        def _execute_analog_evolution(self, functional, readout):
            state = functional.initial_state
            return FunctionalResult(
                readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=state))
            )

    backend = _MockBackend()

    schedule = Schedule(dt=1, hamiltonians={"h": Z(0)}, coefficients={"h": {(0, 1): 1.0}})
    pre = Circuit(1)
    pre.add(H(0))  # no parameters, same signature every time
    reservoir_layer = ReservoirLayer(evolution_dynamics=schedule, input_encoding=pre)
    functional = QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=reservoir_layer,
        input_per_layer=[{}, {}],  # two layers with same circuit params -> cache hit
    )

    to_qtensor_count = [0]
    original = Circuit.to_qtensor

    def counting_to_qtensor(self):
        to_qtensor_count[0] += 1
        return original(self)

    monkeypatch.setattr(Circuit, "to_qtensor", counting_to_qtensor)
    result = backend._execute_quantum_reservoir(functional, [StateTomographyReadout()])
    # First call computes, second should use cache
    assert to_qtensor_count[0] == 1
    assert result is not None


def test_quantum_reservoir_with_qubit_reset(monkeypatch):
    class _MockBackend(Backend):
        def _execute_analog_evolution(self, functional, readout):
            state = functional.initial_state
            return FunctionalResult(
                readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=state))
            )

    backend = _MockBackend()
    schedule = Schedule(dt=1, hamiltonians={"h": Z(0)}, coefficients={"h": {(0, 1): 1.0}})
    reservoir_layer = ReservoirLayer(evolution_dynamics=schedule, qubits_to_reset=[0])
    functional = QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=reservoir_layer,
        input_per_layer=[{}, {}],
    )

    result = backend._execute_quantum_reservoir(functional, [StateTomographyReadout()])
    assert result is not None


def test_variational_program_unsupported_result_type(monkeypatch):
    class _BadCostFunction:
        def compute_cost(self, result):
            return 1 + 2j  # complex with large imaginary part

    backend = Backend()
    circ = Circuit(1)
    param = Parameter("phi", 0.5, bounds=(0.0, np.pi))
    circ.add(RZ(0, phi=param))
    func = DigitalPropagation(circuit=circ)
    readout = [SamplingReadout(nshots=100)]

    class _SingleCallOptimizer:
        has_parameter_constraints = False

        def optimize(self, cost_function, bounds, init_parameters, store_intermediate_results=False):
            cost_function(init_parameters)
            return MockOptimizerResult(optimal_parameters=init_parameters, optimal_value=0.0)

    var_prog = VariationalProgram(functional=func, optimizer=_SingleCallOptimizer(), cost_function=_BadCostFunction())

    monkeypatch.setattr(backend, "_execute_digital_propagation", lambda f, ro: _make_mock_result(f))

    with pytest.raises(ValueError, match="Unsupported result type"):
        backend._execute_variational_program(functional=var_prog, readout=readout)


def test_backend_config_analog_method_direct():

    method = AnalogMethod.direct()
    assert method.evolution_method == "direct"
