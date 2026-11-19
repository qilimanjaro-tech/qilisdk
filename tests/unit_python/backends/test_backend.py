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

from qilisdk.algorithms import Algorithm, QuantumReservoir, ReservoirInput, ReservoirLayer
from qilisdk.algorithms.variational_program import VariationalProgram
from qilisdk.analog import Schedule, Z
from qilisdk.backends.backend import Backend
from qilisdk.backends.backend_config import AnalogMethod
from qilisdk.core import LT, InitialState, Parameter, ket
from qilisdk.digital import Circuit
from qilisdk.digital.gates import CNOT, RX, RY, RZ, H
from qilisdk.functionals import AnalogEvolution, DigitalPropagation, FunctionalResult
from qilisdk.noise import NoiseModel
from qilisdk.readout import Readout
from qilisdk.readout.readout_result import ReadoutCompositeResults, SamplingReadoutResult, StateTomographyReadoutResult


def test_backend_initialization():
    backend = Backend()
    assert isinstance(backend, Backend)


def test_backend_execute():
    backend = Backend()
    functional = DigitalPropagation(Circuit(1))
    readout = Readout().with_sampling(nshots=10)

    with pytest.raises(NotImplementedError, match="has no DigitalPropagation"):
        backend.execute(functional, readout)


def test_backend_execute_delegates_to_algorithm():
    expected = FunctionalResult(
        readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=ket(0)))
    )
    calls = []

    class _EchoAlgorithm(Algorithm):
        def _execute(self, backend, readout):
            calls.append((backend, readout))
            return expected

    backend = Backend()
    readout = Readout().with_state_tomography()

    result = backend.execute(_EchoAlgorithm(), readout)

    assert calls == [(backend, readout)]
    assert result is expected
    assert result.execution_time is not None


def test_backend_native_hook_runs_algorithm_and_its_subclasses():
    class _NativeReservoirBackend(Backend):
        def _execute_quantum_reservoir(self, functional, readout):
            return expected

    class _MyReservoir(QuantumReservoir):
        pass

    expected = FunctionalResult(
        readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=ket(0)))
    )
    reservoir = _MyReservoir(
        initial_state=ket(0),
        reservoir_layer=ReservoirLayer(evolution_dynamics=Circuit(1)),
        input_per_layer=[{}],
    )

    result = _NativeReservoirBackend().execute(reservoir, Readout().with_state_tomography())

    assert result is expected


def test_backend_execute_empty_readout():
    backend = Backend()
    functional = DigitalPropagation(Circuit(1))

    with pytest.raises(ValueError, match="At least one readout method must be provided"):
        backend.execute(functional, Readout())


def test_backend_execute_duplicate_readout():
    with pytest.raises(ValueError, match="Sampling readout already set"):
        Readout().with_sampling(nshots=10).with_sampling(nshots=20)


def _make_mock_result(functional):
    """Create a FunctionalResult with sampling results from functional parameters."""
    param_val = functional.get_parameter_values()[0] if functional.get_parameter_values() else 0.0
    samples = {"0": int(param_val * 100) if param_val > 0 else 100}
    readout_result = SamplingReadoutResult.from_samples(samples=samples)
    return FunctionalResult(
        readout_results=ReadoutCompositeResults(sampling=readout_result, expectation_values=None, state_tomography=None)
    )


class MockCostFunction:
    def compute_cost(self, result):
        samples = result.get_samples()
        if abs(samples.get("0", 0) - 50) < 1:
            return 0.5 + 0j
        return samples.get("0", 0) / 100.0


class MockOptimizerResult:
    def __init__(self, optimal_parameters, optimal_cost):
        self.optimal_parameters = optimal_parameters
        self.optimal_cost = optimal_cost
        self.intermediate_results = []


class MockOptimizer:
    def __init__(self, has_parameter_constraints=False):
        self.has_parameter_constraints = has_parameter_constraints

    def optimize(self, cost_function, bounds, init_parameters, store_intermediate_results=False):
        if self.has_parameter_constraints:
            assert cost_function([3.0]) > 0
        else:
            assert np.isclose(cost_function([0.5]), 0.5)
            assert np.isclose(cost_function([0.7]), 0.7)
        return MockOptimizerResult(optimal_parameters=[0.5], optimal_cost=0.1)


def test_backend_variational_program(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.0, bounds=(0.0, np.pi))
    circ.add(RZ(0, phi=param))
    func = DigitalPropagation(circuit=circ)
    readout = Readout().with_sampling(nshots=100)
    var_prog = VariationalProgram(functional=func, optimizer=MockOptimizer(), cost_function=MockCostFunction())

    monkeypatch.setattr(
        backend,
        "_execute_digital_propagation",
        lambda f, ro: _make_mock_result(f),
    )

    res = backend.execute(var_prog, readout)
    assert np.isclose(res.optimal_parameters[0], 0.5)


def test_variational_program_clamps_out_of_bounds_parameters(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.5, bounds=(0.0, 1.0))
    circ.add(RZ(0, phi=param))
    func = DigitalPropagation(circuit=circ)
    readout = Readout().with_sampling(nshots=100)

    class OutOfBoundsOptimizer:
        def optimize(self, cost_function, bounds, init_parameters, store_intermediate_results=False):
            assert np.isclose(cost_function([2.9]), 1.0)
            assert np.isclose(param.value, 1.0)
            cost_function([-0.3])
            assert np.isclose(param.value, 0.0)
            return MockOptimizerResult(optimal_parameters=[2.9], optimal_cost=1.0)

    var_prog = VariationalProgram(functional=func, optimizer=OutOfBoundsOptimizer(), cost_function=MockCostFunction())

    monkeypatch.setattr(
        backend,
        "_execute_digital_propagation",
        lambda f, ro: _make_mock_result(f),
    )

    result = backend.execute(var_prog, readout)
    assert np.isclose(param.value, 1.0)
    assert np.allclose(result.optimal_parameters, [1.0])


def test_non_parameterized_functional():
    backend = Backend()

    circ = Circuit(1)
    circ.add(H(0))
    func = DigitalPropagation(circuit=circ)
    readout = Readout().with_sampling(nshots=10)
    var_prog = VariationalProgram(functional=func, optimizer=MagicMock(), cost_function=MagicMock())

    with pytest.raises(ValueError, match=r"Functional provided does not contain trainable parameters."):
        backend.execute(var_prog, readout)


def test_variational_program_parameter_constraints(monkeypatch):
    backend = Backend()

    circ = Circuit(1)
    param = Parameter("phi", 0.0, bounds=(0.0, np.pi))
    term = param * 2 + 1
    circ.add(RZ(0, phi=param))
    circ._parameter_constraints.append(LT(term, 0))
    func = DigitalPropagation(circuit=circ)
    readout = Readout().with_sampling(nshots=100)
    var_prog = VariationalProgram(
        functional=func, optimizer=MockOptimizer(has_parameter_constraints=True), cost_function=MockCostFunction()
    )

    monkeypatch.setattr(
        backend,
        "_execute_digital_propagation",
        lambda f, ro: _make_mock_result(f),
    )

    with pytest.raises(ValueError, match="Optimizer Failed at finding"):
        backend.execute(var_prog, readout)


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

    backend.execute(functional, Readout().with_state_tomography())

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
        backend.execute(UnknownFunctional(), Readout().with_sampling(nshots=10))


def test_backend_analog_evolution_not_implemented():
    backend = Backend()
    schedule = Schedule(dt=1, hamiltonians={"h": Z(0)}, coefficients={"h": {(0, 1): 1.0}})
    functional = AnalogEvolution(schedule=schedule, initial_state=ket(0))
    with pytest.raises(NotImplementedError, match="has no AnalogEvolution"):
        backend.execute(functional, Readout().with_state_tomography())


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

    result = backend.execute(functional, Readout().with_state_tomography())
    assert result is not None


def test_quantum_reservoir_accepts_initial_state_enum():
    # When the reservoir's initial state is an InitialState enum (rather than a
    # QTensor), the base backend must materialise it into a density matrix.
    class _MockBackend(Backend):
        def _execute_analog_evolution(self, functional, readout):
            state = functional.initial_state
            return FunctionalResult(
                readout_results=ReadoutCompositeResults(state_tomography=StateTomographyReadoutResult(state=state))
            )

    backend = _MockBackend()
    schedule = Schedule(dt=1, hamiltonians={"h": Z(0)}, coefficients={"h": {(0, 1): 1.0}})
    reservoir_layer = ReservoirLayer(evolution_dynamics=schedule)
    functional = QuantumReservoir(
        initial_state=InitialState.UNIFORM,
        reservoir_layer=reservoir_layer,
        input_per_layer=[{}],
    )
    result = backend.execute(functional, Readout().with_state_tomography())
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
    result = backend.execute(functional, Readout().with_state_tomography())
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

    result = backend.execute(functional, Readout().with_state_tomography())
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
    readout = Readout().with_sampling(nshots=100)

    class _SingleCallOptimizer:
        has_parameter_constraints = False

        def optimize(self, cost_function, bounds, init_parameters, store_intermediate_results=False):
            cost_function(init_parameters)
            return MockOptimizerResult(optimal_parameters=init_parameters, optimal_cost=0.0)

    var_prog = VariationalProgram(functional=func, optimizer=_SingleCallOptimizer(), cost_function=_BadCostFunction())

    monkeypatch.setattr(backend, "_execute_digital_propagation", lambda f, ro: _make_mock_result(f))

    with pytest.raises(ValueError, match="Unsupported result type"):
        backend.execute(var_prog, readout)


def test_backend_config_analog_method_direct():
    method = AnalogMethod.direct()
    assert method.evolution_method == "direct"


def test_quantum_reservoir_with_circuit_dynamics():
    class _DigitalOnlyBackend(Backend):
        pass

    dynamics = Circuit(2)
    dynamics.add(RX(0, theta=ReservoirInput("x", 0.0)))
    dynamics.add(CNOT(0, 1))
    dynamics.add(RY(1, theta=0.7))
    encoding = Circuit(2)
    encoding.add(RY(1, theta=ReservoirInput("y", 0.0)))
    layer = ReservoirLayer(evolution_dynamics=dynamics, input_encoding=encoding)
    inputs = [{"x": 0.3, "y": 0.2}, {"x": 1.1, "y": -0.4}]
    observables = [Z(0), Z(1)]

    z_matrices = [np.kron(np.diag([1, -1]), np.eye(2)), np.kron(np.eye(2), np.diag([1, -1]))]
    rho = ket(0, 0).to_density_matrix().dense()
    expected = []
    for layer_inputs in inputs:
        layer.set_parameters(layer_inputs)
        for step in layer:
            unitary = step.to_qtensor().dense()
            rho = unitary @ rho @ unitary.conj().T
        expected.append([np.real(np.trace(z @ rho)) for z in z_matrices])

    functional = QuantumReservoir(initial_state=ket(0, 0), reservoir_layer=layer, input_per_layer=inputs)
    result = _DigitalOnlyBackend().execute(functional, Readout().with_expectation(observables=observables))

    actual = [*result.get_intermediate_expectation_values(), result.get_expectation_values()]
    assert np.allclose(actual, expected, atol=1e-10)
