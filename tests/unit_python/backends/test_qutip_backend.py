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
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from qutip import Qobj

from qilisdk.analog.hamiltonian import Hamiltonian, PauliX, PauliZ
from qilisdk.analog.schedule import Schedule
from qilisdk.backends.qutip_backend import QutipBackend
from qilisdk.core import ket
from qilisdk.core.interpolator import Interpolation
from qilisdk.core.qtensor import InitialState, QTensor, tensor_prod
from qilisdk.functionals.analog_evolution import AnalogEvolution
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.quantum_reservoirs import QuantumReservoir, ReservoirLayer
from qilisdk.noise import (
    AmplitudeDamping,
    Dephasing,
    Depolarizing,
    KrausChannel,
    LindbladGenerator,
    NoiseModel,
)
from qilisdk.readout import Readout, SamplingReadout

pytest.importorskip("qutip", reason="QuTiP backend tests require the 'qutip' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "qutip_qip",
    reason="QuTiP backend tests require the 'qutip' optional dependency",
    exc_type=ImportError,
)


from qutip_qip.circuit import CircuitSimulator

from qilisdk.backends.qutip_backend import QutipI
from qilisdk.core import bra
from qilisdk.core.model import Constraint, Model, Objective
from qilisdk.core.variables import BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.digital import CNOT, RX, RY, RZ, SWAP, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import Adjoint, BasicGate, Controlled
from qilisdk.functionals.digital_propagation import DigitalPropagation
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.optimizers.optimizer_result import OptimizerResult


@pytest.fixture
def backend():
    return QutipBackend()


def test_basic_gate_handlers_mapping(backend):
    # ensure mapping includes all gates
    expected = {X, H, RZ}
    has = set(backend._basic_gate_handlers.keys())
    for g in expected:
        assert g in has


def test_unsupported_gate_raises(backend):
    class DummyGate(BasicGate):
        """A dummy basic gate to trigger unsupported-gate errors."""

        def __init__(self, qubit: int) -> None:
            super().__init__((qubit,))

        @property
        def name(self) -> str: ...  # type: ignore

        def _generate_matrix(self) -> np.ndarray:
            return np.eye(2, dtype=complex)

    circuit = Circuit(nqubits=1)
    circuit.gates.append(DummyGate(0))
    with pytest.raises(UnsupportedGateError):
        backend.execute(DigitalPropagation(circuit=circuit), Readout().with_sampling(nshots=100))


basic_gate_test_cases = [
    (I(0), ("i", "q0")),
    (X(0), ("x", "q0")),
    (Y(0), ("y", "q0")),
    (Z(0), ("z", "q0")),
    (H(0), ("h", "q0")),
    (S(0), ("s", "q0")),
    (T(0), ("t", "q0")),
    (RX(0, theta=0.5), ("rx", 0.5, "q0")),
    (RY(0, theta=0.6), ("ry", 0.6, "q0")),
    (RZ(0, phi=0.7), ("rz", 0.7, "q0")),
    (U1(0, phi=0.8), ("u3", 0.0, 0.8, 0.0, "q0")),
    (U2(0, phi=0.9, gamma=1.0), ("u3", np.pi / 2, 0.9, 1.0, "q0")),
    (U3(0, theta=1.1, phi=1.2, gamma=1.3), ("u3", 1.1, 1.2, 1.3, "q0")),
]
swap_test_case = [
    (SWAP(0, 1), ("swap", "q0", "q1")),
]


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
def test_adjoint_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=3)
    adjoint_gate = Adjoint(gate_instance)
    circuit._gates.append(adjoint_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name == "Adjoint_" + adjoint_gate.name for g in qutip_circuit.gates)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases + swap_test_case])
def test_controlled_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=3)
    controlled_gate = Controlled(2, basic_gate=gate_instance)
    circuit.add(controlled_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name.startswith(controlled_gate.name) for g in qutip_circuit.gates)


def test_controlled_cnot():
    backend = QutipBackend()
    circuit = Circuit(nqubits=3)
    controlled_gate = Controlled(0, basic_gate=CNOT(1, 2))
    circuit.add(controlled_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)
    assert any(g.name.startswith(controlled_gate.name) for g in qutip_circuit.gates)


def test_cnot():
    backend = QutipBackend()
    circuit = Circuit(nqubits=2)
    circuit.add(CNOT(0, 1))
    qutip_circuit = backend._get_qutip_circuit(circuit)
    assert any(g.name.startswith("CNOT") for g in qutip_circuit.gates)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases + swap_test_case])
def test_multi_controlled_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=5)
    controlled_gate = Controlled(2, 3, 4, basic_gate=gate_instance)
    circuit.add(controlled_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)
    expected_targets = set(controlled_gate.target_qubits).union(set(controlled_gate.control_qubits))

    assert any(g.name.startswith(controlled_gate.name) for g in qutip_circuit.gates)
    assert any(set(g.targets) == expected_targets for g in qutip_circuit.gates)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases + swap_test_case])
def test_handlers(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=3)
    circuit.add(gate_instance)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name == gate_instance.name for g in qutip_circuit.gates)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
def test_adjoint_fully(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=2)
    circuit.add(gate_instance)
    circuit.add(Adjoint(gate_instance))
    result = backend.execute(DigitalPropagation(circuit=circuit), Readout().with_sampling(nshots=100))
    assert isinstance(result, FunctionalResult)
    assert result.get_samples().get("00", 0) == 100


###################
# Parameterized Program
###################


@pytest.fixture
def dummy_optimizer():
    """
    Create a dummy optimizer that, upon optimization, returns a tuple of
    (optimal_cost, optimal_parameters). For testing, we use (0.2, [0.9, 0.1]).
    """
    optimizer = MagicMock()
    optimizer.optimize.side_effect = lambda cost_function, init_parameters, bounds, store_intermediate_results: (
        OptimizerResult(0.2, [0.9, 0.1])
    )
    return optimizer


def test_parameterized_program_properties_assignment(dummy_optimizer):
    """
    Test that the parameterized_program instance correctly stores its initial properties.

    Verifies that the ansatz, initial parameters, and cost function are assigned properly.
    """
    mock_instance = MagicMock(spec=ModelCostFunction)
    circuit = HardwareEfficientAnsatz(2)
    cost_function = ModelCostFunction(mock_instance)

    parameterized_program = VariationalProgram(DigitalPropagation(circuit), dummy_optimizer, cost_function)
    assert isinstance(parameterized_program.functional, DigitalPropagation)
    assert parameterized_program.functional.circuit == circuit
    assert parameterized_program.optimizer == dummy_optimizer
    assert parameterized_program.cost_function == cost_function


def test_obtain_cost_calls_backend(dummy_optimizer):
    """
    Test that the obtain_cost method correctly generates the circuit, calls the backend,
    and applies the cost function.

    This ensures:
      - ansatz.get_circuit is called with the provided parameters.
      - backend.execute is called with the generated circuit and specified number of shots.
      - The returned cost is as defined by the dummy cost function.
    """
    mock_instance = MagicMock(spec=Model)
    mock_instance.variables = mock.Mock(return_value=[BinaryVariable("b0"), BinaryVariable("b1")])

    mock_objective = MagicMock(spec=Objective)
    mock_objective.label = "obj"

    mock_con = MagicMock(spec=Constraint)
    mock_con.label = "con1"

    mock_instance.objective = mock_objective
    mock_instance.constraints = [mock_con]
    mock_instance.evaluate.return_value = {"obj": -2, "con1": 10}

    circuit = HardwareEfficientAnsatz(2)

    cost_function = ModelCostFunction(mock_instance)
    parameterized_program = VariationalProgram(DigitalPropagation(circuit), dummy_optimizer, cost_function)
    # Call obtain_cost with a custom number of shots.
    backend = QutipBackend()
    output = backend.execute(parameterized_program, Readout().with_sampling(nshots=1000))

    # The dummy_cost_function returns 0.7 regardless of input.
    assert np.isclose(output.optimal_cost, 0.2)
    assert np.isclose(cost_function.compute_cost(output.optimal_execution_results), 8.0)


def test_qutip_i():
    i = QutipI(0)
    obj = i.get_compact_qobj()
    assert obj.dims == [[2], [2]]
    assert obj.shape == (2, 2)
    assert np.allclose(obj.full(), np.eye(2))


def test_measurement_gates():
    backend = QutipBackend()
    circuit = Circuit(nqubits=2)
    circuit.add(M(0))
    result = backend.execute(DigitalPropagation(circuit=circuit), Readout().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert all(len(key) == 2 for key in result.get_samples())


def test_measurement_gates_mid_circuit():
    backend = QutipBackend()
    circuit = Circuit(nqubits=2)
    circuit.add(M(0))
    circuit.add(X(0))  # Add a gate after the measurement to trigger the error
    with pytest.raises(ValueError, match="Mid-circuit measurements are not supported"):
        backend.execute(DigitalPropagation(circuit=circuit), Readout().with_sampling(nshots=10))


def test_bad_probability(monkeypatch):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)

    class CircuitSimulatorMockResults:
        def __init__(self):
            self.probabilities = np.array([1.2, 0.2])
            self.cbits = ["0", "1"]

        def get_final_states(self) -> Qobj:
            return [Qobj([[1.2], [0.2]])]

    monkeypatch.setattr(CircuitSimulator, "run_statistics", lambda self, nshots: CircuitSimulatorMockResults())
    result = backend.execute(DigitalPropagation(circuit=circuit), Readout().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert sum(result.get_samples().values()) == 10


class QObjMock:
    def __init__(self, data):
        self.data = data

    def full(self):
        return self.data


class TimeEvolutionMockResults:
    def __init__(self):
        self.expect = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        self.final_state = QObjMock(np.array([[0], [1]]))
        self.states = [self.final_state, self.final_state]


class TimeEvolutionMockResults2Qubits:
    def __init__(self):
        self.expect = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        self.final_state = QObjMock(np.array([[1], [0], [0], [0]]))
        self.states = [self.final_state, self.final_state]


def _build_quantum_reservoir_functional() -> QuantumReservoir:
    schedule = Schedule(
        dt=1,
        hamiltonians={"h": PauliZ(0).to_hamiltonian()},
        coefficients={"h": {(0, 10): lambda t: 1 - t / 10}},
    )
    pre = Circuit(1)
    pre.add(H(0))
    post = Circuit(1)
    post.add(X(0))
    reservoir_layer = ReservoirLayer(
        evolution_dynamics=schedule,
        input_encoding=pre,
        output_encoding=post,
        qubits_to_reset=[0],
    )
    return QuantumReservoir(
        initial_state=ket(0),
        reservoir_layer=reservoir_layer,
        input_per_layer=[{}, {}],
    )


# @pytest.mark.parametrize("initial_state", [ket(0), ket(0).to_density_matrix(), bra(0)])
@pytest.mark.parametrize(
    "initial_state",
    [tensor_prod([ket(0), ket(0)]), tensor_prod([ket(0), ket(0)]).to_density_matrix(), tensor_prod([bra(0), bra(0)])],
)
@pytest.mark.parametrize(
    "ob", [PauliZ(0).to_hamiltonian(), Hamiltonian({(PauliZ(0),): 1.0}), (PauliZ(0) * PauliZ(1)).to_qtensor()]
)
def test_time_evolution(monkeypatch, initial_state, ob):
    monkeypatch.setattr(
        "qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults2Qubits()
    )
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0, (PauliZ(1),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1, total_time=1.0)
    func = AnalogEvolution(schedule=schedule, initial_state=initial_state)
    result = backend.execute(func, Readout().with_expectation(observables=[ob]).with_state_tomography())
    assert np.allclose(result.get_expectation_values(), np.array([1]))
    assert isinstance(result.get_state(), QTensor)


def test_time_evolution_initial_state_enum(monkeypatch):
    monkeypatch.setattr("qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults())
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1, total_time=1.0)

    func = AnalogEvolution(schedule=schedule, initial_state=InitialState.ZERO)
    result = backend.execute(func, Readout().with_state_tomography())
    assert isinstance(result, FunctionalResult)


def test_time_evolution_bad_initial(monkeypatch):
    monkeypatch.setattr("qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults())
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1, total_time=1.0)
    bad_state = QTensor(-np.eye(2))
    func = AnalogEvolution(schedule=schedule, initial_state=bad_state)
    with pytest.raises(ValueError, match="initial state"):
        backend.execute(func, Readout().with_state_tomography())


def test_time_evolution_bad_observable():
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1, total_time=1.0)
    initial_state = ket(0)
    ob = "bad"
    func = AnalogEvolution(schedule=schedule, initial_state=initial_state)
    with pytest.raises(ValueError, match="observable"):
        backend.execute(func, Readout().with_expectation(observables=[ob]))


def test_execute_quantum_reservoir_raises_if_time_evolution_returns_no_state(monkeypatch):
    backend = QutipBackend()
    functional = _build_quantum_reservoir_functional()

    def _mock_execute_analog_evolution(self, f, readout):
        # Return a FunctionalResult with a StateTomographyReadoutResult containing final_state=None
        # This simulates the case where the evolution returns no valid state
        raise ValueError("Reservoir Runtime Error: state repair failed before expectation value computation.")

    monkeypatch.setattr(
        "qilisdk.backends.qutip_backend.QutipBackend._execute_analog_evolution",
        _mock_execute_analog_evolution,
    )

    with pytest.raises(ValueError, match="Reservoir Runtime Error"):
        backend._execute_quantum_reservoir(
            functional, readout=[SamplingReadout(nshots=10)]
        )  # internal method takes list


def test_get_qutip_observable_qtensor(monkeypatch):

    monkeypatch.setattr("qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults())
    backend = QutipBackend()
    z_matrix = QTensor(np.array([[1, 0], [0, -1]]))
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1, total_time=1.0)
    func = AnalogEvolution(schedule=schedule, initial_state=ket(0))
    result = backend.execute(func, Readout().with_expectation(observables=[z_matrix]).with_state_tomography())
    assert result.get_expectation_values() is not None


def test_get_qutip_observable_hamiltonian_smaller_than_system(monkeypatch):

    monkeypatch.setattr(
        "qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults2Qubits()
    )
    backend = QutipBackend()
    small_hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    system_hamiltonian = Hamiltonian({(PauliZ(0),): 1.0, (PauliZ(1),): 1.0})
    schedule = Schedule(hamiltonians={"h": system_hamiltonian}, dt=0.1, total_time=1.0)
    func = AnalogEvolution(schedule=schedule, initial_state=ket(0, 0))
    result = backend.execute(func, Readout().with_expectation(observables=[small_hamiltonian]).with_state_tomography())
    assert result.get_expectation_values() is not None


def test_get_qutip_observable_unsupported_type_raises():

    backend = QutipBackend()
    with pytest.raises(ValueError, match="unsupported observable type"):
        backend._to_qubip_observables(42, nqubits=1)


def test_qutip_digital_propagation_initial_state():
    c = Circuit(1)
    digital = DigitalPropagation(circuit=c, initial_state=InitialState.ONE)
    backend = QutipBackend()
    readout = Readout().with_sampling(10)
    results = backend.execute(digital, readout)
    assert isinstance(results, FunctionalResult)
    assert "1" in results.get_samples()


def test_qutip_digital_propagation_initial_state_qtensor():
    c = Circuit(1)
    initial_state_qtensor = QTensor.one(1)
    digital = DigitalPropagation(circuit=c, initial_state=initial_state_qtensor)
    backend = QutipBackend()
    readout = Readout().with_sampling(10)
    results = backend.execute(digital, readout)
    assert isinstance(results, FunctionalResult)
    assert "1" in results.get_samples()
def test_to_qutip_observable_pauli_operator():
    """A single-qubit PauliOperator must be embedded with identities on a multi-qubit system."""
    obs = QutipBackend._to_qubip_observables(PauliZ(1), nqubits=2)
    # Z on qubit 1 of a 2-qubit register: diag(1, -1, 1, -1).
    expected = np.diag([1.0, -1.0, 1.0, -1.0])
    assert np.allclose(obs.full(), expected)


def test_to_qutip_observable_hamiltonian_padded():
    """A Hamiltonian acting on fewer qubits than the system must be padded with identities."""
    obs = QutipBackend._to_qubip_observables(PauliZ(0).to_hamiltonian(), nqubits=2)
    assert obs.dims == [[2, 2], [2, 2]]
    assert np.allclose(obs.full(), np.diag([1.0, 1.0, -1.0, -1.0]))


def test_to_qutip_observable_qtensor():
    """A QTensor observable must be passed through unchanged."""
    obs = QutipBackend._to_qubip_observables(QTensor(np.diag([1.0, -1.0])), nqubits=1)
    assert np.allclose(obs.full(), np.diag([1.0, -1.0]))


def test_qutip_backend_accepts_noise_model():
    nm = NoiseModel()
    nm.add(LindbladGenerator([QTensor(np.array([[0, 1], [1, 0]]))], rates=[0.1]))
    backend = QutipBackend(noise_model=nm)
    assert backend.noise_model is nm


def test_qutip_backend_analog_static_lindblad_relaxes():
    """A strong amplitude-damping channel must relax ``|1>`` toward ``|0>`` (<Z> -> +1)."""
    schedule = Schedule(
        hamiltonians={"hz": PauliZ(0).to_hamiltonian()},
        coefficients={"hz": {0.0: 1.0, 1.0: 1.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    analog_evolution = AnalogEvolution(schedule=schedule, initial_state=ket(1))

    nm = NoiseModel()
    nm.add(AmplitudeDamping(t1=0.1))

    result = QutipBackend(noise_model=nm).execute(
        analog_evolution, readout=Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()])
    )
    assert result.get_expectation_values()[0] > 0.9


def test_qutip_backend_analog_time_dependent_lindblad():
    """A callable rate ``rate(t)`` must be sampled over the schedule and drive dissipation."""
    schedule = Schedule(
        hamiltonians={"hz": PauliZ(0).to_hamiltonian()},
        coefficients={"hz": {0.0: 1.0, 1.0: 1.0}},
        dt=0.1,
        interpolation=Interpolation.LINEAR,
    )
    sigma_minus = QTensor(np.array([[0, 1], [0, 0]], dtype=complex))

    # rate(t) == 0 -> no dissipation, the |1> state is preserved (<Z> stays -1).
    nm_off = NoiseModel()
    nm_off.add(LindbladGenerator(jump_operators=[sigma_minus], rates=[lambda t: 0.0]), qubits=[0])
    result_off = QutipBackend(noise_model=nm_off).execute(
        AnalogEvolution(schedule=schedule, initial_state=ket(1)),
        readout=Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()]),
    )
    assert result_off.get_expectation_values()[0] < -0.99

    # A large rate(t) drives strong relaxation toward |0> (<Z> -> +1).
    nm_on = NoiseModel()
    nm_on.add(LindbladGenerator(jump_operators=[sigma_minus], rates=[lambda t: 10.0]), qubits=[0])
    result_on = QutipBackend(noise_model=nm_on).execute(
        AnalogEvolution(schedule=schedule, initial_state=ket(1)),
        readout=Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()]),
    )
    assert result_on.get_expectation_values()[0] > 0.9


def _single_qubit_z_schedule(nqubits: int = 1, coefficient: float = 1.0) -> Schedule:
    """Build a simple Z-Hamiltonian schedule over ``[0, 1]`` for noise tests."""
    hz = PauliZ(0).to_hamiltonian()
    for q in range(1, nqubits):
        hz = hz + PauliZ(q).to_hamiltonian()
    return Schedule(
        hamiltonians={"hz": hz},
        coefficients={"hz": {0.0: coefficient, 1.0: coefficient}},
        dt=0.05,
        interpolation=Interpolation.LINEAR,
    )


def test_qutip_backend_digital_noise_ignored_with_warning():
    """A noise model on a digital circuit must be ignored (with a warning) rather than fail."""
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))

    nm = NoiseModel()
    nm.add(AmplitudeDamping(t1=0.1))

    result = QutipBackend(noise_model=nm).execute(
        DigitalPropagation(circuit=circuit), Readout().with_sampling(nshots=100)
    )
    # The circuit is simulated noiselessly: X|0> = |1> deterministically.
    assert result.get_samples() == {"1": 100}


def test_qutip_backend_analog_time_derived_lindblad():
    """Noise exposing only ``as_lindblad_from_duration`` (e.g. Depolarizing) must be applied."""
    nm = NoiseModel()
    nm.add(Depolarizing(probability=0.9))

    result = QutipBackend(noise_model=nm).execute(
        AnalogEvolution(schedule=_single_qubit_z_schedule(), initial_state=ket(0)),
        readout=Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()]),
    )
    # Strong depolarization pushes the state toward the maximally mixed state (<Z> -> 0).
    assert abs(result.get_expectation_values()[0]) < 0.5


def test_qutip_backend_analog_kraus_only_noise_is_skipped():
    """A Kraus-only channel (no Lindblad representation) must be ignored by analog evolution."""
    nm = NoiseModel()
    nm.add(KrausChannel([QTensor(np.eye(2, dtype=complex))]))

    result = QutipBackend(noise_model=nm).execute(
        AnalogEvolution(schedule=_single_qubit_z_schedule(), initial_state=ket(1)),
        readout=Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()]),
    )
    # No dissipation is applied, so |1> is preserved (<Z> stays -1).
    assert result.get_expectation_values()[0] < -0.99


def test_qutip_backend_lindblad_non_square_jump_operator_raises():
    # A ket-shaped (2, 1) jump operator is non-square and must be rejected.
    nm = NoiseModel()
    nm.add(LindbladGenerator(jump_operators=[QTensor(np.zeros((2, 1), dtype=complex))]), qubits=[0])

    backend = QutipBackend(noise_model=nm)
    func = AnalogEvolution(schedule=_single_qubit_z_schedule(), initial_state=ket(0))
    readout = Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()])
    with pytest.raises(ValueError, match="must be square matrices"):
        backend.execute(func, readout)


def test_qutip_backend_lindblad_wrong_size_jump_operator_raises():
    # A two-qubit (4, 4) operator is neither single-qubit nor full-system on a one-qubit schedule.
    nm = NoiseModel()
    nm.add(LindbladGenerator(jump_operators=[QTensor(np.zeros((4, 4), dtype=complex))]), qubits=[0])

    backend = QutipBackend(noise_model=nm)
    func = AnalogEvolution(schedule=_single_qubit_z_schedule(), initial_state=ket(0))
    readout = Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()])
    with pytest.raises(ValueError, match="single-qubit or full-system"):
        backend.execute(func, readout)


def test_qutip_backend_per_qubit_noise_embedding_two_qubits():
    """Per-qubit single-qubit noise on a multi-qubit system must be embedded on its target only.

    With no coherent evolution (zero Hamiltonian coefficient), constant-rate dephasing on qubit 0
    destroys its coherence (<X_0> -> 0) while the noiseless qubit 1 is untouched (<X_1> == 1).
    """
    nm = NoiseModel()
    nm.add(Dephasing(t_phi=0.05), qubits=[0])

    plus = (ket(0) + ket(1)).unit()
    initial_state = tensor_prod([plus, plus])

    result = QutipBackend(noise_model=nm).execute(
        AnalogEvolution(schedule=_single_qubit_z_schedule(nqubits=2, coefficient=0.0), initial_state=initial_state),
        readout=Readout().with_expectation(observables=[PauliX(0).to_hamiltonian(), PauliX(1).to_hamiltonian()]),
    )

    assert abs(result.get_expectation_values()[0]) < 0.1
    assert result.get_expectation_values()[1] > 0.99


def test_qutip_backend_lindblad_with_hamiltonian_delta():
    """A Lindblad generator's coherent Hamiltonian term must be added to the evolution.

    With a zero base coefficient and a zero dissipation rate, the only dynamics comes from the
    generator's ``X`` Hamiltonian, driving Rabi oscillations: ``<Z>(T) = cos(2T)`` for ``T = 1``.
    """
    sigma_minus = QTensor(np.array([[0, 1], [0, 0]], dtype=complex))
    generator = LindbladGenerator(jump_operators=[sigma_minus], rates=[0.0], hamiltonian=PauliX(0).to_hamiltonian())
    nm = NoiseModel()
    nm.add(generator, qubits=[0])

    result = QutipBackend(noise_model=nm).execute(
        AnalogEvolution(schedule=_single_qubit_z_schedule(coefficient=0.0), initial_state=ket(0)),
        readout=Readout().with_expectation(observables=[PauliZ(0).to_hamiltonian()]),
    )

    assert np.isclose(np.real_if_close(result.get_expectation_values()[0]), np.cos(2.0), atol=1e-2)
