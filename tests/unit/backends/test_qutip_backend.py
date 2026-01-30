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

from qilisdk.analog.hamiltonian import Hamiltonian, PauliZ
from qilisdk.analog.schedule import Schedule
from qilisdk.core.qtensor import QTensor, tensor_prod
from qilisdk.functionals.time_evolution import TimeEvolution

pytest.importorskip("qutip", reason="QuTiP backend tests require the 'qutip' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "qutip_qip",
    reason="QuTiP backend tests require the 'qutip' optional dependency",
    exc_type=ImportError,
)


from qutip_qip.circuit import CircuitSimulator

from qilisdk.backends.qutip_backend import QutipBackend, QutipI
from qilisdk.core import bra, ket
from qilisdk.core.model import Constraint, Model, Objective
from qilisdk.core.variables import BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.digital import CNOT, RX, RY, RZ, SWAP, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import Adjoint, BasicGate, Controlled
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.sampling_result import SamplingResult
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
        backend.execute(Sampling(circuit=circuit))


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
    result = backend.execute(Sampling(circuit=circuit, nshots=100))
    assert isinstance(result, SamplingResult)
    assert result.nshots == 100
    assert all(len(key) == 2 for key in result.samples)
    assert result.samples.get("00", 0) == 100


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
    optimizer.optimize.side_effect = (
        lambda cost_function, init_parameters, bounds, store_intermediate_results: OptimizerResult(0.2, [0.9, 0.1])
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

    parameterized_program = VariationalProgram(Sampling(circuit), dummy_optimizer, cost_function)
    assert isinstance(parameterized_program.functional, Sampling)
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
    parameterized_program = VariationalProgram(Sampling(circuit), dummy_optimizer, cost_function)
    # Call obtain_cost with a custom number of shots.
    backend = QutipBackend()
    output = backend.execute(parameterized_program)

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
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)
    assert result.nshots == 10
    assert all(len(key) == 1 for key in result.samples)


def test_bad_probability(monkeypatch):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)

    class CircuitSimulatorMockResults:
        def __init__(self):
            self.probabilities = np.array([1.2, 0.2])
            self.cbits = ["0", "1"]

    monkeypatch.setattr(CircuitSimulator, "run_statistics", lambda self, nshots: CircuitSimulatorMockResults())
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)
    assert result.nshots == 10
    assert sum(result.samples.values()) == 10


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


# @pytest.mark.parametrize("initial_state", [ket(0), ket(0).to_density_matrix(), bra(0)])
@pytest.mark.parametrize(
    "initial_state",
    [tensor_prod([ket(0), ket(0)]), tensor_prod([ket(0), ket(0)]).to_density_matrix(), tensor_prod([bra(0), bra(0)])],
)
@pytest.mark.parametrize("ob", [PauliZ(0), Hamiltonian({(PauliZ(0),): 1.0}), (PauliZ(0) * PauliZ(1)).to_qtensor()])
def test_time_evolution(monkeypatch, initial_state, ob):
    monkeypatch.setattr("qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults())
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0, (PauliZ(1),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    func = TimeEvolution(schedule=schedule, observables=[ob], initial_state=initial_state)
    result = backend.execute(func)
    assert np.allclose(result.final_expected_values, np.array([0]))
    assert isinstance(result.final_state, QTensor)


def test_time_evolution_bad_initial(monkeypatch):
    monkeypatch.setattr("qilisdk.backends.qutip_backend.mesolve", lambda *args, **kwargs: TimeEvolutionMockResults())
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    bad_state = QTensor(-np.eye(2))
    func = TimeEvolution(schedule=schedule, observables=[PauliZ(0)], initial_state=bad_state)
    with pytest.raises(ValueError, match="initial state"):
        backend.execute(func)


def test_time_evolution_bad_observable():
    backend = QutipBackend()
    hamiltonian = Hamiltonian({(PauliZ(0),): 1.0})
    schedule = Schedule(hamiltonians={"h": hamiltonian}, dt=0.1)
    initial_state = ket(0)
    ob = "bad"
    func = TimeEvolution(schedule=schedule, observables=[ob], initial_state=initial_state)
    with pytest.raises(ValueError, match="observable"):
        backend.execute(func)
