from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends import QutipBackend
from qilisdk.common.model import Constraint, Model, Objective
from qilisdk.common.quantum_objects import ket, tensor_prod
from qilisdk.common.variables import BinaryVariable
from qilisdk.digital import RX, RY, RZ, U1, U2, U3, Circuit, H, M, S, T, X, Y, Z
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import CNOT, Adjoint, Controlled
from qilisdk.functionals.parameterized_program import ParameterizedProgram
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer


@pytest.fixture
def backend():
    return QutipBackend()


def test_basic_gate_handlers_mapping(backend):
    # ensure mapping includes all gates
    expected = {X, H, RZ}
    has = set(backend._basic_gate_handlers.keys())
    for g in expected:
        assert g in has


def test_execute_simple_circuit_no_measurement(backend):
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))
    result = backend.execute(Sampling(circuit=circuit, nshots=100))
    # Expect roughly all shots to be '1'
    assert isinstance(result, SamplingResult)
    samples = result.samples
    assert "1" in samples
    assert samples["1"] == 100


def test_execute_with_measurement_gate(backend):
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))
    circuit.add(M(0))
    result = backend.execute(Sampling(circuit=circuit, nshots=50))
    # Still expect only '1'
    assert result.samples == {"1": 50}


def test_unsupported_gate_raises(backend):
    class FakeGate:
        target_qubits = [0]

    circuit = Circuit(nqubits=1)
    circuit.gates.append(FakeGate())
    with pytest.raises(UnsupportedGateError):
        backend.execute(Sampling(circuit=circuit))


def test_controlled_cnot(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(CNOT(control=0, target=1))
    # Expect no error on building or executing
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)
    # All samples should be "00" since X only applies if control=1, but no preparation
    assert result.samples == {"00": 10}


def test_nshots():
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)
    assert result.nshots == 10


basic_gate_test_cases = [
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


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
def test_adjoint_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    adjoint_gate = Adjoint(gate_instance)
    circuit._gates.append(adjoint_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name == "Adjoint_" + adjoint_gate.name for g in qutip_circuit.gates)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
def test_controlled_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    controlled_gate = Controlled(1, basic_gate=gate_instance)
    circuit._gates.append(controlled_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name == "Controlled_" + controlled_gate.name for g in qutip_circuit.gates)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
def test_handlers(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    circuit.add(gate_instance)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name == gate_instance.name for g in qutip_circuit.gates)


def test_constant_hamiltonian():
    x = 2.0
    schedule = Schedule(
        hamiltonians={"hz": x * pauli_z(0)},
        dt=0.1,
        T=1.0,
        schedule={i: {"hz": 1.0} for i in range(int(1.0 / 0.1))},
    )
    psi0 = ket(0)
    obs = [pauli_z(0)]
    backend = QutipBackend()
    res = backend.execute(
        TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs, store_intermediate_results=True)
    )

    assert isinstance(res, TimeEvolutionResult)

    assert pytest.approx(res.final_expected_values, rel=1e-6) == 1.0

    # Intermediate states should replicate constant behavior
    assert res.intermediate_states is not None
    for state in res.intermediate_states:
        psi = state.dense.flatten()
        assert pytest.approx(abs(psi[0]) ** 2, rel=1e-6) == 1.0


def test_time_dependent_hamiltonian():
    o = 1.0
    dt = 0.01
    T = 10

    steps = np.linspace(0, T, int(T / dt))

    schedule = Schedule(
        T,
        dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        schedule={t: {"h1": 1 - steps[t] / T, "h2": steps[t] / T} for t in range(len(steps))},
    )

    psi0 = (ket(0) + ket(1)).unit()
    obs = [
        pauli_z(0),  # measure z
    ]

    backend = QutipBackend()
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    expect_z = res.final_expected_values[0]
    assert pytest.approx(expect_z, rel=1e-2) == 1.0


def test_time_dependent_hamiltonian_with_3_qubits():
    dt = 0.01
    T = 50
    # steps = int(T / dt) - 1

    steps = np.linspace(0, T, int(T / dt))
    h1 = pauli_x(0) + pauli_x(1) + pauli_x(2)
    h2 = -1 * pauli_z(0) - 1 * pauli_z(1) - 2 * pauli_z(2) + 3 * pauli_z(0) * pauli_z(1)

    # Create a schedule for the time evolution
    schedule = Schedule(
        T,
        dt,
        hamiltonians={"h1": h1, "h2": h2},
        schedule={t: {"h1": 1 - steps[t] / T, "h2": steps[t] / T} for t in range(len(steps))},
    )

    psi0 = (ket(0) + ket(1)).unit()
    psi0 = tensor_prod([psi0, psi0, psi0]).unit()
    obs = [pauli_z(0), pauli_z(1), pauli_z(2)]  # measure z

    backend = QutipBackend()
    res = backend.execute(
        TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs, store_intermediate_results=False)
    )

    assert pytest.approx(res.final_expected_values[0], rel=1e-2) == -1.0
    assert pytest.approx(res.final_expected_values[1], rel=1e-2) == -1.0
    assert pytest.approx(res.final_expected_values[2], rel=1e-2) == -1.0


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
    optimizer.optimize.side_effect = lambda cost_function, init_parameters, store_intermediate_results: OptimizerResult(
        0.2, [0.9, 0.1]
    )
    return optimizer


def test_parameterized_program_properties_assignment(dummy_optimizer):
    """
    Test that the VQE instance correctly stores its initial properties.

    Verifies that the ansatz, initial parameters, and cost function are assigned properly.
    """
    mock_instance = MagicMock(spec=Model)
    ansatz = HardwareEfficientAnsatz(2)
    circuit = ansatz.get_circuit([0 for _ in range(ansatz.nparameters)])

    vqe = ParameterizedProgram(Sampling(circuit), dummy_optimizer, mock_instance)
    assert isinstance(vqe.functional, Sampling)
    assert vqe.functional.circuit == circuit
    assert vqe.optimizer == dummy_optimizer
    assert vqe.cost_model == mock_instance


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

    ansatz = HardwareEfficientAnsatz(2)
    circuit = ansatz.get_circuit([0 for _ in range(ansatz.nparameters)])

    vqe = ParameterizedProgram(Sampling(circuit), dummy_optimizer, mock_instance)
    # Call obtain_cost with a custom number of shots.
    backend = QutipBackend()
    output = backend.optimize(vqe)

    # The dummy_cost_function returns 0.7 regardless of input.
    assert output.optimal_cost == 0.2
    assert Sampling(circuit).compute_cost(output.optimal_execution_results, mock_instance) == 8.0


def test_real_example():
    backend = QutipBackend()
    b = BinaryVariable("b")
    model = Model("test")
    model.set_objective(2 * b - 1)

    cr = Circuit(1)
    cr.add(U1(0, phi=0.1))

    output = backend.optimize(ParameterizedProgram(Sampling(cr), SciPyOptimizer(), model))
    assert output.optimal_cost == -1
    assert output.optimal_execution_results.samples == {"0": 1000}
