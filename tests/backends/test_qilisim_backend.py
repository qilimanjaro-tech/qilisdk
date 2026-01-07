from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core.model import Constraint, Model, Objective
from qilisdk.core.qtensor import ket, tensor_prod
from qilisdk.core.variables import BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.digital import RX, RY, RZ, SWAP, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.gates import CNOT, Controlled
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer


@pytest.fixture
def backend():
    return QiliSim()


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


def test_controlled_cnot(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(CNOT(control=0, target=1))
    # Expect no error on building or executing
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)
    # All samples should be "00" since X only applies if control=1, but no preparation
    assert result.samples == {"00": 10}


def test_nshots():
    backend = QiliSim()
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)
    assert result.nshots == 10


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


@pytest.mark.parametrize("gate", [basic_gate for basic_gate, _ in basic_gate_test_cases])
def test_basic_gates(backend, gate):
    circuit = Circuit(nqubits=1)
    circuit.add(gate)
    # Expect no error on building or executing
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    assert isinstance(result, SamplingResult)


def test_multi_controlled_execution():
    # Create two Xs then a multi-controlled X (Toffoli) gate
    # Expect roughly all shots to be '111'
    backend = QiliSim()
    circuit = Circuit(nqubits=3)
    circuit.add(X(0))
    circuit.add(X(1))
    circuit.add(Controlled(0, 1, basic_gate=X(2)))
    result = backend.execute(Sampling(circuit=circuit, nshots=100))
    assert isinstance(result, SamplingResult)
    samples = result.samples
    assert "111" in samples
    assert samples["111"] == 100


def test_constant_hamiltonian():
    x = 2.0
    schedule = Schedule(
        hamiltonians={"hz": x * pauli_z(0)},
        dt=1,
        total_time=10,
        coefficients={"hz": dict.fromkeys(range(int(1.0 / 0.1)), 1.0)},
    )
    psi0 = ket(0)
    obs = [pauli_z(0)]
    backend = QiliSim()
    res = backend.execute(
        TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs, store_intermediate_results=True)
    )

    assert isinstance(res, TimeEvolutionResult)
    assert np.isclose(res.final_expected_values[0], 1.0, rtol=1e-6)

    # Intermediate states should replicate constant behavior
    assert res.intermediate_states is not None
    for state in res.intermediate_states:
        psi = state.dense().flatten()
        assert np.isclose(abs(psi[0]) ** 2, 1.0, rtol=1e-6)


simulation_types = ["direct", "arnoldi", "integrate"]


@pytest.mark.parametrize("method", simulation_types)
def test_time_dependent_hamiltonian(method):
    o = 1.0
    dt = 0.1
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    obs = [
        pauli_z(0),  # measure z
    ]

    backend = QiliSim(evolution_method=method)
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    expect_z = res.final_expected_values[0]
    assert res.final_state.is_ket()
    assert np.isclose(expect_z, -1.0, rtol=1e-2)


@pytest.mark.parametrize("method", simulation_types)
def test_time_dependent_hamiltonian_density_mat(method):
    o = 1.0
    dt = 0.1
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    psi0 = psi0.to_density_matrix()
    obs = [
        pauli_z(0),  # measure z
    ]

    backend = QiliSim(evolution_method=method)
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    expect_z = res.final_expected_values[0]
    assert res.final_state.shape == (2, 2)
    assert np.isclose(expect_z, -1.0, rtol=1e-2)


@pytest.mark.parametrize("method", simulation_types)
def test_monte_carlo_time_evolution(method):
    o = 1.0
    dt = 0.1
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = ((ket(0) - ket(1)).unit()).to_density_matrix()
    psi1 = ket(1).to_density_matrix()
    mix = 0.2
    psi0 = ((1 - mix) * psi0 + mix * psi1).unit()
    obs = [
        pauli_z(0),  # measure z
    ]

    backend = QiliSim(evolution_method=method, monte_carlo=True)
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)

    expect_z = res.final_expected_values[0]
    assert res.final_state.shape == (2, 2)
    assert np.isclose(expect_z, -0.8, rtol=1e-2)


def test_qilisim_params():
    with pytest.raises(ValueError, match="Unknown time evolution method: something-else"):
        QiliSim(evolution_method="something-else")

    with pytest.raises(ValueError, match="num_arnoldi_substeps must be a positive integer"):
        QiliSim(num_arnoldi_substeps=-1)

    with pytest.raises(ValueError, match="arnoldi_dim must be a positive integer"):
        QiliSim(arnoldi_dim=0)

    with pytest.raises(ValueError, match="num_integrate_substeps must be a positive integer"):
        QiliSim(num_integrate_substeps=-5)

    with pytest.raises(ValueError, match="num_monte_carlo_trajectories must be a positive integer"):
        QiliSim(num_monte_carlo_trajectories=0)


def test_time_dependent_hamiltonian_with_3_qubits():
    dt = 0.01
    T = 50

    h1 = pauli_x(0) + pauli_x(1) + pauli_x(2)
    h2 = -1 * pauli_z(0) - 1 * pauli_z(1) - 2 * pauli_z(2) + 3 * pauli_z(0) * pauli_z(1)

    # Create a schedule for the time evolution
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": h1, "h2": h2},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) + ket(1)).unit()
    psi0 = tensor_prod([psi0, psi0, psi0]).unit()
    obs = [pauli_z(0), pauli_z(1), pauli_z(2)]  # measure z

    backend = QiliSim()
    res = backend.execute(
        TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs, store_intermediate_results=False)
    )

    assert np.isclose(res.final_expected_values[0], -1.0, rtol=1e-2)
    assert np.isclose(res.final_expected_values[1], -1.0, rtol=1e-2)
    assert np.isclose(res.final_expected_values[2], -1.0, rtol=1e-2)


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
    backend = QiliSim()
    output = backend.execute(parameterized_program)

    assert np.isclose(output.optimal_cost, 0.2)
    assert np.isclose(cost_function.compute_cost(output.optimal_execution_results), 8.0)


def test_real_example():
    backend = QiliSim()
    b = BinaryVariable("b")
    model = Model("test")
    model.set_objective(2 * b - 1)

    cr = Circuit(1)
    cr.add(U1(0, phi=0.1))

    output = backend.execute(VariationalProgram(Sampling(cr), SciPyOptimizer(), ModelCostFunction(model)))
    assert output.optimal_cost == -1
    assert output.optimal_execution_results.samples == {"0": 1000}
