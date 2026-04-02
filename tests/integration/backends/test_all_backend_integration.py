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

from qilisdk.backends.backend import Backend

pytest.importorskip("qutip", reason="QuTiP backend tests require the 'qutip' optional dependency", exc_type=ImportError)
pytest.importorskip(
    "qutip_qip",
    reason="QuTiP backend tests require the 'qutip' optional dependency",
    exc_type=ImportError,
)
pytest.importorskip("qilisim_module", reason="Backend integration tests require the 'qilisim_module' C++ extension")

from unittest.mock import MagicMock

import pytest

from qilisdk.analog.hamiltonian import Hamiltonian, PauliX, PauliZ
from qilisdk.analog.hamiltonian import X as pauli_x
from qilisdk.analog.hamiltonian import Y as pauli_y
from qilisdk.analog.hamiltonian import Z as pauli_z
from qilisdk.analog.schedule import Schedule
from qilisdk.backends import QutipBackend
from qilisdk.backends.backend_config import ExecutionConfig
from qilisdk.backends.qilisim import QiliSim
from qilisdk.core.model import Constraint, Model, Objective
from qilisdk.core.qtensor import QTensor, ket, tensor_prod
from qilisdk.core.variables import BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.digital import RX, RY, RZ, SWAP, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.ansatz import HardwareEfficientAnsatz, TrotterizedSchedule
from qilisdk.digital.gates import CNOT, Controlled
from qilisdk.functionals import AnalogEvolution, DigitalPropagation, FunctionalResult, QuantumReservoir, ReservoirLayer
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
from qilisdk.readout import ReadoutSpec, StateTomographyReadout

pytest.importorskip(
    "cudaq",
    reason="CUDA backend tests require the 'cuda' optional dependency",
    exc_type=ImportError,
)

from qilisdk.backends.cuda_backend import CudaBackend

backends = [QutipBackend(), QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1))]
if pytest.importorskip(
    "cudaq",
    reason="CUDA backend tests require the 'cuda' optional dependency",
    exc_type=ImportError,
):
    backends.append(CudaBackend())
backends_no_cuda = [QutipBackend(), QiliSim(execution_config=ExecutionConfig(seed=42, num_threads=1))]


def _build_quantum_reservoir_functional() -> QuantumReservoir:
    schedule = Schedule(
        dt=1,
        hamiltonians={"h": pauli_z(0)},
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


@pytest.mark.parametrize("backend", backends)
def test_execute_simple_circuit_no_measurement(backend):
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=100))
    assert isinstance(result, FunctionalResult)
    samples = result.samples
    assert "1" in samples
    assert samples["1"] == 100


@pytest.mark.parametrize("backend", backends)
def test_execute_with_measurement_gate(backend):
    circuit = Circuit(nqubits=1)
    circuit.add(X(0))
    circuit.add(M(0))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=50))
    assert result.samples == {"1": 50}


@pytest.mark.parametrize("backend", backends)
def test_controlled_cnot(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(CNOT(control=0, target=1))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert result.samples == {"00": 10}


@pytest.mark.parametrize("backend", backends)
def test_nshots(backend):
    circuit = Circuit(nqubits=1)
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    total_shots = sum(result.samples.values())
    assert total_shots == 10


@pytest.mark.parametrize("backend", backends)
def test_multi_controlled_execution(backend):
    if type(backend) is QiliSim and backend.get_config()["sampling_method"] == "statevector_matrix_free":
        pytest.skip(
            "Multi-controlled gates are not currently supported in statevector_matrix_free sampling method of QiliSim."
        )
    circuit = Circuit(nqubits=3)
    circuit.add(X(0))
    circuit.add(X(1))
    circuit.add(Controlled(0, 1, basic_gate=X(2)))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=100))
    assert isinstance(result, FunctionalResult)
    samples = result.samples
    assert "111" in samples
    assert samples["111"] == 100


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_constant_hamiltonian(backend):
    x = 2.0
    schedule = Schedule(
        hamiltonians={"hz": x * pauli_z(0)},
        dt=1,
        total_time=10,
        coefficients={"hz": dict.fromkeys(range(int(1.0 / 0.1)), 1.0)},
    )
    psi0 = ket(0)

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0, store_intermediate_results=True),
        ReadoutSpec().with_expectation(observables=[pauli_z(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    assert np.isclose(res.expected_values[0], 1.0, rtol=1e-6)

    # Intermediate states should replicate constant behavior
    assert len(res.intermediate_results) > 0
    for inter in res.intermediate_results:
        state = inter.state_tomography.state
        psi = state.dense().flatten()
        assert np.isclose(abs(psi[0]) ** 2, 1.0, rtol=1e-6)


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_time_dependent_hamiltonian(backend):
    o = 1.0
    dt = 1
    T = 1000

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        ReadoutSpec().with_expectation(observables=[pauli_z(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    expect_z = res.expected_values[0]
    assert np.isclose(expect_z, -1.0, rtol=1e-2)


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_time_dependent_hamiltonian_with_3_qubits(backend):
    dt = 0.01
    T = 50

    h1 = pauli_x(0) + pauli_x(1) + pauli_x(2)
    h2 = -1 * pauli_z(0) - 1 * pauli_z(1) - 2 * pauli_z(2) + 3 * pauli_z(0) * pauli_z(1)

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": h1, "h2": h2},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) + ket(1)).unit()
    psi0 = tensor_prod([psi0, psi0, psi0]).unit()

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0, store_intermediate_results=False),
        ReadoutSpec().with_expectation(observables=[pauli_z(0), pauli_z(1), pauli_z(2)]),
    )

    assert np.isclose(res.expected_values[0], -1.0, rtol=1e-2)
    assert np.isclose(res.expected_values[1], -1.0, rtol=1e-2)
    assert np.isclose(res.expected_values[2], -1.0, rtol=1e-2)


@pytest.mark.parametrize("backend", backends)
def test_real_example(backend):
    b = BinaryVariable("b")
    model = Model("test")
    model.set_objective(2 * b - 1)

    cr = Circuit(1)
    cr.add(U1(0, phi=0.1))

    readout = ReadoutSpec().with_sampling(nshots=1000)
    output = backend.execute(
        VariationalProgram(DigitalPropagation(cr), SciPyOptimizer(), ModelCostFunction(model)),
        readout,
    )
    assert np.isclose(output.optimal_cost, -1.0, rtol=1e-6)
    assert output.optimal_execution_results.samples == {"0": 1000}


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_trotterized_time_evolution_results(backend: Backend):
    """TrotterizedSchedule should honor schedule dt and trotter_steps."""

    h0 = Hamiltonian({(PauliX(0),): -1})
    h1 = Hamiltonian({(PauliZ(0),): 1})
    schedule = Schedule(
        hamiltonians={"h0": h0, "h1": h1},
        coefficients={"h0": {(0, 1): lambda t: 1 - t}, "h1": {(0, 1): lambda t: t}},
        dt=0.01,
        total_time=10,
    )
    ansatz = TrotterizedSchedule(schedule, trotter_steps=5)
    ansatz.insert([H(0)], 0)
    te_res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=(ket(0) + ket(1)).unit()),
        ReadoutSpec().with_state_tomography(),
    )
    nshots = 10_000
    sam_res = backend.execute(DigitalPropagation(ansatz), ReadoutSpec().with_sampling(nshots=nshots))
    assert all(np.isclose(list(te_res.probabilities.values()), list(sam_res.probabilities.values()), atol=1e-2))


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
@pytest.mark.parametrize("backend", backends)
def test_basic_gates(backend, gate):
    circuit = Circuit(nqubits=1)
    circuit.add(gate)
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)


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


@pytest.mark.parametrize("backend", backends)
def test_obtain_cost_calls_backend(dummy_optimizer, backend):
    mock_instance = MagicMock(spec=Model)
    mock_instance.variables = MagicMock(return_value=[BinaryVariable("b0"), BinaryVariable("b1")])

    mock_objective = MagicMock(spec=Objective)
    mock_objective.label = "obj"

    mock_con = MagicMock(spec=Constraint)
    mock_con.label = "con1"

    mock_instance.objective = mock_objective
    mock_instance.constraints = [mock_con]
    mock_instance.evaluate.return_value = {"obj": -2, "con1": 10}

    circuit = HardwareEfficientAnsatz(2)

    cost_function = ModelCostFunction(mock_instance)
    readout = ReadoutSpec().with_sampling(nshots=1000)
    parameterized_program = VariationalProgram(DigitalPropagation(circuit), dummy_optimizer, cost_function)
    output = backend.execute(parameterized_program, readout)

    assert np.isclose(output.optimal_cost, 0.2)
    assert np.isclose(cost_function.compute_cost(output.optimal_execution_results), 8.0)


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_time_dependent_hamiltonian_pauli_observable(backend):
    o = 1.0
    dt = 0.5
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * (pauli_x(0) + pauli_x(1)), "h2": o * (pauli_z(0) + pauli_z(1))},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    psi0 = tensor_prod([psi0, ket(0)]).unit()

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        ReadoutSpec().with_expectation(observables=[pauli_z(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    expect_z = res.expected_values[0]
    assert res.state.is_ket()
    assert np.isclose(expect_z, -1.0, rtol=1e-2)


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_time_dependent_hamiltonian_imaginary(backend):
    o = 1.0
    dt = 0.5
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_y(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()
    psi0 = psi0.to_density_matrix()

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        ReadoutSpec().with_expectation(observables=[pauli_y(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    expect_y = res.expected_values[0]
    assert res.state.shape == (2, 2)
    assert np.isclose(expect_y, -1.0, rtol=1e-2)

    # check that it's hermitian
    final_rho = res.state.dense()
    assert np.allclose(final_rho, final_rho.conj().T, rtol=1e-6)


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_time_dependent_hamiltonian_qtensor_observable(backend):
    o = 1.0
    dt = 0.5
    T = 100

    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )

    psi0 = (ket(0) - ket(1)).unit()

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        ReadoutSpec().with_expectation(observables=[QTensor(pauli_z(0).to_matrix())]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    expect_z = res.expected_values[0]
    assert res.state.is_ket()
    assert np.isclose(expect_z, -1.0, rtol=1e-2)


@pytest.mark.parametrize("backend", backends)
def test_cnot(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(CNOT(control=0, target=1))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert result.samples == {"00": 10}


@pytest.mark.parametrize("backend", backends)
def test_multiple_parameterized_gates(backend):
    c = Circuit(nqubits=1)
    c.add(RX(qubit=0, theta=np.pi / 4))
    c.add(RX(qubit=0, theta=np.pi / 4))
    c.add(RX(qubit=0, theta=np.pi / 2))
    result = backend.execute(DigitalPropagation(circuit=c), ReadoutSpec().with_sampling(nshots=100))
    assert isinstance(result, FunctionalResult)
    samples = result.samples
    assert "1" in samples
    assert samples["1"] == 100


@pytest.mark.parametrize("backend", backends)
def test_many_gates(backend):
    c = Circuit.random(nqubits=2, single_qubit_gates={H, X, Y, Z, T, RX, RZ}, two_qubit_gates={CNOT}, ngates=1000)
    result = backend.execute(DigitalPropagation(circuit=c), ReadoutSpec().with_sampling(nshots=1000))
    assert isinstance(result, FunctionalResult)


@pytest.mark.parametrize("backend", backends)
def test_measurement_gates(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(X(0))
    circuit.add(M(0))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=50))
    assert isinstance(result, FunctionalResult)
    samples = result.samples
    assert "1" in samples
    assert samples["1"] == 50


@pytest.mark.parametrize("backend", backends_no_cuda)
def test_time_dependent_hamiltonian_density_mat(backend):
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

    res = backend.execute(
        AnalogEvolution(schedule=schedule, initial_state=psi0),
        ReadoutSpec().with_expectation(observables=[pauli_z(0)]).with_state_tomography(),
    )

    assert isinstance(res, FunctionalResult)
    expect_z = res.expected_values[0]
    assert res.state.shape == (2, 2)
    assert np.isclose(expect_z, -1.0, rtol=1e-2)

    # check that it's hermitian
    final_rho = res.state.dense()
    assert np.allclose(final_rho, final_rho.conj().T, rtol=1e-6)


def test_execute_quantum_reservoir_qutip():
    backend = QutipBackend()
    functional = _build_quantum_reservoir_functional()
    readout = [StateTomographyReadout()]

    result = backend._execute_quantum_reservoir(functional, readout)

    assert result.state is not None
    assert len(result) >= 2


@pytest.mark.parametrize("backend", backends)
def test_cnot_on_one(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(X(0))
    circuit.add(CNOT(control=0, target=1))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert result.samples == {"11": 10}


@pytest.mark.parametrize("backend", backends)
def test_controlled_u3(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(U3(1, theta=1.0, phi=0.5, gamma=0.25).controlled(0))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert result.samples == {"00": 10}


@pytest.mark.parametrize("backend", backends)
def test_swap(backend):
    circuit = Circuit(nqubits=2)
    circuit.add(X(0))
    circuit.add(SWAP(0, 1))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert result.samples == {"01": 10}


@pytest.mark.parametrize("backend", backends)
def test_toffoli(backend):
    circuit = Circuit(nqubits=3)
    circuit.add(X(0))
    circuit.add(X(1))
    circuit.add(X(2).controlled(0, 1))
    result = backend.execute(DigitalPropagation(circuit=circuit), ReadoutSpec().with_sampling(nshots=10))
    assert isinstance(result, FunctionalResult)
    assert result.samples == {"111": 10}
