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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qilisdk.analog.hamiltonian import Hamiltonian
from qilisdk.core.qtensor import ket
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

pytest.importorskip(
    "cudaq",
    reason="CUDA backend tests require the 'cuda' optional dependency",
    exc_type=ImportError,
)


from qilisdk.analog import I as pauli_i
from qilisdk.analog import Schedule
from qilisdk.analog import X as pauli_x
from qilisdk.analog import Y as pauli_y
from qilisdk.analog import Z as pauli_z
from qilisdk.analog.hamiltonian import PauliI, PauliX, PauliY, PauliZ
from qilisdk.backends.cuda_backend import CudaBackend, CudaSamplingMethod
from qilisdk.core.model import Model
from qilisdk.core.variables import BinaryVariable
from qilisdk.cost_functions.model_cost_function import ModelCostFunction
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.circuit import Circuit
from qilisdk.digital.circuit_transpiler_passes import DecomposeMultiControlledGatesPass
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import RX, RY, RZ, SWAP, U1, U2, U3, Adjoint, BasicGate, Controlled, H, I, M, S, T, X, Y, Z
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.optimizers.scipy_optimizer import SciPyOptimizer
from qilisdk.settings import Precision, get_settings

COMPLEX_DTYPE = get_settings().complex_precision.dtype

# --- Dummy classes and helper functions ---


class DummyKernel:
    """A dummy kernel that records method calls."""

    def __init__(self):
        self.calls = []
        self.qubits = []

    def qalloc(self, n):
        self.qubits = [f"q{i}" for i in range(n)]
        return self.qubits

    def i(self, qubit):
        self.calls.append(("i", qubit))

    def x(self, qubit):
        self.calls.append(("x", qubit))

    def y(self, qubit):
        self.calls.append(("y", qubit))

    def z(self, qubit):
        self.calls.append(("z", qubit))

    def h(self, qubit):
        self.calls.append(("h", qubit))

    def s(self, qubit):
        self.calls.append(("s", qubit))

    def t(self, qubit):
        self.calls.append(("t", qubit))

    def rx(self, angle, qubit):
        self.calls.append(("rx", angle, qubit))

    def ry(self, angle, qubit):
        self.calls.append(("ry", angle, qubit))

    def rz(self, angle, qubit):
        self.calls.append(("rz", angle, qubit))

    def u3(self, theta, phi, delta, target):
        self.calls.append(("u3", theta, phi, delta, target))

    def swap(self, qubit_0, qubit_1):
        self.calls.append(("swap", qubit_0, qubit_1))

    def control(self, target_kernel, control_qubit, target_qubit):
        self.calls.append(("control", control_qubit, target_qubit))

    def adjoint(self, target_kernel, target_qubit):
        self.calls.append(("adjoint", target_qubit))

    def mz(self, qubit):
        self.calls.append(("mz", qubit))


# This dummy make_kernel function returns a singleton when called without arguments.
def dummy_make_kernel(*args, **kwargs):
    if not args:
        return dummy_make_kernel.main_kernel
    return DummyKernel(), "dummy_qubit"


dummy_make_kernel.main_kernel = DummyKernel()


class DummyGate(BasicGate):
    """A dummy basic gate to trigger unsupported-gate errors."""

    def __init__(self, qubit: int) -> None:
        super().__init__((qubit,))

    @property
    def name(self) -> str: ...  # type: ignore

    def _generate_matrix(self) -> np.ndarray:
        return np.eye(2, dtype=COMPLEX_DTYPE)


# --- Parameterized test cases for basic gate handler ---
# For each case, we create an instance and note the expected call on the dummy kernel.
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
swap_test_case: list[tuple[BasicGate, tuple]] = [(SWAP(0, 1), ("swap", "q0", "q1"))]


# --- Simulation method tests ---


@patch("cudaq.num_available_gpus", return_value=0)
@patch("cudaq.set_target")
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
def test_state_vector_no_gpu(mock_sample, mock_make_kernel, mock_set_target, mock_num_gpus):
    backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit=circuit, nshots=10))
    mock_set_target.assert_called_with("qpp-cpu")
    assert isinstance(result, SamplingResult)
    assert result.samples == {"0": 1000}
    assert result.nshots == 10


@patch("cudaq.num_available_gpus", return_value=1)
@patch("cudaq.set_target")
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
def test_state_vector_with_gpu(mock_sample, mock_make_kernel, mock_set_target, mock_num_gpus):
    backend = CudaBackend(sampling_method=CudaSamplingMethod.STATE_VECTOR)
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit, nshots=10))
    float_precision = "fp32" if get_settings().complex_precision == Precision.COMPLEX_64 else "fp64"
    mock_set_target.assert_called_with("nvidia", option=float_precision)
    assert isinstance(result, SamplingResult)
    assert result.samples == {"0": 1000}
    assert result.nshots == 10


@patch("cudaq.set_target")
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
def test_tensornet(mock_sample, mock_make_kernel, mock_set_target):
    backend = CudaBackend(sampling_method=CudaSamplingMethod.TENSOR_NETWORK)
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit, nshots=10))
    mock_set_target.assert_called_with("tensornet")
    assert isinstance(result, SamplingResult)
    assert result.samples == {"0": 1000}
    assert result.nshots == 10


@patch("cudaq.set_target")
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
def test_matrix_product_state(mock_sample, mock_make_kernel, mock_set_target):
    backend = CudaBackend(sampling_method=CudaSamplingMethod.MATRIX_PRODUCT_STATE)
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit, nshots=10))
    mock_set_target.assert_called_with("tensornet-mps")
    assert isinstance(result, SamplingResult)
    assert result.samples == {"0": 1000}
    assert result.nshots == 10


# --- Parameterized tests for basic gate execution ---


@pytest.mark.parametrize(("gate_instance", "expected_call"), basic_gate_test_cases + swap_test_case)
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_basic_gate_handler(mock_set_target, mock_sample, mock_make_kernel, gate_instance, expected_call):
    # Reset the main dummy kernel for a clean slate.
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=2)
    circuit._gates.append(gate_instance)
    backend.execute(Sampling(circuit, nshots=10))
    calls = dummy_make_kernel.main_kernel.calls
    assert expected_call in calls


# --- Parameterized tests for controlled gate execution ---
# In controlled mode the main kernel should receive a ('control', 'q0', 'q1') call.
@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_controlled_handler(mock_set_target, mock_sample, mock_make_kernel, gate_instance):
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=2)
    controlled_gate = Controlled(1, basic_gate=gate_instance)
    circuit._gates.append(controlled_gate)
    backend.execute(Sampling(circuit, nshots=10))
    calls = dummy_make_kernel.main_kernel.calls
    assert ("control", "q1", "q0") in calls


# --- Parameterized tests for adjoint gate execution ---
# In adjoint mode the main kernel should receive an ('adjoint', 'q0') call.
@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_adjoint_handler(mock_set_target, mock_sample, mock_make_kernel, gate_instance):
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=1)
    adjoint_gate = Adjoint(gate_instance)
    circuit._gates.append(adjoint_gate)
    backend.execute(Sampling(circuit, nshots=10))
    calls = dummy_make_kernel.main_kernel.calls
    assert ("adjoint", "q0") in calls


# --- Tests for measurement handling ---


@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_measurement_full(mock_set_target, mock_sample, mock_make_kernel):
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=2)
    measurement_gate = M(0, 1)
    circuit._gates.append(measurement_gate)
    backend.execute(Sampling(circuit, nshots=10))
    calls = dummy_make_kernel.main_kernel.calls
    # Full measurement: kernel.mz is called once with the full qubit list.
    assert ("mz", dummy_make_kernel.main_kernel.qubits) in calls


@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_measurement_partial(mock_set_target, mock_sample, mock_make_kernel):
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=3)
    measurement_gate = M(1, 2)
    circuit._gates.append(measurement_gate)
    backend.execute(Sampling(circuit, nshots=10))
    calls = dummy_make_kernel.main_kernel.calls
    assert ("mz", dummy_make_kernel.main_kernel.qubits[1]) in calls
    assert ("mz", dummy_make_kernel.main_kernel.qubits[2]) in calls


# --- Tests for unsupported gate errors ---


@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_unsupported_gate(mock_set_target, mock_sample, mock_make_kernel):
    backend = CudaBackend()
    circuit = Circuit(nqubits=1)
    circuit._gates.append(DummyGate(0))
    with pytest.raises(UnsupportedGateError):
        backend.execute(Sampling(circuit, nshots=10))


def test_controlled_with_unsupported_basic_gate_raises(monkeypatch):
    class BadGate(BasicGate):
        name = "Bad"

        def __init__(self, q=0):
            super().__init__((q,))

        def _generate_matrix(self):
            return np.eye(2)

    be = CudaBackend()
    circuit = Circuit(2)  # small helper from Backend superclass
    circuit._gates.append(Controlled(1, basic_gate=BadGate(0)))

    with pytest.raises(UnsupportedGateError):
        be.execute(Sampling(circuit=circuit, nshots=10))


@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_controlled_multiple_controls_are_transpiled(mock_set_target, mock_sample, mock_make_kernel):
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=3)
    controlled_gate = Controlled(0, 1, basic_gate=X(2))
    circuit._gates.append(controlled_gate)

    transpiled = DecomposeMultiControlledGatesPass().run(circuit)
    expected_control_calls = [
        ("control", f"q{gate.control_qubits[0]}", f"q{gate.target_qubits[0]}")
        for gate in transpiled.gates
        if isinstance(gate, Controlled)
    ]

    backend.execute(Sampling(circuit, nshots=10))
    actual_control_calls = [call for call in dummy_make_kernel.main_kernel.calls if call[0] == "control"]

    assert actual_control_calls == expected_control_calls


@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_adjoint_unsupported_gate_error(mock_set_target, mock_sample, mock_make_kernel):
    backend = CudaBackend()
    circuit = Circuit(nqubits=1)
    adjoint_gate = Adjoint(DummyGate(0))
    circuit._gates.append(adjoint_gate)
    with pytest.raises(UnsupportedGateError):
        backend.execute(Sampling(circuit, nshots=10))


def test_hamiltonian_to_cuda_computes_expected_sum(monkeypatch):
    be = CudaBackend()

    # Replace the Pauli → spin handler mapping with predictable numbers
    be._pauli_operator_handlers = {
        PauliX: lambda op: 2,
        PauliY: lambda op: 3,
        PauliZ: lambda op: 4,
        PauliI: lambda op: 1,
    }

    # Minimal dummy “Hamiltonian” iterable
    class DummyHam(Hamiltonian):
        def __iter__(self):
            # note: 2 * 2  +  3 * (3*4)  = 4 + 36 = 40
            yield 2, [PauliX(0)]
            yield 3, [PauliY(0), PauliZ(0)]

    assert be._hamiltonian_to_cuda(DummyHam()) == 40


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
    Test that the parameterized_program instance correctly stores its initial properties.

    Verifies that the ansatz, initial parameters, and cost function are assigned properly.
    """
    mock_instance = MagicMock(spec=ModelCostFunction)
    circuit = HardwareEfficientAnsatz(2)

    parameterized_program = VariationalProgram(Sampling(circuit), dummy_optimizer, mock_instance)
    assert isinstance(parameterized_program.functional, Sampling)
    assert parameterized_program.functional.circuit == circuit
    assert parameterized_program.optimizer == dummy_optimizer
    assert parameterized_program.cost_function == mock_instance


def test_real_example():
    backend = CudaBackend()
    b = BinaryVariable("b")
    model = Model("test")
    model.set_objective(2 * b - 1)

    cr = Circuit(1)
    cr.add(U1(0, phi=0.1))

    output = backend.execute(VariationalProgram(Sampling(cr), SciPyOptimizer(), ModelCostFunction(model)))
    assert output.optimal_cost == -1
    assert output.optimal_execution_results.samples == {"0": 1000}


def test_integer_gates():
    backend = CudaBackend()
    circuit = Circuit(1)
    circuit.add(RX(0, theta=1))
    circuit.add(RY(0, theta=1))
    circuit.add(RZ(0, phi=1))
    circuit.add(U1(0, phi=1))
    circuit.add(U2(0, phi=1, gamma=1))
    circuit.add(U3(0, theta=1, phi=1, gamma=1))
    result = backend.execute(Sampling(circuit, nshots=1000))
    assert isinstance(result, SamplingResult)


def test_multi_qubit_controls_no_decompose(monkeypatch):
    # need to patch DecomposeMultiControlledGatesPass to not decompose
    monkeypatch.setattr(
        "qilisdk.digital.circuit_transpiler_passes.DecomposeMultiControlledGatesPass.run", lambda self, circuit: circuit
    )

    backend = CudaBackend()
    circuit = Circuit(3)
    gate = Controlled(0, 1, basic_gate=X(2))
    assert gate.control_qubits == (0, 1)
    circuit.add(gate)
    with pytest.raises(UnsupportedGateError):
        backend.execute(Sampling(circuit, nshots=1000))


def test_time_dependent_hamiltonian_cuda(monkeypatch):
    # monkeypatch the evolve that we import from cudaq in cuda_backend
    dummy_return = MagicMock()
    dummy_return.final_state = MagicMock(return_value=np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]))
    dummy_evolve = MagicMock(return_value=dummy_return)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.evolve", dummy_evolve)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.cudaq.set_target", lambda target: None)
    dummy_state = MagicMock(return_value=None)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.State.from_data", dummy_state)

    o = 1.0
    dt = 1
    T = 1000
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0) + pauli_y(0), "h2": o * pauli_z(0) + pauli_i(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )
    psi0 = (ket(0) - ket(1)).unit()
    obs = [
        pauli_z(0),
        PauliZ(0),
    ]

    backend = CudaBackend()
    res = backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))

    assert isinstance(res, TimeEvolutionResult)
    assert dummy_evolve.called
    assert dummy_state.called


def test_bad_observable_raises(monkeypatch):
    # monkeypatch the evolve that we import from cudaq in cuda_backend
    dummy_return = MagicMock()
    dummy_return.final_state = MagicMock(return_value=np.array([1 / np.sqrt(2), -1 / np.sqrt(2)]))
    dummy_evolve = MagicMock(return_value=dummy_return)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.evolve", dummy_evolve)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.cudaq.set_target", lambda target: None)
    dummy_state = MagicMock(return_value=None)
    monkeypatch.setattr("qilisdk.backends.cuda_backend.State.from_data", dummy_state)

    o = 1.0
    dt = 1
    T = 1000
    schedule = Schedule(
        dt=dt,
        hamiltonians={"h1": o * pauli_x(0), "h2": o * pauli_z(0)},
        coefficients={"h1": {(0, T): lambda t: 1 - t / T}, "h2": {(0, T): lambda t: t / T}},
    )
    psi0 = (ket(0) - ket(1)).unit()
    obs = [
        "bad observable",  # measure z
    ]

    backend = CudaBackend()
    with pytest.raises(ValueError, match="unsupported observable type"):
        backend.execute(TimeEvolution(schedule=schedule, initial_state=psi0, observables=obs))
