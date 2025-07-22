from unittest.mock import patch

import numpy as np
import pytest

from qilisdk.analog.hamiltonian import PauliI, PauliX, PauliY, PauliZ
from qilisdk.digital import (
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    Circuit,
    H,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import Adjoint, BasicGate, Controlled
from qilisdk.digital.sampling import Sampling
from qilisdk.digital.sampling_result import SamplingResult
from qilisdk.extras.cuda.cuda_backend import CudaBackend, DigitalSimulationMethod

# --- Dummy classes and helper functions ---


class DummyKernel:
    """A dummy kernel that records method calls."""

    def __init__(self):
        self.calls = []
        self.qubits = []

    def qalloc(self, n):
        self.qubits = [f"q{i}" for i in range(n)]
        return self.qubits

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


dummy_make_kernel.main_kernel = DummyKernel()  # type: ignore[attr-defined]


class DummyGate(BasicGate):
    """A dummy basic gate to trigger unsupported-gate errors."""

    def __init__(self, qubit: int) -> None:
        super().__init__((qubit,))

    @property
    def name(self) -> str:
        return "Dummy"

    def _generate_matrix(self) -> np.ndarray:
        return np.eye(2, dtype=complex)


# --- Parameterized test cases for basic gate handler ---
# For each case, we create an instance and note the expected call on the dummy kernel.
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


# --- Simulation method tests ---


@patch("cudaq.num_available_gpus", return_value=0)
@patch("cudaq.set_target")
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
def test_state_vector_no_gpu(mock_sample, mock_make_kernel, mock_set_target, mock_num_gpus):
    backend = CudaBackend(digital_simulation_method=DigitalSimulationMethod.STATE_VECTOR)
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
    backend = CudaBackend(digital_simulation_method=DigitalSimulationMethod.STATE_VECTOR)
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit, nshots=10))
    mock_set_target.assert_called_with("nvidia")
    assert isinstance(result, SamplingResult)
    assert result.samples == {"0": 1000}
    assert result.nshots == 10


@patch("cudaq.set_target")
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
def test_tensornet(mock_sample, mock_make_kernel, mock_set_target):
    backend = CudaBackend(digital_simulation_method=DigitalSimulationMethod.TENSOR_NETWORK)
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
    backend = CudaBackend(digital_simulation_method=DigitalSimulationMethod.MATRIX_PRODUCT_STATE)
    circuit = Circuit(nqubits=1)
    result = backend.execute(Sampling(circuit, nshots=10))
    mock_set_target.assert_called_with("tensornet-mps")
    assert isinstance(result, SamplingResult)
    assert result.samples == {"0": 1000}
    assert result.nshots == 10


# --- Parameterized tests for basic gate execution ---


@pytest.mark.parametrize(("gate_instance", "expected_call"), basic_gate_test_cases)
@patch("cudaq.make_kernel", side_effect=dummy_make_kernel)
@patch("cudaq.sample", return_value={"0": 1000})
@patch("cudaq.set_target")
def test_execute_basic_gate_handler(mock_set_target, mock_sample, mock_make_kernel, gate_instance, expected_call):
    # Reset the main dummy kernel for a clean slate.
    dummy_make_kernel.main_kernel = DummyKernel()
    backend = CudaBackend()
    circuit = Circuit(nqubits=1)
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
def test_controlled_multiple_controls_error(mock_set_target, mock_sample, mock_make_kernel):
    backend = CudaBackend()
    circuit = Circuit(nqubits=3)
    controlled_gate = Controlled(0, 1, basic_gate=X(2))
    circuit._gates.append(controlled_gate)
    with pytest.raises(UnsupportedGateError):
        backend.execute(Sampling(circuit, nshots=10))


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


@patch("qilisdk.extras.cuda.cuda_backend.spin.x", lambda *, target: f"x{target}")
@patch("qilisdk.extras.cuda.cuda_backend.spin.y", lambda *, target: f"y{target}")
@patch("qilisdk.extras.cuda.cuda_backend.spin.z", lambda *, target: f"z{target}")
@patch("qilisdk.extras.cuda.cuda_backend.spin.i", lambda *, target: f"i{target}")
def test_pauli_operator_handlers_call_spin():
    assert CudaBackend._handle_PauliX(PauliX(1)) == "x1"
    assert CudaBackend._handle_PauliY(PauliY(2)) == "y2"
    assert CudaBackend._handle_PauliZ(PauliZ(3)) == "z3"
    assert CudaBackend._handle_PauliI(PauliI(4)) == "i4"


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
    class DummyHam:
        def __iter__(self):
            # 2 * 2  +  3 * (3*4)  = 4 + 36 = 40
            yield 2, [PauliX(0)]
            yield 3, [PauliY(0), PauliZ(0)]

    assert be._hamiltonian_to_cuda(DummyHam()) == 40
