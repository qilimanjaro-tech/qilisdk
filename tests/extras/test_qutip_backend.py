import numpy as np
import pytest

from qilisdk.digital import RX, RY, RZ, U1, U2, U3, Circuit, H, M, S, T, X, Y, Z
from qilisdk.digital.digital_result import DigitalResult
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import CNOT, Adjoint, Controlled
from qilisdk.extras.qutip import QutipBackend, QutipDigitalResult  # adjust import as needed


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
    circ = Circuit(nqubits=1)
    circ.add(X(0))
    result = backend.execute(circ, nshots=100)
    # Expect roughly all shots to be '1'
    assert isinstance(result, QutipDigitalResult)
    samples = result.samples
    assert "1" in samples
    assert samples["1"] == 100


def test_execute_with_measurement_gate(backend):
    circ = Circuit(nqubits=1)
    circ.add(X(0))
    circ.add(M(0))
    result = backend.execute(circ, nshots=50)
    # Still expect only '1'
    assert result.samples == {"1": 50}


def test_unsupported_gate_raises(backend):
    class FakeGate:
        target_qubits = [0]

    circ = Circuit(nqubits=1)
    circ.gates.append(FakeGate())
    with pytest.raises(UnsupportedGateError):
        backend.execute(circ)


def test_controlled_cnot(backend):
    circ = Circuit(nqubits=2)
    circ.add(CNOT(control=0, target=1))
    # Expect no error on building or executing
    result = backend.execute(circ, nshots=10)
    assert isinstance(result, QutipDigitalResult)
    # All samples should be "00" since X only applies if control=1, but no preparation
    assert result.samples == {"00": 10}


def test_nshots():
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    result = backend.execute(circuit, nshots=10)
    assert isinstance(result, DigitalResult)
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
def test_execute_adjoint_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    adjoint_gate = Adjoint(gate_instance)
    circuit._gates.append(adjoint_gate)
    with pytest.raises(NotImplementedError):
        backend.execute(circuit, nshots=10)


@pytest.mark.parametrize("gate_instance", [case[0] for case in basic_gate_test_cases])
def test_execute_controlled_handler(gate_instance):
    backend = QutipBackend()
    circuit = Circuit(nqubits=1)
    controlled_gate = Controlled(1, basic_gate=gate_instance)
    circuit._gates.append(controlled_gate)
    qutip_circuit = backend._get_qutip_circuit(circuit)

    assert any(g.name == "User_" + controlled_gate.name for g in qutip_circuit.gates)
