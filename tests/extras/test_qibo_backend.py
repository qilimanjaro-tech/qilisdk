from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qibo.gates import gates as QiboGates

from qilisdk.digital import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    Circuit,
    Gate,
    H,
    M,
    S,
    SimulationDigitalResults,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.extras.qibo_backend import QiboBackend


@pytest.mark.parametrize(
    (
        "method",
        "gate_instance",
        "expected_class",
        "expected_target_qubits",
        "expected_control_qubits",
        "expected_parameters",
    ),
    [
        # ----------------------------------------------------------------
        # Non-parameterized single-qubit gates
        (QiboBackend._to_qibo_X, X(0), QiboGates.X, (0,), (), {}),
        (QiboBackend._to_qibo_Y, Y(1), QiboGates.Y, (1,), (), {}),
        (QiboBackend._to_qibo_Z, Z(2), QiboGates.Z, (2,), (), {}),
        (QiboBackend._to_qibo_H, H(0), QiboGates.H, (0,), (), {}),
        (QiboBackend._to_qibo_S, S(1), QiboGates.S, (1,), (), {}),
        (QiboBackend._to_qibo_T, T(2), QiboGates.T, (2,), (), {}),
        # ----------------------------------------------------------------
        # Parameterized single-qubit gates
        (QiboBackend._to_qibo_RX, RX(qubit=0, theta=np.pi / 4), QiboGates.RX, (0,), (), {"theta": np.pi / 4}),
        (QiboBackend._to_qibo_RY, RY(qubit=1, theta=1.234), QiboGates.RY, (1,), (), {"theta": 1.234}),
        (QiboBackend._to_qibo_RZ, RZ(qubit=2, phi=-0.5), QiboGates.RZ, (2,), (), {"theta": -0.5}),
        (QiboBackend._to_qibo_U1, U1(qubit=0, phi=np.e), QiboGates.U1, (0,), (), {"theta": np.e}),
        (QiboBackend._to_qibo_U2, U2(qubit=1, phi=0.1, gamma=0.2), QiboGates.U2, (1,), (), {"phi": 0.1, "lam": 0.2}),
        (
            QiboBackend._to_qibo_U3,
            U3(qubit=2, theta=0.3, phi=0.4, gamma=0.5),
            QiboGates.U3,
            (2,),
            (),
            {"theta": 0.3, "phi": 0.4, "lam": 0.5},
        ),
        # ----------------------------------------------------------------
        # Two-qubit gates
        (QiboBackend._to_qibo_CNOT, CNOT(control=0, target=1), QiboGates.CNOT, (1,), (0,), {}),
        (QiboBackend._to_qibo_CZ, CZ(control=1, target=2), QiboGates.CZ, (2,), (1,), {}),
        # ----------------------------------------------------------------
        # Measurement gate (supports multiple qubits, but we'll test single-qubit for simplicity)
        (QiboBackend._to_qibo_M, M(0), QiboGates.M, (0,), (), {}),
    ],
)
@pytest.mark.qibo_backend
def test_private_to_qibo_methods(
    method, gate_instance, expected_class, expected_target_qubits, expected_control_qubits, expected_parameters
):
    """
    Tests each _to_qibo_XXX private static method by calling it directly with
    the appropriate gate instance, ensuring the Qibo gate has the correct
    type, qubits, and parameters.
    """
    qibo_gate = method(gate_instance)

    # 1) Check Qibo gate class
    assert isinstance(qibo_gate, expected_class), (
        f"Expected Qibo gate of type {expected_class.__name__}, got {type(qibo_gate).__name__}"
    )

    # 2) Check target qubits
    assert qibo_gate.target_qubits == expected_target_qubits, (
        f"Expected target qubits {expected_target_qubits}, got {qibo_gate.target_qubits}"
    )

    # 3) Check control qubits (only relevant for multi-qubit gates)
    if hasattr(qibo_gate, "control_qubits"):
        assert qibo_gate.control_qubits == expected_control_qubits, (
            f"Expected control qubits {expected_control_qubits}, got {qibo_gate.control_qubits}"
        )

    # 4) Check parameters for parameterized gates
    if hasattr(qibo_gate, "parameters"):
        for param_name, param_value in expected_parameters.items():
            actual_value = qibo_gate.parameters[qibo_gate.parameter_names.index(param_name)]
            assert actual_value == param_value, (
                f"Expected parameter '{param_name}' to be {param_value}, got {actual_value}"
            )


@pytest.mark.qibo_backend
def test_to_qibo():
    """
    Test conversion of multiple gates in a single circuit.
    """
    circuit = Circuit(nqubits=3)
    circuit.add(X(0))
    circuit.add(CNOT(0, 1))
    circuit.add(RZ(2, phi=0.5))

    qibo_circuit = QiboBackend.to_qibo(circuit)
    assert qibo_circuit.nqubits == 3

    qibo_gates = qibo_circuit.queue
    assert len(qibo_gates) == 3

    # 1) X gate on qubit 0
    assert qibo_gates[0].__class__.__name__ == "X"
    assert qibo_gates[0].target_qubits == (0,)

    # 2) CNOT gate (control=0, target=1)
    assert qibo_gates[1].__class__.__name__ == "CNOT"
    assert qibo_gates[1].control_qubits == (0,)
    assert qibo_gates[1].target_qubits == (1,)

    # 3) RZ on qubit 2
    assert qibo_gates[2].__class__.__name__ == "RZ"
    assert qibo_gates[2].target_qubits == (2,)
    assert qibo_gates[2].parameters == (0.5,)


@pytest.mark.qibo_backend
def test_to_qibo_unsupported_gate():
    """
    Test that a ValueError is raised when the circuit has a gate
    not in QiboBackend's to_qibo_converters map.
    """

    # Create a dummy gate type that QiboBackend doesn't support
    class MyCustomGate(Gate):
        _NAME = "MyGate"
        _NQUBITS = 1
        _PARAMETER_NAMES = []

        def __init__(self, qubit: int):
            super().__init__()
            self._target_qubits = (qubit,)

    circuit = Circuit(nqubits=1)
    custom_gate = MyCustomGate(0)
    circuit.add(custom_gate)

    with pytest.raises(UnsupportedGateError) as excinfo:
        QiboBackend.to_qibo(circuit)

    assert "Unsupported gate type: MyCustomGate" in str(excinfo.value)


@pytest.mark.qibo_backend
@patch("qibo.models.circuit.Circuit.execute")
def test_execute(mock_qibo_execute):
    """
    Test QiboBackend.execute() using a mock QiboCircuit.execute() to avoid
    requiring an actual Qibo simulator environment.
    """
    # Mock the QiboCircuit.execute() result
    mock_result = MagicMock()
    mock_result.state.return_value = "mock_state"
    mock_result.probabilities.return_value = "mock_probabilities"
    mock_result.samples.return_value = "mock_samples"
    mock_result.frequencies.return_value = {"00": 10, "11": 90}

    # When the mock Qibo circuit calls .execute(), return the mock_result
    mock_qibo_execute.return_value = mock_result

    # Prepare a simple circuit
    circuit = Circuit(nqubits=2)
    circuit.add(X(0))
    backend = QiboBackend()

    # Execute with a custom nshots
    result = backend.execute(circuit, nshots=500)

    # Check that we called QiboCircuit.execute(...) with nshots=500
    mock_qibo_execute.assert_called_once_with(nshots=500)

    # Verify the returned SimulationDigitalResults content
    assert isinstance(result, SimulationDigitalResults)
    assert result.state == "mock_state"
    assert result.probabilities == "mock_probabilities"
    assert result.samples == "mock_samples"
    assert result.frequencies == {"00": 10, "11": 90}
    assert result.nshots == 500
