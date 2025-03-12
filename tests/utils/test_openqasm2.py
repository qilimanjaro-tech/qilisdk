# test_openqasm2.py

import math
import re

import pytest

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import CNOT, RX, H, M
from qilisdk.utils.openqasm2 import from_qasm2, from_qasm2_file, to_qasm2, to_qasm2_file


# --- Helper functions for tests ---
def create_empty_circuit(nqubits=3) -> Circuit:
    """Return a circuit with nqubits and no gates."""
    return Circuit(nqubits)


def create_circuit_with_gates() -> Circuit:
    """
    Create a circuit with several types of gates:
      - A Hadamard gate on qubit 0.
      - A parameterized RX gate on qubit 1.
      - A two-qubit CNOT gate on qubits 0 and 1.
      - A measurement on qubit 2.
    """
    circuit = Circuit(3)
    circuit.add(H(0))
    circuit.add(RX(1, theta=math.pi))
    circuit.add(CNOT(0, 1))
    circuit.add(M(*range(3)))
    return circuit


# --- Test cases ---
def test_empty_circuit_to_qasm():
    """Test conversion of an empty circuit (with no gates)."""
    circuit = create_empty_circuit(nqubits=3)
    qasm_str = to_qasm2(circuit)
    # Check header lines and qreg declaration.
    assert 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3]' in qasm_str
    # No classical register since no measurement.
    assert "creg" not in qasm_str
    # No gate instructions should be present.
    # (Only header and qreg declaration exist.)
    # Splitting into lines: there should be exactly 3 lines.
    assert len(qasm_str.strip().splitlines()) == 3


def test_single_gate_to_qasm():
    """Test conversion of a circuit with a single one-qubit gate (Hadamard)."""
    circuit = Circuit(1)
    circuit.add(H(0))
    qasm_str = to_qasm2(circuit)
    # Check that the gate instruction for Hadamard is present.
    # According to the mapping, H should be converted to "h".
    assert re.search(r"^\s*h\s+q\[0\];\s*$", qasm_str, re.MULTILINE)


def test_parameterized_gate_to_qasm():
    """Test conversion of a parameterized gate (RX) on a single qubit."""
    circuit = Circuit(1)
    circuit.add(RX(0, theta=math.pi))
    qasm_str = to_qasm2(circuit)
    # Expect the RX gate to be converted to "rx(3.14) q[0];"
    assert re.search(rf"^\s*rx\(\s*{math.pi}\s*\)\s+q\[0\];\s*$", qasm_str, re.MULTILINE)


def test_two_qubit_gate_to_qasm():
    """Test conversion of a two-qubit gate (CNOT)."""
    circuit = Circuit(2)
    circuit.add(CNOT(0, 1))
    qasm_str = to_qasm2(circuit)
    # CNOT should map to "cx". Expect: "cx q[0], q[1];"
    assert re.search(r"^\s*cx\s+q\[0\],\s*q\[1\];\s*$", qasm_str, re.MULTILINE)


def test_measurement_to_qasm():
    """Test conversion of a measurement gate (M)."""
    circuit = Circuit(2)
    circuit.add(M(1))
    qasm_str = to_qasm2(circuit)
    # Expect a classical register declaration since a measurement is present.
    assert "creg c[2];" in qasm_str
    # Expect the measurement instruction.
    assert re.search(r"^\s*measure\s+q\[1\]\s*->\s*c\[1\];\s*$", qasm_str, re.MULTILINE)


def test_measurement_all_qubits_to_qasm():
    """Test conversion of a circuit with a single one-qubit gate (Hadamard)."""
    circuit = Circuit(2)
    circuit.add(M(*range(2)))
    qasm_str = to_qasm2(circuit)
    # Expect a classical register declaration since a measurement is present.
    assert "creg c[2];" in qasm_str
    # Expect the measurement instruction.
    assert re.search(r"^\s*measure\s+q\s*->\s*c\s*;\s*$", qasm_str, re.MULTILINE)


def test_full_circuit_to_qasm_and_from_qasm():
    """Test converting a full circuit to QASM and then parsing it back."""
    original_circuit = create_circuit_with_gates()
    qasm_str = to_qasm2(original_circuit)
    reconstructed_circuit = from_qasm2(qasm_str)
    # Check that the number of qubits is preserved.
    assert reconstructed_circuit.nqubits == original_circuit.nqubits
    # Check that the same number of gates are present.
    assert len(reconstructed_circuit.gates) == len(original_circuit.gates)
    # Check gate types and qubit assignments.
    for orig_gate, recon_gate in zip(original_circuit.gates, reconstructed_circuit.gates):
        # Compare the gate names.
        assert orig_gate.name == recon_gate.name
        # Compare qubit assignments.
        assert orig_gate.qubits == recon_gate.qubits
        # Compare parameters.
        assert orig_gate.parameter_values == recon_gate.parameter_values


def test_from_qasm2_no_qreg():
    """Test that from_qasm2 raises a ValueError when no quantum register is declared."""
    qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";'
    with pytest.raises(ValueError, match="No quantum register declaration found"):
        from_qasm2(qasm_str)


def test_from_qasm2_unsupported_gate():
    """Test that an unsupported gate in QASM raises an UnsupportedGateError."""
    # Create a QASM string with an unsupported gate "foo".
    qasm_str = "\n".join(["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[1];", "foo q[0];"])
    with pytest.raises(UnsupportedGateError, match="Unknown gate: foo"):
        from_qasm2(qasm_str)


def test_to_qasm2_file_and_from_qasm2_file(tmp_path):
    """Test file I/O: writing to a file and reading it back."""
    circuit = create_circuit_with_gates()
    file_path = tmp_path / "test_circuit.qasm"
    # Write QASM to file.
    to_qasm2_file(circuit, str(file_path))
    # Read the file and reconstruct the circuit.
    reconstructed_circuit = from_qasm2_file(str(file_path))
    # Verify the reconstructed circuit matches the original.
    assert reconstructed_circuit.nqubits == circuit.nqubits
    assert len(reconstructed_circuit.gates) == len(circuit.gates)
    for orig_gate, recon_gate in zip(circuit.gates, reconstructed_circuit.gates):
        assert orig_gate.name == recon_gate.name
        assert orig_gate.qubits == recon_gate.qubits
        assert orig_gate.parameter_values == recon_gate.parameter_values
