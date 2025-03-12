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
import re
from pathlib import Path

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import CNOT, CZ, RX, RY, RZ, U1, U2, U3, Gate, H, M, S, T, X, Y, Z

OPENQASM2_MAP: dict[type[Gate], str] = {
    X: "x",
    Y: "y",
    Z: "z",
    H: "h",
    S: "s",
    T: "t",
    RX: "rx",
    RY: "ry",
    RZ: "rz",
    U1: "u1",
    U2: "u2",
    U3: "u3",
    CNOT: "cx",
    CZ: "cz",
}


def to_qasm2(circuit: Circuit) -> str:
    """
    Convert the circuit to an OpenQASM 2.0 formatted string.

    Args:
        circuit: The circuit to convert to OpenQASM 2.0.

    Returns:
        str: The OpenQASM 2.0 representation of the circuit.
    """
    qasm_lines: list[str] = []
    # QASM header, standard library and quantum register.
    qasm_lines.extend(("OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.nqubits}];"))

    # If any measurement is present, declare a classical register.
    if any(isinstance(gate, M) for gate in circuit.gates):
        qasm_lines.append(f"creg c[{circuit.nqubits}];")

    # Process each gate.
    for gate in circuit.gates:
        # Special conversion for measurement.
        if isinstance(gate, M):
            if len(gate.target_qubits) == circuit.nqubits:
                qasm_lines.append("measure q -> c;")
            else:
                # Generate a measurement for each target qubit.
                measurements = (f"measure q[{q}] -> c[{q}];" for q in gate.target_qubits)
                qasm_lines.extend(measurements)
        else:
            # Map the internal gate name to its QASM equivalent.
            qasm_name = OPENQASM2_MAP.get(type(gate), gate.name.lower())
            # Format parameter string, if any.
            param_str = ""
            if gate.is_parameterized:
                parameters = ", ".join(str(p) for p in gate.parameter_values)
                param_str = f"({parameters})"
            # Format qubit operands.
            qubit_str = ", ".join(f"q[{q}]" for q in gate.qubits)
            qasm_lines.append(f"{qasm_name}{param_str} {qubit_str};")

    return "\n".join(qasm_lines)


def to_qasm2_file(circuit: Circuit, filename: str) -> None:
    """
    Save the QASM representation to a file.

    Args:
        circuit: The circuit to convert to OpenQASM 2.0.
        filename (str): The path to the file where the QASM code will be saved.
    """
    qasm_code = to_qasm2(circuit)
    Path(filename).write_text(qasm_code, encoding="utf-8")


# TODO(vyron): Add full support for OpenQASM 2.0 grammar.
def from_qasm2(qasm_str: str) -> Circuit:
    """
    Parse an OpenQASM 2.0 string and create a corresponding Circuit instance.

    This parser supports the following instructions:
        - Quantum register declaration (e.g., "qreg q[3];")
        - Classical register declaration (ignored)
        - Gate instructions (one-qubit and two-qubit gates)
        - Measurement instructions (e.g., "measure q[0] -> c[0];")

    Args:
        qasm_str (str): The QASM string to parse.

    Returns:
        Circuit: The constructed Circuit object.
    """  # noqa: DOC501
    # Mapping from QASM gate names (lowercase) to internal gate names.
    reverse_qasm2_map = {v: k for k, v in OPENQASM2_MAP.items()}

    circuit = None
    lines = qasm_str.splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        # Skip header and include lines.
        if line.startswith(("OPENQASM", "include")):
            continue
        # Parse quantum register declaration.
        if line.startswith("qreg"):
            # e.g., "qreg q[3];"
            m = re.match(r"qreg\s+\w+\[(\d+)\];", line)
            if m:
                nqubits = int(m.group(1))
                circuit = Circuit(nqubits)
            continue
        # Skip classical register declaration.
        if line.startswith("creg"):
            continue
        # Process measurement instructions.
        if line.startswith("measure"):
            # e.g., "measure q[0] -> c[0];"
            m = re.match(r"measure\s+q\[(\d+)\]\s*->\s*c\[\d+\];", line)
            if m:
                # TODO(vyron): Check consecutive lines of measurement and combine into single M.
                q_index = int(m.group(1))
                if circuit is None:
                    raise ValueError("Quantum register must be declared before measurement.")
                circuit.add(M(q_index))
            else:
                # Special case: "measure q -> c;" means measure all qubits
                m_all = re.match(r"measure\s+q\s*->\s*c\s*;", line)
                if m_all:
                    if circuit is None:
                        raise ValueError("Quantum register must be declared before measurement.")
                    circuit.add(M(*list(range(circuit.nqubits))))
            continue
        # Process gate instructions.
        # Pattern breakdown:
        #   Group 1: gate name (e.g., "h", "rx", "cx")
        #   Group 2: optional parameters (inside parentheses)
        #   Group 3: operand list (e.g., "q[0]" or "q[0], q[1]")
        m = re.match(r"^(\w+)(?:\(([^)]*)\))?\s+(.+);$", line)
        if m:
            qasm_gate_name = m.group(1)
            params_str = m.group(2)
            operands_str = m.group(3)

            # Convert QASM gate name to internal gate name.
            gate_class = reverse_qasm2_map.get(qasm_gate_name.lower())
            if gate_class is None:
                raise UnsupportedGateError(f"Unknown gate: {qasm_gate_name}")

            # Extract qubit indices.
            qubit_matches = re.findall(r"q\[(\d+)\]", operands_str)
            qubits = [int(q) for q in qubit_matches]

            # Parse parameters, if any.
            parameters = []
            if params_str:
                parameters = [float(p.strip()) for p in params_str.split(",") if p.strip()]

            # Instantiate the gate based on the number of qubits.
            # For one-qubit gates.
            if len(qubits) == 1:
                if gate_class.PARAMETER_NAMES:
                    # Build a dictionary of parameter names to values.
                    param_dict = {name: parameters[i] for i, name in enumerate(gate_class.PARAMETER_NAMES)}
                    gate_instance = gate_class(qubits[0], **param_dict)  # type: ignore[call-arg]
                else:
                    gate_instance = gate_class(qubits[0])  # type: ignore[call-arg]
            # For two-qubit gates.
            elif len(qubits) == 2:  # noqa: PLR2004
                if gate_class.PARAMETER_NAMES:
                    param_dict = {name: parameters[i] for i, name in enumerate(gate_class.PARAMETER_NAMES)}
                    gate_instance = gate_class(qubits[0], qubits[1], **param_dict)  # type: ignore[call-arg]
                else:
                    gate_instance = gate_class(qubits[0], qubits[1])  # type: ignore[call-arg]
            else:
                raise UnsupportedGateError("Only one- and two-qubit gates are supported.")

            if circuit is None:
                raise ValueError("Quantum register must be declared before adding gates.")
            circuit.add(gate_instance)
    if circuit is None:
        raise ValueError("No quantum register declaration found in QASM.")
    return circuit


def from_qasm2_file(filename: str) -> Circuit:
    """
    Read an OpenQASM 2.0 file and create a corresponding Circuit instance.

    Args:
        filename (str): The path to the QASM file.

    Returns:
        Circuit: The reconstructed Circuit object.
    """
    qasm_str = Path(filename).read_text(encoding="utf-8")
    return from_qasm2(qasm_str)
