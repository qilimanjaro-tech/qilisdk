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
from pathlib import Path

from numpy import pi
from openqasm3.ast import (
    BinaryOperator,
    ClassicalAssignment,
    ClassicalDeclaration,
    ConstantDeclaration,
    ImaginaryLiteral,
    Include,
    QuantumGate,
    QubitDeclaration,
)
from openqasm3.parser import parse

from qilisdk.digital import CNOT, CZ, RX, RY, RZ, U1, U2, U3, Circuit, H, M, S, T, X, Y, Z


def _evaluate_expression(expr: object, var_list: dict) -> any:

    # If it's an imaginary literal
    if isinstance(expr, ImaginaryLiteral) and hasattr(expr, "value"):
        return 1j * expr.value

    # If we have a value, perfect
    if hasattr(expr, "value"):
        return expr.value

    # If it's a variable
    if hasattr(expr, "name"):
        var_name = expr.name
        if isinstance(var_name, str) and var_name in var_list:
            return var_list[var_name]["value"]
        raise ValueError(f"Undefined variable: {var_name}")

    # If it's an operation, recurse
    if hasattr(expr, "op"):
        if expr.op == BinaryOperator["+"]:
            return _evaluate_expression(expr.lhs, var_list) + _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["-"]:
            return _evaluate_expression(expr.lhs, var_list) - _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["/"]:
            return _evaluate_expression(expr.lhs, var_list) / _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["*"]:
            return _evaluate_expression(expr.lhs, var_list) * _evaluate_expression(expr.rhs, var_list)
        raise ValueError(f"Unsupported operator: {expr.op}")

    raise ValueError(f"Unsupported expression type: {type(expr)}")


def _to_qilisdk_gate(gate_name: str, qubits: list) -> object:
    gate_name = gate_name.lower()
    if gate_name == "x":
        return X(*qubits)
    if gate_name == "y":
        return Y(*qubits)
    if gate_name == "z":
        return Z(*qubits)
    if gate_name == "h":
        return H(*qubits)
    if gate_name == "s":
        return S(*qubits)
    if gate_name == "t":
        return T(*qubits)
    if gate_name == "rx":
        return RX(*qubits)
    if gate_name == "ry":
        return RY(*qubits)
    if gate_name == "rz":
        return RZ(*qubits)
    if gate_name == "u1":
        return U1(*qubits)
    if gate_name == "u2":
        return U2(*qubits)
    if gate_name == "u3":
        return U3(*qubits)
    if gate_name == "cx":
        return CNOT(*qubits)
    if gate_name == "cz":
        return CZ(*qubits)
    if gate_name == "m":
        return M(*qubits)

    raise ValueError(f"Unsupported gate: {gate_name}")


def from_qasm3(qasm3: str) -> Circuit:
    """
    Convert an OpenQASM 3.0 string representation of a quantum circuit into a Circuit object.

    Args:
        qasm3 (str): The OpenQASM 3.0 string representation of the quantum circuit.

    Returns:
        Circuit: The reconstructed Circuit object.

    Raises:
        ValueError: If the QASM string contains unsupported statements, gates, or expressions.
    """

    # Use the official OpenQASM 3.0 parser to parse the text and get the AST nodes
    ast = parse(qasm3)

    # The vars to fill as we parse the tree
    nqubits = 0
    reg_name_to_start_end = {}
    var_list = {
        "π": {"size": 1, "value": pi},
    }
    gates_to_add = []

    # Go through the tree and determine what to do
    for statement in ast.statements:
        print()
        print(statement)

        # Initializing a qubit
        if isinstance(statement, QubitDeclaration):
            reg_name = statement.qubit.name
            reg_size = 1
            if hasattr(statement, "size") and statement.size is not None:
                reg_size = _evaluate_expression(statement.size, var_list)
            reg_name_to_start_end[reg_name] = (nqubits, nqubits + reg_size - 1)
            var_list[reg_name] = {"size": reg_size, "value": 0}
            nqubits = max(nqubits, nqubits + reg_size)

        # Initializing a classical variable
        elif isinstance(statement, (ClassicalDeclaration, ConstantDeclaration)):
            var_name = statement.identifier.name
            var_size = 1
            var_value = 0
            if hasattr(statement, "size") and statement.size is not None:
                var_size = _evaluate_expression(statement.size, var_list)
            if hasattr(statement, "init_expression") and statement.init_expression is not None:
                var_value = _evaluate_expression(statement.init_expression, var_list)
            var_list[var_name] = {"size": var_size, "value": var_value}

        # Classical assignment
        elif isinstance(statement, ClassicalAssignment):
            var_name = statement.lvalue.name
            var_list[var_name]["value"] = _evaluate_expression(statement.rvalue, var_list)

        # Quantum gates
        elif isinstance(statement, QuantumGate):
            gate_name = statement.name.name
            qubits = [_evaluate_expression(qb, var_list) for qb in statement.qubits]
            gates_to_add.append(_to_qilisdk_gate(gate_name, qubits))

        # Otherwise raise an error for now - we can add more statement types later
        elif not isinstance(statement, (Include)):
            raise ValueError(f"Unsupported statement type: {type(statement)}")

    print()
    print(f"Determined number of qubits: {nqubits}")
    print(f"Register name to qubit index mapping: {reg_name_to_start_end}")
    print(f"All variables: {var_list}")
    print(f"Gates to add: {gates_to_add}")

    # Create a Circuit with the determined number of qubits
    c = Circuit(nqubits)
    for gate in gates_to_add:
        c.add(gate)

    return c


def to_qasm3(circuit: Circuit) -> str:
    """
    Convert a Circuit object to its OpenQASM 3.0 string representation.

    Args:
        circuit: The Circuit object to convert.

    Returns:
        str: The OpenQASM 3.0 representation of the circuit.
    """
    qasm3 = "OPENQASM 3.0;\n"
    qasm3 += 'include "stdgates.inc";\n\n'
    if circuit.nqubits > 0:
        qasm3 += f"qubit[{circuit.nqubits}] q;\n\n"
    for gate in circuit.gates:
        qasm_gate_name = gate.name.lower()
        qasm_control_str = ["ctrl @ " for _ in gate.controls].join("")
        qasm_qubits_str = ", ".join([f"q[{qb}]" for qb in gate.qubits])
        qasm_gate_string = f"{qasm_control_str} {qasm_gate_name} {qasm_qubits_str}"
        qasm_gate_string = qasm_gate_string.strip()
        qasm3 += f"{qasm_gate_string};\n"
    return qasm3


def from_qasm3_file(filename: str) -> Circuit:
    """
    Read an OpenQASM 3.0 file and create a corresponding Circuit instance.

    Args:
        filename (str): The path to the QASM file.

    Returns:
        Circuit: The reconstructed Circuit object.
    """
    qasm_str = Path(filename).read_text(encoding="utf-8")
    return from_qasm3(qasm_str)


def to_qasm3_file(circuit: Circuit, filename: str) -> None:
    """
    Save the OpenQASM 3.0 representation of a circuit to a file.

    Args:
        circuit: The circuit to convert to OpenQASM 3.0.
        filename (str): The path to the file where the QASM code will be saved.
    """
    qasm_code = to_qasm3(circuit)
    Path(filename).write_text(qasm_code, encoding="utf-8")
