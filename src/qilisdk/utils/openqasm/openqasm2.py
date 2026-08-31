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
import ast
import math
import re
from pathlib import Path

from loguru import logger

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.exceptions import UnsupportedGateError
from qilisdk.digital.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    U2,
    U3,
    Adjoint,
    BasicGate,
    Controlled,
    Gate,
    H,
    I,
    M,
    S,
    T,
    X,
    Y,
    Z,
)

OPENQASM2_MAP: dict[type[Gate], str] = {
    I: "id",
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
    SWAP: "swap",
}

# Gate names accepted on import on top of those in OPENQASM2_MAP, here the identity as QiliSDK used to export it.
_QASM2_IMPORT_ALIASES: dict[str, type[Gate]] = {"i": I}

# OpenQASM 2.0 has no inverse modifier, so the adjoint of these gates is written under a dedicated name instead.
_QASM2_ADJOINT_MAP: dict[type[BasicGate], str] = {S: "sdg", T: "tdg"}

# Gates that are their own adjoint, and so are written out unchanged.
_QASM2_SELF_ADJOINT: frozenset[type[Gate]] = frozenset({I, X, Y, Z, H, CNOT, CZ, SWAP})

# Gates whose adjoint is the same gate with every parameter negated.
_QASM2_NEGATED_ADJOINT: frozenset[type[Gate]] = frozenset({RX, RY, RZ, U1})

# Lookups from an OpenQASM 2.0 gate name to the gate, or to the gate whose adjoint it is.
_REVERSE_QASM2_MAP: dict[str, type[Gate]] = {name: gate for gate, name in OPENQASM2_MAP.items()} | _QASM2_IMPORT_ALIASES
_REVERSE_QASM2_ADJOINT_MAP: dict[str, type[BasicGate]] = {name: gate for gate, name in _QASM2_ADJOINT_MAP.items()}

# The Toffoli gate is the only gate acting on more than two qubits that is supported.
_QASM2_TOFFOLI_NQUBITS = 3

# A valid OpenQASM 2.0 gate name, which every gate name is checked against before being written out.
_QASM2_GATE_NAME = re.compile(r"[a-z]\w*", re.ASCII)

_ALLOWED_QASM2_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "ln": math.log,
    "sqrt": math.sqrt,
}

_MAX_DEPTH = 2


def _evaluate_qasm2_expression(expr: str) -> float:
    """Safely evaluate a numeric OpenQASM 2.0 parameter expression.
    Raises:
        ValueError: if the parameter expression in the input string is empty or the invalid.
    Returns:
        float: the parameter expression.
    """
    normalized_expr = expr.strip().replace("^", "**")
    if not normalized_expr:
        raise ValueError("Empty parameter expression.")
    try:
        parsed = ast.parse(normalized_expr, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid parameter expression: {expr}") from error
    return _evaluate_qasm2_ast(parsed.body, expr)


def _evaluate_qasm2_ast_binary(node: ast.BinOp, original_expr: str) -> float:
    left = _evaluate_qasm2_ast(node.left, original_expr)
    right = _evaluate_qasm2_ast(node.right, original_expr)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.Pow):
        return left**right
    raise ValueError(f"Unsupported operator in parameter expression: {original_expr}")


def _evaluate_qasm2_ast(node: ast.AST, original_expr: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        raise ValueError(f"Unsupported symbol in parameter expression: {original_expr}")

    if isinstance(node, ast.BinOp):
        return _evaluate_qasm2_ast_binary(node, original_expr)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_qasm2_ast(node.operand, original_expr)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator in parameter expression: {original_expr}")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Unsupported function in parameter expression: {original_expr}")
        func = _ALLOWED_QASM2_FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unsupported function in parameter expression: {original_expr}")
        if len(node.args) != 1 or node.keywords:
            raise ValueError(f"Unsupported function signature in parameter expression: {original_expr}")
        argument = _evaluate_qasm2_ast(node.args[0], original_expr)
        return float(func(argument))

    raise ValueError(f"Unsupported parameter expression: {original_expr}")


def _parse_qasm2_gate_line(line: str) -> tuple[str, str | None, str] | None:
    """Parse a QASM gate line with bounded parenthesis nesting.

    Returns:
        tuple[str, str | None, str] | None: the gate name, the string representing the parameter if available, and the rest.

    Raises:
        ValueError: if the parameter expression is nested for deeper than the max depth allowed, or the expression is invalid.
    """
    if not line.endswith(";"):
        return None

    body = line[:-1].strip()
    if not body:
        return None

    name_match = re.match(r"^(\w+)", body)
    if name_match is None:
        return None
    gate_name = name_match.group(1)

    # Anything other than a parameter list or a separator after the name means the name itself is malformed.
    separator = body[name_match.end() : name_match.end() + 1]
    if separator not in {"", "(", " ", "\t"}:
        raise ValueError(f"Invalid gate name in: {line}")

    rest = body[name_match.end() :].strip()
    params_str = None
    if rest.startswith("("):
        depth = 0
        closing_index = None
        for index, char in enumerate(rest):
            if char == "(":
                depth += 1
                if depth > _MAX_DEPTH:
                    raise ValueError(f"Parameter expression nesting deeper than {_MAX_DEPTH} levels is not supported.")
            elif char == ")":
                depth -= 1
                if depth == 0:
                    closing_index = index
                    break

        if closing_index is None:
            raise ValueError(f"Unclosed parameter expression in gate: {line}")
        params_str = rest[1:closing_index].strip()
        rest = rest[closing_index + 1 :].strip()

    if not rest:
        return None

    return gate_name, params_str, rest


def _qasm2_name_and_parameters(gate: Gate) -> tuple[str, list[float]]:
    """Resolve the OpenQASM 2.0 name and parameters of a gate.

    OpenQASM 2.0 has no inverse modifier, so the adjoint of a gate has to be expressed as another gate instead.

    Args:
        gate: the gate to export.

    Raises:
        UnsupportedGateError: if the gate cannot be expressed in OpenQASM 2.0.

    Returns:
        tuple[str, list[float]]: the OpenQASM 2.0 gate name, and the parameters to export it with.
    """
    if isinstance(gate, Adjoint):
        inner = gate.basic_gate
        inner_type = type(inner)
        if isinstance(inner, Adjoint):
            # The adjoint of an adjoint is the gate itself.
            return _qasm2_name_and_parameters(inner.basic_gate)
        if inner_type in _QASM2_ADJOINT_MAP:
            return _QASM2_ADJOINT_MAP[inner_type], []
        if inner_type in _QASM2_SELF_ADJOINT:
            return OPENQASM2_MAP[inner_type], []
        if inner_type in _QASM2_NEGATED_ADJOINT:
            return OPENQASM2_MAP[inner_type], [-value for value in inner.get_parameter_values()]
        if inner_type is U3:
            theta, phi, gamma = inner.get_parameter_values()
            return "u3", [-theta, -gamma, -phi]
        if inner_type is U2:
            phi, gamma = inner.get_parameter_values()
            return "u3", [-math.pi / 2, -gamma, -phi]
        raise UnsupportedGateError(f"Cannot express the adjoint of the {inner.name} gate in OpenQASM 2.0.")

    name = OPENQASM2_MAP.get(type(gate), gate.name.lower())
    if _QASM2_GATE_NAME.fullmatch(name) is None:
        raise UnsupportedGateError(f"Cannot express the {gate.name} gate in OpenQASM 2.0.")
    return name, gate.get_parameter_values() if gate.is_parameterized else []


def to_qasm2(circuit: Circuit) -> str:
    """
    Convert the circuit to an OpenQASM 2.0 formatted string.

    Args:
        circuit: The circuit to convert to OpenQASM 2.0.

    Raises:
        UnsupportedGateError: If a gate of the circuit cannot be expressed in OpenQASM 2.0.

    Returns:
        str: The OpenQASM 2.0 representation of the circuit.
    """
    logger.info("[OpenQASM2] Exporting circuit to OpenQASM 2.0")
    logger.debug("[OpenQASM2] Exporting {} gates on {} qubits", len(circuit.gates), circuit.nqubits)
    qasm_lines: list[str] = []
    # QASM header, standard library and quantum register.
    qasm_lines.extend(("OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{circuit.nqubits}];"))

    # If any measurement is present, declare a classical register.
    if any(isinstance(gate, M) for gate in circuit.gates):
        qasm_lines.append(f"creg c[{circuit.nqubits}];")

    # Process each gate.
    for gate in circuit.gates:
        logger.trace("[OpenQASM2] Exporting gate {} on qubits {}", gate.name, gate.qubits)
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
            qasm_name, parameters = _qasm2_name_and_parameters(gate)
            # Format parameter string, if any.
            param_str = f"({', '.join(str(p) for p in parameters)})" if parameters else ""
            # An adjoint does not expose the control qubits of the gate it wraps, so take the qubits from that gate.
            gate_to_export = gate
            while isinstance(gate_to_export, Adjoint):
                gate_to_export = gate_to_export.basic_gate
            # Format qubit operands.
            qubit_str = ", ".join(f"q[{q}]" for q in gate_to_export.qubits)
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
    logger.debug("[OpenQASM2] Writing OpenQASM 2.0 to file {}", filename)
    Path(filename).write_text(qasm_code, encoding="utf-8")


# TODO(vyron): Add full support for OpenQASM 2.0 grammar.
def from_qasm2(qasm_str: str) -> Circuit:
    """
    Parse an OpenQASM 2.0 string and create a corresponding Circuit instance.

    This parser supports the following instructions:
        - Quantum register declaration (e.g., "qreg q[3];"), of which there may be only one
        - Classical register declaration (e.g., "creg c[3];")
        - Gate instructions (one-qubit and two-qubit gates, plus the three-qubit "ccx" gate)
        - Measurement instructions (e.g., "measure q[0] -> c[0];")

    The registers may be given any name, and any instruction that refers to a register that was never declared
    raises rather than being skipped.

    Args:
        qasm_str (str): The QASM string to parse.

    Returns:
        Circuit: The constructed Circuit object.
    """  # ruff: ignore[docstring-missing-exception]
    logger.info("[OpenQASM2] Importing circuit from OpenQASM 2.0")
    circuit = None
    qreg_name = ""
    creg_names: set[str] = set()
    lines = qasm_str.splitlines()
    logger.debug("[OpenQASM2] Parsing {} lines of OpenQASM 2.0", len(lines))
    for raw_line in lines:
        line = raw_line.strip()
        logger.trace("[OpenQASM2] Parsing line: {}", line)
        if "//" in line:
            line = line.split("//", 1)[0].strip()
        if not line or line.startswith("//"):
            continue

        # Skip header and include lines.
        if line.startswith(("OPENQASM", "include")):
            continue

        # Parse quantum register declaration.
        if line.startswith("qreg"):
            # e.g., "qreg q[3];"
            m = re.match(r"qreg\s+(\w+)\s*\[(\d+)\]\s*;", line)
            if m:
                if circuit is not None:
                    raise ValueError("Only a single quantum register is supported.")
                qreg_name = m.group(1)
                circuit = Circuit(int(m.group(2)))
            continue

        # Parse classical register declaration, whose name is needed to recognise measurements.
        if line.startswith("creg"):
            # e.g., "creg c[3];"
            m = re.match(r"creg\s+(\w+)\s*\[\d+\]\s*;", line)
            if m:
                creg_names.add(m.group(1))
            continue

        # Process measurement instructions.
        if line.startswith("measure"):
            if circuit is None:
                raise ValueError("Quantum register must be declared before measurement.")
            # e.g., "measure q[0] -> c[0];"
            m = re.fullmatch(r"measure\s+(\w+)\s*\[(\d+)\]\s*->\s*(\w+)\s*\[\d+\]\s*;", line)
            if m is not None:
                # TODO(vyron): Check consecutive lines of measurement and combine into single M.
                quantum_name, classical_name, measured = m.group(1), m.group(3), (int(m.group(2)),)
            else:
                # Special case: "measure q -> c;" means measure all qubits.
                m = re.fullmatch(r"measure\s+(\w+)\s*->\s*(\w+)\s*;", line)
                if m is None:
                    raise ValueError(f"Invalid measurement instruction: {line}")
                quantum_name, classical_name, measured = m.group(1), m.group(2), tuple(range(circuit.nqubits))
            if quantum_name != qreg_name:
                raise ValueError(f"Undeclared quantum register '{quantum_name}' in measurement: {line}")
            if classical_name not in creg_names:
                raise ValueError(f"Undeclared classical register '{classical_name}' in measurement: {line}")
            circuit.add(M(*measured))
            continue

        # Process gate instructions.
        gate_data = _parse_qasm2_gate_line(line)
        if gate_data:
            qasm_gate_name, params_str, operands_str = gate_data
            if circuit is None:
                raise ValueError("Quantum register must be declared before adding gates.")
            gate_name = qasm_gate_name.lower()

            # Extract qubit indices, which have to belong to the declared quantum register.
            qubits = [int(index) for index in re.findall(rf"{re.escape(qreg_name)}\s*\[(\d+)\]", operands_str)]
            if not qubits:
                raise ValueError(f"Gate operands do not refer to the quantum register '{qreg_name}': {line}")

            # Parse parameters, if any.
            parameters = []
            if params_str:
                parameters = [_evaluate_qasm2_expression(p) for p in params_str.split(",") if p.strip()]

            # The Toffoli gate is the only supported gate acting on more than two qubits.
            if gate_name == "ccx":
                if len(qubits) != _QASM2_TOFFOLI_NQUBITS:
                    raise UnsupportedGateError(f"The ccx gate acts on three qubits, got {len(qubits)}.")
                circuit.add(Controlled(qubits[0], qubits[1], basic_gate=X(qubits[2])))
                continue
            if len(qubits) not in {1, 2}:
                raise UnsupportedGateError("Only one- and two-qubit gates are supported.")

            # Gates that OpenQASM 2.0 names in place of an inverse modifier, such as "sdg" for the adjoint of S.
            adjoint_of = _REVERSE_QASM2_ADJOINT_MAP.get(gate_name)
            if adjoint_of is not None:
                circuit.add(Adjoint(adjoint_of(qubits[0])))  # ty: ignore[invalid-argument-type]
                continue

            # Convert QASM gate name to internal gate name.
            gate_class = _REVERSE_QASM2_MAP.get(gate_name)
            if gate_class is None:
                raise UnsupportedGateError(f"Unknown gate: {qasm_gate_name}")

            # Build a dictionary of parameter names to values.
            param_dict = {name: parameters[i] for i, name in enumerate(gate_class.PARAMETER_NAMES)}
            circuit.add(gate_class(*qubits, **param_dict))

    if circuit is None:
        raise ValueError("No quantum register declaration found in QASM.")
    logger.debug("[OpenQASM2] Imported circuit with {} qubits and {} gates", circuit.nqubits, len(circuit.gates))
    return circuit


def from_qasm2_file(filename: str) -> Circuit:
    """
    Read an OpenQASM 2.0 file and create a corresponding Circuit instance.

    Args:
        filename (str): The path to the QASM file.

    Returns:
        Circuit: The reconstructed Circuit object.
    """
    logger.debug("[OpenQASM2] Reading OpenQASM 2.0 from file {}", filename)
    qasm_str = Path(filename).read_text(encoding="utf-8")
    return from_qasm2(qasm_str)
