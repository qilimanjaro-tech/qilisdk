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
import math
from copy import copy
from pathlib import Path

from numpy import e, pi
from openqasm3.ast import (
    AliasStatement,
    AngleType,
    ArrayType,
    AssignmentOperator,
    BinaryOperator,
    BitType,
    BoolType,
    BranchingStatement,
    BreakStatement,
    Cast,
    ClassicalAssignment,
    ClassicalDeclaration,
    ComplexType,
    ConstantDeclaration,
    ContinueStatement,
    DiscreteSet,
    DurationType,
    ExpressionStatement,
    FloatType,
    ForInLoop,
    FunctionCall,
    GateModifierName,
    Identifier,
    ImaginaryLiteral,
    Include,
    IndexExpression,
    IntType,
    IODeclaration,
    IOKeyword,
    QuantumGate,
    QuantumGateDefinition,
    QuantumMeasurementStatement,
    QubitDeclaration,
    RangeDefinition,
    ReturnStatement,
    StretchType,
    SubroutineDefinition,
    SwitchStatement,
    TimeUnit,
    UintType,
    UnaryOperator,
    WhileLoop,
)
from openqasm3.parser import parse

from qilisdk.core import Parameter
from qilisdk.digital import CNOT, CZ, RX, RY, RZ, U1, U2, U3, Adjoint, Circuit, Controlled, Gate, H, M, S, T, X, Y, Z


class OpenQasmParser:
    """
    Internal class for parsing OpenQASM 3.0.

    Use the external methods instead:
     - `from qilisdk.utils.openqasm import to_qasm3_file`
     - `from qilisdk.utils.openqasm import from_qasm3_file`
     - `from qilisdk.utils.openqasm import to_qasm3`
     - `from qilisdk.utils.openqasm import from_qasm3`

    Understanding the following table:
    - ✅: The feature is fully supported
    - 🟡: The feature is partially supported, see the note for explanation
    - ❌: The feature is not supported at all

    Feature Table
    OpenQASM 3 Feature          QiliSDK       Notes
    comments                    ✅
    QASM vstring                ✅
    include                     ✅
    unicode names               ✅
    qubit                       ✅
    bit                         🟡            1
    bool                        🟡            1
    int                         🟡            1
    uint                        🟡            1
    float                       🟡            1
    angle                       🟡            1
    complex                     🟡            1
    const                       ✅
    pi/π/tau/τ/euler/ℇ          ✅
    Aliasing: let               ✅
    register concatenation      ❌
    casting expr.Cast           ✅
    duration                    🟡            1
    durationof                  ❌
    ns/μs/us/ms/s/dt            ✅
    stretch expr.Stretch        🟡            1
    delay                       ❌
    barrier                     ❌
    box                         ❌
    Built-in U                  ✅
    gate                        🟡            1
    gphase                      ❌
    ctrl @                      ✅
    negctrl @                   🟡            1
    inv @                       ✅
    pow(k) @                    🟡            1, 2
    reset                       ❌
    measure                     🟡            3
    bit operations              🟡            1
    boolean operations          🟡            1
    arithmetic expressions      🟡            1
    comparisons                 🟡            1
    if                          🟡            1
    else                        🟡            1
    else if                     🟡            1
    for loops                   🟡            1
    switch                      🟡            1
    while loops                 🟡            1
    continue                    🟡            1
    break                       🟡            1
    extern                      ❌
    def subroutines             🟡            1
    return                      🟡            1
    input                       🟡            1
    output                      ❌

    1) Reading these operations is fully supported, but the expressions will not be stored within the circuit object, and thus
       will not written back out if you convert back to OpenQASM. For example, if you declare "int x = 5;", the variable "x"
       will be evaluated and used during the rest of the parsing, but in the circuit object itself "x" will not appear,
       and thus converting back to OpenQASM will not include the declaration of "x".

    2) pow(k) is only supported in QiliSDK when k is an integer, and is done by doing repeated gate applications.

    3) Mid-circuit measurements are not supported in QiliSDK.

    """

    def __init__(self) -> None:
        self.reg_name_to_start_end = {}
        self.var_list = {}
        self.custom_gate_definitions = {}
        self.subroutine_definitions = {}
        self.gates_to_add = []
        self.nqubits = 0

    def _recursive_replace(self, statement: object, replacement_map: dict) -> object:
        new_statement = copy(statement)

        # If the statement is in the replacement map, replace it
        if (
            hasattr(new_statement, "name")
            and new_statement.name is not None
            and isinstance(new_statement.name, str)
            and not hasattr(new_statement.name, "name")
            and new_statement.name in replacement_map
        ):
            # If we're changing to an identifier
            if isinstance(replacement_map[new_statement.name], Identifier):
                new_statement = copy(replacement_map[new_statement.name])

            # If we're changing to a string, change the name
            elif isinstance(replacement_map[new_statement.name], str):
                setattr(new_statement, "name", replacement_map[new_statement.name])

            # If changing to something else (i.e. a number), change the value
            else:
                setattr(new_statement, "value", replacement_map[new_statement.name])

        # For any other attributes, as long as it's not a default attribute or a callable, recurse
        for attr_name in dir(statement):
            if (
                attr_name.startswith("__")
                or callable(getattr(new_statement, attr_name))
                or isinstance(getattr(new_statement, attr_name), (str, int, float, type(None)))
            ):
                continue
            attr_value = getattr(new_statement, attr_name)
            if isinstance(attr_value, list):
                new_list = []
                for sub_value in attr_value:
                    new_list.append(self._recursive_replace(sub_value, replacement_map))
                setattr(new_statement, attr_name, new_list)
            else:
                setattr(new_statement, attr_name, self._recursive_replace(attr_value, replacement_map))
        return new_statement

    @staticmethod
    def _parse_return_val(return_str: str | complex | bool) -> str | int | float | complex | bool:

        # If we have "return:" at the start, remove it
        if isinstance(return_str, str):
            return_str = return_str.removeprefix("return:")
            return_str = return_str.strip()
            if not return_str:
                return 0

        # Try to interpret as an int
        try:
            return int(str(return_str))
        except ValueError:
            pass

        # Try to interpret as a float
        try:
            return float(str(return_str))
        except ValueError:
            pass

        # Try to interpret as a complex
        try:
            return complex(str(return_str))
        except ValueError:
            pass

        # Try to interpret as a bool
        if return_str == "True":
            return True
        if return_str == "False":
            return False

        # Otherwise, just return the string
        return return_str

    @staticmethod
    def _handle_standard_functions(
        func_name: str, args_evalled: list
    ) -> list | str | int | float | complex | bool | None:

        # With two int arguments
        if (
            len(args_evalled) >= 2  # noqa: PLR2004
            and isinstance(args_evalled[0], int)
            and isinstance(args_evalled[1], int)
        ):
            match func_name:
                case "rotl":
                    return ((args_evalled[0] << args_evalled[1]) | (args_evalled[0] >> (32 - args_evalled[1]))) % (
                        2**32
                    )
                case "rotr":
                    return ((args_evalled[0] >> args_evalled[1]) | (args_evalled[0] << (32 - args_evalled[1]))) % (
                        2**32
                    )
                case "mod":
                    return args_evalled[0] % args_evalled[1]

        # One arg, at most a float
        if len(args_evalled) >= 1 and isinstance(args_evalled[0], (int, float)):
            match func_name:
                case "sin":
                    return math.sin(args_evalled[0])
                case "cos":
                    return math.cos(args_evalled[0])
                case "tan":
                    return math.tan(args_evalled[0])
                case "arcsin":
                    return math.asin(args_evalled[0])
                case "arccos":
                    return math.acos(args_evalled[0])
                case "arctan":
                    return math.atan(args_evalled[0])
                case "floor":
                    return math.floor(args_evalled[0])
                case "ceiling":
                    return math.ceil(args_evalled[0])
                case "sqrt":
                    return math.sqrt(args_evalled[0])
                case "exp":
                    return math.exp(args_evalled[0])

        # One arg, at most a complex
        if len(args_evalled) >= 1 and isinstance(args_evalled[0], (int, float, complex)):
            match func_name:
                case "real":
                    return float(args_evalled[0].real)
                case "imag":
                    return float(args_evalled[0].imag)
                case "log":
                    return math.log(args_evalled[0])

        return None

    @staticmethod
    def _try_generic_operators(
        lhs: complex, rhs: complex, op: BinaryOperator
    ) -> list | str | int | float | complex | bool | None:
        if op == BinaryOperator["+"]:
            return lhs + rhs
        if op == BinaryOperator["-"]:
            return lhs - rhs
        if op == BinaryOperator["/"]:
            return lhs / rhs
        if op == BinaryOperator["*"]:
            return lhs * rhs
        if op == BinaryOperator["=="]:
            return lhs == rhs
        if op == BinaryOperator["!="]:
            return lhs != rhs
        if op == BinaryOperator["&&"]:
            return lhs and rhs
        if op == BinaryOperator["||"]:
            return lhs or rhs
        if op == BinaryOperator["**"]:
            return lhs**rhs
        return None

    @staticmethod
    def _try_non_complex_operators(
        lhs: float, rhs: float, op: BinaryOperator
    ) -> list | str | int | float | complex | bool | None:
        if op == BinaryOperator[">"]:
            return lhs > rhs
        if op == BinaryOperator[">="]:
            return lhs >= rhs
        if op == BinaryOperator["<"]:
            return lhs < rhs
        if op == BinaryOperator["<="]:
            return lhs <= rhs
        return None

    @staticmethod
    def _try_int_operators(lhs: int, rhs: int, op: BinaryOperator) -> list | str | int | float | complex | bool | None:
        if op == BinaryOperator["<<"]:
            return lhs << rhs
        if op == BinaryOperator[">>"]:
            return lhs >> rhs
        if op == BinaryOperator["&"]:
            return lhs & rhs
        if op == BinaryOperator["|"]:
            return lhs | rhs
        if op == BinaryOperator["%"]:
            return lhs % rhs
        return None  # pragma: no cover

    def _handle_expression_lhs_rhs(
        self, lhs: object, rhs: object, op: BinaryOperator
    ) -> list | str | int | float | complex | bool:
        if not (isinstance(lhs, (int, float, complex)) and isinstance(rhs, (int, float, complex))):
            raise ValueError(f"Unsupported operands for operator {op}: {lhs} and {rhs}")  # pragma: no cover

        res = self._try_generic_operators(lhs, rhs, op)
        if res is not None:
            return res

        if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
            res = self._try_non_complex_operators(lhs, rhs, op)
            if res is not None:
                return res

        if isinstance(lhs, int) and isinstance(rhs, int):
            res = self._try_int_operators(lhs, rhs, op)
            if res is not None:
                return res

        raise ValueError(f"Unsupported operator: {op}")  # pragma: no cover

    def _handle_expression_with_op(self, expr: object) -> list | str | int | float | complex | bool:
        if hasattr(expr, "op"):
            if hasattr(expr, "lhs") and hasattr(expr, "rhs"):
                lhs = self._evaluate_expression(expr.lhs)
                rhs = self._evaluate_expression(expr.rhs)
                return self._handle_expression_lhs_rhs(lhs, rhs, expr.op)  # ty:ignore[invalid-argument-type]
            if hasattr(expr, "expression"):
                expr_val = self._evaluate_expression(expr.expression)
                if isinstance(expr_val, (bool, int, float, complex)):
                    if expr.op == UnaryOperator["-"]:
                        return -expr_val
                    if expr.op == UnaryOperator["!"]:
                        return not expr_val
            raise ValueError(f"Unsupported operator: {expr.op}")  # pragma: no cover
        raise ValueError("Expression does not have an operator")  # pragma: no cover

    @staticmethod
    def _handle_cast(expr: Cast, value_to_cast: object) -> list | str | int | float | complex | bool:
        if isinstance(expr.type, (UintType, IntType, BitType)) and isinstance(value_to_cast, (int, float)):
            return int(value_to_cast) % (2**32)
        if isinstance(expr.type, (FloatType, AngleType, DurationType, StretchType)) and isinstance(
            value_to_cast, (int, float)
        ):
            return float(value_to_cast)
        if isinstance(expr.type, ComplexType) and isinstance(value_to_cast, (int, float, complex)):
            return complex(value_to_cast)
        if isinstance(expr.type, BoolType) and isinstance(value_to_cast, (str, int, float, complex, bool)):
            return bool(value_to_cast)
        raise ValueError(f"Unsupported cast type: {expr.type}")  # pragma: no cover

    def _handle_expression_function_call(self, expr: FunctionCall) -> list | str | int | float | complex | bool:
        func_name = expr.name.name
        args_evalled = [self._evaluate_expression(arg) for arg in expr.arguments]

        # Standard functions
        standard_func_result = self._handle_standard_functions(func_name, args_evalled)
        if standard_func_result is not None:
            return standard_func_result

        # Custom functions
        if func_name in self.subroutine_definitions:
            func_def = self.subroutine_definitions[func_name]
            replacement_map = {}
            for actual_arg in expr.arguments:
                arg_name_in_body = func_def["args"][expr.arguments.index(actual_arg)]
                if isinstance(actual_arg, (IndexExpression)):
                    replacement_map[arg_name_in_body] = actual_arg
                else:
                    replacement_map[arg_name_in_body] = self._evaluate_expression(actual_arg)
            for func_statement in func_def["body"]:
                res = self._process_statement(self._recursive_replace(func_statement, replacement_map))
                if res:
                    return self._parse_return_val(res)
            return 0

        raise ValueError(f"Unsupported function: {func_name}")

    @staticmethod
    def _get_value_with_unit(expr: object) -> list | str | int | float | complex | bool | None:
        if hasattr(expr, "value") and expr.value is not None and isinstance(expr.value, (int, float, complex)):
            value_with_unit = expr.value
            if hasattr(expr, "unit") and expr.unit is not None:
                if expr.unit == TimeUnit["ns"]:
                    value_with_unit *= 1e-9
                elif expr.unit == TimeUnit["us"]:
                    value_with_unit *= 1e-6
                elif expr.unit == TimeUnit["ms"]:
                    value_with_unit *= 1e-3
                elif expr.unit != TimeUnit["s"]:
                    raise ValueError(f"Unsupported time unit: {expr.unit}")  # pragma: no cover
            return value_with_unit
        return None

    def _handle_expression_index_expression(self, expr: IndexExpression) -> list | str | int | float | complex | bool:
        if not isinstance(expr.index, list) or not hasattr(expr.collection, "name") or expr.collection.name is None:
            raise ValueError(f"Invalid index expression: {expr}")  # pragma: no cover
        var_name = expr.collection.name
        var_indices = [self._evaluate_expression(index) for index in expr.index]
        if isinstance(var_name, str) and var_name in self.var_list and "value" in self.var_list[var_name]:
            value = self.var_list[var_name]["value"]
            for index in var_indices:
                if isinstance(index, int) and isinstance(value, list) and index < len(value):
                    value = value[index]
            return value
        raise ValueError(f"Undefined variable for index expression: {var_name}")  # pragma: no cover

    def _evaluate_expression(self, expr: object) -> list | str | int | float | complex | bool:

        # If it's a list, evaluate each element
        if isinstance(expr, list):
            return [self._evaluate_expression(element) for element in expr]

        # If it's a function call
        if isinstance(expr, FunctionCall):
            return self._handle_expression_function_call(expr)

        # Scale by the unit if we have it
        value_with_unit = self._get_value_with_unit(expr)

        # If it's an imaginary literal
        if isinstance(expr, ImaginaryLiteral) and isinstance(value_with_unit, (int, float, complex)):
            return 1j * value_with_unit

        # If we have a value, perfect
        if value_with_unit is not None:
            return value_with_unit

        # If it's a variable
        if hasattr(expr, "name"):
            var_name = expr.name
            if isinstance(var_name, str) and var_name in self.var_list:
                return self.var_list[var_name]["value"]
            raise ValueError(f"Undefined variable: {var_name}")

        # If it's an operation, recurse
        if hasattr(expr, "op"):
            return self._handle_expression_with_op(expr)

        # If it's a cast
        if isinstance(expr, Cast):
            value_to_cast = self._evaluate_expression(expr.argument)
            return self._handle_cast(expr, value_to_cast)

        # If it's an array literal
        if hasattr(expr, "values") and expr.values is not None and isinstance(expr.values, list):
            return [self._evaluate_expression(element) for element in expr.values]

        # If it's a range definition
        if hasattr(expr, "start") and expr.start is not None and hasattr(expr, "end") and expr.end is not None:
            start = self._evaluate_expression(expr.start)
            end = self._evaluate_expression(expr.end)
            step = 1
            if hasattr(expr, "step") and expr.step is not None:
                step = self._evaluate_expression(expr.step)
            if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
                return list(range(start, end, step))

        # If it's an index expression, evaluate the base and the indices and return the indexed value
        if isinstance(expr, IndexExpression):
            return self._handle_expression_index_expression(expr)

        raise ValueError(f"Unsupported expression: {expr}")  # pragma: no cover

    def _flatten(self, lst: list) -> list:
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend(self._flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    def _evaluate_register(self, qb: object) -> list[int]:

        # We should always have a name for the register
        if hasattr(qb, "name") and qb.name is not None:
            # Get the reg info
            reg_name = qb.name
            if hasattr(reg_name, "name") and reg_name.name is not None:
                reg_name = reg_name.name
            if isinstance(reg_name, str) and "$" in reg_name:
                start = int(reg_name.split("$")[1])
                end = start
            elif reg_name in self.reg_name_to_start_end:
                start, end = self.reg_name_to_start_end[reg_name]
            else:
                raise ValueError(f"Undefined register: {reg_name}")

            # If we have indices, get those specific qubits
            if hasattr(qb, "indices") and isinstance(qb.indices, list):
                indices = [self._evaluate_expression(index_list) for index_list in qb.indices]
                indices = self._flatten(indices)

                # If only given one index, get at that location
                qubits_to_return = []
                for index in indices:
                    if index < end - start + 1:
                        qubits_to_return.append(start + index)
                    else:
                        raise ValueError(
                            f"Index {index} out of bounds for register {reg_name} with start {start} and end {end}"
                        )
                return qubits_to_return

            # If we don't have indices, return the whole register as a list of qubits
            qubits_to_return = []
            for i in range(start, end + 1):
                qubits_to_return.append(i)
            return qubits_to_return

        raise ValueError(f"Unsupported qubit specification: {qb}")  # pragma: no cover

    @staticmethod
    def _str_to_gate(gate_name: str, qubit: int, arguments: list[float]) -> Gate:
        if gate_name == "x":
            return X(qubit)
        if gate_name == "y":
            return Y(qubit)
        if gate_name == "z":
            return Z(qubit)
        if gate_name == "h":
            return H(qubit)
        if gate_name == "s":
            return S(qubit)
        if gate_name == "t":
            return T(qubit)
        if gate_name == "rx":
            return RX(qubit, theta=arguments[0])
        if gate_name == "ry":
            return RY(qubit, theta=arguments[0])
        if gate_name == "rz":
            return RZ(qubit, phi=arguments[0])
        if gate_name == "u1":
            return U1(qubit, phi=arguments[0])
        if gate_name == "u2":
            return U2(qubit, phi=arguments[0], gamma=arguments[1])
        if gate_name in {"u", "u3"}:
            return U3(qubit, theta=arguments[0], phi=arguments[1], gamma=arguments[2])
        raise ValueError(f"Unsupported gate: {gate_name}")  # pragma: no cover

    def _to_qilisdk_gate(
        self, gate_name: str, qubits: list, arguments: list[float] = [], modifiers: list[str] = []
    ) -> list[Gate]:

        # Process the gate info
        gate_name = gate_name.lower()
        gates_to_return = []
        num_controls_total = 0
        for modifier in modifiers:
            if modifier in {"ctrl", "negctrl"}:
                num_controls_total += 1
        qubits_without_controls = qubits[num_controls_total:]

        # The gate itself
        if gate_name == "cx":
            gates_to_return.append(CNOT(*qubits_without_controls))
        elif gate_name == "cz":
            gates_to_return.append(CZ(*qubits_without_controls))
        else:
            for qubit in qubits_without_controls:
                gates_to_return.append(self._str_to_gate(gate_name, qubit, arguments))

        # Add controls if we have them
        main_gates = copy(gates_to_return)
        gates_to_return = []
        for j in range(len(main_gates)):
            main_gate = main_gates[j]
            gates_to_prepend = []
            gates_to_append = []
            num_repeats = 1
            for i in range(len(modifiers) - 1, -1, -1):
                if modifiers[i] == "ctrl":
                    main_gate = Controlled(qubits[i], basic_gate=main_gate)
                elif modifiers[i] == "negctrl":
                    main_gate = Controlled(qubits[i], basic_gate=main_gate)
                    gates_to_prepend.append(X(qubits[i]))
                    gates_to_append.append(X(qubits[i]))
                elif modifiers[i] == "inv":
                    main_gate = Adjoint(main_gate)  # ty:ignore[invalid-argument-type]
                elif modifiers[i] == "pow":
                    num_repeats += 1
                else:
                    raise ValueError(f"Unsupported gate modifier: {modifiers[i]}")  # pragma: no cover
            for _ in range(num_repeats):
                gates_to_return.extend(gates_to_prepend)
                gates_to_return.append(main_gate)
                gates_to_return.extend(gates_to_append)

        return gates_to_return

    def _cast_to_type(self, var_name: str | Identifier) -> None:

        # If we have a variable name object, get the name string
        if isinstance(var_name, Identifier):
            var_name = var_name.name  # pragma: no cover

        # Make sure the type is correct
        if "type" in self.var_list[var_name] and self.var_list[var_name]["type"] is not None:
            var_type = self.var_list[var_name]["type"]
            if var_type in {"uint", "int", "bit"}:
                self.var_list[var_name]["value"] = int(self.var_list[var_name]["value"])
            elif var_type in {"float", "angle", "duration", "stretch"}:
                self.var_list[var_name]["value"] = float(self.var_list[var_name]["value"])
            elif var_type == "complex":
                self.var_list[var_name]["value"] = complex(self.var_list[var_name]["value"])  # ty: ignore[invalid-assignment]
            elif var_type == "bool":
                self.var_list[var_name]["value"] = bool(self.var_list[var_name]["value"])

            # Truncate to the variable size if needed
            if "size" in self.var_list[var_name] and self.var_list[var_name]["size"] is not None:
                var_size = self.var_list[var_name]["size"]
                if var_type in {"uint", "bit"}:
                    self.var_list[var_name]["value"] %= 2**var_size
                elif var_type == "int":
                    self.var_list[var_name]["value"] = (
                        (self.var_list[var_name]["value"] + 2 ** (var_size - 1)) % (2**var_size)
                    ) - 2 ** (var_size - 1)

    def _handle_statement_qubit_declaration(self, statement: QubitDeclaration) -> None:
        reg_name = statement.qubit.name
        reg_size = 1
        if hasattr(statement, "size") and statement.size is not None:
            reg_size = self._evaluate_expression(statement.size)
        if isinstance(reg_size, int):
            self.reg_name_to_start_end[reg_name] = (self.nqubits, self.nqubits + reg_size - 1)
            self.var_list[reg_name] = {"size": reg_size, "value": 0, "type": "qubit"}  # ty: ignore[invalid-assignment]
            self.nqubits = max(self.nqubits, self.nqubits + reg_size)

    def _handle_statement_classical_declaration(self, statement: ClassicalDeclaration | ConstantDeclaration) -> None:
        var_name = statement.identifier.name
        var_size = 0
        var_value = 0
        var_type = "bit"
        if hasattr(statement, "init_expression") and statement.init_expression is not None:
            var_value = self._evaluate_expression(statement.init_expression)
        if hasattr(statement, "type") and statement.type is not None:
            if isinstance(statement.type, UintType):
                var_type = "uint"
                var_size = 32
            elif isinstance(statement.type, BitType):
                var_type = "bit"
                var_size = 1
            elif isinstance(statement.type, BoolType):
                var_type = "bool"
                var_size = 1
            elif isinstance(statement.type, IntType):
                var_type = "int"
                var_size = 32
            elif isinstance(statement.type, FloatType):
                var_type = "float"
                var_size = 64
            elif isinstance(statement.type, AngleType):
                var_type = "angle"
                var_size = 64
            elif isinstance(statement.type, DurationType):
                var_type = "duration"
                var_size = 32
            elif isinstance(statement.type, StretchType):
                var_type = "stretch"
                var_size = 32
            elif isinstance(statement.type, ComplexType):
                var_type = "complex"
                var_size = 128
            elif isinstance(statement.type, ArrayType):
                var_type = "array"
            else:
                raise ValueError(f"Unsupported variable type: {statement.type}")  # pragma: no cover
            if hasattr(statement.type, "size") and statement.type.size is not None:
                var_size = self._evaluate_expression(statement.type.size)
        self.var_list[var_name] = {
            "size": var_size,
            "value": var_value,
            "type": var_type,
        }  # ty: ignore[invalid-assignment]
        self._cast_to_type(var_name)

    def _handle_statement_classical_assignment(self, statement: ClassicalAssignment) -> None:
        var_name = statement.lvalue.name
        new_value = self._evaluate_expression(statement.rvalue)

        # Depending on the assignment type
        if not isinstance(var_name, Identifier):
            if statement.op == AssignmentOperator["="]:
                self.var_list[var_name]["value"] = new_value  # ty: ignore[invalid-assignment]
            elif statement.op == AssignmentOperator["+="] and not isinstance(new_value, (str, list)):
                self.var_list[var_name]["value"] += new_value
            elif statement.op == AssignmentOperator["-="] and not isinstance(new_value, (str, list)):
                self.var_list[var_name]["value"] -= new_value

        # Cast to the variable type if needed
        self._cast_to_type(var_name)

    def _handle_modifier(self, modifier: object, modifiers: list[str]) -> None:
        modifier_name = ""
        if not hasattr(modifier, "modifier"):
            return  # pragma: no cover
        if modifier.modifier == GateModifierName["ctrl"]:
            modifier_name = "ctrl"
        elif modifier.modifier == GateModifierName["negctrl"]:
            modifier_name = "negctrl"
        elif modifier.modifier == GateModifierName["inv"]:
            modifier_name = "inv"
        elif modifier.modifier == GateModifierName["pow"]:
            modifier_name = "pow"
        else:
            raise ValueError(f"Unsupported gate modifier: {modifier}")  # pragma: no cover

        # Repeat if needed
        repeats_needed = 1
        if hasattr(modifier, "argument") and modifier.argument is not None:
            repeats_needed = self._evaluate_expression(modifier.argument)
        if modifier_name == "pow":
            if (
                repeats_needed == 0
                or not isinstance(repeats_needed, int)
                or not hasattr(modifier, "argument")
                or modifier.argument is None
            ):
                raise ValueError(f"Invalid value for pow modifier: {modifier}")
            if repeats_needed < 0:
                repeats_needed = -repeats_needed - 1
                modifiers.append("inv")
            else:
                repeats_needed -= 1

        if isinstance(repeats_needed, int):
            for _ in range(repeats_needed):
                modifiers.append(modifier_name)

    def _get_modifiers_for_statement(self, statement: QuantumGate) -> tuple[list[str], int]:

        modifiers = []
        num_controls = 0

        # Return early if we don't have any modifiers
        if not (hasattr(statement, "modifiers") and isinstance(statement.modifiers, list)):
            return modifiers, num_controls  # pragma: no cover

        # For each modifier, get the name
        for modifier in statement.modifiers:
            self._handle_modifier(modifier, modifiers)

        # Count the controls
        for modifier in modifiers:
            if modifier in {"ctrl", "negctrl"}:
                num_controls += 1

        return modifiers, num_controls

    def _handle_statement_quantum_gate(
        self, statement: QuantumGate, extra_qubits: list[int] = [], extra_modifiers: list[str] = []
    ) -> None:

        # Get info about the gates
        gate_name = statement.name.name
        qubits = extra_qubits.copy()
        for qubit in statement.qubits:
            qubits.extend(self._evaluate_register(qubit))
        arguments = []
        for argument in statement.arguments:
            arguments.append(self._evaluate_expression(argument))
        modifiers = extra_modifiers.copy()
        new_modifiers, num_controls = self._get_modifiers_for_statement(statement)
        modifiers.extend(new_modifiers)

        # If it's a custom
        if gate_name in self.custom_gate_definitions:
            gate_def = self.custom_gate_definitions[gate_name]
            replacement_map = {}
            for actual_arg in arguments:
                arg_name_in_body = gate_def["args"][arguments.index(actual_arg)]
                replacement_map[arg_name_in_body] = actual_arg
            reg_names = [qubit.name for qubit in statement.qubits]
            for reg_name in reg_names[num_controls : num_controls + len(gate_def["qubits"])]:
                reg_name_in_body = gate_def["qubits"][reg_names.index(reg_name) - num_controls]
                replacement_map[reg_name_in_body] = reg_name
            control_qubits = qubits[:num_controls]
            for gate_statement in gate_def["body"]:
                self._process_statement(
                    self._recursive_replace(gate_statement, replacement_map),
                    extra_modifiers=modifiers,
                    extra_qubits=control_qubits,
                )

        # Otherwise process normally
        else:
            self.gates_to_add.extend(self._to_qilisdk_gate(gate_name, qubits, arguments, modifiers))

    def _handle_statement_branching_statement(
        self, statement: BranchingStatement, extra_modifiers: list[str] = [], extra_qubits: list[int] = []
    ) -> complex | bool | int | float | str | None:

        # Check the condition
        condition_value = self._evaluate_expression(statement.condition)

        # If the condition is true, process the true body
        if condition_value:
            for branched_statement in statement.if_block:
                res = self._process_statement(
                    branched_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
                )
                if res:
                    return self._parse_return_val(res)
        else:
            for branched_statement in statement.else_block:
                res = self._process_statement(
                    branched_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
                )
                if res:
                    return self._parse_return_val(res)

        return ""

    def _handle_statement_switch_statement(
        self, statement: SwitchStatement, extra_modifiers: list[str] = [], extra_qubits: list[int] = []
    ) -> complex | bool | int | float | str | None:

        # Get the value of the target expression
        target_val = self._evaluate_expression(statement.target)
        found_case = False

        # Check each case
        for case in statement.cases:
            case_val = self._evaluate_expression(case[0][0])
            if target_val == case_val:
                found_case = True
                for case_statement in case[1].statements:
                    res = self._process_statement(
                        case_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
                    )
                    if res:
                        return self._parse_return_val(res)
                break

        # If not, then use the default case
        if not found_case and hasattr(statement, "default") and statement.default is not None:
            for default_statement in statement.default.statements:
                res = self._process_statement(
                    default_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
                )
                if res:
                    return self._parse_return_val(res)

        return ""

    def _handle_statement_for_in_loop(
        self, statement: ForInLoop, extra_modifiers: list[str] = [], extra_qubits: list[int] = []
    ) -> complex | bool | int | float | str | None:

        # The new variable to declare for the loop variable
        loop_var_name = statement.identifier.name
        loop_var_type = statement.type
        loop_var_size = 1
        if hasattr(statement.type, "size") and statement.type.size is not None:
            loop_var_size = self._evaluate_expression(statement.type.size)
        loop_range = []

        # If it's a range-based for loop
        if isinstance(statement.set_declaration, RangeDefinition):
            loop_var_starting_value = self._evaluate_expression(statement.set_declaration.start)
            loop_var_step = 1
            if hasattr(statement.set_declaration, "step") and statement.set_declaration.step is not None:
                loop_var_step = self._evaluate_expression(statement.set_declaration.step)
            loop_var_final_value = self._evaluate_expression(statement.set_declaration.end)
            if (
                not isinstance(loop_var_step, int)
                or not isinstance(loop_var_starting_value, int)
                or not isinstance(loop_var_final_value, int)
            ):
                raise ValueError(f"Invalid loop setup: {statement.set_declaration}")
            loop_range = range(loop_var_starting_value, loop_var_final_value, loop_var_step)

        # If it's looping over an array
        elif isinstance(statement.set_declaration, Identifier):
            array_var_name = statement.set_declaration.name
            loop_range = self.var_list[array_var_name]["value"]

        # If it's a set
        elif isinstance(statement.set_declaration, DiscreteSet):
            values = statement.set_declaration.values
            loop_range = self._evaluate_expression(values)

        else:
            raise ValueError(f"Unsupported for loop declaration: {statement.set_declaration}")  # pragma: no cover

        # Make the variable
        self.var_list[loop_var_name] = {
            "size": loop_var_size,
            "value": 0,
            "type": loop_var_type,
        }  # ty: ignore[invalid-assignment]

        # Loop through the values and process the body with the loop variable set to the current value
        res = None
        if not isinstance(loop_range, (list, range)):
            raise ValueError(f"Loop statement is not iterable: {statement}")
        for i in loop_range:
            self.var_list[loop_var_name]["value"] = i
            for loop_statement in statement.block:
                res = self._process_statement(
                    loop_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
                )
                if res:
                    break
            if res == "break":
                break
            if res == "continue":
                continue
            if res:
                return self._parse_return_val(res)

        # Remove the loop variable from the var list after the loop is done
        del self.var_list[loop_var_name]

        return ""

    def _handle_statement_while_loop(
        self, statement: WhileLoop, extra_modifiers: list[str] = [], extra_qubits: list[int] = []
    ) -> complex | bool | int | float | str | None:
        res = None
        while self._evaluate_expression(statement.while_condition):
            for loop_statement in statement.block:
                res = self._process_statement(
                    loop_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
                )
                if res:
                    break
            if res == "break":
                break
            if res == "continue":
                continue
            if res:
                return self._parse_return_val(res)

        return ""

    def _handle_statement_quantum_measurement(self, statement: QuantumMeasurementStatement) -> None:
        qubit_statement = statement.measure.qubit
        qubits_to_measure = self._evaluate_register(qubit_statement)
        for qubit in qubits_to_measure:
            self.gates_to_add.append(M(qubit))
        if hasattr(statement, "target") and statement.target is not None:
            raise ValueError("Measurement statements with targets are not currently supported")

    def _process_statement(
        self, statement: object, extra_modifiers: list[str] = [], extra_qubits: list[int] = []
    ) -> str | int | float | complex | bool | None:

        # Initializing a qubit
        if isinstance(statement, QubitDeclaration):
            self._handle_statement_qubit_declaration(statement)

        # Initializing a classical variable
        elif isinstance(statement, (ClassicalDeclaration, ConstantDeclaration)):
            self._handle_statement_classical_declaration(statement)

        # Classical assignment
        elif isinstance(statement, ClassicalAssignment):
            self._handle_statement_classical_assignment(statement)

        # Quantum gates
        elif isinstance(statement, QuantumGate):
            self._handle_statement_quantum_gate(statement, extra_qubits=extra_qubits, extra_modifiers=extra_modifiers)

        # Branching statements
        elif isinstance(statement, BranchingStatement):
            return self._handle_statement_branching_statement(
                statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
            )

        # Switch statements
        elif isinstance(statement, SwitchStatement):
            return self._handle_statement_switch_statement(
                statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
            )

        # For Loops
        elif isinstance(statement, ForInLoop):
            return self._handle_statement_for_in_loop(
                statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
            )

        # While Loops
        elif isinstance(statement, WhileLoop):
            return self._handle_statement_while_loop(
                statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits
            )

        # Break statement
        elif isinstance(statement, BreakStatement):
            return "break"

        # Continue statement
        elif isinstance(statement, ContinueStatement):
            return "continue"

        # Return statement
        elif isinstance(statement, ReturnStatement):
            return_val = (
                str(self._evaluate_expression(statement.expression))
                if hasattr(statement, "expression") and statement.expression is not None
                else ""
            )
            return "return:" + return_val

        # Custom gate definitions
        elif isinstance(statement, QuantumGateDefinition):
            gate_name = statement.name.name
            gate_args = [arg.name for arg in statement.arguments]
            gate_qubits = [arg.name for arg in statement.qubits]
            gate_body = statement.body
            self.custom_gate_definitions[gate_name] = {"args": gate_args, "qubits": gate_qubits, "body": gate_body}

        # Subroutine definitions
        elif isinstance(statement, SubroutineDefinition):
            sub_name = statement.name.name
            sub_args = [arg.name.name for arg in statement.arguments]
            sub_body = statement.body
            self.subroutine_definitions[sub_name] = {"args": sub_args, "body": sub_body}

        # Measurements
        elif isinstance(statement, QuantumMeasurementStatement):
            self._handle_statement_quantum_measurement(statement)

        # If it's an expression, evaluate it just in case there's a subroutine call inside
        elif isinstance(statement, ExpressionStatement):
            self._evaluate_expression(statement.expression)

        # If it's an input type, make a parameter
        elif isinstance(statement, IODeclaration):
            # For now we only support input, not output
            if statement.io_identifier != IOKeyword["input"]:
                raise ValueError(f"Unsupported IO statement: {statement}")  # pragma: no cover

            # Otherwise, declare a parameter
            param_name = statement.identifier.name
            new_param = Parameter(param_name, 0.0)
            self.var_list[param_name] = {
                "size": 1,
                "value": new_param,
                "type": "parameter",
            }  # ty: ignore[invalid-assignment]

        # Otherwise raise an error for now - we can add more statement types later
        elif not isinstance(statement, (Include, AliasStatement)):
            raise ValueError(f"Unsupported statement type: {type(statement)}")  # pragma: no cover

        return ""

    def from_qasm3(self, qasm3: str, directory: str = "") -> Circuit:

        # Check for includes, if so, add their text to the qasm3 text and re-parse to get a full AST with all statements
        qasm3_with_includes = qasm3
        qasm3_with_includes = qasm3_with_includes.replace(";", ";\n")
        new_qasm3 = ""
        for line in qasm3.splitlines():
            if line.strip().startswith("include"):
                include_filename = line.strip().split(" ")[1].replace('"', "").replace("'", "").replace(";", "")
                if include_filename != "stdgates.inc":
                    include_path = Path(directory) / include_filename
                    if include_path.is_file():
                        include_qasm = include_path.read_text(encoding="utf-8")
                        include_qasm_lines = include_qasm.splitlines()
                        include_qasm_lines = [
                            line
                            for line in include_qasm_lines
                            if not line.strip().startswith("OPENQASM")
                            and not line.strip().startswith('include "stdgates.inc"')
                        ]
                        include_qasm = "\n".join(include_qasm_lines)
                        new_qasm3 += "\n" + include_qasm
                    else:
                        raise ValueError(f"Unsupported include statement: {line}")  # pragma: no cover
            else:
                new_qasm3 += line + "\n"
        qasm3_with_includes = new_qasm3

        # Find any let statements and do a find/replace for the alias in the qasm3 text
        as_lines = qasm3_with_includes.splitlines()
        for i, line in enumerate(as_lines):
            if line.strip().startswith("let "):
                let_statement = line.strip()[len("let ") :].rstrip(";")
                alias_name, alias_value = let_statement.split(" = ")
                alias_name = alias_name.strip()
                alias_value = alias_value.strip()
                for j in range(i + 1, len(as_lines)):
                    as_lines[j] = as_lines[j].replace(alias_name, alias_value)
        qasm3_with_includes = "\n".join(as_lines)

        # Use the official OpenQASM 3.0 parser to parse the text and get the AST nodes
        ast = parse(qasm3_with_includes)

        # The vars to fill as we parse the tree
        self.nqubits = 0
        self.reg_name_to_start_end = {}
        self.var_list = {
            "π": {"size": 1, "value": pi},
            "pi": {"size": 1, "value": pi},
            "τ": {"size": 1, "value": 2 * pi},
            "tau": {"size": 1, "value": 2 * pi},
            "euler": {"size": 1, "value": e},
            "ℇ": {"size": 1, "value": e},
        }
        self.custom_gate_definitions = {}
        self.subroutine_definitions = {}
        self.gates_to_add = []

        # Go through the tree and determine what to do
        for statement in ast.statements:
            self._process_statement(statement)

        # Create a Circuit with the determined number of qubits
        c = Circuit(self.nqubits)
        for gate in self.gates_to_add:
            c.add(gate)

        return c

    @staticmethod
    def to_qasm3(circuit: Circuit) -> str:
        qasm3 = "OPENQASM 3.0;\n"
        qasm3 += 'include "stdgates.inc";\n'
        if circuit.nqubits > 0:
            qasm3 += f"qubit[{circuit.nqubits}] q;\n"
        for gate in circuit.gates:
            qasm_gate_name = gate.name.lower()
            qasm_parameter_str = ""
            if gate.is_parameterized:
                qasm_parameter_str = "(" + ", ".join([str(param) for param in gate.get_parameter_values()]) + ")"
            qasm_control_str = "".join(["ctrl @ " for _ in gate.control_qubits])
            qasm_qubits_str = ", ".join([f"q[{qb}]" for qb in gate.qubits])
            qasm_gate_string = f"{qasm_control_str} {qasm_gate_name}{qasm_parameter_str} {qasm_qubits_str}"
            qasm_gate_string = qasm_gate_string.strip()
            qasm3 += f"{qasm_gate_string};\n"
        return qasm3.strip()


def from_qasm3(qasm3: str, directory: str = "") -> Circuit:
    """
    Convert an OpenQASM 3.0 string representation of a quantum circuit into a Circuit object.

    Args:
        qasm3 (str): The OpenQASM 3.0 string representation of the quantum circuit.
        directory (str): The directory to resolve include statements from. Defaults to the current directory.

    Returns:
        Circuit: The reconstructed Circuit object.

    Raises:
        ValueError: If the QASM string contains unsupported statements, gates, or expressions.
    """
    return OpenQasmParser().from_qasm3(qasm3, directory)


def to_qasm3(circuit: Circuit) -> str:
    """
    Convert a Circuit object to its OpenQASM 3.0 string representation.

    Args:
        circuit: The Circuit object to convert.

    Returns:
        str: The OpenQASM 3.0 representation of the circuit.
    """
    return OpenQasmParser.to_qasm3(circuit)


def from_qasm3_file(filename: str) -> Circuit:
    """
    Read an OpenQASM 3.0 file and create a corresponding Circuit instance.

    Args:
        filename (str): The path to the QASM file.

    Returns:
        Circuit: The reconstructed Circuit object.
    """
    qasm_str = Path(filename).read_text(encoding="utf-8")
    directory = str(Path(filename).parent)
    return from_qasm3(qasm_str, directory)


def to_qasm3_file(circuit: Circuit, filename: str) -> None:
    """
    Save the OpenQASM 3.0 representation of a circuit to a file.

    Args:
        circuit: The circuit to convert to OpenQASM 3.0.
        filename (str): The path to the file where the QASM code will be saved.
    """
    qasm_code = to_qasm3(circuit)
    Path(filename).write_text(qasm_code, encoding="utf-8")
