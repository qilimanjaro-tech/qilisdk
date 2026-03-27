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
    DurationType,
    ExpressionStatement,
    FloatType,
    ForInLoop,
    GateModifierName,
    ImaginaryLiteral,
    Include,
    IntType,
    QuantumGate,
    QuantumGateDefinition,
    QuantumMeasurementStatement,
    QubitDeclaration,
    StretchType,
    SwitchStatement,
    TimeUnit,
    UintType,
    UnaryOperator,
    WhileLoop,
)
from openqasm3.parser import parse

from qilisdk.digital import CNOT, CZ, RX, RY, RZ, U1, U2, U3, Adjoint, Circuit, Controlled, H, M, S, T, X, Y, Z


def _recursive_replace(statement: object, replacement_map: dict) -> object:
    new_statement = copy(statement)

    # If the statement is in the replacement map, replace it
    if hasattr(new_statement, "name") and new_statement.name is not None and isinstance(new_statement.name, str) and not hasattr(new_statement.name, "name") and new_statement.name in replacement_map:

        # If it's a string, change the name
        if isinstance(replacement_map[new_statement.name], str):
            setattr(new_statement, "name", replacement_map[new_statement.name])

        # If it's something else (i.e. a number), change the value
        else:
            setattr(new_statement, "value", replacement_map[new_statement.name])

    # For any other attributes, as long as it's not a default attribute or a callable, recurse
    for attr_name in dir(statement):
        if attr_name.startswith("__") or callable(getattr(new_statement, attr_name)) or isinstance(getattr(new_statement, attr_name), (str, int, float, type(None))):
            continue
        attr_value = getattr(new_statement, attr_name)
        if isinstance(attr_value, list):
            new_list = []
            for sub_value in attr_value:
                new_list.append(_recursive_replace(sub_value, replacement_map))
            setattr(new_statement, attr_name, new_list)
        else:
            setattr(new_statement, attr_name, _recursive_replace(attr_value, replacement_map))
    return new_statement


def _evaluate_expression(expr: object, var_list: dict) -> any:

    # Scale by the unit if we have it
    value_with_unit = None
    if hasattr(expr, "value") and expr.value is not None:
        value_with_unit = expr.value
        if hasattr(expr, "unit") and expr.unit is not None:
            if expr.unit == TimeUnit["ns"]:
                value_with_unit *= 1e-9
            elif expr.unit == TimeUnit["us"]:
                value_with_unit *= 1e-6
            elif expr.unit == TimeUnit["ms"]:
                value_with_unit *= 1e-3
            elif expr.unit != TimeUnit["s"]:
                raise ValueError(f"Unsupported time unit: {expr.unit}")

    # If it's an imaginary literal
    if isinstance(expr, ImaginaryLiteral) and value_with_unit is not None:
        return 1j * value_with_unit

    # If we have a value, perfect
    if value_with_unit is not None:
        return value_with_unit

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
        if expr.op == BinaryOperator["<<"]:
            return _evaluate_expression(expr.lhs, var_list) << _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator[">>"]:
            return _evaluate_expression(expr.lhs, var_list) >> _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["&"]:
            return _evaluate_expression(expr.lhs, var_list) & _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["|"]:
            return _evaluate_expression(expr.lhs, var_list) | _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["=="]:
            return _evaluate_expression(expr.lhs, var_list) == _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["!="]:
            return _evaluate_expression(expr.lhs, var_list) == _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator[">"]:
            return _evaluate_expression(expr.lhs, var_list) > _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator[">="]:
            return _evaluate_expression(expr.lhs, var_list) >= _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["<"]:
            return _evaluate_expression(expr.lhs, var_list) < _evaluate_expression(expr.rhs, var_list)
        if expr.op == BinaryOperator["<="]:
            return _evaluate_expression(expr.lhs, var_list) <= _evaluate_expression(expr.rhs, var_list)
        if expr.op == UnaryOperator["-"]:
            return -_evaluate_expression(expr.expression, var_list)
        if expr.op == UnaryOperator["!"]:
            return not _evaluate_expression(expr.expression, var_list)
        if expr.op == BinaryOperator["**"]:
            return _evaluate_expression(expr.lhs, var_list) ** _evaluate_expression(expr.rhs, var_list)
        raise ValueError(f"Unsupported operator: {expr.op}")

    # If it's a cast
    if isinstance(expr, Cast):
        value_to_cast = _evaluate_expression(expr.argument, var_list)
        if isinstance(expr.type, (UintType, IntType, BitType)):
            return int(value_to_cast) % (2 ** 32)
        if isinstance(expr.type, (FloatType, AngleType, DurationType, StretchType)):
            return float(value_to_cast)
        if isinstance(expr.type, ComplexType):
            return complex(value_to_cast)
        if isinstance(expr.type, BoolType):
            return bool(value_to_cast)
        raise ValueError(f"Unsupported cast type: {expr.type}")

    # If it's an array literal
    if hasattr(expr, "values") and expr.values is not None:
        return [_evaluate_expression(element, var_list) for element in expr.values]

    raise ValueError(f"Unsupported expression: {expr}")


def _evaluate_register(qb: object, var_list: dict, reg_name_to_start_end: dict) -> list[int]:

    # We should always have a name for the register
    if hasattr(qb, "name") and qb.name is not None:

        # Get the reg info
        reg_name = qb.name
        if hasattr(reg_name, "name") and reg_name.name is not None:
            reg_name = reg_name.name
        if "$" in reg_name:
            start = int(reg_name.split("$")[1])
            end = start
        elif reg_name in reg_name_to_start_end:
            start, end = reg_name_to_start_end[reg_name]
        else:
            raise ValueError(f"Undefined register: {reg_name}")

        # If we have indices, get those specific qubits
        if hasattr(qb, "indices") and qb.indices is not None:
            indices = [[_evaluate_expression(index, var_list) for index in index_list] for index_list in qb.indices]

            # If only given one index, get at that location
            if len(indices) == 1 and len(indices[0]) == 1 and indices[0][0] <= end - start + 1:
                return [start + indices[0][0]]
            raise ValueError(f"Index {indices[0][0]} out of bounds for register {reg_name} with start {start} and end {end}")

        # If we don't have indices, return the whole register as a list of qubits
        qubits_to_return = []
        for i in range(start, end + 1):
            qubits_to_return.append(i)
        return qubits_to_return

    raise ValueError(f"Unsupported qubit specification: {qb}")


def _to_qilisdk_gate(gate_name: str, qubits: list, arguments: list[float] = [], modifiers: list[str] = []) -> object:

    # Process the gate info
    gate_name = gate_name.lower()
    gates_to_return = []
    num_controls_total = 0
    for modifier in modifiers:
        if modifier in {"ctrl", "negctrl"}:
            num_controls_total += 1
    qubits_without_controls = qubits[num_controls_total:]

    # The gate itself
    if gate_name == "x":
        for qubit in qubits_without_controls:
            gates_to_return.append(X(qubit))
    elif gate_name == "y":
        for qubit in qubits_without_controls:
            gates_to_return.append(Y(qubit))
    elif gate_name == "z":
        for qubit in qubits_without_controls:
            gates_to_return.append(Z(qubit))
    elif gate_name == "h":
        for qubit in qubits_without_controls:
            gates_to_return.append(H(qubit))
    elif gate_name == "s":
        for qubit in qubits_without_controls:
            gates_to_return.append(S(qubit))
    elif gate_name == "t":
        for qubit in qubits_without_controls:
            gates_to_return.append(T(qubit))
    elif gate_name == "rx":
        for qubit in qubits_without_controls:
            gates_to_return.append(RX(qubit, theta=arguments[0]))
    elif gate_name == "ry":
        for qubit in qubits_without_controls:
            gates_to_return.append(RY(qubit, theta=arguments[0]))
    elif gate_name == "rz":
        for qubit in qubits_without_controls:
            gates_to_return.append(RZ(qubit, phi=arguments[0]))
    elif gate_name == "u1":
        for qubit in qubits_without_controls:
            gates_to_return.append(U1(qubit, phi=arguments[0]))
    elif gate_name == "u2":
        for qubit in qubits_without_controls:
            gates_to_return.append(U2(qubit, phi=arguments[0], gamma=arguments[1]))
    elif gate_name in {"u", "u3"}:
        for qubit in qubits_without_controls:
            gates_to_return.append(U3(qubit, theta=arguments[0], phi=arguments[1], gamma=arguments[2]))
    elif gate_name == "cx":
        gates_to_return.append(CNOT(*qubits_without_controls))
    elif gate_name == "cz":
        gates_to_return.append(CZ(*qubits_without_controls))
    else:
        raise ValueError(f"Unsupported gate: {gate_name}")

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
                main_gate = Adjoint(main_gate)
            elif modifiers[i] == "pow":
                num_repeats += 1
            else:
                raise ValueError(f"Unsupported gate modifier: {modifiers[i]}")
        for _ in range(num_repeats):
            gates_to_return.extend(gates_to_prepend)
            gates_to_return.append(main_gate)
            gates_to_return.extend(gates_to_append)

    return gates_to_return


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

    # Check for includes, if so, add their text to the qasm3 text and re-parse to get a full AST with all statements
    qasm3_with_includes = qasm3
    qasm3_with_includes = qasm3_with_includes.replace(';', ';\n')
    new_qasm3 = ""
    for line in qasm3.splitlines():
        if line.strip().startswith("include"):
            include_filename = line.strip().split(" ")[1].replace('"', '').replace("'", "").replace(";", "")
            if include_filename != "stdgates.inc":
                include_path = Path(directory) / include_filename
                if include_path.is_file():
                    include_qasm = include_path.read_text(encoding="utf-8")
                    include_qasm_lines = include_qasm.splitlines()
                    include_qasm_lines = [line for line in include_qasm_lines if not line.strip().startswith("OPENQASM") and not line.strip().startswith('include "stdgates.inc"')]
                    include_qasm = "\n".join(include_qasm_lines)
                    new_qasm3 += "\n" + include_qasm
                else:
                    raise ValueError(f"Unsupported include statement: {line}")
        else:
            new_qasm3 += line + "\n"
    qasm3_with_includes = new_qasm3

    # Find any let statements and do a find/replace for the alias in the qasm3 text
    as_lines = qasm3_with_includes.splitlines()
    for i, line in enumerate(as_lines):
        if line.strip().startswith("let "):
            let_statement = line.strip()[len("let "):].rstrip(";")
            alias_name, alias_value = let_statement.split(" = ")
            alias_name = alias_name.strip()
            alias_value = alias_value.strip()
            for j in range(i + 1, len(as_lines)):
                as_lines[j] = as_lines[j].replace(alias_name, alias_value)
    qasm3_with_includes = "\n".join(as_lines)

    # Use the official OpenQASM 3.0 parser to parse the text and get the AST nodes
    ast = parse(qasm3_with_includes)

    # The vars to fill as we parse the tree
    nqubits = 0
    reg_name_to_start_end = {}
    var_list = {
        "π": {"size": 1, "value": pi},
        "pi": {"size": 1, "value": pi},
        "τ": {"size": 1, "value": 2 * pi},
        "tau": {"size": 1, "value": 2 * pi},
        "euler": {"size": 1, "value": e},
        "ℇ": {"size": 1, "value": e},
    }
    custom_gate_definitions = {}
    gates_to_add = []

    def _process_statement(statement: object, extra_modifiers: list[str] = [], extra_qubits: list[int] = []) -> str:
        print()  # noqa: T201
        print(statement)  # noqa: T201
        nonlocal nqubits, reg_name_to_start_end, var_list, custom_gate_definitions, gates_to_add

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
            var_size = 0
            var_value = 0
            var_type = "bit"
            if hasattr(statement, "size") and statement.size is not None:
                var_size = _evaluate_expression(statement.size, var_list)
            if hasattr(statement, "init_expression") and statement.init_expression is not None:
                var_value = _evaluate_expression(statement.init_expression, var_list)
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
                    raise ValueError(f"Unsupported variable type: {statement.type}")
                if hasattr(statement.type, "size") and statement.type.size is not None:
                    var_size = _evaluate_expression(statement.type.size, var_list)
            var_list[var_name] = {"size": var_size, "value": var_value, "type": var_type}

        # Classical assignment
        elif isinstance(statement, ClassicalAssignment):
            var_name = statement.lvalue.name
            new_value = _evaluate_expression(statement.rvalue, var_list)
            if statement.op == AssignmentOperator["="]:
                var_list[var_name]["value"] = new_value
            elif statement.op == AssignmentOperator["+="]:
                var_list[var_name]["value"] += new_value
            elif statement.op == AssignmentOperator["-="]:
                var_list[var_name]["value"] -= new_value

            # Truncate to the variable size if needed
            if "size" in var_list[var_name] and var_list[var_name]["size"] is not None:
                var_size = var_list[var_name]["size"]
                if var_list[var_name]["type"] == "uint":
                    var_list[var_name]["value"] %= (2 ** var_size)
                elif var_list[var_name]["type"] == "int":
                    var_list[var_name]["value"] = ((var_list[var_name]["value"] + 2 ** (var_size - 1)) % (2 ** var_size)) - 2 ** (var_size - 1)

            # Cast to the variable type if needed
            if "type" in var_list[var_name] and var_list[var_name]["type"] is not None:
                var_type = var_list[var_name]["type"]
                if var_type in {"uint", "int", "bit"}:
                    var_list[var_name]["value"] = int(var_list[var_name]["value"])
                elif var_type in {"float", "angle", "duration", "stretch"}:
                    var_list[var_name]["value"] = float(var_list[var_name]["value"])
                elif var_type == "complex":
                    var_list[var_name]["value"] = complex(var_list[var_name]["value"])
                elif var_type == "bool":
                    var_list[var_name]["value"] = bool(var_list[var_name]["value"])

        # Quantum gates
        elif isinstance(statement, QuantumGate):

            # Get info about the gates
            gate_name = statement.name.name
            qubits = extra_qubits.copy()
            for qubit in statement.qubits:
                qubits.extend(_evaluate_register(qubit, var_list, reg_name_to_start_end))
            arguments = []
            for argument in statement.arguments:
                arguments.append(_evaluate_expression(argument, var_list))
            modifiers = extra_modifiers.copy()
            num_controls = 0
            if hasattr(statement, "modifiers") and statement.modifiers is not None:
                for modifier in statement.modifiers:

                    # Get the modifier name
                    modifier_name = ""
                    if hasattr(modifier, "modifier") and modifier.modifier == GateModifierName["ctrl"]:
                        modifier_name = "ctrl"
                        num_controls += 1
                    elif hasattr(modifier, "modifier") and modifier.modifier == GateModifierName["negctrl"]:
                        modifier_name = "negctrl"
                        num_controls += 1
                    elif hasattr(modifier, "modifier") and modifier.modifier == GateModifierName["inv"]:
                        modifier_name = "inv"
                    elif hasattr(modifier, "modifier") and modifier.modifier == GateModifierName["pow"]:
                        modifier_name = "pow"
                    else:
                        raise ValueError(f"Unsupported gate modifier: {modifier}")

                    # Repeat if needed
                    repeats_needed = 1
                    value_given = False
                    if hasattr(modifier, "argument") and modifier.argument is not None and hasattr(modifier.argument, "value") and modifier.argument.value is not None:
                        repeats_needed = int(modifier.argument.value)
                        value_given = True
                    if modifier_name == "pow":
                        if not value_given:
                            raise ValueError("Missing value for pow modifier")
                        if (repeats_needed == 0 or not isinstance(repeats_needed, int)):
                            raise ValueError(f"Invalid value for pow modifier: {modifier.argument.value}")
                        if repeats_needed < 0:
                            repeats_needed = -repeats_needed
                            modifiers.append("inv")
                        else:
                            repeats_needed -= 1

                    for _ in range(repeats_needed):
                        modifiers.append(modifier_name)

            # If it's a custom
            if gate_name in custom_gate_definitions:
                gate_def = custom_gate_definitions[gate_name]
                replacement_map = {}
                for actual_arg in arguments:
                    arg_name_in_body = gate_def["args"][arguments.index(actual_arg)]
                    replacement_map[arg_name_in_body] = actual_arg
                reg_names = [qubit.name for qubit in statement.qubits]
                for reg_name in reg_names[num_controls:num_controls + len(gate_def["qubits"])]:
                    reg_name_in_body = gate_def["qubits"][reg_names.index(reg_name) - num_controls]
                    replacement_map[reg_name_in_body] = reg_name
                control_qubits = qubits[:num_controls]
                for gate_statement in gate_def["body"]:
                    _process_statement(_recursive_replace(gate_statement, replacement_map), extra_modifiers=modifiers, extra_qubits=control_qubits)

            # Otherwise process normally
            else:
                gates_to_add.extend(_to_qilisdk_gate(gate_name, qubits, arguments, modifiers))

        # Include statements
        elif isinstance(statement, Include):
            if statement.filename.value != "stdgates.inc":
                include_path = Path(statement.filename.value)
                if include_path.is_file():
                    include_qasm = include_path.read_text(encoding="utf-8")
                    include_circuit = from_qasm3(include_qasm)
                    gates_to_add.extend(include_circuit.gates)

        # Branching statements
        elif isinstance(statement, BranchingStatement):

            # Check the condition
            condition_value = _evaluate_expression(statement.condition, var_list)

            # If the condition is true, process the true body
            if condition_value:
                for branched_statement in statement.if_block:
                    _process_statement(branched_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits)
            else:
                for branched_statement in statement.else_block:
                    _process_statement(branched_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits)

        # Switch statements TODO
        elif isinstance(statement, SwitchStatement):
            target_val = _evaluate_expression(statement.target, var_list)
            found_case = False
            for case in statement.cases:
                case_val = _evaluate_expression(case[0][0], var_list)
                if target_val == case_val:
                    found_case = True
                    for case_statement in case[1].statements:
                        _process_statement(case_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits)
                    break
            if not found_case and hasattr(statement, "default") and statement.default is not None:
                for default_statement in statement.default.statements:
                    _process_statement(default_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits)

        # For Loops
        elif isinstance(statement, ForInLoop):

            # The new variable to declare for the loop variable
            loop_var_name = statement.identifier.name
            loop_var_type = statement.type
            loop_var_size = statement.type.size

            # If it's a range-based for loop
            if hasattr(statement.set_declaration, "start") and statement.set_declaration.start is not None:
                loop_var_starting_value = _evaluate_expression(statement.set_declaration.start, var_list)
                loop_var_step = 1
                if hasattr(statement.set_declaration, "step") and statement.set_declaration.step is not None:
                    loop_var_step = _evaluate_expression(statement.set_declaration.step, var_list)
                loop_var_final_value = _evaluate_expression(statement.set_declaration.end, var_list)
                loop_range = range(loop_var_starting_value, loop_var_final_value, loop_var_step)

            # If it's looping over an array
            else:
                array_var_name = statement.set_declaration.name
                loop_range = var_list[array_var_name]["value"]

            # Make the variable
            var_list[loop_var_name] = {"size": loop_var_size, "value": 0, "type": loop_var_type}

            # Loop through the values and process the body with the loop variable set to the current value
            res = None
            for i in loop_range:
                var_list[loop_var_name]["value"] = i
                for loop_statement in statement.block:
                    res = _process_statement(loop_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits)
                    if res:
                        break
                if res == "break":
                    break
                if res == "continue":
                    continue

            # Remove the loop variable from the var list after the loop is done
            del var_list[loop_var_name]

        # While Loops
        elif isinstance(statement, WhileLoop):
            res = None
            while _evaluate_expression(statement.while_condition, var_list):
                for loop_statement in statement.block:
                    res = _process_statement(loop_statement, extra_modifiers=extra_modifiers, extra_qubits=extra_qubits)
                    if res:
                        break
                if res == "break":
                    break
                if res == "continue":
                    continue

        # Break statement TODO
        elif isinstance(statement, BreakStatement):
            return "break"

        # Continue statement TODO
        elif isinstance(statement, ContinueStatement):
            return "continue"

        # Custom gate definitions
        elif isinstance(statement, QuantumGateDefinition):
            gate_name = statement.name.name
            gate_args = [arg.name for arg in statement.arguments]
            gate_qubits = [arg.name for arg in statement.qubits]
            gate_body = statement.body
            custom_gate_definitions[gate_name] = {"args": gate_args, "qubits": gate_qubits, "body": gate_body}

        # Measurements
        elif isinstance(statement, QuantumMeasurementStatement):
            qubit_statement = statement.measure.qubit
            qubits_to_measure = _evaluate_register(qubit_statement, var_list, reg_name_to_start_end)
            for qubit in qubits_to_measure:
                gates_to_add.append(M(qubit))
            if hasattr(statement, "target") and statement.target is not None:
                raise ValueError("Measurement statements with targets are not currently supported")

        # Otherwise raise an error for now - we can add more statement types later
        elif not isinstance(statement, (AliasStatement, ExpressionStatement)):
            raise ValueError(f"Unsupported statement type: {type(statement)}")

        return ""

    # TODO: remove all prints

    # Go through the tree and determine what to do
    for statement in ast.statements:
        _process_statement(statement)

    print()  # noqa: T201
    print(f"Determined number of qubits: {nqubits}")  # noqa: T201
    print("Register name to qubit index mapping:")  # noqa: T201
    for reg_name, (start, end) in reg_name_to_start_end.items():
        print(f"  {reg_name}: qubits {start} to {end}")  # noqa: T201
    print("All variables:")  # noqa: T201
    for var_name, var_info in var_list.items():
        print(f"  {var_name}: type={var_info.get('type', 'unknown')}, size={var_info['size']}, value={var_info['value']}")  # noqa: T201
    print("Custom gate definitions:")  # noqa: T201
    for gate_name, gate_info in custom_gate_definitions.items():
        print(f"  {gate_name}: args={gate_info['args']}, qubits={gate_info['qubits']}")  # noqa: T201
    print("Gates to add:")  # noqa: T201
    for gate in gates_to_add:
        print(f"  {gate}")  # noqa: T201

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


"""
Feature Table
OpenQASM 3 Feature     Qiskit SDK       IBM Qiskit Runtime      QiliSDK       Qiskit Notes
comments               ✅               ✅                      ✅
QASM vstring           ✅               ✅                      ✅            1
include                🟡               ❌                      ✅            1, 7
unicode names          ✅               ✅                      ✅
qubit                  ✅               🟡                      ✅            2
bit                    ✅               ✅                      ✅            3
bool                   🟡               ✅                      ✅            4
int                    ❌               ✅                      ✅            4
uint                   🟡               ✅                      ✅            4
float                  🟡               🟡                      ✅            4
angle                  ❌               🟡                      ✅            4
complex                ❌               ❌                      ✅            4
const                  ❌               ❌                      ✅            4
pi/π/tau/τ/euler/ℇ     ✅               ✅                      ✅
Aliasing: let          🟡               ❌                      ✅            5
register concatenation 🟡               ❌                      ❌            5
casting expr.Cast      🟡               🟡                      ✅            4
duration               ❌               ❌                      ✅
durationof             ❌               ❌                      ❌
ns/μs/us/ms/s/dt       ✅               ✅                      ✅            6
stretch expr.Stretch   🟡               🟡                      ✅            4, 6
delay                  ✅               ✅                      ❌            6
barrier                ✅               ✅                      ❌
box                    ✅               ❌                      ❌            6
Built-in U             ✅               ✅                      ✅
gate                   🟡               🟡                      ✅            7
gphase                 🟡               ❌                      ❌            7
ctrl @/ negctrl @      🟡               ❌                      ✅            7
inv @                  🟡               ❌                      ✅            7
pow(k) @               🟡               ❌                      🟡            7
reset                  ✅               ✅                      ❌
measure                ✅               ✅                      🟡
bit operations         🟡               ✅                      ✅            4
boolean operations     🟡               ✅                      ✅            4
arithmetic expressions 🟡               🟡                      ✅            4
comparisons            🟡               ✅                      ✅            4
if                     ✅               ✅                      ✅            8
else                   ✅               ✅                      ✅            8
else if                ✅               ❌                      ✅            8
for loops              🟡               ❌                      ✅            8
switch                 ❌               ❌                      ✅
while loops            ✅               ❌                      ✅            8
continue               🟡               ❌                      ✅            8
break                  🟡               ❌                      ✅            8
extern                 ❌               ❌                      ❌
def subroutines        ❌               ❌                      TODO
return                 ❌               ❌                      TODO
input                  ✅               🟡                      ❌            4, 9
output                 ❌               ❌                      ❌

1) These OpenQASM 3 program features have no impact on the execution and Qiskit strips
them out as part of parsing the files. Files that use them can be submitted but they
will have no effect. For include files, stdgates.inc is currently supported as input
to Qiskit, and backend execution always requires circuits to have been compiled to
the backend Instruction Set Architecture (ISA), where include files are irrelevant.

2) Qiskit SDK supports parsing and dumping OpenQASM 3 files with any qubit declarations.
For execution on hardware, only circuits defined in terms of hardware qubits
(for example, $0) are valid. Qiskit SDK automatically outputs OpenQASM 3 in terms
of the supported hardware-qubit identifiers if the circuit was transpiled for a
backend with layout information.

3) bit- and bit[n]-typed variable declarations in Qiskit SDK correspond to Clbit
and ClassicalRegister declarations.

4) As of July 2025, Qiskit SDK can represent local variables of a restricted
set of types, can represent many runtime operations on these objects, and supports
outputting them to OpenQASM 3. However, Qiskit SDK (through qiskit-qasm3-import v0.6.0)
does not support parsing OpenQASM 3 files that contain variable declarations, and has very
limited support for parsing variable expressions. In general, most of what Qiskit can
represent in its expression system can be executed on suitable dynamic circuits hardware,
even if the expression cannot yet be parsed by Qiskit SDK. See the Qiskit documentation
of the qiskit.circuit.classical module for the most up-to-date information.

5) Qiskit SDK can represent register aliasing for both quantum and classical registers,
but it is strongly discouraged to use aliasing of classical registers. Most expressions
on classical registers do not work with aliases, and aliased classical registers are not
supported for execution on hardware. The Qiskit OpenQASM 3 parser can resolve let alias
statements that bind the result of register concantenation.

6) Qiskit SDK supports explicit delays via QuantumCircuit.delay, and circuit boxes
(QuantumCircuit.box) can also have explicit durations. These durations can include
classical expressions of stretch variables. Qiskit SDK (as of July 2025 through
qiskit-qasm3-import v0.6.0) does not support parsing declarations of type duration
or type stretch from OpenQASM 3 files. Hardware has limited support for durations including stretch.

7) Circuits must be transpiled to the backend ISA to run on IBM hardware. This precludes
custom gate definitions and higher-level constructs like gate modifiers (such as inv @)
from being valid for execution on hardware verbatim, but the transpile process resolves
them into valid ISA circuits. Qiskit SDK (as of July 2025, through qiskit-qasm3-import v0.6.0)
will eagerly evaluate gate modifiers during the parse, so these will not be evident
in the resulting QuantumCircuit, potentially at a runtime cost.

8) Qiskit SDK can represent structured control flow and export this to OpenQASM 3. The
continue and break statements can technically be represented by Qiskit, but are not well
supported even within Qiskit SDK. for loops in Qiskit v2.1.0 are not well supported. Nested
control flow (such as an if inside another if, or an else if statement) is not
eligible for execution on hardware.

9) Qiskit SDK supports declaring any supported classical type as an input variable on the
circuit. Such variables are not currently eligible for execution on hardware, and cannot
be loaded by the Qiskit OpenQASM 3 importer. Unbound Parameter objects present in the
QuantumCircuit are exported as input float[64] variables. Certain runtime configuration
options can enable executing such circuits on some backends.

"""
