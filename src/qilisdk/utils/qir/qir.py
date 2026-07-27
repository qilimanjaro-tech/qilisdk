# Copyright 2026 Qilimanjaro Quantum Tech
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
"""QIR (Quantum Intermediate Representation) Base-Profile import / export.

Bridges :class:`qilisdk.digital.Circuit` and the `QIR Base Profile
<https://github.com/qir-alliance/qir-spec/blob/main/specification/under_development/profiles/Base_Profile.md>`_
via Microsoft's `pyqir <https://pypi.org/project/pyqir/>`_ library.

The Base Profile is a flat, statically-allocated subset of QIR: a single entry
point with no classical control flow, all qubits and result registers declared
up front, measurements (typically) at the end. This maps cleanly onto the
existing :class:`Circuit` / :class:`Gate` model.

The :mod:`pyqir` dependency is optional; install with ``qilisdk[qir]``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from pyqir import (
    BasicQisBuilder,
    Call,
    Context,
    FloatConstant,
    Module,
    SimpleModule,
    Value,
    is_entry_point,
    ptr_id,
    required_num_qubits,
)
from pyqir.qis import swap as qis_swap

from qilisdk.digital import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    Adjoint,
    Circuit,
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

# === Gate ↔ QIR intrinsic mapping =================================================
#
# Each entry maps a qilisdk gate class to the matching method name on pyqir's
# BasicQisBuilder. The Adjoint-of-S/T case is handled inline in `_emit_gate`
# because it pattern-matches on the inner gate.

_SIMPLE_GATE_EMITTERS: dict[type[Gate], str] = {
    X: "x",
    Y: "y",
    Z: "z",
    H: "h",
    S: "s",
    T: "t",
}


def _qubit_index(gate: Gate) -> int:
    """Return the single target qubit of ``gate``.

    Args:
        gate (Gate): A single-qubit gate.

    Returns:
        int: The target qubit index.

    Raises:
        ValueError: If the gate targets a different number of qubits.
    """
    if len(gate.qubits) != 1:
        raise ValueError(f"Expected a single-qubit gate, got {gate} on qubits {gate.qubits}")
    return gate.qubits[0]


def _scalar_param(gate: Gate, name: str) -> float:
    """Return the numeric value of a parameterized gate's angle.

    Args:
        gate (Gate): The parameterized gate.
        name (str): Parameter name (``theta`` / ``phi`` / ``lam``).

    Returns:
        float: The angle value in radians.
    """
    return float(getattr(gate, name))


def _produce_gate(
    module: SimpleModule,
    qis: BasicQisBuilder,
    qubits: list[Value],
    results: list[Value],
    gate: Gate,
) -> None:
    """Translate a single qilisdk :class:`Gate` into one (or more) QIS calls.

    Args:
        module (SimpleModule): The pyqir module whose builder backs ``qis``;
            used for intrinsics like ``swap`` that only have a function-style
            binding.
        qis (BasicQisBuilder): A QIS builder bound to the entry point.
        qubits (list[Value]): The module's statically allocated qubit values.
        results (list[Value]): The module's statically allocated result values.
        gate (Gate): The gate to produce.

    Raises:
        NotImplementedError: If the gate has no QIR Base-Profile mapping.
    """
    cls = type(gate)
    if cls is I:
        return  # identity is a no-op in QIR
    if cls in _SIMPLE_GATE_EMITTERS:
        getattr(qis, _SIMPLE_GATE_EMITTERS[cls])(qubits[_qubit_index(gate)])
        return
    if cls is RX:
        qis.rx(_scalar_param(gate, "theta"), qubits[_qubit_index(gate)])
        return
    if cls is RY:
        qis.ry(_scalar_param(gate, "theta"), qubits[_qubit_index(gate)])
        return
    if cls is RZ:
        qis.rz(_scalar_param(gate, "phi"), qubits[_qubit_index(gate)])
        return
    if cls is CNOT:
        qis.cx(qubits[gate.control_qubits[0]], qubits[gate.target_qubits[0]])
        return
    if cls is CZ:
        qis.cz(qubits[gate.control_qubits[0]], qubits[gate.target_qubits[0]])
        return
    if cls is SWAP:
        qis_swap(module.builder, qubits[gate.qubits[0]], qubits[gate.qubits[1]])
        return
    if isinstance(gate, Adjoint):
        inner_cls = type(gate.basic_gate)
        if inner_cls is S:
            qis.s_adj(qubits[_qubit_index(gate.basic_gate)])  # ty:ignore[invalid-argument-type]
            return
        if inner_cls is T:
            qis.t_adj(qubits[_qubit_index(gate.basic_gate)])  # ty:ignore[invalid-argument-type]
            return
    if cls is M:
        for q in gate.qubits:
            qis.mz(qubits[q], results[q])
        return
    raise NotImplementedError(
        f"Gate {gate.name!r} (type {cls.__name__}) has no QIR Base-Profile "
        "intrinsic. Decompose it into supported primitives before export."
    )


def _populate_module(circuit: Circuit, module: SimpleModule) -> None:
    """Walk ``circuit`` and emit each gate into ``module``'s entry point.

    Args:
        circuit (Circuit): The source circuit.
        module (SimpleModule): A pyqir :class:`SimpleModule` pre-allocated with
            enough qubits and results for the circuit.
    """
    qis = BasicQisBuilder(module.builder)
    qubits = list(module.qubits)
    results = list(module.results)
    logger.debug("[QIR] Emitting {} gates into module", len(circuit.gates))
    for gate in circuit.gates:
        logger.trace("[QIR] Emitting gate {} on qubits {}", gate.name, gate.qubits)
        _produce_gate(module, qis, qubits, results, gate)


# === QIR intrinsic → Gate factories (import side) =================================


def _qid(arg: Value) -> int:
    """Return the qubit/result pointer ID for a QIR call argument.

    ``pyqir.ptr_id`` returns ``None`` for the implicit zero pointer (``null``),
    which QIR uses to encode qubit/result index ``0``.

    Args:
        arg (Value): A pyqir value representing a qubit or result pointer.

    Returns:
        int: The numeric ID; ``0`` for the null pointer.
    """
    qid_value = ptr_id(arg)
    return 0 if qid_value is None else qid_value


def _float_value(arg: Value) -> float:
    """Extract the numeric value of a :class:`FloatConstant` rotation angle.

    Args:
        arg (Value): The first argument of a QIS rotation call.

    Returns:
        float: The rotation angle.

    Raises:
        TypeError: If ``arg`` is not a constant floating-point value.
    """
    if not isinstance(arg, FloatConstant):
        raise TypeError(f"Expected FloatConstant rotation angle, got {type(arg).__name__}")
    return arg.value


def _gate_from_call(callee: str, args: list[Value]) -> Gate:
    """Reconstruct a :class:`Gate` from a parsed QIS call.

    Args:
        callee (str): The mangled intrinsic name, e.g. ``__quantum__qis__h__body``.
        args (list[Value]): The argument list; for rotations the first entry is
            a :class:`FloatConstant` and the remaining entries are qubit /
            result pointer constants.

    Returns:
        Gate: The reconstructed gate.

    Raises:
        NotImplementedError: If the intrinsic is not recognized.
    """
    if callee == "__quantum__qis__x__body":
        return X(_qid(args[0]))
    if callee == "__quantum__qis__y__body":
        return Y(_qid(args[0]))
    if callee == "__quantum__qis__z__body":
        return Z(_qid(args[0]))
    if callee == "__quantum__qis__h__body":
        return H(_qid(args[0]))
    if callee == "__quantum__qis__s__body":
        return S(_qid(args[0]))
    if callee == "__quantum__qis__t__body":
        return T(_qid(args[0]))
    if callee == "__quantum__qis__s__adj":
        return Adjoint(S(_qid(args[0])))
    if callee == "__quantum__qis__t__adj":
        return Adjoint(T(_qid(args[0])))
    if callee == "__quantum__qis__rx__body":
        return RX(_qid(args[1]), theta=_float_value(args[0]))
    if callee == "__quantum__qis__ry__body":
        return RY(_qid(args[1]), theta=_float_value(args[0]))
    if callee == "__quantum__qis__rz__body":
        return RZ(_qid(args[1]), phi=_float_value(args[0]))
    if callee == "__quantum__qis__cnot__body":
        return CNOT(_qid(args[0]), _qid(args[1]))
    if callee == "__quantum__qis__cz__body":
        return CZ(_qid(args[0]), _qid(args[1]))
    if callee == "__quantum__qis__swap__body":
        return SWAP(_qid(args[0]), _qid(args[1]))
    if callee == "__quantum__qis__mz__body":
        return M(_qid(args[0]))
    raise NotImplementedError(
        f"QIS intrinsic {callee!r} is not supported. Add a mapping in "
        "`qilisdk.utils.qir._gate_from_call` to extend the supported set."
    )


# === Internal: module → Circuit walk =============================================


def _circuit_from_module(module: Module) -> Circuit:
    """Walk a parsed QIR module's entry point and rebuild a :class:`Circuit`.

    Args:
        module (Module): A pyqir :class:`Module` produced by
            :meth:`Module.from_ir` or :meth:`Module.from_bitcode`.

    Returns:
        Circuit: The reconstructed circuit.

    Raises:
        ValueError: If no Base-Profile entry-point function is present.
    """
    entry = next((f for f in module.functions if is_entry_point(f)), None)
    if entry is None:
        raise ValueError(
            "QIR module has no entry-point function; expected a Base-Profile "
            "module with one `entry_point`-attributed function."
        )
    nqubits = required_num_qubits(entry) or 0
    circuit = Circuit(nqubits=nqubits)
    for block in entry.basic_blocks:
        for instr in block.instructions:
            if isinstance(instr, Call):
                logger.trace("[QIR] Reconstructing gate from call {}", instr.callee.name)
                circuit.add(_gate_from_call(instr.callee.name, list(instr.args)))
    logger.debug("[QIR] Reconstructed circuit with {} qubits and {} gates", circuit.nqubits, len(circuit.gates))
    return circuit


# === Public API ===================================================================


def to_qir(circuit: Circuit, *, name: str = "circuit") -> str:
    """Serialize a :class:`Circuit` to a QIR Base-Profile textual LLVM IR string.

    Every qubit in the circuit gets its own pre-allocated result register, so
    measurements can target any qubit. The module advertises itself as a Base
    Profile module via the standard ``required_num_qubits`` /
    ``required_num_results`` attributes that QIR consumers expect.

    Args:
        circuit (Circuit): The circuit to serialize.
        name (str): Module name embedded in the IR. Defaults to ``"circuit"``.

    Returns:
        str: The QIR textual LLVM IR (``.ll`` syntax).

    Raises:
        NotImplementedError: If the circuit contains a gate with no Base-Profile
            mapping (e.g. ``U1`` / ``U2`` / ``U3``, three-qubit gates).
    """
    logger.info("[QIR] Exporting circuit to QIR")
    module = SimpleModule(name=name, num_qubits=circuit.nqubits, num_results=circuit.nqubits)
    _populate_module(circuit, module)
    return module.ir()


def to_qir_file(circuit: Circuit, filename: str, *, name: str | None = None) -> None:
    """Serialize a :class:`Circuit` to a QIR file.

    Dispatches on the file extension: ``.ll`` is written as textual LLVM IR and
    ``.bc`` as LLVM bitcode. Any other suffix is treated as textual.

    Args:
        circuit (Circuit): The circuit to serialize.
        filename (str): Destination path.
        name (str | None): Module name; defaults to the file stem.
    """
    path = Path(filename)
    module_name = name if name is not None else path.stem
    logger.debug("[QIR] Writing QIR to file {}", filename)
    if path.suffix.lower() == ".bc":
        module = SimpleModule(
            name=module_name,
            num_qubits=circuit.nqubits,
            num_results=circuit.nqubits,
        )
        _populate_module(circuit, module)
        path.write_bytes(module.bitcode())
        return
    path.write_text(to_qir(circuit, name=module_name), encoding="utf-8")


def from_qir(qir_text: str) -> Circuit:
    """Parse a QIR Base-Profile textual LLVM IR string into a :class:`Circuit`.

    Args:
        qir_text (str): The QIR textual LLVM IR.

    Returns:
        Circuit: The reconstructed circuit.

    Raises:
        ValueError: If the module has no Base-Profile entry-point function.
        NotImplementedError: If the IR uses an unsupported QIS intrinsic.
    """
    logger.info("[QIR] Importing circuit from QIR")
    module = Module.from_ir(Context(), qir_text)
    return _circuit_from_module(module)


def from_qir_file(filename: str) -> Circuit:
    """Read a QIR file (``.ll`` or ``.bc``) and parse it into a :class:`Circuit`.

    Args:
        filename (str): Path to a ``.ll`` or ``.bc`` QIR file.

    Returns:
        Circuit: The reconstructed circuit.
    """
    logger.debug("[QIR] Reading QIR from file {}", filename)
    path = Path(filename)
    ctx = Context()
    if path.suffix.lower() == ".bc":
        module = Module.from_bitcode(ctx, path.read_bytes())
    else:
        module = Module.from_ir(ctx, path.read_text(encoding="utf-8"))
    return _circuit_from_module(module)
