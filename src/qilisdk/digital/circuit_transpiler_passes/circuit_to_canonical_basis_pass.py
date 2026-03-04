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

from __future__ import annotations

import math
from typing import TypeGuard

from qilisdk.digital import Circuit
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
    Exponential,
    Gate,
    H,
    I,
    M,
    X,
    Y,
    Z,
)

from .circuit_transpiler_pass import CircuitTranspilerPass
from .numeric_helpers import (
    _mat_U3,
    _unitary_sqrt_2x2,
    _wrap_angle,
    _zyz_from_unitary,
)


def _is_controlled(gate: Gate) -> TypeGuard[Controlled[BasicGate]]:
    return isinstance(gate, Controlled)


def _is_exponential(gate: Gate) -> TypeGuard[Exponential[BasicGate]]:
    return isinstance(gate, Exponential)


def _is_adjoint(gate: Gate) -> TypeGuard[Adjoint[BasicGate]]:
    return isinstance(gate, Adjoint)


# ======================= Small basis building blocks =======================


def _H_as_U3(qubit: int) -> list[Gate]:
    """Return a U3-only realization of a Hadamard on one qubit.

    Args:
        qubit (int): Target qubit.

    Returns:
        list[Gate]: Equivalent one-gate sequence in canonical basis.
    """
    # H = U2(0, π) = U3(π/2, 0, π) up to a global phase.
    return [U3(qubit, theta=math.pi / 2.0, phi=0.0, gamma=math.pi)]


def _CNOT_as_CZ_plus_1q(control_qubit: int, target_qubit: int) -> list[Gate]:
    """Return a CNOT decomposition using CZ and single-qubit gates.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.

    Returns:
        list[Gate]: Equivalent gate sequence in the canonical basis.
    """
    # CNOT = (I ⊗ H) · CZ · (I ⊗ H).
    return [*_H_as_U3(target_qubit), CZ(control_qubit, target_qubit), *_H_as_U3(target_qubit)]


def _CRZ_using_CNOT(control_qubit: int, target_qubit: int, lambda_angle: float) -> list[Gate]:
    """Return a controlled-RZ decomposition using CNOT-based primitives.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        lambda_angle (float): RZ rotation angle.

    Returns:
        list[Gate]: Equivalent controlled-RZ sequence in canonical basis.
    """
    # CRZ(λ) = (I ⊗ RZ(λ/2)) · CNOT · (I ⊗ RZ(-λ/2)) · CNOT.
    return [
        RZ(target_qubit, phi=_wrap_angle(lambda_angle / 2.0)),
        *_CNOT_as_CZ_plus_1q(control_qubit, target_qubit),
        RZ(target_qubit, phi=_wrap_angle(-lambda_angle / 2.0)),
        *_CNOT_as_CZ_plus_1q(control_qubit, target_qubit),
    ]


def _CRX_using_CRZ(control_qubit: int, target_qubit: int, theta: float) -> list[Gate]:
    """Return a controlled-RX decomposition via controlled-RZ.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        theta (float): RX rotation angle.

    Returns:
        list[Gate]: Equivalent controlled-RX sequence in canonical basis.
    """
    # RX(θ) = (I ⊗ RY(-π/2)) · CRZ(θ) · (I ⊗ RY(π/2)).
    return [
        RY(target_qubit, theta=-math.pi / 2.0),
        *_CRZ_using_CNOT(control_qubit, target_qubit, theta),
        RY(target_qubit, theta=math.pi / 2.0),
    ]


def _CRY_using_CRZ(control_qubit: int, target_qubit: int, theta: float) -> list[Gate]:
    """Return a controlled-RY decomposition via controlled-RZ.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        theta (float): RY rotation angle.

    Returns:
        list[Gate]: Equivalent controlled-RY sequence in canonical basis.
    """
    # RY(θ) = (I ⊗ RX(π/2)) · CRZ(θ) · (I ⊗ RX(-π/2)).
    return [
        RX(target_qubit, theta=math.pi / 2.0),
        *_CRZ_using_CNOT(control_qubit, target_qubit, theta),
        RX(target_qubit, theta=-math.pi / 2.0),
    ]


def _CU3_using_CNOT(
    control_qubit: int,
    target_qubit: int,
    theta: float,
    phi: float,
    lambda_angle: float,
) -> list[Gate]:
    """Return a controlled-U3 decomposition using two CNOT skeletons.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        theta (float): U3 theta parameter.
        phi (float): U3 phi parameter.
        lambda_angle (float): U3 gamma/lambda parameter.

    Returns:
        list[Gate]: Equivalent controlled-U3 sequence in canonical basis.
    """
    # Two-CX synthesis with CX realized by H-CZ-H.
    return [
        RZ(control_qubit, phi=_wrap_angle((lambda_angle + phi) / 2.0)),
        U3(target_qubit, theta=theta / 2.0, phi=phi, gamma=0.0),
        *_CNOT_as_CZ_plus_1q(control_qubit, target_qubit),
        U3(target_qubit, theta=-theta / 2.0, phi=0.0, gamma=_wrap_angle(-(lambda_angle + phi) / 2.0)),
        *_CNOT_as_CZ_plus_1q(control_qubit, target_qubit),
        RZ(target_qubit, phi=_wrap_angle((lambda_angle - phi) / 2.0)),
    ]


def _invert_basis_gate(gate: Gate) -> list[Gate]:
    """Return the inverse of a basis-rewritten gate as a gate sequence.

    Args:
        gate (Gate): Gate assumed to be in canonical basis or equivalent simple primitive.

    Returns:
        list[Gate]: Inverse sequence that undoes ``gate``.
    """
    if isinstance(gate, U3):
        return [U3(gate.qubits[0], theta=-gate.theta, phi=-gate.gamma, gamma=-gate.phi)]
    if isinstance(gate, RX):
        return [RX(gate.qubits[0], theta=-gate.theta)]
    if isinstance(gate, RY):
        return [RY(gate.qubits[0], theta=-gate.theta)]
    if isinstance(gate, RZ):
        return [RZ(gate.qubits[0], phi=-gate.phi)]
    if isinstance(gate, CZ):
        return [CZ(gate.control_qubits[0], gate.target_qubits[0])]
    if isinstance(gate, M):
        return [gate]
    if isinstance(gate, H):
        return _H_as_U3(gate.qubits[0])[::-1]
    if isinstance(gate, X):
        return [RX(gate.qubits[0], theta=-math.pi)]
    if isinstance(gate, Y):
        return [RY(gate.qubits[0], theta=-math.pi)]
    if isinstance(gate, Z):
        return [RZ(gate.qubits[0], phi=-math.pi)]
    # Should not happen after canonicalization.
    return [Adjoint(gate)]  # type: ignore[type-var]


# ======================= Canonicalization (mapping-only) =======================


def _as_basis_1q(gate: Gate) -> Gate:
    """Return an equivalent one-qubit gate expressed in canonical one-qubit basis.

    Args:
        gate (Gate): One-qubit gate to canonicalize.

    Returns:
        Gate: Equivalent gate in ``{U3, RX, RY, RZ}``.

    Raises:
        NotImplementedError: If ``gate`` is unsupported for one-qubit canonicalization.
    """
    qubit = gate.qubits[0]
    if isinstance(gate, (RX, RY, RZ, U3)):
        return gate
    if isinstance(gate, U1):
        return RZ(qubit, phi=gate.phi)
    if isinstance(gate, U2):
        return U3(qubit, theta=math.pi / 2.0, phi=gate.phi, gamma=gate.gamma)
    if isinstance(gate, H):
        return _H_as_U3(qubit)[0]
    if isinstance(gate, X):
        return RX(qubit, theta=math.pi)
    if isinstance(gate, Y):
        return RY(qubit, theta=math.pi)
    if isinstance(gate, Z):
        return RZ(qubit, phi=math.pi)
    if isinstance(gate, BasicGate) and gate.nqubits == 1:
        theta, phi, lambda_angle = _zyz_from_unitary(gate.matrix)
        return U3(qubit, theta=theta, phi=phi, gamma=lambda_angle)
    raise NotImplementedError(f"Unsupported 1-qubit gate type {type(gate).__name__} in _as_basis_1q")


def _sqrt_1q_gate_as_basis(gate: Gate) -> Gate:
    """Return a canonical one-qubit square root gate.

    Args:
        gate (Gate): One-qubit gate whose principal square root is requested.

    Returns:
        Gate: Canonical one-qubit gate ``V`` such that ``V^2`` equals ``gate`` up to global phase.
    """
    qubit = gate.qubits[0]
    # Fast paths
    if isinstance(gate, RZ):
        return RZ(qubit, phi=gate.phi / 2.0)
    if isinstance(gate, RX):
        return RX(qubit, theta=gate.theta / 2.0)
    if isinstance(gate, RY):
        return RY(qubit, theta=gate.theta / 2.0)
    if isinstance(gate, Z):
        return RZ(qubit, phi=math.pi / 2.0)
    if isinstance(gate, X):
        return RX(qubit, theta=math.pi / 2.0)
    if isinstance(gate, Y):
        return RY(qubit, theta=math.pi / 2.0)
    if isinstance(gate, U1):
        return RZ(qubit, phi=gate.phi / 2.0)

    # General 1q unitary
    if isinstance(gate, U2):
        unitary_matrix = _mat_U3(math.pi / 2.0, gate.phi, gate.gamma)
    elif isinstance(gate, U3):
        unitary_matrix = _mat_U3(gate.theta, gate.phi, gate.gamma)
    elif isinstance(gate, BasicGate) and gate.nqubits == 1:
        unitary_matrix = gate.matrix
    else:
        return _sqrt_1q_gate_as_basis(_as_basis_1q(gate))

    square_root_unitary = _unitary_sqrt_2x2(unitary_matrix)
    theta, phi, lambda_angle = _zyz_from_unitary(square_root_unitary)
    return U3(qubit, theta=theta, phi=phi, gamma=lambda_angle)


def _adjoint_1q(gate: Gate) -> Gate:
    """Return the adjoint of a one-qubit gate in canonical basis.

    Args:
        gate (Gate): One-qubit gate to invert.

    Returns:
        Gate: Adjoint gate expressed in canonical one-qubit basis.
    """
    qubit = gate.qubits[0]
    if isinstance(gate, RX):
        return RX(qubit, theta=-gate.theta)
    if isinstance(gate, RY):
        return RY(qubit, theta=-gate.theta)
    if isinstance(gate, RZ):
        return RZ(qubit, phi=-gate.phi)
    if isinstance(gate, U3):
        return U3(qubit, theta=-gate.theta, phi=-gate.gamma, gamma=-gate.phi)
    return _adjoint_1q(_as_basis_1q(gate))


class CircuitToCanonicalBasisPass(CircuitTranspilerPass):
    """
    Map an arbitrary circuit to the circuit basis {U3, RX, RY, RZ, CZ} (+ M).

    - Eliminates CNOT / SWAP to CZ + 1Q.
    - Controlled with any #controls (target 1-qubit) → ancilla-free recursive synthesis.
    - Adjoint(g) → canonicalize(g) then reverse+invert.
    - Exponential(1q) → ZYZ → U3.

    NOTE: This pass does *not* perform any 1-qubit fusion/merging.
    """

    def run(self, circuit: Circuit) -> Circuit:
        """Rewrite a circuit into the canonical digital basis.

        Args:
            circuit (Circuit): Input circuit to canonicalize.

        Returns:
            Circuit: New circuit containing only canonical gates and measurements.
        """
        rewritten_sequence = self._rewrite_list(circuit.gates)

        output_circuit = Circuit(circuit.nqubits)
        for gate in rewritten_sequence:
            output_circuit.add(gate)

        self.append_circuit_to_context(output_circuit)
        return output_circuit

    def _rewrite_list(self, gates: list[Gate]) -> list[Gate]:
        """Rewrite a gate list into canonical basis.

        Args:
            gates (list[Gate]): Input gate list.

        Returns:
            list[Gate]: Canonicalized gate list.
        """
        rewritten_gates: list[Gate] = []
        for gate in gates:
            rewritten_gates += self._rewrite_gate(gate)
        return rewritten_gates

    def _rewrite_gate(self, gate: Gate) -> list[Gate]:
        """Rewrite a single gate into an equivalent canonical sequence.

        Args:
            gate (Gate): Input gate to canonicalize.

        Returns:
            list[Gate]: Equivalent canonical gate sequence.

        Raises:
            NotImplementedError: If no rewrite rule exists for ``gate``.
        """
        # measurement passes through
        if isinstance(gate, M):
            return [gate]

        # already basis
        if isinstance(gate, (U3, RX, RY, RZ, CZ)):
            return [gate]

        # simple 1q
        if isinstance(gate, I):
            return []
        if isinstance(gate, H):
            return _H_as_U3(gate.qubits[0])
        if isinstance(gate, X):
            return [RX(gate.qubits[0], theta=math.pi)]
        if isinstance(gate, Y):
            return [RY(gate.qubits[0], theta=math.pi)]
        if isinstance(gate, Z):
            return [RZ(gate.qubits[0], phi=math.pi)]

        # param 1q
        if isinstance(gate, U1):
            return [RZ(gate.qubits[0], phi=gate.phi)]
        if isinstance(gate, U2):
            return [U3(gate.qubits[0], theta=math.pi / 2.0, phi=gate.phi, gamma=gate.gamma)]

        # 2q
        if isinstance(gate, CNOT):
            control_qubit, target_qubit = gate.control_qubits[0], gate.target_qubits[0]
            return _CNOT_as_CZ_plus_1q(control_qubit, target_qubit)
        if isinstance(gate, SWAP):
            first_target_qubit, second_target_qubit = gate.target_qubits
            return (
                _CNOT_as_CZ_plus_1q(first_target_qubit, second_target_qubit)
                + _CNOT_as_CZ_plus_1q(second_target_qubit, first_target_qubit)
                + _CNOT_as_CZ_plus_1q(first_target_qubit, second_target_qubit)
            )

        # Controlled (k controls) over a 1-qubit target
        if _is_controlled(gate):
            control_qubits = list(gate.control_qubits)
            basic_gate: BasicGate = gate.basic_gate
            if basic_gate.nqubits != 1:
                raise NotImplementedError("Controlled of multi-qubit gates not supported.")
            target_qubit = basic_gate.qubits[0]
            basis_one_qubit_gate = _as_basis_1q(basic_gate)
            if basis_one_qubit_gate.qubits[0] != target_qubit:
                if isinstance(basis_one_qubit_gate, U3):
                    basis_one_qubit_gate = U3(
                        target_qubit,
                        theta=basis_one_qubit_gate.theta,
                        phi=basis_one_qubit_gate.phi,
                        gamma=basis_one_qubit_gate.gamma,
                    )
                elif isinstance(basis_one_qubit_gate, RX):
                    basis_one_qubit_gate = RX(target_qubit, theta=basis_one_qubit_gate.theta)
                elif isinstance(basis_one_qubit_gate, RY):
                    basis_one_qubit_gate = RY(target_qubit, theta=basis_one_qubit_gate.theta)
                elif isinstance(basis_one_qubit_gate, RZ):
                    basis_one_qubit_gate = RZ(target_qubit, phi=basis_one_qubit_gate.phi)
            return _multi_controlled(control_qubits, basis_one_qubit_gate)

        # Adjoint
        if _is_adjoint(gate):
            rewritten_base_sequence = self._rewrite_gate(gate.basic_gate)
            inverse_sequence: list[Gate] = []
            for base_gate in reversed(rewritten_base_sequence):
                inverse_sequence += _invert_basis_gate(base_gate)
            return inverse_sequence

        # Exponential(1q)
        if _is_exponential(gate):
            basic_gate = gate.basic_gate
            if basic_gate.nqubits != 1:
                raise NotImplementedError("Exponential of multi-qubit gates not supported.")
            unitary_matrix = gate.matrix
            theta, phi, lambda_angle = _zyz_from_unitary(unitary_matrix)
            return [U3(basic_gate.qubits[0], theta=theta, phi=phi, gamma=lambda_angle)]

        # generic 1q
        if isinstance(gate, BasicGate) and gate.nqubits == 1:
            theta, phi, lambda_angle = _zyz_from_unitary(gate.matrix)
            return [U3(gate.qubits[0], theta=theta, phi=phi, gamma=lambda_angle)]

        raise NotImplementedError(f"No canonicalization rule for {type(gate).__name__}")

    # --- multi-control synthesis (uses helpers above) ---


def _single_controlled(control_qubit: int, target_gate: Gate) -> list[Gate]:
    """Return a single-control decomposition for a one-qubit target gate.

    Args:
        control_qubit (int): Control qubit index.
        target_gate (Gate): One-qubit target gate to control.

    Returns:
        list[Gate]: Equivalent sequence with canonical two-qubit primitives.
    """
    target_qubit = target_gate.qubits[0]
    if isinstance(target_gate, RZ):
        return _CRZ_using_CNOT(control_qubit, target_qubit, target_gate.phi)
    if isinstance(target_gate, RX):
        return _CRX_using_CRZ(control_qubit, target_qubit, target_gate.theta)
    if isinstance(target_gate, RY):
        return _CRY_using_CRZ(control_qubit, target_qubit, target_gate.theta)
    if isinstance(target_gate, U3):
        return _CU3_using_CNOT(
            control_qubit,
            target_qubit,
            target_gate.theta,
            target_gate.phi,
            target_gate.gamma,
        )
    return _single_controlled(control_qubit, _as_basis_1q(target_gate))


def _multi_controlled(control_qubits: list[int], base_one_qubit_gate: Gate) -> list[Gate]:
    """Recursively synthesize a multi-controlled one-qubit gate.

    Args:
        control_qubits (list[int]): Ordered control qubit indices.
        base_one_qubit_gate (Gate): One-qubit target gate in canonical one-qubit basis.

    Returns:
        list[Gate]: Canonical gate sequence implementing the same multi-controlled operation.
    """
    if not control_qubits:
        return [base_one_qubit_gate]
    if len(control_qubits) == 1:
        return _single_controlled(control_qubits[0], base_one_qubit_gate)

    last_control_qubit = control_qubits[-1]
    remaining_control_qubits = control_qubits[:-1]

    square_root_gate = _sqrt_1q_gate_as_basis(base_one_qubit_gate)
    square_root_adjoint_gate = _adjoint_1q(square_root_gate)

    target_qubit = base_one_qubit_gate.qubits[0]
    decomposition_sequence: list[Gate] = []
    decomposition_sequence += _multi_controlled(remaining_control_qubits, square_root_gate)
    decomposition_sequence += _CNOT_as_CZ_plus_1q(last_control_qubit, target_qubit)
    decomposition_sequence += _multi_controlled(remaining_control_qubits, square_root_adjoint_gate)
    decomposition_sequence += _CNOT_as_CZ_plus_1q(last_control_qubit, target_qubit)
    decomposition_sequence += _multi_controlled(remaining_control_qubits, square_root_gate)
    return decomposition_sequence
