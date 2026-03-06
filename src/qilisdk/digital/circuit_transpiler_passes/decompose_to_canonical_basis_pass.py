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
from enum import Enum
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
    _wrap_angle,
    _zyz_from_unitary,
)


def _is_controlled(gate: Gate) -> TypeGuard[Controlled[BasicGate]]:
    return isinstance(gate, Controlled)


def _is_exponential(gate: Gate) -> TypeGuard[Exponential[BasicGate]]:
    return isinstance(gate, Exponential)


def _is_adjoint(gate: Gate) -> TypeGuard[Adjoint[BasicGate]]:
    return isinstance(gate, Adjoint)


class TwoQubitGateBasis(Enum):
    """Selectable two-qubit entangler used by canonicalization rewrites."""

    CZ = "CZ"
    CNOT = "CNOT"


class SingleQubitGateBasis(Enum):
    """Selectable single-qubit basis used by canonicalization rewrites."""

    RxRyRz = "RxRyRz"
    U3 = "U3"


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
    # CNOT = (I ⊗ RY(-π/2)) · CZ · (I ⊗ RY(π/2)).
    return [
        RY(target_qubit, theta=-math.pi / 2.0),
        CZ(control_qubit, target_qubit),
        RY(target_qubit, theta=math.pi / 2.0),
    ]


def _CZ_as_CNOT_plus_1q(control_qubit: int, target_qubit: int) -> list[Gate]:
    """Return a CZ decomposition using CNOT and single-qubit gates.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.

    Returns:
        list[Gate]: Equivalent gate sequence in the canonical basis.
    """
    # CZ = (I ⊗ RY(π/2)) · CNOT · (I ⊗ RY(-π/2)).
    return [
        RY(target_qubit, theta=math.pi / 2.0),
        CNOT(control_qubit, target_qubit),
        RY(target_qubit, theta=-math.pi / 2.0),
    ]


def _cnot_in_basis(control_qubit: int, target_qubit: int, basis: TwoQubitGateBasis) -> list[Gate]:
    """Emit a CNOT in the selected two-qubit basis.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        basis (TwoQubitGateBasis): Selected two-qubit basis.

    Returns:
        list[Gate]: Sequence implementing CNOT in the selected basis.
    """
    if basis == TwoQubitGateBasis.CNOT:
        return [CNOT(control_qubit, target_qubit)]
    return _CNOT_as_CZ_plus_1q(control_qubit, target_qubit)


def _cz_in_basis(control_qubit: int, target_qubit: int, basis: TwoQubitGateBasis) -> list[Gate]:
    """Emit a CZ in the selected two-qubit basis.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        basis (TwoQubitGateBasis): Selected two-qubit basis.

    Returns:
        list[Gate]: Sequence implementing CZ in the selected basis.
    """
    if basis == TwoQubitGateBasis.CZ:
        return [CZ(control_qubit, target_qubit)]
    return _CZ_as_CNOT_plus_1q(control_qubit, target_qubit)


def _CRZ_using_CNOT(
    control_qubit: int,
    target_qubit: int,
    gamma: float,
    basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
) -> list[Gate]:
    """Return a controlled-RZ decomposition using CNOT-based primitives.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        gamma (float): RZ rotation angle.

    Returns:
        list[Gate]: Equivalent controlled-RZ sequence in canonical basis.
    """
    # CRZ(λ) = (I ⊗ RZ(λ/2)) · CNOT · (I ⊗ RZ(-λ/2)) · CNOT.
    return [
        RZ(target_qubit, phi=_wrap_angle(gamma / 2.0)),
        *_cnot_in_basis(control_qubit, target_qubit, basis),
        RZ(target_qubit, phi=_wrap_angle(-gamma / 2.0)),
        *_cnot_in_basis(control_qubit, target_qubit, basis),
    ]


def _CRX_using_CRZ(
    control_qubit: int,
    target_qubit: int,
    theta: float,
    basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
) -> list[Gate]:
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
        *_CRZ_using_CNOT(control_qubit, target_qubit, theta, basis),
        RY(target_qubit, theta=math.pi / 2.0),
    ]


def _CRY_using_CRZ(
    control_qubit: int,
    target_qubit: int,
    theta: float,
    basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
) -> list[Gate]:
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
        *_CRZ_using_CNOT(control_qubit, target_qubit, theta, basis),
        RX(target_qubit, theta=-math.pi / 2.0),
    ]


def _CU3_using_CNOT(
    control_qubit: int,
    target_qubit: int,
    theta: float,
    phi: float,
    gamma: float,
    basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
) -> list[Gate]:
    """Return a controlled-U3 decomposition using two CNOT skeletons.

    Args:
        control_qubit (int): Control qubit index.
        target_qubit (int): Target qubit index.
        theta (float): U3 theta parameter.
        phi (float): U3 phi parameter.
        gamma (float): U3 gamma/lambda parameter.

    Returns:
        list[Gate]: Equivalent controlled-U3 sequence in canonical basis.
    """
    # Exact single-control synthesis:
    # CU3(θ, φ, λ) = P((λ+φ)/2)_c · P((λ-φ)/2)_t · CX ·
    #                U3(-θ/2, 0, -(φ+λ)/2)_t · CX · U3(θ/2, φ, 0)_t
    return [
        U1(control_qubit, phi=_wrap_angle((gamma + phi) / 2.0)),
        U1(target_qubit, phi=_wrap_angle((gamma - phi) / 2.0)),
        *_cnot_in_basis(control_qubit, target_qubit, basis),
        U3(target_qubit, theta=-theta / 2.0, phi=0.0, gamma=_wrap_angle(-(gamma + phi) / 2.0)),
        *_cnot_in_basis(control_qubit, target_qubit, basis),
        U3(target_qubit, theta=theta / 2.0, phi=phi, gamma=0.0),
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
    if isinstance(gate, CNOT):
        return [CNOT(gate.control_qubits[0], gate.target_qubits[0])]
    if isinstance(gate, M):
        return [gate]
    if isinstance(gate, H):
        return _H_as_U3(gate.qubits[0])[::-1]
    if isinstance(gate, X):
        return [U3(gate.qubits[0], theta=math.pi, phi=0.0, gamma=math.pi)]
    if isinstance(gate, Y):
        return [U3(gate.qubits[0], theta=math.pi, phi=math.pi / 2.0, gamma=math.pi / 2.0)]
    if isinstance(gate, Z):
        return [U3(gate.qubits[0], theta=0.0, phi=0.0, gamma=math.pi)]
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
        return U3(qubit, theta=0.0, phi=0.0, gamma=gate.phi)
    if isinstance(gate, U2):
        return U3(qubit, theta=math.pi / 2.0, phi=gate.phi, gamma=gate.gamma)
    if isinstance(gate, H):
        return _H_as_U3(qubit)[0]
    if isinstance(gate, X):
        return U3(qubit, theta=math.pi, phi=0.0, gamma=math.pi)
    if isinstance(gate, Y):
        return U3(qubit, theta=math.pi, phi=math.pi / 2.0, gamma=math.pi / 2.0)
    if isinstance(gate, Z):
        return U3(qubit, theta=0.0, phi=0.0, gamma=math.pi)
    if isinstance(gate, BasicGate) and gate.nqubits == 1:
        theta, phi, gamma = _zyz_from_unitary(gate.matrix)
        return U3(qubit, theta=theta, phi=phi, gamma=gamma)
    raise NotImplementedError(f"Unsupported 1-qubit gate type {type(gate).__name__} in _as_basis_1q")


def _normalize_single_qubit_gate(gate: Gate, basis: SingleQubitGateBasis) -> list[Gate]:
    """Rewrite one-qubit gates to the selected single-qubit basis.

    Args:
        gate (Gate): Gate to normalize.
        basis (SingleQubitGateBasis): Target single-qubit basis.

    Raises:
        TypeError: If an unsupported basis is given.

    Returns:
        list[Gate]: Equivalent sequence where one-qubit unitary gates are in ``basis``.
    """
    if gate.nqubits != 1 or isinstance(gate, M):
        return [gate]

    qubit = gate.qubits[0]

    if basis == SingleQubitGateBasis.U3:
        if isinstance(gate, U3):
            return [gate]
        if isinstance(gate, RX):
            return [U3(qubit, theta=gate.theta, phi=-math.pi / 2.0, gamma=math.pi / 2.0)]
        if isinstance(gate, RY):
            return [U3(qubit, theta=gate.theta, phi=0.0, gamma=0.0)]
        if isinstance(gate, RZ):
            return [U3(qubit, theta=0.0, phi=gate.phi, gamma=0.0)]
        return _normalize_single_qubit_gate(_as_basis_1q(gate), basis)

    if basis == SingleQubitGateBasis.RxRyRz:
        if isinstance(gate, (RX, RY, RZ)):
            return [gate]
        if isinstance(gate, U3):
            return [
                RZ(qubit, phi=_wrap_angle(gate.phi)),
                RY(qubit, theta=_wrap_angle(gate.theta)),
                RZ(qubit, phi=_wrap_angle(gate.gamma)),
            ]
        return _normalize_single_qubit_gate(_as_basis_1q(gate), basis)

    raise TypeError(f"Unsupported single-qubit basis {basis}")


def _normalize_single_qubit_sequence(gates: list[Gate], basis: SingleQubitGateBasis) -> list[Gate]:
    """Normalize all one-qubit gates in a sequence to the selected basis.

    Args:
        gates (list[Gate]): Gate sequence to normalize.
        basis (SingleQubitGateBasis): Target single-qubit basis.

    Returns:
        list[Gate]: Sequence where every one-qubit gate is expressed in ``basis``.
    """
    normalized: list[Gate] = []
    for gate in gates:
        normalized += _normalize_single_qubit_gate(gate, basis)
    return normalized


class DecomposeToCanonicalBasisPass(CircuitTranspilerPass):
    """
    Map an arbitrary circuit to the circuit basis {1Q, 2Q} (+ M).

    The two-qubit basis gate is selectable via ``two_qubit_basis``:
    ``TwoQubitGateBasis.CZ`` or ``TwoQubitGateBasis.CNOT``.
    The one-qubit basis set is selectable via ``single_qubit_basis``:
    ``SingleQubitGateBasis.RxRyRz`` or ``SingleQubitGateBasis.U3``.

    - Rewrites CNOT / CZ / SWAP to the selected two-qubit basis + 1Q gates.
    - Controlled with one control (target 1-qubit) → selected two-qubit basis + 1Q synthesis.
    - Multi-controlled gates are intentionally out of scope for this pass.
    - Adjoint(g) → canonicalize(g) then reverse+invert.
    - Exponential(1q) → ZYZ, then normalized to the selected 1Q basis.

    NOTE: This pass does *not* perform any 1-qubit fusion/merging.
    """

    def __init__(
        self,
        single_qubit_basis: SingleQubitGateBasis = SingleQubitGateBasis.U3,
        two_qubit_basis: TwoQubitGateBasis = TwoQubitGateBasis.CNOT,
    ) -> None:
        """Initialize the canonical-basis decomposition pass.

        Args:
            single_qubit_basis (SingleQubitGateBasis): Single-qubit basis set used by this pass.
            two_qubit_basis (TwoQubitGateBasis): Two-qubit entangler basis used by this pass.

        Raises:
            TypeError: If ``single_qubit_basis`` or ``two_qubit_basis`` is not a valid enum value.
        """
        if not isinstance(single_qubit_basis, SingleQubitGateBasis):
            raise TypeError(
                f"single_qubit_basis must be a SingleQubitGateBasis value (got {type(single_qubit_basis).__name__})."
            )
        if not isinstance(two_qubit_basis, TwoQubitGateBasis):
            raise TypeError(
                f"two_qubit_basis must be a TwoQubitGateBasis value (got {type(two_qubit_basis).__name__})."
            )
        self._single_qubit_basis = single_qubit_basis
        self._two_qubit_basis = two_qubit_basis

    @property
    def single_qubit_basis(self) -> SingleQubitGateBasis:
        """Return the single-qubit basis set used by this pass.

        Returns:
            SingleQubitGateBasis: Configured single-qubit basis.
        """
        return self._single_qubit_basis

    @property
    def two_qubit_basis(self) -> TwoQubitGateBasis:
        """Return the two-qubit basis gate used by this pass.

        Returns:
            TwoQubitGateBasis: Configured two-qubit basis.
        """
        return self._two_qubit_basis

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
        if isinstance(gate, (U3, RX, RY, RZ)):
            return _normalize_single_qubit_sequence([gate], self._single_qubit_basis)
        if self._two_qubit_basis == TwoQubitGateBasis.CZ and isinstance(gate, CZ):
            return [gate]
        if self._two_qubit_basis == TwoQubitGateBasis.CNOT and isinstance(gate, CNOT):
            return [gate]

        # simple 1q
        if isinstance(gate, I):
            return []
        if isinstance(gate, (H, X, Y, Z)):
            return _normalize_single_qubit_sequence([_as_basis_1q(gate)], self._single_qubit_basis)

        # param 1q
        if isinstance(gate, U1):
            return _normalize_single_qubit_sequence([_as_basis_1q(gate)], self._single_qubit_basis)
        if isinstance(gate, U2):
            return _normalize_single_qubit_sequence([_as_basis_1q(gate)], self._single_qubit_basis)

        # 2q
        if isinstance(gate, CNOT):
            control_qubit, target_qubit = gate.control_qubits[0], gate.target_qubits[0]
            return _normalize_single_qubit_sequence(
                _cnot_in_basis(control_qubit, target_qubit, self._two_qubit_basis),
                self._single_qubit_basis,
            )
        if isinstance(gate, CZ):
            control_qubit, target_qubit = gate.control_qubits[0], gate.target_qubits[0]
            return _normalize_single_qubit_sequence(
                _cz_in_basis(control_qubit, target_qubit, self._two_qubit_basis),
                self._single_qubit_basis,
            )
        if isinstance(gate, SWAP):
            first_target_qubit, second_target_qubit = gate.target_qubits
            return _normalize_single_qubit_sequence(
                _cnot_in_basis(first_target_qubit, second_target_qubit, self._two_qubit_basis)
                + _cnot_in_basis(second_target_qubit, first_target_qubit, self._two_qubit_basis)
                + _cnot_in_basis(first_target_qubit, second_target_qubit, self._two_qubit_basis),
                self._single_qubit_basis,
            )

        # Controlled (single control) over a 1-qubit target
        if _is_controlled(gate):
            control_qubits = list(gate.control_qubits)
            basic_gate: BasicGate = gate.basic_gate
            if basic_gate.nqubits != 1:
                raise NotImplementedError("Controlled of multi-qubit gates not supported.")
            if len(control_qubits) != 1:
                raise NotImplementedError(
                    "CircuitToCanonicalBasisPass supports only single-control gates. "
                    "Run DecomposeMultiControlledGatesPass first."
                )
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
            return _single_controlled(
                control_qubits[0],
                basis_one_qubit_gate,
                self._two_qubit_basis,
                self._single_qubit_basis,
            )

        # Adjoint
        if _is_adjoint(gate):
            rewritten_base_sequence = self._rewrite_gate(gate.basic_gate)
            inverse_sequence: list[Gate] = []
            for base_gate in reversed(rewritten_base_sequence):
                inverse_sequence += _invert_basis_gate(base_gate)
            return _normalize_single_qubit_sequence(inverse_sequence, self._single_qubit_basis)

        # Exponential(1q)
        if _is_exponential(gate):
            basic_gate = gate.basic_gate
            if basic_gate.nqubits != 1:
                raise NotImplementedError("Exponential of multi-qubit gates not supported.")
            unitary_matrix = gate.matrix
            theta, phi, gamma = _zyz_from_unitary(unitary_matrix)
            return _normalize_single_qubit_sequence(
                [U3(basic_gate.qubits[0], theta=theta, phi=phi, gamma=gamma)],
                self._single_qubit_basis,
            )

        # generic 1q
        if isinstance(gate, BasicGate) and gate.nqubits == 1:
            theta, phi, gamma = _zyz_from_unitary(gate.matrix)
            return _normalize_single_qubit_sequence(
                [U3(gate.qubits[0], theta=theta, phi=phi, gamma=gamma)],
                self._single_qubit_basis,
            )

        raise NotImplementedError(f"No canonicalization rule for {type(gate).__name__}")

    # --- single-control synthesis ---


def _single_controlled(
    control_qubit: int,
    target_gate: Gate,
    basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
    single_qubit_basis: SingleQubitGateBasis = SingleQubitGateBasis.U3,
) -> list[Gate]:
    """Return a single-control decomposition for a one-qubit target gate.

    Args:
        control_qubit (int): Control qubit index.
        target_gate (Gate): One-qubit target gate to control.
        basis (TwoQubitGateBasis): Selected two-qubit basis used in the decomposition.
        single_qubit_basis (SingleQubitGateBasis): Selected one-qubit basis used to normalize emitted 1Q gates.

    Returns:
        list[Gate]: Equivalent sequence with canonical two-qubit primitives.
    """
    target_qubit = target_gate.qubits[0]
    if isinstance(target_gate, RZ):
        return _normalize_single_qubit_sequence(
            _CRZ_using_CNOT(control_qubit, target_qubit, target_gate.phi, basis),
            single_qubit_basis,
        )
    if isinstance(target_gate, RX):
        return _normalize_single_qubit_sequence(
            _CRX_using_CRZ(control_qubit, target_qubit, target_gate.theta, basis),
            single_qubit_basis,
        )
    if isinstance(target_gate, RY):
        return _normalize_single_qubit_sequence(
            _CRY_using_CRZ(control_qubit, target_qubit, target_gate.theta, basis),
            single_qubit_basis,
        )
    if isinstance(target_gate, U3):
        return _normalize_single_qubit_sequence(
            _CU3_using_CNOT(
                control_qubit,
                target_qubit,
                target_gate.theta,
                target_gate.phi,
                target_gate.gamma,
                basis,
            ),
            single_qubit_basis,
        )
    return _single_controlled(control_qubit, _as_basis_1q(target_gate), basis, single_qubit_basis)
