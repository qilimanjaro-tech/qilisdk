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

from __future__ import annotations

import math
from enum import Enum
from typing import List

from qilisdk.digital import Circuit
from qilisdk.digital.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    S,
    SWAP,
    T,
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

ANGLE_TOL = 1e-9


class CanonicalBasis(Enum):
    """Enumerates the supported hardware bases for circuit canonicalization."""

    CLIFFORD_T = "clifford_t"
    XYZ_ROTATIONS = "xyz_rotations"
    ZX_EULER = "zx_euler"
    U3_CX = "u3_cx"


# ======================= Small basis building blocks =======================
# All helpers here should eventually emit only RX/RZ/CZ/M, possibly via
# intermediate U3/CNOT/etc. that are *immediately* reduced by other helpers.


# ---------- 1-qubit: generic + U-gates ----------


def _U3_to_RXRZ(q: int, theta: float, phi: float, lam: float) -> List[Gate]:
    """
    Decompose U3(θ, φ, λ) into RZ–RX–RZ using

        U3(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
        RY(θ) = RZ(π/2) RX(θ) RZ(-π/2)

    ⇒  U3(θ, φ, λ) = RZ(φ + π/2) RX(θ) RZ(λ - π/2)  (up to a global phase).
    """

    return [
        RZ(q, phi=_wrap_angle(phi + math.pi / 2.0)),
        RX(q, theta=_wrap_angle(theta)),
        RZ(q, phi=_wrap_angle(lam - math.pi / 2.0)),
    ]


def _U1_as_RZ(q: int, lam: float) -> List[Gate]:
    # U1(λ) = RZ(λ)
    return [RZ(q, phi=_wrap_angle(lam))]


def _U2_as_RXRZ(q: int, phi: float, lam: float) -> List[Gate]:
    # U2(φ, λ) = U3(π/2, φ, λ)
    return _U3_to_RXRZ(q, math.pi / 2.0, phi, lam)


def _normalized_u3_gate(q: int, theta: float, phi: float, lam: float) -> U3:
    """Return a normalized U3 instance with wrapped angles."""

    return U3(q, theta=_wrap_angle(theta), phi=_wrap_angle(phi), gamma=_wrap_angle(lam))


# ---------- 1-qubit: named Clifford / Pauli / rotations ----------


def _RX_as_RX(q: int, theta: float) -> List[Gate]:
    return [RX(q, theta=_wrap_angle(theta))]


def _RZ_as_RZ(q: int, phi: float) -> List[Gate]:
    return [RZ(q, phi=_wrap_angle(phi))]


def _RY_as_RXRZ(q: int, theta: float) -> List[Gate]:
    # RY(θ) = RZ(π/2) RX(θ) RZ(-π/2)
    return [
        RZ(q, phi=_wrap_angle(math.pi / 2.0)),
        RX(q, theta=_wrap_angle(theta)),
        RZ(q, phi=_wrap_angle(-math.pi / 2.0)),
    ]


def _H_as_RXRZ(q: int) -> List[Gate]:
    # H = U3(π/2, 0, π) → RX/RZ via _U3_to_RXRZ
    return _U3_to_RXRZ(q, math.pi / 2.0, 0.0, math.pi)


def _X_as_RX(q: int) -> List[Gate]:
    # X = RX(π)
    return _RX_as_RX(q, math.pi)


def _Y_as_RXRZ(q: int) -> List[Gate]:
    # Y = RY(π) = RZ(π/2) RX(π) RZ(-π/2)
    return _RY_as_RXRZ(q, math.pi)


def _Z_as_RZ(q: int) -> List[Gate]:
    # Z = RZ(π)
    return _RZ_as_RZ(q, math.pi)


# ---------- 2-qubit: CNOT <-> CZ, SWAP ----------


def _CNOT_as_CZ_plus_1q(c: int, t: int) -> List[Gate]:
    """
    Realize CNOT from CZ using single-qubit H on the target:

        CNOT = (I ⊗ H) · CZ · (I ⊗ H)
    """

    return [*_H_as_RXRZ(t), CZ(c, t), *_H_as_RXRZ(t)]


def _CZ_as_CNOT_plus_1q(c: int, t: int) -> List[Gate]:
    """
    Realize CZ from CNOT using single-qubit H on the target:

        CZ = (I ⊗ H) · CNOT · (I ⊗ H)
    """

    return [*_H_as_RXRZ(t), CNOT(c, t), *_H_as_RXRZ(t)]


def _SWAP_as_CNOT(a: int, b: int) -> List[Gate]:
    """
    SWAP decomposition via CNOT:

        SWAP(a, b) = CNOT(a, b) · CNOT(b, a) · CNOT(a, b)
    """

    return [CNOT(a, b), CNOT(b, a), CNOT(a, b)]


def _SWAP_as_CZ(a: int, b: int) -> List[Gate]:
    """
    SWAP using only CZ and 1Q, via the CNOT-based decomposition.
    """

    return (
        _CNOT_as_CZ_plus_1q(a, b)
        + _CNOT_as_CZ_plus_1q(b, a)
        + _CNOT_as_CZ_plus_1q(a, b)
    )


# ---------- Controlled 1-qubit gates via CNOT/CZ ----------


def _CRZ_using_CNOT(c: int, t: int, lam: float) -> List[Gate]:
    """
    CRZ(λ) using two CNOTs:

        CRZ(λ) = (I⊗RZ(λ/2)) · CNOT · (I⊗RZ(-λ/2)) · CNOT
    """

    return [
        *_RZ_as_RZ(t, lam / 2.0),
        *_CNOT_as_CZ_plus_1q(c, t),
        *_RZ_as_RZ(t, -lam / 2.0),
        *_CNOT_as_CZ_plus_1q(c, t),
    ]


def _CRX_using_CNOT(c: int, t: int, theta: float) -> List[Gate]:
    """
    CRX(θ) using CRZ(θ) and H on target:

        RX(θ) = H RZ(θ) H
        ⇒ CRX(θ) = (I ⊗ H) CRZ(θ) (I ⊗ H)
    """

    return [
        *_H_as_RXRZ(t),
        *_CRZ_using_CNOT(c, t, theta),
        *_H_as_RXRZ(t),
    ]


def _CRY_using_CNOT(c: int, t: int, theta: float) -> List[Gate]:
    """
    CRY(θ) using CRX(θ) and Z-phase rotations on target:

        RY(θ) = RZ(π/2) RX(θ) RZ(-π/2)
        ⇒ CRY(θ) = (I⊗RZ(π/2)) · CRX(θ) · (I⊗RZ(-π/2))
    """

    return [
        *_RZ_as_RZ(t, math.pi / 2.0),
        *_CRX_using_CNOT(c, t, theta),
        *_RZ_as_RZ(t, -math.pi / 2.0),
    ]


def _CU3_using_CNOT(c: int, t: int, theta: float, phi: float, lam: float) -> List[Gate]:
    """
    Standard two-CNOT synthesis of CU3 (here CX realized by H-CZ-H):

        (Based on the common Qiskit-style decomposition, but with all 1q
         gates written via RX/RZ/U3 helpers.)
    """

    return [
        # RZ on control
        RZ(c, phi=_wrap_angle((lam + phi) / 2.0)),
        # U3(theta/2, phi, 0) on target
        *_U3_to_RXRZ(t, theta / 2.0, phi, 0.0),
        *_CNOT_as_CZ_plus_1q(c, t),
        # U3(-theta/2, 0, -(lam+phi)/2) on target
        *_U3_to_RXRZ(t, -theta / 2.0, 0.0, _wrap_angle(-(lam + phi) / 2.0)),
        *_CNOT_as_CZ_plus_1q(c, t),
        # final RZ on target
        RZ(t, phi=_wrap_angle((lam - phi) / 2.0)),
    ]


def _invert_basis_gate(g: Gate) -> List[Gate]:
    """Invert gate types produced by ``_rewrite_gate`` prior to hardware mapping."""

    if isinstance(g, RX):
        return [RX(g.qubits[0], theta=_wrap_angle(-g.theta))]
    if isinstance(g, RY):
        return [RY(g.qubits[0], theta=_wrap_angle(-g.theta))]
    if isinstance(g, RZ):
        return [RZ(g.qubits[0], phi=_wrap_angle(-g.phi))]
    if isinstance(g, H):
        return [H(g.qubits[0])]
    if isinstance(g, S):
        return _RZ_as_RZ(g.qubits[0], -math.pi / 2.0)
    if isinstance(g, T):
        return _RZ_as_RZ(g.qubits[0], -math.pi / 4.0)
    if isinstance(g, U3):
        return [_normalized_u3_gate(g.qubits[0], -g.theta, -g.gamma, -g.phi)]
    if isinstance(g, CNOT):
        return [CNOT(g.control_qubits[0], g.target_qubits[0])]
    if isinstance(g, CZ):
        return [CZ(g.control_qubits[0], g.target_qubits[0])]
    if isinstance(g, M):
        return [g]
    # Should not happen after canonicalization.
    return [Adjoint(g)]  # type: ignore[type-var]


# ======================= Canonicalization (mapping-only) =======================


def _as_basis_1q(g: Gate) -> Gate:
    """
    Return an equivalent 1-qubit gate expressed as RX/RZ/U3.
    U3 is used as an intermediate form and is later lowered to RX/RZ.
    """

    q = g.qubits[0]
    if isinstance(g, (RX, RZ, U3)):
        return g
    if isinstance(g, U1):
        return RZ(q, phi=g.phi)
    if isinstance(g, U2):
        return U3(q, theta=math.pi / 2.0, phi=g.phi, gamma=g.gamma)
    if isinstance(g, H):
        # Represent as U3; will be lowered when actually rewritten.
        return U3(q, theta=math.pi / 2.0, phi=0.0, gamma=math.pi)
    if isinstance(g, X):
        return RX(q, theta=math.pi)
    if isinstance(g, Y):
        # Y = U3(π, π/2, -π/2)
        return U3(q, theta=math.pi, phi=math.pi / 2.0, gamma=-math.pi / 2.0)
    if isinstance(g, Z):
        return RZ(q, phi=math.pi)
    if isinstance(g, BasicGate) and g.nqubits == 1:
        th, ph, lam = _zyz_from_unitary(g.matrix)
        return U3(q, theta=th, phi=ph, gamma=lam)
    raise NotImplementedError(f"Unsupported 1-qubit gate type {type(g).__name__} in _as_basis_1q")


def _convert_cz_to_cnot(gates: List[Gate]) -> List[Gate]:
    """Replace CZ gates with CNOT plus the necessary local rotations."""

    out: List[Gate] = []
    for g in gates:
        if isinstance(g, CZ):
            c, t = g.control_qubits[0], g.target_qubits[0]
            out.extend(_CZ_as_CNOT_plus_1q(c, t))
        else:
            out.append(g)
    return out


def _angles_equivalent(value: float, target: float) -> bool:
    return abs(_wrap_angle(value - target)) <= ANGLE_TOL


def _RZ_as_clifford_t(q: int, phi: float) -> List[Gate]:
    """Express RZ(φ) as a sequence of S and T gates."""

    phi = _wrap_angle(phi)
    k = round(phi / (math.pi / 4.0))
    approx = k * (math.pi / 4.0)
    if not math.isclose(phi, approx, abs_tol=ANGLE_TOL):
        raise ValueError(
            "RZ rotations must be multiples of π/4 to express them with Clifford+T gates."
        )

    steps = int(k % 8)
    seq: List[Gate] = []
    for _ in range(steps // 2):
        seq.append(S(q))
    if steps % 2:
        seq.append(T(q))
    return seq


def _RX_as_clifford_t(q: int, theta: float) -> List[Gate]:
    """Express RX(θ) as H · RZ(θ) · H using Clifford+T-compatible rotations."""

    phase = _RZ_as_clifford_t(q, theta)
    return [H(q), *phase, H(q)]


def _recover_ry_sequences(gates: List[Gate]) -> List[Gate]:
    """Collapse Z-X-Z patterns back into RY for the XYZ basis."""

    out: List[Gate] = []
    i = 0
    while i < len(gates):
        if i + 2 < len(gates):
            g0, g1, g2 = gates[i : i + 3]
            if (
                isinstance(g0, RZ)
                and isinstance(g1, RX)
                and isinstance(g2, RZ)
                and g0.qubits[0] == g1.qubits[0] == g2.qubits[0]
                and _angles_equivalent(g0.phi, math.pi / 2.0)
                and _angles_equivalent(g2.phi, -math.pi / 2.0)
            ):
                out.append(RY(g0.qubits[0], theta=_wrap_angle(g1.theta)))
                i += 3
                continue
        out.append(gates[i])
        i += 1
    return out


class CircuitToCanonicalBasisPass(CircuitTranspilerPass):
    """
    Map an arbitrary circuit to a configurable universal gate basis.

    The pass first rewrites every gate into the canonical {RX, RZ, CZ} set. The resulting
    sequence is then converted into one of the supported hardware bases:

    1. ``CanonicalBasis.CLIFFORD_T`` → {CNOT, H, S, T}
    2. ``CanonicalBasis.XYZ_ROTATIONS`` → {CNOT, RX, RY, RZ}
    3. ``CanonicalBasis.ZX_EULER`` → {CNOT, RX, RZ}
    4. ``CanonicalBasis.U3_CX`` → {CNOT, U3}

    Measurement gates always passthrough unchanged. All helper routines continue to emit
    RX/RZ/CZ internally so existing decompositions (_CNOT↔CZ_, SWAP, CRZ, etc.) remain valid.
    """

    def __init__(self, basis: CanonicalBasis = CanonicalBasis.ZX_EULER) -> None:
        self._basis = basis

    def run(self, circuit: Circuit) -> Circuit:
        seq = self._rewrite_list(circuit.gates)
        seq = self._map_to_target_basis(seq)

        out = Circuit(circuit.nqubits)
        for g in seq:
            out.add(g)

        return out

    def _rewrite_list(self, gates: List[Gate]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            out += self._rewrite_gate(g)
        return out

    def _rewrite_gate(self, g: Gate) -> List[Gate]:
        basis = self._basis
        # measurement passes through
        if isinstance(g, M):
            return [g]

        # Explicit U3/RY are not in the basis: decompose them.
        if isinstance(g, U3):
            q = g.qubits[0]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, g.theta, g.phi, g.gamma)]
            return _U3_to_RXRZ(q, g.theta, g.phi, g.gamma)
        if isinstance(g, RY):
            q = g.qubits[0]
            if basis == CanonicalBasis.XYZ_ROTATIONS:
                return [RY(q, theta=_wrap_angle(g.theta))]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, g.theta, math.pi / 2.0, -math.pi / 2.0)]
            return _RY_as_RXRZ(q, g.theta)

        # already basis
        if isinstance(g, (RX, RZ, CZ)):
            return [g]

        # simple 1q
        if isinstance(g, I):
            return []
        if isinstance(g, H):
            q = g.qubits[0]
            if basis == CanonicalBasis.CLIFFORD_T:
                return [H(q)]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, math.pi / 2.0, 0.0, math.pi)]
            return _H_as_RXRZ(q)
        if isinstance(g, X):
            return _X_as_RX(g.qubits[0])
        if isinstance(g, Y):
            q = g.qubits[0]
            if basis == CanonicalBasis.XYZ_ROTATIONS:
                return [RY(q, theta=_wrap_angle(math.pi))]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, math.pi, math.pi / 2.0, -math.pi / 2.0)]
            return _Y_as_RXRZ(q)
        if isinstance(g, Z):
            return _Z_as_RZ(g.qubits[0])
        if isinstance(g, S):
            q = g.qubits[0]
            if basis == CanonicalBasis.CLIFFORD_T:
                return [S(q)]
            return _RZ_as_RZ(q, math.pi / 2.0)
        if isinstance(g, T):
            q = g.qubits[0]
            if basis == CanonicalBasis.CLIFFORD_T:
                return [T(q)]
            return _RZ_as_RZ(q, math.pi / 4.0)

        # param 1q
        if isinstance(g, U1):
            q = g.qubits[0]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, 0.0, g.phi, 0.0)]
            return _U1_as_RZ(q, g.phi)
        if isinstance(g, U2):
            q = g.qubits[0]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, math.pi / 2.0, g.phi, g.gamma)]
            return _U2_as_RXRZ(q, g.phi, g.gamma)

        # 2q
        if isinstance(g, CNOT):
            c, t = g.control_qubits[0], g.target_qubits[0]
            return [CNOT(c, t)]
        if isinstance(g, SWAP):
            a, b = g.target_qubits
            return _SWAP_as_CNOT(a, b)

        # Controlled (k controls) over a 1-qubit target
        if isinstance(g, Controlled):
            ctrls = list(g.control_qubits)
            base: BasicGate = g.basic_gate
            if base.nqubits != 1:
                raise NotImplementedError("Controlled of multi-qubit gates not supported.")
            if len(ctrls) > 1:
                raise NotImplementedError(
                    "Multi-controlled gates are not supported. "
                    "Please run DecomposeMultiControlledGatesPass beforehand."
                )
            t = base.qubits[0]
            base1q = _as_basis_1q(base)
            # ensure target qubit index is correct
            if base1q.qubits[0] != t:
                if isinstance(base1q, U3):
                    base1q = U3(t, theta=base1q.theta, phi=base1q.phi, gamma=base1q.gamma)
                elif isinstance(base1q, RX):
                    base1q = RX(t, theta=base1q.theta)
                elif isinstance(base1q, RZ):
                    base1q = RZ(t, phi=base1q.phi)
            if not ctrls:
                # no controls: act as underlying 1q gate (which will itself be rewritten)
                return self._rewrite_gate(base1q)
            return _single_controlled(ctrls[0], base1q)

        # Adjoint
        if isinstance(g, Adjoint):
            base_seq = self._rewrite_gate(g.basic_gate)
            inv: List[Gate] = []
            for x in reversed(base_seq):
                inv += _invert_basis_gate(x)
            return inv

        # Exponential(1q)
        if isinstance(g, Exponential):
            base = g.basic_gate
            if base.nqubits != 1:
                raise NotImplementedError("Exponential of multi-qubit gates not supported.")
            U = g.matrix
            th, ph, lam = _zyz_from_unitary(U)
            q = base.qubits[0]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, th, ph, lam)]
            return _U3_to_RXRZ(q, th, ph, lam)

        # generic 1q
        if isinstance(g, BasicGate) and g.nqubits == 1:
            th, ph, lam = _zyz_from_unitary(g.matrix)
            q = g.qubits[0]
            if basis == CanonicalBasis.U3_CX:
                return [_normalized_u3_gate(q, th, ph, lam)]
            return _U3_to_RXRZ(q, th, ph, lam)

        raise NotImplementedError(f"No canonicalization rule for {type(g).__name__}")

    def _map_to_target_basis(self, gates: List[Gate]) -> List[Gate]:
        if self._basis == CanonicalBasis.ZX_EULER:
            return _convert_cz_to_cnot(gates)
        if self._basis == CanonicalBasis.XYZ_ROTATIONS:
            seq = _convert_cz_to_cnot(gates)
            return _recover_ry_sequences(seq)
        if self._basis == CanonicalBasis.U3_CX:
            seq = _convert_cz_to_cnot(gates)
            return self._to_u3_basis(seq)
        if self._basis == CanonicalBasis.CLIFFORD_T:
            seq = _convert_cz_to_cnot(gates)
            return self._to_clifford_t_basis(seq)
        raise ValueError(f"Unsupported canonical basis {self._basis}.")

    def _to_clifford_t_basis(self, gates: List[Gate]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            if isinstance(g, RX):
                out.extend(_RX_as_clifford_t(g.qubits[0], g.theta))
            elif isinstance(g, RZ):
                out.extend(_RZ_as_clifford_t(g.qubits[0], g.phi))
            else:
                out.append(g)
        return out

    def _to_u3_basis(self, gates: List[Gate]) -> List[Gate]:
        out: List[Gate] = []
        for g in gates:
            if isinstance(g, RX):
                out.append(U3(g.qubits[0], theta=_wrap_angle(g.theta), phi=-math.pi / 2.0, gamma=math.pi / 2.0))
            elif isinstance(g, RZ):
                out.append(U3(g.qubits[0], theta=0.0, phi=_wrap_angle(g.phi), gamma=0.0))
            else:
                out.append(g)
        return out


def _single_controlled(c: int, target_gate: Gate) -> List[Gate]:
    """
    Implement a single-controlled 1-qubit gate using CNOT + 1q, then
    CNOT is realized as H-CZ-H. All 1q work is done via RX/RZ helpers.
    """

    t = target_gate.qubits[0]
    if isinstance(target_gate, RZ):
        return _CRZ_using_CNOT(c, t, target_gate.phi)
    if isinstance(target_gate, RX):
        return _CRX_using_CNOT(c, t, target_gate.theta)
    if isinstance(target_gate, RY):
        return _CRY_using_CNOT(c, t, target_gate.theta)
    if isinstance(target_gate, U3):
        return _CU3_using_CNOT(c, t, target_gate.theta, target_gate.phi, target_gate.gamma)
    # Fallback: reduce to RX/RZ/U3 and try again.
    return _single_controlled(c, _as_basis_1q(target_gate))
