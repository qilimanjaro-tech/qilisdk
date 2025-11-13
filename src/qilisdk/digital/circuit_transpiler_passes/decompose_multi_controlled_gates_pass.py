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
from typing import List

from qilisdk.digital import RX, RY, RZ, U1, U2, U3, Circuit, Gate, H, I, S, T, X, Y, Z
from qilisdk.digital.gates import BasicGate, Controlled

from .circuit_transpiler_pass import CircuitTranspilerPass
from .numeric_helpers import (
    _mat_U3,
    _unitary_sqrt_2x2,
    _zyz_from_unitary,
)


class DecomposeMultiControlledGatesPass(CircuitTranspilerPass):
    """Decompose multi-controlled (k >= 2) 1-qubit gates."""

    def run(self, circuit: Circuit) -> Circuit:
        out = Circuit(circuit.nqubits)
        for g in circuit.gates:
            for h in self._rewrite_gate(g):
                out.add(h)

        return out

    def _rewrite_gate(self, gate: Gate) -> List[Gate]:
        # --- Multi-controlled gates ---
        if isinstance(gate, Controlled):

            base: BasicGate = gate.basic_gate
            if base.nqubits != 1:
                raise NotImplementedError("Controlled version of multi-qubit gates is not supported.")

            return _multi_controlled(gate)

        # Everything else is untouched.
        return [gate]


def _multi_controlled(gate: Controlled) -> List[Gate]:
    if len(gate.control_qubits) == 1:
        return [gate]

    c_last = gate.control_qubits[-1]
    rest = gate.control_qubits[:-1]

    V = _sqrt_of_gate(gate.basic_gate)
    Vd = _adjoint_1q(V)

    seq: List[Gate] = []
    seq += _multi_controlled(Controlled(c_last, basic_gate=V))
    seq += _multi_controlled(X(c_last).controlled(*rest))
    seq += _multi_controlled(Controlled(c_last, basic_gate=Vd))
    seq += _multi_controlled(X(c_last).controlled(*rest))
    seq += _multi_controlled(Controlled(*rest, basic_gate=V))

    return seq


def _sqrt_of_gate(gate: BasicGate) -> BasicGate:
    """Return V such that V^2 == gate."""
    q = gate.qubits[0]

    # Identity: sqrt(I) = I
    if isinstance(gate, I):
        return I(q)

    # Direct parametric rotations.
    if isinstance(gate, RZ):
        return RZ(q, phi=gate.phi / 2.0)
    if isinstance(gate, RX):
        return RX(q, theta=gate.theta / 2.0)
    if isinstance(gate, RY):
        return RY(q, theta=gate.theta / 2.0)

    # Pauli gates via half-angle rotations.
    if isinstance(gate, Z):
        return RZ(q, phi=math.pi / 2.0)
    if isinstance(gate, X):
        return RX(q, theta=math.pi / 2.0)
    if isinstance(gate, Y):
        return RY(q, theta=math.pi / 2.0)

    # Phase gate U1(φ) = diag(1, e^{iφ}), sqrt is U1(φ/2).
    if isinstance(gate, U1):
        return RZ(q, phi=gate.phi / 2.0)

    # S and T: phase gates with known relation to RZ
    # S = RZ(π/2) ⇒ sqrt(S) = RZ(π/4) ≡ T
    if isinstance(gate, S):
        return T(q)

    # T = RZ(π/4) ⇒ sqrt(T) = RZ(π/8)
    if isinstance(gate, T):
        return RZ(q, phi=math.pi / 8.0)

    # Build the 2x2 unitary matrix for gate
    if isinstance(gate, U2):
        # U2(φ, λ) = U3(π/2, φ, λ) up to a global phase.
        U = _mat_U3(math.pi / 2.0, gate.phi, gate.gamma)
    elif isinstance(gate, U3):
        U = _mat_U3(gate.theta, gate.phi, gate.gamma)
    elif isinstance(gate, (H, BasicGate)):
        U = gate.matrix
    else:
        raise NotImplementedError(
            f"_sqrt_1q_gate_as_basis only supports 1-qubit gates; got {type(gate).__name__}"
        )

    # Compute a matrix square root V such that V @ V ≈ U.
    Vs = _unitary_sqrt_2x2(U)

    # Express V as a U3 on the same qubit. This introduces a new gate in U3 form
    # for the *square root*, but leaves the original g untouched.
    th, ph, lam = _zyz_from_unitary(Vs)
    return U3(q, theta=th, phi=ph, gamma=lam)


def _adjoint_1q(g: BasicGate) -> BasicGate:
    """Return the 1-qubit adjoint (inverse) of gate."""
    q = g.qubits[0]

    # Identity: self-adjoint.
    if isinstance(g, I):
        return I(q)

    if isinstance(g, RX):
        return RX(q, theta=-g.theta)
    if isinstance(g, RY):
        return RY(q, theta=-g.theta)
    if isinstance(g, RZ):
        return RZ(q, phi=-g.phi)

    if isinstance(g, U3):
        # U3(θ, φ, λ)† = U3(-θ, -λ, -φ) (up to global phase).
        return U3(q, theta=-g.theta, phi=-g.gamma, gamma=-g.phi)

    if isinstance(g, U1):
        # U1(λ)† = U1(-λ)
        return RZ(q, phi=-g.phi)

    # ---------- Named 1q gates ----------

    # Pauli & Hadamard: self-adjoint.
    if isinstance(g, X):
        return X(q)
    if isinstance(g, Y):
        return Y(q)
    if isinstance(g, Z):
        return Z(q)
    if isinstance(g, H):
        return H(q)

    # S, T: phase gates about Z.
    # S = RZ(π/2)  ⇒ S† = RZ(-π/2)
    if isinstance(g, S):
        return RZ(q, phi=-math.pi / 2.0)

    # T = RZ(π/4)  ⇒ T† = RZ(-π/4)
    if isinstance(g, T):
        return RZ(q, phi=-math.pi / 4.0)

    # ---------- Generic 1-qubit unitary via matrix adjoint ----------

    # U2(φ, λ) we handle through its matrix; same idea as in sqrt.
    if isinstance(g, U2):
        U = _mat_U3(math.pi / 2.0, g.phi, g.gamma)
    elif isinstance(g, BasicGate) and g.nqubits == 1:
        U = g.matrix
    else:
        raise NotImplementedError(
            f"_adjoint_1q only supports 1-qubit gates; got {type(g).__name__}"
        )

    # Take the matrix adjoint U† and convert to ZYZ → U3.
    U_dag = U.conj().T
    th, ph, lam = _zyz_from_unitary(U_dag)
    return U3(q, theta=th, phi=ph, gamma=lam)
