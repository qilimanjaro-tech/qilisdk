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
    _unitary_sqrt_2x2,
    _zyz_from_unitary,
)


class DecomposeMultiControlledGatesPass(CircuitTranspilerPass):
    """Decompose multi-controlled (k >= 2) single-qubit gates.

    The construction follows Lemma 7.5 from Barenco et al., *Elementary Gates for Quantum Computation*,
    recursively replacing a k-controlled unitary with five layers of (k-1)-controlled operations built
    from sqrt(U), its adjoint, and multi-controlled Pauli-X gates.
    """

    def run(self, circuit: Circuit) -> Circuit:
        """Rewrite the circuit while decomposing multi-controlled gates.

        Args:
            circuit (Circuit): Circuit whose gates should be rewritten.
        Returns:
            Circuit: Newly built circuit containing only supported primitives.
        """
        out = Circuit(circuit.nqubits)
        for g in circuit.gates:
            for h in self._rewrite_gate(g):
                out.add(h)

        return out

    def _rewrite_gate(self, gate: Gate) -> List[Gate]:  # noqa: PLR6301
        """Expand unsupported gates into equivalent elementary gates.

        Args:
            gate (Gate): Candidate gate potentially containing multiple controls.
        Returns:
            list[Gate]: Sequence of equivalent gates that rely on supported primitives.
        """
        # --- Multi-controlled gates ---
        if isinstance(gate, Controlled):
            base: BasicGate = gate.basic_gate
            if base.nqubits != 1:
                raise NotImplementedError("Controlled version of multi-qubit gates is not supported.")

            return _decompose(gate)

        # Everything else is untouched.
        return [gate]


def _decompose(gate: Controlled) -> List[Gate]:
    """Recursively decompose a multi-controlled single-qubit gate.

    Args:
        gate (Controlled): Controlled gate whose target operation is single-qubit.
    Returns:
        list[Gate]: Gate sequence computing the same unitary as `gate`.
    """
    if len(gate.control_qubits) == 1:
        return [gate]

    c_last = gate.control_qubits[-1]
    rest = gate.control_qubits[:-1]

    V = _sqrt_of(gate.basic_gate)
    Vd = _adjoint_of(V)

    seq: List[Gate] = []
    seq += _decompose(Controlled(c_last, basic_gate=V))
    seq += _decompose(X(c_last).controlled(*rest))
    seq += _decompose(Controlled(c_last, basic_gate=Vd))
    seq += _decompose(X(c_last).controlled(*rest))
    seq += _decompose(Controlled(*rest, basic_gate=V))

    return seq


def _sqrt_of(gate: BasicGate) -> BasicGate:
    """Return a gate V whose square equals the provided gate.

    Args:
        gate (BasicGate): Single-qubit gate to compute the principal square root for.
    Returns:
        BasicGate: New primitive V that satisfies V · V ≡ gate.
    """
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

    # Phase gate U1(phi) = diag(1, e^{iphi}), sqrt is U1(phi/2).
    if isinstance(gate, U1):
        return RZ(q, phi=gate.phi / 2.0)

    # S and T: phase gates with known relation to RZ
    # S = RZ(pi/2) ⇒ sqrt(S) = RZ(pi/4) ≡ T
    if isinstance(gate, S):
        return T(q)

    # T = RZ(pi/4) ⇒ sqrt(T) = RZ(pi/8)
    if isinstance(gate, T):
        return RZ(q, phi=math.pi / 8.0)

    # Build the 2x2 unitary matrix for gate
    if isinstance(gate, (U2, U3, H, BasicGate)):
        U = gate.matrix
    else:
        raise NotImplementedError(f"_sqrt_1q_gate_as_basis only supports 1-qubit gates; got {type(gate).__name__}")

    # Compute a matrix square root V such that V @ V ≈ U.
    Vs = _unitary_sqrt_2x2(U)

    # Express V as a U3 on the same qubit. This introduces a new gate in U3 form
    # for the *square root*, but leaves the original g untouched.
    th, ph, lam = _zyz_from_unitary(Vs)
    return U3(q, theta=th, phi=ph, gamma=lam)


def _adjoint_of(gate: BasicGate) -> BasicGate:
    """Return the single-qubit adjoint (inverse) of a gate.

    Args:
        gate (BasicGate): Gate whose inverse should be produced.
    Returns:
        BasicGate: Gate that when composed with `gate` yields the identity.
    """
    q = gate.qubits[0]

    # Identity: self-adjoint.
    if isinstance(gate, I):
        return I(q)

    # Pauli & Hadamard: self-adjoint.
    if isinstance(gate, X):
        return X(q)
    if isinstance(gate, Y):
        return Y(q)
    if isinstance(gate, Z):
        return Z(q)
    if isinstance(gate, H):
        return H(q)

    if isinstance(gate, RX):
        return RX(q, theta=-gate.theta)
    if isinstance(gate, RY):
        return RY(q, theta=-gate.theta)
    if isinstance(gate, RZ):
        return RZ(q, phi=-gate.phi)

    if isinstance(gate, U1):
        # U1(gamma)† = U1(-gamma)
        return RZ(q, phi=-gate.phi)
    if isinstance(gate, U2):
        # U2(phi, gamma)† = U3(pi/2, phi, gamma)† = U3(-pi/2, -phi, -gamma)
        return U3(q, theta=-math.pi / 2.0, phi=-gate.gamma, gamma=-gate.phi)
    if isinstance(gate, U3):
        # U3(theta, phi, gamma)† = U3(-theta, -gamma, -phi)
        return U3(q, theta=-gate.theta, phi=-gate.gamma, gamma=-gate.phi)

    # S, T: phase gates about Z.
    # S = RZ(pi/2)  ⇒ S† = RZ(-pi/2)
    if isinstance(gate, S):
        return RZ(q, phi=-math.pi / 2.0)

    # T = RZ(pi/4)  ⇒ T† = RZ(-pi/4)
    if isinstance(gate, T):
        return RZ(q, phi=-math.pi / 4.0)

    # ---------- Generic 1-qubit unitary via matrix adjoint ----------

    if isinstance(gate, BasicGate) and gate.nqubits == 1:
        U = gate.matrix
    else:
        raise NotImplementedError(f"_adjoint_1q only supports 1-qubit gates; got {type(gate).__name__}")

    # Take the matrix adjoint U† and convert to ZYZ → U3.
    U_dag = U.conj().T
    theta, phi, gamma = _zyz_from_unitary(U_dag)
    return U3(q, theta=theta, phi=phi, gamma=gamma)
