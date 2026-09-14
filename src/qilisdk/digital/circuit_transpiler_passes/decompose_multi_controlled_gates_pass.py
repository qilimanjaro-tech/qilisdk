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
from typing import TypeGuard

from loguru import logger

from qilisdk.digital import RX, RY, RZ, U1, U2, U3, Circuit, Gate, H, I, S, T, X, Y, Z
from qilisdk.digital.gates import BasicGate, Controlled

from .circuit_transpiler_pass import CircuitTranspilerPass
from .numeric_helpers import (
    _EPS,
    _u3_and_phase_from_unitary,
    _unitary_sqrt_2x2,
    _wrap_angle,
)


def _is_controlled(gate: Gate) -> TypeGuard[Controlled[BasicGate]]:
    return isinstance(gate, Controlled)


# A gate paired with a global phase
PhasedGate = tuple[BasicGate, float]


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
        logger.debug("[DecomposeMultiControlledGatesPass] Running on circuit with {} gates", len(circuit.gates))
        output_circuit = Circuit(circuit.nqubits)
        for gate in circuit.gates:
            for rewritten_gate in self._rewrite_gate(gate):
                output_circuit.add(rewritten_gate)

        logger.debug("[DecomposeMultiControlledGatesPass] Produced circuit with {} gates", len(output_circuit.gates))
        self.append_circuit_to_context(output_circuit)

        return output_circuit

    def _rewrite_gate(self, gate: Gate) -> list[Gate]:  # ruff: ignore[no-self-use]
        """Expand unsupported gates into equivalent elementary gates.

        Args:
            gate (Gate): Candidate gate potentially containing multiple controls.

        Returns:
            list[Gate]: Sequence of equivalent gates that rely on supported primitives.
        """
        # --- Multi-controlled gates ---
        if _is_controlled(gate):
            basic_gate: BasicGate = gate.basic_gate
            if basic_gate.nqubits != 1:
                raise NotImplementedError("Controlled version of multi-qubit gates is not supported.")

            logger.trace(
                "[DecomposeMultiControlledGatesPass] Decomposing {}-controlled {}",
                len(gate.control_qubits),
                type(basic_gate).__name__,
            )
            return _decompose(gate)

        # Everything else is untouched.
        return [gate]


def _decompose(gate: Controlled) -> list[Gate]:
    """Recursively decompose a multi-controlled single-qubit gate.

    Args:
        gate (Controlled): Controlled gate whose target operation is single-qubit.

    Returns:
        list[Gate]: Gate sequence computing the same unitary as `gate`.
    """
    if len(gate.control_qubits) == 1:
        return [gate]

    last_control_qubit = gate.control_qubits[-1]
    remaining_control_qubits = gate.control_qubits[:-1]

    # We need the square root of the target gate, and its adjoint, to build the decomposition
    # sqrt(U) = e^{i·square_root_phase} · square_root_gate
    # square_root_gate† = e^{i·adjoint_phase} · square_root_adjoint_gate,
    # hence sqrt(U)† = e^{i·(adjoint_phase - square_root_phase)} · square_root_adjoint_gate
    square_root_gate, square_root_phase = _sqrt_of(gate.basic_gate)
    square_root_adjoint_gate, adjoint_phase = _adjoint_of(square_root_gate)
    square_root_adjoint_phase = adjoint_phase - square_root_phase

    # Each C(e^{i·alpha}·G) equals C(G) preceded by U1(alpha) on the control qubits, so the leftover phases of the0
    # square root and of its adjoint are emitted as multi-controlled U1 gates instead of being silently dropped
    decomposition_sequence: list[Gate] = []
    decomposition_sequence += _decompose(Controlled(last_control_qubit, basic_gate=square_root_gate))
    decomposition_sequence += _phase_on_controls(square_root_phase, (last_control_qubit,))
    decomposition_sequence += _decompose(X(last_control_qubit).controlled(*remaining_control_qubits))
    decomposition_sequence += _decompose(Controlled(last_control_qubit, basic_gate=square_root_adjoint_gate))
    decomposition_sequence += _phase_on_controls(square_root_adjoint_phase, (last_control_qubit,))
    decomposition_sequence += _decompose(X(last_control_qubit).controlled(*remaining_control_qubits))
    decomposition_sequence += _decompose(Controlled(*remaining_control_qubits, basic_gate=square_root_gate))
    decomposition_sequence += _phase_on_controls(square_root_phase, remaining_control_qubits)

    return decomposition_sequence


def _phase_on_controls(phase: float, control_qubits: tuple[int, ...]) -> list[Gate]:
    """Build the gates applying `e^{i·phase}` only when every control qubit is set.

    Args:
        phase (float): Phase to apply, in radians.
        control_qubits (tuple[int, ...]): Qubits that must all be set for the phase to apply.

    Returns:
        list[Gate]: Empty list when the phase is negligible, otherwise a `C^(k-1)(U1(phase))` decomposition.
    """
    if abs(_wrap_angle(phase)) < _EPS:
        return []

    # A phase conditioned on k qubits is a U1 on any one of them, controlled by the remaining k - 1.
    phase_gate = U1(control_qubits[-1], phi=phase)
    if len(control_qubits) == 1:
        return [phase_gate]
    return _decompose(Controlled(*control_qubits[:-1], basic_gate=phase_gate))


def _sqrt_of(gate: BasicGate) -> PhasedGate:
    """Return a gate V and a phase alpha such that `(e^{i·alpha}·V)² ` equals the provided gate.

    The phase is reported rather than discarded because a global phase on V becomes a relative phase once V is
    placed under a control, which would otherwise make the decomposition compute a different unitary.

    Args:
        gate (BasicGate): Single-qubit gate to compute the principal square root for.
    Returns:
        PhasedGate: New primitive V and the residual phase alpha.
    """
    target_qubit = gate.qubits[0]

    # Identity: sqrt(I) = I
    if isinstance(gate, I):
        return I(target_qubit), 0.0

    # Direct parametric rotations: RX/RY/RZ are exactly half-angle closed.
    if isinstance(gate, RZ):
        return RZ(target_qubit, phi=gate.phi / 2.0), 0.0
    if isinstance(gate, RX):
        return RX(target_qubit, theta=gate.theta / 2.0), 0.0
    if isinstance(gate, RY):
        return RY(target_qubit, theta=gate.theta / 2.0), 0.0

    # Pauli gates via half-angle rotations. Z is diagonal so it has an exact square root, while
    # sqrt(X) = e^{i·pi/4}·RX(pi/2) and sqrt(Y) = e^{i·pi/4}·RY(pi/2) keep a quarter-turn phase.
    if isinstance(gate, Z):
        return S(target_qubit), 0.0
    if isinstance(gate, X):
        return RX(target_qubit, theta=math.pi / 2.0), math.pi / 4.0
    if isinstance(gate, Y):
        return RY(target_qubit, theta=math.pi / 2.0), math.pi / 4.0

    # Phase gate U1(phi) = diag(1, e^{iphi}), sqrt is U1(phi/2).
    if isinstance(gate, U1):
        return U1(target_qubit, phi=gate.phi / 2.0), 0.0

    # S = U1(pi/2) ⇒ sqrt(S) = U1(pi/4) ≡ T
    if isinstance(gate, S):
        return T(target_qubit), 0.0

    # T = U1(pi/4) ⇒ sqrt(T) = U1(pi/8)
    if isinstance(gate, T):
        return U1(target_qubit, phi=math.pi / 8.0), 0.0

    # Build the 2x2 unitary matrix for gate
    if isinstance(gate, BasicGate) and gate.nqubits == 1:
        unitary_matrix = gate.matrix.dense()
    else:
        raise NotImplementedError(f"_sqrt_1q_gate_as_basis only supports 1-qubit gates; got {type(gate).__name__}")

    # Compute a matrix square root V such that V @ V ≈ U.
    square_root_unitary = _unitary_sqrt_2x2(unitary_matrix)

    # Express V as a phase times a U3 on the same qubit. This introduces a new gate in U3 form
    # for the *square root*, but leaves the original gate untouched.
    theta, phi, gamma, alpha = _u3_and_phase_from_unitary(square_root_unitary)
    return U3(target_qubit, theta=theta, phi=phi, gamma=gamma), alpha


def _adjoint_of(gate: BasicGate) -> PhasedGate:
    """Return a gate W and a phase alpha such that `e^{i·alpha}·W` is the adjoint (inverse) of a gate.

    Args:
        gate (BasicGate): Gate whose inverse should be produced.
    Returns:
        PhasedGate: Gate W and the residual phase alpha.
    """
    target_qubit = gate.qubits[0]

    # Identity: self-adjoint.
    if isinstance(gate, I):
        return I(target_qubit), 0.0

    # Pauli & Hadamard: self-adjoint.
    if isinstance(gate, X):
        return X(target_qubit), 0.0
    if isinstance(gate, Y):
        return Y(target_qubit), 0.0
    if isinstance(gate, Z):
        return Z(target_qubit), 0.0
    if isinstance(gate, H):
        return H(target_qubit), 0.0

    if isinstance(gate, RX):
        return RX(target_qubit, theta=-gate.theta), 0.0
    if isinstance(gate, RY):
        return RY(target_qubit, theta=-gate.theta), 0.0
    if isinstance(gate, RZ):
        return RZ(target_qubit, phi=-gate.phi), 0.0

    if isinstance(gate, U1):
        # U1(phi)† = U1(-phi)
        return U1(target_qubit, phi=-gate.phi), 0.0
    if isinstance(gate, U2):
        # U2(phi, gamma)† = U3(pi/2, phi, gamma)† = U3(-pi/2, -gamma, -phi)
        return U3(target_qubit, theta=-math.pi / 2.0, phi=-gate.gamma, gamma=-gate.phi), 0.0
    if isinstance(gate, U3):
        # U3(theta, phi, gamma)† = U3(-theta, -gamma, -phi)
        return U3(target_qubit, theta=-gate.theta, phi=-gate.gamma, gamma=-gate.phi), 0.0

    # S, T: diagonal phase gates about Z.
    # S = U1(pi/2) ⇒ S† = U1(-pi/2)
    if isinstance(gate, S):
        return U1(target_qubit, phi=-math.pi / 2.0), 0.0

    # T = U1(pi/4) ⇒ T† = U1(-pi/4)
    if isinstance(gate, T):
        return U1(target_qubit, phi=-math.pi / 4.0), 0.0

    # ---------- Generic 1-qubit unitary via matrix adjoint ----------

    if isinstance(gate, BasicGate) and gate.nqubits == 1:
        unitary_matrix = gate.matrix.dense()
    else:
        raise NotImplementedError(f"_adjoint_1q only supports 1-qubit gates; got {type(gate).__name__}")

    # Take the matrix adjoint U† and convert to a phase times a U3.
    unitary_adjoint = unitary_matrix.conj().T
    theta, phi, gamma, alpha = _u3_and_phase_from_unitary(unitary_adjoint)
    return U3(target_qubit, theta=theta, phi=phi, gamma=gamma), alpha
