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
from typing import Callable

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
    S,
    T,
    X,
    Y,
    Z,
)

from .circuit_transpiler_pass import CircuitTranspilerPass

ANGLE_TOL = 1e-9


class UniversalSet(Enum):
    """Enumerate the supported universal sets for single and two qubit gates."""

    CLIFFORD_T = "CliffordT"
    RZ_RX_CX = "RzRxCX"
    U3_CX = "U3CX"


class DecomposeToUniversalSetPass(CircuitTranspilerPass):
    """Rewrite every gate in a circuit to a configurable universal set."""

    def __init__(self, target_set: UniversalSet = UniversalSet.RZ_RX_CX) -> None:
        """Initialize the transpiler pass.

        Args:
            target_set (UniversalSet): Universal set that the circuit must use after decomposition.
        """

        self._target_set = target_set

    def run(self, circuit: Circuit) -> Circuit:
        """Return a circuit equivalent to ``circuit`` using the selected universal set.

        Args:
            circuit (Circuit): Circuit to transpile.

        Returns:
            Circuit: Newly constructed circuit expressed in the target universal set.
        """

        out = Circuit(circuit.nqubits)
        for gate in circuit.gates:
            primitives = decompose_gate_for_universal_set(gate, self._target_set.value)
            for primitive in primitives:
                out.add(primitive)
        return out


# ======================= Universal set decomposition helpers =======================


def _wrap_angle(angle: float) -> float:
    """Wrap an angle into the ``(-pi, pi]`` interval.

    Args:
        angle (float): Angle expressed in radians.

    Returns:
        float: Value mapped to the open closed interval ``(-pi, pi]``.
    """

    angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
    if angle <= -math.pi:
        angle = math.pi
    return angle


def _normalized_u3(qubit: int, theta: float, phi: float, gamma: float) -> U3:
    """Create a U3 gate while normalizing its parameters.

    Args:
        qubit (int): Target qubit index.
        theta (float): Polar rotation expressed in radians.
        phi (float): First azimuthal rotation expressed in radians.
        gamma (float): Second azimuthal rotation expressed in radians.

    Returns:
        U3: Normalized U3 instance acting on ``qubit``.
    """

    return U3(qubit, theta=_wrap_angle(theta), phi=_wrap_angle(phi), gamma=_wrap_angle(gamma))


def _ry_as_rxrz_sequence(qubit: int, theta: float) -> list[Gate]:
    """Decompose a RY gate into the RX RZ RX product.

    Args:
        qubit (int): Target qubit index.
        theta (float): Rotation angle expressed in radians.

    Returns:
        list[Gate]: Sequence equivalent to ``RY(theta)`` using RX and RZ gates.
    """

    return [
        RZ(qubit, phi=_wrap_angle(math.pi / 2.0)),
        RX(qubit, theta=_wrap_angle(theta)),
        RZ(qubit, phi=_wrap_angle(-math.pi / 2.0)),
    ]


def _u3_to_rxrz_sequence(qubit: int, theta: float, phi: float, lam: float) -> list[Gate]:
    """Convert a U3 gate into the canonical RZ RX RZ pattern.

    Args:
        qubit (int): Target qubit index.
        theta (float): Polar rotation expressed in radians.
        phi (float): First azimuthal rotation expressed in radians.
        lam (float): Second azimuthal rotation expressed in radians.

    Returns:
        list[Gate]: Sequence composed of RX and RZ gates equivalent to the original U3 gate.
    """

    return [
        RZ(qubit, phi=_wrap_angle(phi + math.pi / 2.0)),
        RX(qubit, theta=_wrap_angle(theta)),
        RZ(qubit, phi=_wrap_angle(lam - math.pi / 2.0)),
    ]


def _h_as_rxrz_sequence(qubit: int) -> list[Gate]:
    """Express a Hadamard gate via RX and RZ rotations.

    Args:
        qubit (int): Target qubit index.

    Returns:
        list[Gate]: Sequence of RX and RZ gates equivalent to a Hadamard gate.
    """

    return _u3_to_rxrz_sequence(qubit, math.pi / 2.0, 0.0, math.pi)


def _rz_sequence_for_clifford(qubit: int, angle: float) -> list[Gate]:
    """Rewrite an RZ rotation using S and T gates when possible.

    Args:
        qubit (int): Target qubit index.
        angle (float): Rotation angle expressed in radians.

    Returns:
        list[Gate]: Sequence containing S and T gates equivalent to ``RZ(angle)``.

    Raises:
        ValueError: If ``angle`` is not a multiple of ``pi / 4``.
    """

    angle = _wrap_angle(angle)
    step = math.pi / 4.0
    k = round(angle / step)
    approx = k * step
    if not math.isclose(angle, approx, abs_tol=ANGLE_TOL):
        raise ValueError("RZ rotations must be multiples of pi/4 for Clifford+T decompositions.")
    steps = int(k % 8)
    sequence: list[Gate] = []
    for _ in range(steps // 2):
        sequence.append(S(qubit))
    if steps % 2:
        sequence.append(T(qubit))
    return sequence


def _swap_via_cnot(qubit_a: int, qubit_b: int) -> list[Gate]:
    """Decompose a SWAP gate using three CNOT gates.

    Args:
        qubit_a (int): First qubit label.
        qubit_b (int): Second qubit label.

    Returns:
        list[Gate]: Sequence of CNOT gates implementing the SWAP unitary.
    """

    return [CNOT(qubit_a, qubit_b), CNOT(qubit_b, qubit_a), CNOT(qubit_a, qubit_b)]


def _M_for_CliffordT(gate: M) -> list[Gate]:
    """Preserve measurement gates when targeting the Clifford+T universal set.

    Args:
        gate (M): Measurement gate to keep.

    Returns:
        list[Gate]: List containing the original measurement gate.
    """

    return [gate]


def _M_for_RzRxCX(gate: M) -> list[Gate]:
    """Preserve measurement gates when targeting the Rz Rx CX universal set.

    Args:
        gate (M): Measurement gate to keep.

    Returns:
        list[Gate]: List containing the original measurement gate.
    """

    return [gate]


def _M_for_U3CX(gate: M) -> list[Gate]:
    """Preserve measurement gates when targeting the U3 CX universal set.

    Args:
        gate (M): Measurement gate to keep.

    Returns:
        list[Gate]: List containing the original measurement gate.
    """

    return [gate]


def _I_for_CliffordT(gate: I) -> list[Gate]:
    """Eliminate explicit identity gates in the Clifford+T universal set.

    Args:
        gate (I): Identity gate to drop.

    Returns:
        list[Gate]: Empty list because the identity does not contribute to the circuit.
    """

    return []


def _I_for_RzRxCX(gate: I) -> list[Gate]:
    """Eliminate explicit identity gates in the Rz Rx CX universal set.

    Args:
        gate (I): Identity gate to drop.

    Returns:
        list[Gate]: Empty list because the identity does not contribute to the circuit.
    """

    return []


def _I_for_U3CX(gate: I) -> list[Gate]:
    """Eliminate explicit identity gates in the U3 CX universal set.

    Args:
        gate (I): Identity gate to drop.

    Returns:
        list[Gate]: Empty list because the identity does not contribute to the circuit.
    """

    return []


def _RZ_for_CliffordT(gate: RZ) -> list[Gate]:
    """Rewrite an RZ gate using Clifford+T primitives.

    Args:
        gate (RZ): Rotation gate expressed around Z.

    Returns:
        list[Gate]: Sequence containing S and T gates equivalent to the original RZ gate.
    """

    qubit = gate.qubits[0]
    return _rz_sequence_for_clifford(qubit, gate.phi)


def _RZ_for_RzRxCX(gate: RZ) -> list[Gate]:
    """Keep RZ gates intact in the Rz Rx CX universal set.

    Args:
        gate (RZ): Rotation gate expressed around Z.

    Returns:
        list[Gate]: List containing a normalized RZ gate.
    """

    qubit = gate.qubits[0]
    return [RZ(qubit, phi=_wrap_angle(gate.phi))]


def _RZ_for_U3CX(gate: RZ) -> list[Gate]:
    """Convert an RZ rotation into its U3 equivalent.

    Args:
        gate (RZ): Rotation gate expressed around Z.

    Returns:
        list[Gate]: List containing a single U3 gate describing the rotation.
    """

    qubit = gate.qubits[0]
    return [_normalized_u3(qubit, 0.0, gate.phi, 0.0)]


def _RX_for_CliffordT(gate: RX) -> list[Gate]:
    """Rewrite an RX gate using Clifford+T primitives.

    Args:
        gate (RX): Rotation gate expressed around X.

    Returns:
        list[Gate]: Sequence of H, S and T gates equivalent to the RX gate.
    """

    qubit = gate.qubits[0]
    rz_sequence = _rz_sequence_for_clifford(qubit, gate.theta)
    return [H(qubit), *rz_sequence, H(qubit)]


def _RX_for_RzRxCX(gate: RX) -> list[Gate]:
    """Keep RX gates intact in the Rz Rx CX universal set.

    Args:
        gate (RX): Rotation gate expressed around X.

    Returns:
        list[Gate]: List containing a normalized RX gate.
    """

    qubit = gate.qubits[0]
    return [RX(qubit, theta=_wrap_angle(gate.theta))]


def _RX_for_U3CX(gate: RX) -> list[Gate]:
    """Convert an RX rotation into its U3 equivalent.

    Args:
        gate (RX): Rotation gate expressed around X.

    Returns:
        list[Gate]: List containing a single U3 gate describing the rotation.
    """

    qubit = gate.qubits[0]
    return [_normalized_u3(qubit, gate.theta, -math.pi / 2.0, math.pi / 2.0)]


def _convert_rxrz_sequence_to_clifford(sequence: list[Gate]) -> list[Gate]:
    """Rewrite a sequence of RX and RZ gates using Clifford+T primitives.

    Args:
        sequence (list[Gate]): Sequence containing RX and RZ gates only.

    Returns:
        list[Gate]: Sequence that uses the Clifford+T universal set exclusively.
    """

    expanded: list[Gate] = []
    for primitive in sequence:
        if isinstance(primitive, RX):
            expanded.extend(_RX_for_CliffordT(primitive))
        elif isinstance(primitive, RZ):
            expanded.extend(_RZ_for_CliffordT(primitive))
        else:
            msg = f"Only RX and RZ gates can be converted, received {type(primitive).__name__}."
            raise ValueError(msg)
    return expanded


def _RY_for_CliffordT(gate: RY) -> list[Gate]:
    """Rewrite an RY gate using Clifford+T primitives.

    Args:
        gate (RY): Rotation gate expressed around Y.

    Returns:
        list[Gate]: Sequence of Clifford+T gates equivalent to the RY gate.
    """

    sequence = _ry_as_rxrz_sequence(gate.qubits[0], gate.theta)
    return _convert_rxrz_sequence_to_clifford(sequence)


def _RY_for_RzRxCX(gate: RY) -> list[Gate]:
    """Decompose an RY gate into RX and RZ gates.

    Args:
        gate (RY): Rotation gate expressed around Y.

    Returns:
        list[Gate]: Sequence of RX and RZ gates equivalent to the RY gate.
    """

    return _ry_as_rxrz_sequence(gate.qubits[0], gate.theta)


def _RY_for_U3CX(gate: RY) -> list[Gate]:
    """Convert an RY rotation into its U3 equivalent.

    Args:
        gate (RY): Rotation gate expressed around Y.

    Returns:
        list[Gate]: List containing a single U3 gate describing the rotation.
    """

    qubit = gate.qubits[0]
    return [_normalized_u3(qubit, gate.theta, 0.0, 0.0)]


def _U1_for_CliffordT(gate: U1) -> list[Gate]:
    """Rewrite a U1 gate using Clifford+T primitives.

    Args:
        gate (U1): Azimuthal rotation gate.

    Returns:
        list[Gate]: Sequence of Clifford+T gates equivalent to the U1 gate.
    """

    rz_gate = RZ(gate.qubits[0], phi=gate.phi)
    return _RZ_for_CliffordT(rz_gate)


def _U1_for_RzRxCX(gate: U1) -> list[Gate]:
    """Preserve U1 gates as RZ rotations in the Rz Rx CX universal set.

    Args:
        gate (U1): Azimuthal rotation gate.

    Returns:
        list[Gate]: Sequence containing a normalized RZ gate.
    """

    rz_gate = RZ(gate.qubits[0], phi=gate.phi)
    return _RZ_for_RzRxCX(rz_gate)


def _U1_for_U3CX(gate: U1) -> list[Gate]:
    """Convert a U1 gate into its U3 equivalent.

    Args:
        gate (U1): Azimuthal rotation gate.

    Returns:
        list[Gate]: List containing a single U3 gate describing the rotation.
    """

    return [_normalized_u3(gate.qubits[0], 0.0, gate.phi, 0.0)]


def _U2_for_CliffordT(gate: U2) -> list[Gate]:
    """Rewrite a U2 gate using Clifford+T primitives.

    Args:
        gate (U2): Two parameter single qubit gate.

    Returns:
        list[Gate]: Sequence of Clifford+T gates equivalent to the U2 gate.
    """

    sequence = _u3_to_rxrz_sequence(gate.qubits[0], math.pi / 2.0, gate.phi, gate.gamma)
    return _convert_rxrz_sequence_to_clifford(sequence)


def _U2_for_RzRxCX(gate: U2) -> list[Gate]:
    """Express a U2 gate using RX and RZ rotations.

    Args:
        gate (U2): Two parameter single qubit gate.

    Returns:
        list[Gate]: Sequence of RX and RZ gates equivalent to the U2 gate.
    """

    return _u3_to_rxrz_sequence(gate.qubits[0], math.pi / 2.0, gate.phi, gate.gamma)


def _U2_for_U3CX(gate: U2) -> list[Gate]:
    """Convert a U2 gate into its U3 equivalent.

    Args:
        gate (U2): Two parameter single qubit gate.

    Returns:
        list[Gate]: List containing a single U3 gate describing the rotation.
    """

    return [_normalized_u3(gate.qubits[0], math.pi / 2.0, gate.phi, gate.gamma)]


def _U3_for_CliffordT(gate: U3) -> list[Gate]:
    """Rewrite a U3 gate using Clifford+T primitives.

    Args:
        gate (U3): Three parameter single qubit gate.

    Returns:
        list[Gate]: Sequence of Clifford+T gates equivalent to the U3 gate.
    """

    sequence = _u3_to_rxrz_sequence(gate.qubits[0], gate.theta, gate.phi, gate.gamma)
    return _convert_rxrz_sequence_to_clifford(sequence)


def _U3_for_RzRxCX(gate: U3) -> list[Gate]:
    """Express a U3 gate using RX and RZ rotations.

    Args:
        gate (U3): Three parameter single qubit gate.

    Returns:
        list[Gate]: Sequence of RX and RZ gates equivalent to the U3 gate.
    """

    return _u3_to_rxrz_sequence(gate.qubits[0], gate.theta, gate.phi, gate.gamma)


def _U3_for_U3CX(gate: U3) -> list[Gate]:
    """Normalize a U3 gate in the U3 CX universal set.

    Args:
        gate (U3): Three parameter single qubit gate.

    Returns:
        list[Gate]: List containing a normalized U3 gate equivalent to the original one.
    """

    return [_normalized_u3(gate.qubits[0], gate.theta, gate.phi, gate.gamma)]


def _X_for_CliffordT(gate: X) -> list[Gate]:
    """Rewrite an X gate using Clifford+T primitives.

    Args:
        gate (X): Pauli-X gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the X gate in the Clifford+T basis.
    """

    rx_gate = RX(gate.qubits[0], theta=math.pi)
    return _RX_for_CliffordT(rx_gate)


def _X_for_RzRxCX(gate: X) -> list[Gate]:
    """Rewrite an X gate as a single RX rotation.

    Args:
        gate (X): Pauli-X gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single RX gate equivalent to the X gate.
    """

    rx_gate = RX(gate.qubits[0], theta=math.pi)
    return _RX_for_RzRxCX(rx_gate)


def _X_for_U3CX(gate: X) -> list[Gate]:
    """Rewrite an X gate as a U3 rotation.

    Args:
        gate (X): Pauli-X gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single U3 gate equivalent to the X gate.
    """

    rx_gate = RX(gate.qubits[0], theta=math.pi)
    return _RX_for_U3CX(rx_gate)


def _Y_for_CliffordT(gate: Y) -> list[Gate]:
    """Rewrite a Y gate using Clifford+T primitives.

    Args:
        gate (Y): Pauli-Y gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the Y gate in the Clifford+T basis.
    """

    ry_gate = RY(gate.qubits[0], theta=math.pi)
    return _RY_for_CliffordT(ry_gate)


def _Y_for_RzRxCX(gate: Y) -> list[Gate]:
    """Rewrite a Y gate using RX and RZ rotations.

    Args:
        gate (Y): Pauli-Y gate to decompose.

    Returns:
        list[Gate]: Sequence of RX and RZ gates equivalent to the Y gate.
    """

    ry_gate = RY(gate.qubits[0], theta=math.pi)
    return _RY_for_RzRxCX(ry_gate)


def _Y_for_U3CX(gate: Y) -> list[Gate]:
    """Rewrite a Y gate as a U3 rotation.

    Args:
        gate (Y): Pauli-Y gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single U3 gate equivalent to the Y gate.
    """

    return [_normalized_u3(gate.qubits[0], math.pi, math.pi / 2.0, -math.pi / 2.0)]


def _Z_for_CliffordT(gate: Z) -> list[Gate]:
    """Rewrite a Z gate using Clifford+T primitives.

    Args:
        gate (Z): Pauli-Z gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the Z gate in the Clifford+T basis.
    """

    rz_gate = RZ(gate.qubits[0], phi=math.pi)
    return _RZ_for_CliffordT(rz_gate)


def _Z_for_RzRxCX(gate: Z) -> list[Gate]:
    """Rewrite a Z gate using a single RZ rotation.

    Args:
        gate (Z): Pauli-Z gate to decompose.

    Returns:
        list[Gate]: Sequence containing an RZ gate equivalent to the Z gate.
    """

    rz_gate = RZ(gate.qubits[0], phi=math.pi)
    return _RZ_for_RzRxCX(rz_gate)


def _Z_for_U3CX(gate: Z) -> list[Gate]:
    """Rewrite a Z gate as a U3 rotation.

    Args:
        gate (Z): Pauli-Z gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single U3 gate equivalent to the Z gate.
    """

    return [_normalized_u3(gate.qubits[0], 0.0, math.pi, 0.0)]


def _H_for_CliffordT(gate: H) -> list[Gate]:
    """Preserve Hadamard gates in the Clifford+T universal set.

    Args:
        gate (H): Hadamard gate to keep.

    Returns:
        list[Gate]: List containing the original Hadamard gate.
    """

    return [gate]


def _H_for_RzRxCX(gate: H) -> list[Gate]:
    """Rewrite a Hadamard gate using RX and RZ rotations.

    Args:
        gate (H): Hadamard gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the Hadamard gate in the Rz Rx CX basis.
    """

    return _h_as_rxrz_sequence(gate.qubits[0])


def _H_for_U3CX(gate: H) -> list[Gate]:
    """Rewrite a Hadamard gate as a U3 rotation.

    Args:
        gate (H): Hadamard gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single U3 gate equivalent to the Hadamard gate.
    """

    return [_normalized_u3(gate.qubits[0], math.pi / 2.0, 0.0, math.pi)]


def _S_for_CliffordT(gate: S) -> list[Gate]:
    """Preserve phase gates in the Clifford+T universal set.

    Args:
        gate (S): Phase gate to keep.

    Returns:
        list[Gate]: List containing the original S gate.
    """

    return [gate]


def _S_for_RzRxCX(gate: S) -> list[Gate]:
    """Rewrite a phase gate using a single RZ rotation.

    Args:
        gate (S): Phase gate to decompose.

    Returns:
        list[Gate]: Sequence containing an RZ gate equivalent to the phase gate.
    """

    rz_gate = RZ(gate.qubits[0], phi=math.pi / 2.0)
    return _RZ_for_RzRxCX(rz_gate)


def _S_for_U3CX(gate: S) -> list[Gate]:
    """Rewrite a phase gate as a U3 rotation.

    Args:
        gate (S): Phase gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single U3 gate equivalent to the phase gate.
    """

    return [_normalized_u3(gate.qubits[0], 0.0, math.pi / 2.0, 0.0)]


def _T_for_CliffordT(gate: T) -> list[Gate]:
    """Preserve T gates in the Clifford+T universal set.

    Args:
        gate (T): T gate to keep.

    Returns:
        list[Gate]: List containing the original T gate.
    """

    return [gate]


def _T_for_RzRxCX(gate: T) -> list[Gate]:
    """Rewrite a T gate using a single RZ rotation.

    Args:
        gate (T): T gate to decompose.

    Returns:
        list[Gate]: Sequence containing an RZ gate equivalent to the T gate.
    """

    rz_gate = RZ(gate.qubits[0], phi=math.pi / 4.0)
    return _RZ_for_RzRxCX(rz_gate)


def _T_for_U3CX(gate: T) -> list[Gate]:
    """Rewrite a T gate as a U3 rotation.

    Args:
        gate (T): T gate to decompose.

    Returns:
        list[Gate]: Sequence containing a single U3 gate equivalent to the T gate.
    """

    return [_normalized_u3(gate.qubits[0], 0.0, math.pi / 4.0, 0.0)]


def _CNOT_for_CliffordT(gate: CNOT) -> list[Gate]:
    """Preserve CNOT gates in the Clifford+T universal set.

    Args:
        gate (CNOT): Controlled NOT gate to keep.

    Returns:
        list[Gate]: List containing a freshly constructed CNOT gate.
    """

    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    return [CNOT(control, target)]


def _CNOT_for_RzRxCX(gate: CNOT) -> list[Gate]:
    """Preserve CNOT gates in the Rz Rx CX universal set.

    Args:
        gate (CNOT): Controlled NOT gate to keep.

    Returns:
        list[Gate]: List containing a freshly constructed CNOT gate.
    """

    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    return [CNOT(control, target)]


def _CNOT_for_U3CX(gate: CNOT) -> list[Gate]:
    """Preserve CNOT gates in the U3 CX universal set.

    Args:
        gate (CNOT): Controlled NOT gate to keep.

    Returns:
        list[Gate]: List containing a freshly constructed CNOT gate.
    """

    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    return [CNOT(control, target)]


def _CZ_for_CliffordT(gate: CZ) -> list[Gate]:
    """Rewrite a CZ gate using CNOT and Hadamard gates in the Clifford+T basis.

    Args:
        gate (CZ): Controlled-Z gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the CZ gate.
    """

    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    return [H(target), CNOT(control, target), H(target)]


def _CZ_for_RzRxCX(gate: CZ) -> list[Gate]:
    """Rewrite a CZ gate using the Rz Rx CX primitive set.

    Args:
        gate (CZ): Controlled-Z gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the CZ gate.
    """

    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    local_h = _H_for_RzRxCX(H(target))
    return [*local_h, CNOT(control, target), *local_h]


def _CZ_for_U3CX(gate: CZ) -> list[Gate]:
    """Rewrite a CZ gate using the U3 CX primitive set.

    Args:
        gate (CZ): Controlled-Z gate to decompose.

    Returns:
        list[Gate]: Sequence equivalent to the CZ gate.
    """

    control = gate.control_qubits[0]
    target = gate.target_qubits[0]
    local_h = _H_for_U3CX(H(target))
    return [*local_h, CNOT(control, target), *local_h]


def _SWAP_for_CliffordT(gate: SWAP) -> list[Gate]:
    """Rewrite a SWAP gate using CNOT gates in the Clifford+T basis.

    Args:
        gate (SWAP): SWAP gate to decompose.

    Returns:
        list[Gate]: Sequence of CNOT gates equivalent to the SWAP gate.
    """

    a, b = gate.target_qubits
    return _swap_via_cnot(a, b)


def _SWAP_for_RzRxCX(gate: SWAP) -> list[Gate]:
    """Rewrite a SWAP gate using CNOT gates in the Rz Rx CX basis.

    Args:
        gate (SWAP): SWAP gate to decompose.

    Returns:
        list[Gate]: Sequence of CNOT gates equivalent to the SWAP gate.
    """

    a, b = gate.target_qubits
    return _swap_via_cnot(a, b)


def _SWAP_for_U3CX(gate: SWAP) -> list[Gate]:
    """Rewrite a SWAP gate using CNOT gates in the U3 CX basis.

    Args:
        gate (SWAP): SWAP gate to decompose.

    Returns:
        list[Gate]: Sequence of CNOT gates equivalent to the SWAP gate.
    """

    a, b = gate.target_qubits
    return _swap_via_cnot(a, b)


_DECOMPOSERS: dict[type[Gate], dict[UniversalSet, Callable[..., list[Gate]]]] = {
    M: {
        UniversalSet.CLIFFORD_T: _M_for_CliffordT,
        UniversalSet.RZ_RX_CX: _M_for_RzRxCX,
        UniversalSet.U3_CX: _M_for_U3CX,
    },
    I: {
        UniversalSet.CLIFFORD_T: _I_for_CliffordT,
        UniversalSet.RZ_RX_CX: _I_for_RzRxCX,
        UniversalSet.U3_CX: _I_for_U3CX,
    },
    RX: {
        UniversalSet.CLIFFORD_T: _RX_for_CliffordT,
        UniversalSet.RZ_RX_CX: _RX_for_RzRxCX,
        UniversalSet.U3_CX: _RX_for_U3CX,
    },
    RY: {
        UniversalSet.CLIFFORD_T: _RY_for_CliffordT,
        UniversalSet.RZ_RX_CX: _RY_for_RzRxCX,
        UniversalSet.U3_CX: _RY_for_U3CX,
    },
    RZ: {
        UniversalSet.CLIFFORD_T: _RZ_for_CliffordT,
        UniversalSet.RZ_RX_CX: _RZ_for_RzRxCX,
        UniversalSet.U3_CX: _RZ_for_U3CX,
    },
    U1: {
        UniversalSet.CLIFFORD_T: _U1_for_CliffordT,
        UniversalSet.RZ_RX_CX: _U1_for_RzRxCX,
        UniversalSet.U3_CX: _U1_for_U3CX,
    },
    U2: {
        UniversalSet.CLIFFORD_T: _U2_for_CliffordT,
        UniversalSet.RZ_RX_CX: _U2_for_RzRxCX,
        UniversalSet.U3_CX: _U2_for_U3CX,
    },
    U3: {
        UniversalSet.CLIFFORD_T: _U3_for_CliffordT,
        UniversalSet.RZ_RX_CX: _U3_for_RzRxCX,
        UniversalSet.U3_CX: _U3_for_U3CX,
    },
    X: {
        UniversalSet.CLIFFORD_T: _X_for_CliffordT,
        UniversalSet.RZ_RX_CX: _X_for_RzRxCX,
        UniversalSet.U3_CX: _X_for_U3CX,
    },
    Y: {
        UniversalSet.CLIFFORD_T: _Y_for_CliffordT,
        UniversalSet.RZ_RX_CX: _Y_for_RzRxCX,
        UniversalSet.U3_CX: _Y_for_U3CX,
    },
    Z: {
        UniversalSet.CLIFFORD_T: _Z_for_CliffordT,
        UniversalSet.RZ_RX_CX: _Z_for_RzRxCX,
        UniversalSet.U3_CX: _Z_for_U3CX,
    },
    H: {
        UniversalSet.CLIFFORD_T: _H_for_CliffordT,
        UniversalSet.RZ_RX_CX: _H_for_RzRxCX,
        UniversalSet.U3_CX: _H_for_U3CX,
    },
    S: {
        UniversalSet.CLIFFORD_T: _S_for_CliffordT,
        UniversalSet.RZ_RX_CX: _S_for_RzRxCX,
        UniversalSet.U3_CX: _S_for_U3CX,
    },
    T: {
        UniversalSet.CLIFFORD_T: _T_for_CliffordT,
        UniversalSet.RZ_RX_CX: _T_for_RzRxCX,
        UniversalSet.U3_CX: _T_for_U3CX,
    },
    CNOT: {
        UniversalSet.CLIFFORD_T: _CNOT_for_CliffordT,
        UniversalSet.RZ_RX_CX: _CNOT_for_RzRxCX,
        UniversalSet.U3_CX: _CNOT_for_U3CX,
    },
    CZ: {
        UniversalSet.CLIFFORD_T: _CZ_for_CliffordT,
        UniversalSet.RZ_RX_CX: _CZ_for_RzRxCX,
        UniversalSet.U3_CX: _CZ_for_U3CX,
    },
    SWAP: {
        UniversalSet.CLIFFORD_T: _SWAP_for_CliffordT,
        UniversalSet.RZ_RX_CX: _SWAP_for_RzRxCX,
        UniversalSet.U3_CX: _SWAP_for_U3CX,
    },
}


def decompose_gate_for_universal_set(gate: Gate, basis: UniversalSet) -> list[Gate]:
    """Decompose a gate into one of the supported universal sets.

    Args:
        gate (Gate): Gate to decompose.
        basis (UniversalSetName): Target universal set name.

    Returns:
        list[Gate]: Sequence of gates implementing ``gate`` using the ``basis`` primitives.

    Raises:
        NotImplementedError: If the gate type or basis combination is not supported.
    """

    gate_type = type(gate)
    try:
        basis_map = _DECOMPOSERS[gate_type]
    except KeyError as exc:
        msg = f"Unsupported gate type {gate_type.__name__}"
        raise NotImplementedError(msg) from exc
    try:
        transform = basis_map[basis]
    except KeyError as exc:
        msg = f"Unsupported basis {basis} for gate type {gate_type.__name__}"
        raise NotImplementedError(msg) from exc
    return transform(gate)
