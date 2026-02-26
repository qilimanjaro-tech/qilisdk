# Copyright 2023 Qilimanjaro Quantum Tech
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

import numpy as np

from qilisdk.digital import Circuit
from qilisdk.digital.exceptions import GateHasNoMatrixError
from qilisdk.digital.gates import (
    RX,
    RY,
    RZ,
    U3,
    Gate,
)

from .circuit_transpiler_pass import CircuitTranspilerPass
from .numeric_helpers import (
    _is_close_mod_2pi,
    _wrap_angle,
    _zyz_from_unitary,
)


class FuseSingleQubitGatesPass(CircuitTranspilerPass):
    """
    Fuse maximal adjacent runs of 1-qubit unitary gates per wire into a single gate:

      - If the fused unitary is a pure Z: emit RZ(phi).
      - Recognizable canonical forms: RY(theta) if (phi≈0, gamma≈0), RX(theta) if (phi≈-π/2, gamma≈π/2).
      - Otherwise emit a single U3(theta, phi, gamma).

    Any gate that is not a 1-qubit unitary matrix (including measurements or
    unsupported/no-matrix gates) acts as a boundary for the touched qubits.
    Always returns a NEW circuit.
    """

    @staticmethod
    def _try_single_qubit_unitary(gate: Gate) -> np.ndarray | None:
        """Return the 2x2 unitary matrix for a 1-qubit gate, or None if unavailable."""
        if gate.nqubits != 1:
            return None

        try:
            unitary = np.array(gate.matrix, dtype=complex, copy=True)
        except GateHasNoMatrixError:
            return None

        if unitary.shape != (2, 2):
            return None

        identity = np.eye(2, dtype=complex)
        if not np.allclose(unitary.conj().T @ unitary, identity, atol=1e-10):
            return None
        return unitary

    def run(self, circuit: Circuit) -> Circuit:
        output_gates: list[Gate] = []
        pending: dict[int, np.ndarray] = {}

        def flush_pending_qubit(qubit: int) -> None:
            """Emit the fused gate accumulated for a qubit, if any.

            This helper mutates `pending` and appends at most one gate to
            `seq_out`.

            Args:
                qubit (int): Target qubit whose pending 1-qubit unitary should
                    be converted back into a canonical gate.
            """
            if qubit not in pending:
                return
            U = pending.pop(qubit)
            theta, phi, gamma = _zyz_from_unitary(U)
            theta, phi, gamma = float(theta), float(phi), float(gamma)
            theta = _wrap_angle(theta)
            phi = _wrap_angle(phi)
            gamma = _wrap_angle(gamma)
            if _is_close_mod_2pi(theta, 0.0):
                # pure Z: RZ(ph + lam)
                output_gates.append(RZ(qubit, phi=_wrap_angle(phi + gamma)))
                return
            if _is_close_mod_2pi(phi, 0.0) and _is_close_mod_2pi(gamma, 0.0):
                output_gates.append(RY(qubit, theta=theta))
                return
            if _is_close_mod_2pi(phi, math.pi) and _is_close_mod_2pi(gamma, math.pi):
                output_gates.append(RY(qubit, theta=-theta))
                return
            if _is_close_mod_2pi(phi, -math.pi / 2.0) and _is_close_mod_2pi(gamma, math.pi / 2.0):
                output_gates.append(RX(qubit, theta=theta))
                return
            if _is_close_mod_2pi(phi, math.pi / 2.0) and _is_close_mod_2pi(gamma, -math.pi / 2.0):
                output_gates.append(RX(qubit, theta=-theta))
                return
            output_gates.append(U3(qubit, theta=theta, phi=phi, gamma=gamma))

        for gate in circuit.gates:
            unitary = self._try_single_qubit_unitary(gate)
            if unitary is not None:
                qubit = gate.qubits[0]
                pending[qubit] = unitary @ pending[qubit] if qubit in pending else unitary
                continue

            # Boundary: flush pending runs that touch gate qubits.
            for qubit in gate.qubits:
                flush_pending_qubit(qubit)
            output_gates.append(gate)

        # flush any remaining
        for qubit in list(pending):
            flush_pending_qubit(qubit)

        output_circuit = Circuit(circuit.nqubits)
        for gate in output_gates:
            output_circuit.add(gate)

        self.append_circuit_to_context(output_circuit)
        return output_circuit
