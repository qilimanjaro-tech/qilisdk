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
from qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass import SingleQubitGateBasis
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
    Fuse maximal adjacent runs of 1-qubit unitary gates per wire into a basis-respecting sequence:

      - For ``SingleQubitGateBasis.U3``: emit a single ``U3``.
      - For ``SingleQubitGateBasis.RxRyRz``: emit ``RZ``, ``RY``, ``RX`` when recognizable, otherwise emit
        the equivalent ``RZ(phi) · RY(theta) · RZ(gamma)`` sequence.

    Any gate that is not a 1-qubit unitary matrix (including measurements or
    unsupported/no-matrix gates) acts as a boundary for the touched qubits.
    Always returns a NEW circuit.
    """

    def __init__(self, single_qubit_basis: SingleQubitGateBasis = SingleQubitGateBasis.U3) -> None:
        if not isinstance(single_qubit_basis, SingleQubitGateBasis):
            raise TypeError(
                f"single_qubit_basis must be a SingleQubitGateBasis value (got {type(single_qubit_basis).__name__})."
            )
        self._single_qubit_basis = single_qubit_basis

    @property
    def single_qubit_basis(self) -> SingleQubitGateBasis:
        return self._single_qubit_basis

    @staticmethod
    def _try_single_qubit_unitary(gate: Gate) -> np.ndarray | None:
        """Return a 2x2 unitary matrix for a single-qubit gate when available.

        Args:
            gate (Gate): Gate candidate to validate and convert to matrix form.

        Returns:
            np.ndarray | None: Complex 2x2 unitary matrix when ``gate`` is a valid one-qubit unitary; otherwise ``None``.
        """
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
        """Fuse adjacent one-qubit unitary sequences into canonical single gates.

        Non-unitary or multi-qubit gates define boundaries and force flushing of pending fused unitaries on touched qubits.

        Args:
            circuit (Circuit): Input circuit to optimize.

        Returns:
            Circuit: New circuit with fused one-qubit runs and unchanged behavior.
        """
        output_gates: list[Gate] = []
        pending: dict[int, np.ndarray] = {}

        def flush_pending_qubit(qubit: int) -> None:
            """Emit the fused gate accumulated for a qubit, if any.

            This helper mutates ``pending`` and appends a basis-respecting canonical sequence to ``output_gates``.

            Args:
                qubit (int): Qubit whose pending fused unitary is materialized in the configured single-qubit basis.
            """
            if qubit not in pending:
                return
            U = pending.pop(qubit)
            output_gates.extend(self._emit_fused_basis_gates(qubit, U))

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
        while pending:
            flush_pending_qubit(next(iter(pending)))

        output_circuit = Circuit(circuit.nqubits)
        for gate in output_gates:
            output_circuit.add(gate)

        self.append_circuit_to_context(output_circuit)
        return output_circuit

    def _emit_fused_basis_gates(self, qubit: int, unitary: np.ndarray) -> list[Gate]:
        theta, phi, gamma = _zyz_from_unitary(unitary)
        theta, phi, gamma = float(theta), float(phi), float(gamma)
        theta = _wrap_angle(theta)
        phi = _wrap_angle(phi)
        gamma = _wrap_angle(gamma)

        if self._single_qubit_basis == SingleQubitGateBasis.U3:
            return [U3(qubit, theta=theta, phi=phi, gamma=gamma)]

        if self._single_qubit_basis == SingleQubitGateBasis.RxRyRz:
            if _is_close_mod_2pi(theta, 0.0):
                # pure Z: RZ(ph + lam)
                return [RZ(qubit, phi=_wrap_angle(phi + gamma))]
            if _is_close_mod_2pi(phi, 0.0) and _is_close_mod_2pi(gamma, 0.0):
                return [RY(qubit, theta=theta)]
            if _is_close_mod_2pi(phi, math.pi) and _is_close_mod_2pi(gamma, math.pi):
                return [RY(qubit, theta=-theta)]
            if _is_close_mod_2pi(phi, -math.pi / 2.0) and _is_close_mod_2pi(gamma, math.pi / 2.0):
                return [RX(qubit, theta=theta)]
            if _is_close_mod_2pi(phi, math.pi / 2.0) and _is_close_mod_2pi(gamma, -math.pi / 2.0):
                return [RX(qubit, theta=-theta)]
            return [
                RZ(qubit, phi=gamma),
                RY(qubit, theta=theta),
                RZ(qubit, phi=phi),
            ]

        raise TypeError(f"Unsupported single-qubit basis {self._single_qubit_basis}")
