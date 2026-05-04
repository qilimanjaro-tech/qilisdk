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

from qilisdk.digital import Circuit
from qilisdk.digital.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    U3,
    Gate,
    M,
)
from qilisdk.digital.native_gates import Rmw

from .circuit_transpiler_pass import CircuitTranspilerPass
from .decompose_to_canonical_basis_pass import TwoQubitGateBasis
from .numeric_helpers import _wrap_angle


class NativeSingleQubitGateBasis(Enum):
    """Selectable native single-qubit gate used by the lowering pass."""

    Rmw = "Rmw"


class CanonicalBasisToNativeSetPass(CircuitTranspilerPass):
    """
    Lower from the canonical basis ``{CZ|CNOT, U3, RX, RY, RZ, M}`` to the
    native set ``{<native 1Q>, CZ|CNOT, M}`` (+ optional virtual RZ).

    The native single-qubit gate is selectable via ``single_qubit_basis``
    (``NativeSingleQubitGateBasis``). The two-qubit gate is selectable via
    ``two_qubit_basis`` (``TwoQubitGateBasis``); the input circuit is expected
    to already use the matching two-qubit basis.

    Mapping (Rmw native):
      - U3(theta, phi, gamma) → Rmw(theta, phase=phi - π/2) ; RZ(phi + gamma)
      - RX(theta)             → Rmw(theta, phase=0)
      - RY(theta)             → Rmw(theta, phase=π/2)
      - CZ / CNOT             → forwarded as-is
      - M                     → M

    Options:
      - keep_virtual_rz: keep RZ as a virtual Z (default True).
      - merge_consecutive_rz: sum consecutive RZ on each wire (default True).
      - drop_rz_before_measure: drop any pending RZ on measured qubits (default True).
      - angle_tol: numerical tolerance for dropping near-zero angles.
    """

    def __init__(
        self,
        single_qubit_basis: NativeSingleQubitGateBasis = NativeSingleQubitGateBasis.Rmw,
        two_qubit_basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
        *,
        keep_virtual_rz: bool = True,
        merge_consecutive_rz: bool = True,
        drop_rz_before_measure: bool = True,
        angle_tol: float = 1e-12,
    ) -> None:
        if not isinstance(single_qubit_basis, NativeSingleQubitGateBasis):
            raise TypeError(
                "single_qubit_basis must be a NativeSingleQubitGateBasis value "
                f"(got {type(single_qubit_basis).__name__})."
            )
        if not isinstance(two_qubit_basis, TwoQubitGateBasis):
            raise TypeError(
                f"two_qubit_basis must be a TwoQubitGateBasis value (got {type(two_qubit_basis).__name__})."
            )
        self._single_qubit_basis = single_qubit_basis
        self._two_qubit_basis = two_qubit_basis
        self.keep_virtual_rz = keep_virtual_rz
        self.merge_consecutive_rz = merge_consecutive_rz
        self.drop_rz_before_measure = drop_rz_before_measure
        self.angle_tol = float(angle_tol)

    @property
    def single_qubit_basis(self) -> NativeSingleQubitGateBasis:
        return self._single_qubit_basis

    @property
    def two_qubit_basis(self) -> TwoQubitGateBasis:
        return self._two_qubit_basis

    def run(self, circuit: Circuit) -> Circuit:
        out = Circuit(circuit.nqubits)

        # Pending virtual Z rotation per qubit (for simple RZ merging / scheduling)
        pending_rz: dict[int, float] = {}

        def emit_rz(q: int) -> None:
            """Flush a pending RZ on q, if any (and if we keep RZ at all)."""
            if not self.keep_virtual_rz:
                pending_rz.pop(q, None)
                return
            if q in pending_rz:
                phi = _wrap_angle(pending_rz[q])
                pending_rz.pop(q, None)
                if abs(phi) > self.angle_tol:
                    out.add(RZ(q, phi=phi))

        def touch(*qubits: int) -> None:
            """Before emitting any non-commuting gate on given qubits, flush their pending RZ."""
            for q in qubits:
                emit_rz(q)

        def add_rz(q: int, phi: float) -> None:
            """Accumulate a virtual Z on q (merged later)."""
            if not self.keep_virtual_rz:
                return
            if self.merge_consecutive_rz:
                pending_rz[q] = _wrap_angle(pending_rz.get(q, 0.0) + phi)
            else:
                emit_rz(q)
                if abs(phi) > self.angle_tol:
                    out.add(RZ(q, phi=_wrap_angle(phi)))

        def lower_1q(g: Gate) -> None:
            q = g.qubits[0]
            if self._single_qubit_basis == NativeSingleQubitGateBasis.Rmw:
                if isinstance(g, RX):
                    touch(q)
                    out.add(Rmw(q, theta=g.theta, phase=0.0))
                elif isinstance(g, RY):
                    touch(q)
                    out.add(Rmw(q, theta=g.theta, phase=math.pi / 2.0))
                elif isinstance(g, RZ):
                    add_rz(q, g.phi)
                elif isinstance(g, U3):
                    # U3(theta, phi, gamma) = RZ(phi) RY(theta) RZ(gamma)
                    touch(q)
                    out.add(Rmw(q, theta=g.theta, phase=_wrap_angle(g.phi - math.pi / 2)))
                    add_rz(q, _wrap_angle(g.phi + g.gamma))
                else:
                    raise NotImplementedError(f"Unexpected 1-qubit gate in native lowering: {type(g).__name__}")
                return
            raise TypeError(f"Unsupported single-qubit native basis {self._single_qubit_basis}")

        for g in circuit.gates:
            # 1-qubit gates
            if isinstance(g, (U3, RX, RY, RZ)):
                lower_1q(g)
                continue

            # 2-qubit entangler — forward only the configured native two-qubit gate
            if self._two_qubit_basis == TwoQubitGateBasis.CZ and isinstance(g, CZ):
                touch(g.control_qubits[0], g.target_qubits[0])
                out.add(CZ(g.control_qubits[0], g.target_qubits[0]))
                continue
            if self._two_qubit_basis == TwoQubitGateBasis.CNOT and isinstance(g, CNOT):
                touch(g.control_qubits[0], g.target_qubits[0])
                out.add(CNOT(g.control_qubits[0], g.target_qubits[0]))
                continue

            # Measurement
            if isinstance(g, M):
                if self.drop_rz_before_measure:
                    for q in g.qubits:
                        pending_rz.pop(q, None)
                else:
                    for q in g.qubits:
                        emit_rz(q)
                out.add(M(*g.qubits))
                continue

            raise NotImplementedError(f"Gate {type(g).__name__} is not supported at this lowering stage.")

        # Flush any remaining pending Z
        for q in range(out.nqubits):
            emit_rz(q)

        self.append_circuit_to_context(out)

        return out
