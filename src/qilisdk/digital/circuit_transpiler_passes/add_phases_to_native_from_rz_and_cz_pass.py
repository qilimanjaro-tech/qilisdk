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

from typing import Callable

from qilisdk.digital import Circuit
from qilisdk.digital.gates import CNOT, CZ, RZ, M
from qilisdk.digital.native_gates import Rmw

from .canonical_basis_to_native_set_pass import NativeSingleQubitGateBasis
from .circuit_transpiler_pass import CircuitTranspilerPass
from .decompose_to_canonical_basis_pass import TwoQubitGateBasis
from .numeric_helpers import _wrap_angle

PhaseCorrectionProvider = Callable[[int, int], tuple[float, float]]
"""Callable returning per-qubit phase corrections for a two-qubit native gate.

Given the (control, target) qubit pair of a two-qubit native gate, returns the
``(control_correction, target_correction)`` Z-frame increments to fold into
subsequent native single-qubit pulses.
"""


def _no_phase_corrections(_control: int, _target: int) -> tuple[float, float]:
    return (0.0, 0.0)


class AddPhasesToNativeFromRZAndCZPass(CircuitTranspilerPass):
    """Fold all Z-axis phases from RZs and two-qubit phase corrections into subsequent
    native single-qubit pulses as *virtual-Z* updates applied directly to the pulse phase.

    Key points:
      - RZ(φ) commuting forward adds +φ to the axis of every later XY pulse on that qubit.
      - The two-qubit native gate (e.g. CZ) may have per-qubit phase corrections supplied
        by ``phase_correction_provider``; both are accumulated.
      - The per-qubit Z-frame persists; it is NOT reset after a native single-qubit pulse.
      - Any residual Z-frame at the end is irrelevant to Z-basis measurement, so trailing
        RZs can be deleted.

    Sign convention:
      Because the VZ is realized by rotating the *pulse* (logical axis) rather than an NCO frame,
      the phase emitted for a native single-qubit pulse becomes ``phase_out = wrap(phase_in - shift[q])``.

    Phases are wrapped to ``[-π, π)`` for numerical hygiene.
    For background on persistent virtual-Z / frame updates, see https://arxiv.org/abs/1612.00858
    """

    def __init__(
        self,
        phase_correction_provider: PhaseCorrectionProvider | None = None,
        single_qubit_basis: NativeSingleQubitGateBasis = NativeSingleQubitGateBasis.Rmw,
        two_qubit_basis: TwoQubitGateBasis = TwoQubitGateBasis.CZ,
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
        self._phase_correction_provider: PhaseCorrectionProvider = (
            phase_correction_provider if phase_correction_provider is not None else _no_phase_corrections
        )
        self._single_qubit_basis = single_qubit_basis
        self._two_qubit_basis = two_qubit_basis

    @property
    def single_qubit_basis(self) -> NativeSingleQubitGateBasis:
        return self._single_qubit_basis

    @property
    def two_qubit_basis(self) -> TwoQubitGateBasis:
        return self._two_qubit_basis

    def run(self, circuit: Circuit) -> Circuit:
        nqubits = circuit.nqubits
        out_circuit = Circuit(nqubits)
        shift: dict[int, float] = dict.fromkeys(range(nqubits), 0.0)

        native_two_qubit_type: type = CZ if self._two_qubit_basis == TwoQubitGateBasis.CZ else CNOT

        for gate in circuit.gates:
            out_gate: M | Rmw | CZ | CNOT | None = None

            # Accumulate phase shifts from commuting RZ to the end, to discard them as VirtualZ.
            if isinstance(gate, RZ):
                qubit = gate.target_qubits[0]
                shift[qubit] = _wrap_angle(shift[qubit] + gate.phi)

            # Pass through the native two-qubit gate, accumulating its per-qubit phase corrections
            elif isinstance(gate, native_two_qubit_type):
                control_qubit, target_qubit = gate.control_qubits[0], gate.target_qubits[0]
                control_correction, target_correction = self._phase_correction_provider(control_qubit, target_qubit)
                shift[control_qubit] = _wrap_angle(shift[control_qubit] + control_correction)
                shift[target_qubit] = _wrap_angle(shift[target_qubit] + target_correction)
                out_gate = native_two_qubit_type(control_qubit, target_qubit)

            # Apply VZ by rotating the *pulse* axis: phase_out = phase_in - shift[q]
            elif isinstance(gate, Rmw):
                if self._single_qubit_basis != NativeSingleQubitGateBasis.Rmw:
                    raise ValueError(
                        f"Encountered {type(gate).__name__} but configured single_qubit_basis is "
                        f"{self._single_qubit_basis}."
                    )
                qubit = gate.qubits[0]
                out_gate = Rmw(qubit, theta=gate.theta, phase=_wrap_angle(gate.phase - shift[qubit]))

            # Measurement gates pass through unchanged
            elif isinstance(gate, M):
                out_gate = M(*gate.qubits)

            else:
                native_1q_name = self._single_qubit_basis.value
                native_2q_name = self._two_qubit_basis.value
                raise ValueError(
                    f"Unsupported gate {gate!r} (name={getattr(gate, 'name', type(gate).__name__)}) "
                    f"— supported: {RZ.__name__}, {native_1q_name}, {native_2q_name}, {M.__name__}"
                )

            if out_gate is not None:
                out_circuit.add(out_gate)

        # (Residual Z-frames are harmless; measurement in Z basis is invariant.)
        self.append_circuit_to_context(out_circuit)
        return out_circuit
