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

import pytest

from qilisdk.digital import RY, RZ, SWAP, U3, Adjoint, Controlled, Exponential, Gate, H, M, X
from qilisdk.digital.circuit_transpiler_passes.retargeting_helpers import _remap_gate_qubits_inplace, _retarget_gate


@pytest.mark.parametrize(
    ("gate", "mapped_qubits"),
    [
        (H(0), (2,)),
        (U3(0, theta=0.4, phi=0.5, gamma=0.6), (2,)),
        (SWAP(0, 1), (2, 0)),
        (M(0, 1), (1, 3)),
        (Controlled(0, basic_gate=X(1)), (3, 2)),
        (Adjoint(RY(0, theta=0.3)), (1,)),
        (Exponential(RZ(0, phi=0.2)), (4,)),
    ],
)
def test_retarget_gate_returns_remapped_copy(gate: Gate, mapped_qubits: tuple[int, ...]) -> None:
    original_qubits = gate.qubits

    retargeted = _retarget_gate(gate, mapped_qubits)

    assert type(retargeted) is type(gate)
    assert retargeted.qubits == mapped_qubits
    assert gate.qubits == original_qubits


def test_remap_gate_qubits_rejects_unsupported_type() -> None:
    class UnsupportedGate(Gate):
        @property
        def name(self) -> str:
            return "Unsupported"

        @property
        def matrix(self):
            return None

        @property
        def target_qubits(self) -> tuple[int, ...]:
            return (0,)

    gate = UnsupportedGate()
    with pytest.raises(NotImplementedError, match="UnsupportedGate"):
        _remap_gate_qubits_inplace(gate, {0: 1})
