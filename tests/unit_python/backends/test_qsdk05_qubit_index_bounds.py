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

"""Regression tests for QSDK-05 / QSDK-06: out-of-range qubit indices must be
rejected at the trust boundary instead of producing an out-of-bounds
read/write in the matrix-free kernels or the measurement vector."""

import pytest

from qilisdk.analog import PauliX
from qilisdk.backends.qilisim import QiliSim
from qilisdk.digital import Circuit, H, X
from qilisdk.digital.exceptions import QubitOutOfRangeError
from qilisdk.functionals import DigitalPropagation
from qilisdk.readout import Readout


@pytest.mark.parametrize("bad_qubit", [-1, 2, 5])
def test_circuit_rejects_out_of_range_qubit(bad_qubit):
    """QSDK-05: the Python layer must reject both negative and too-large qubit
    indices (the negative case was previously unguarded)."""
    circuit = Circuit(2)
    with pytest.raises(QubitOutOfRangeError):
        circuit.add(X(bad_qubit))


def test_pauli_operator_rejects_negative_qubit():
    """QSDK-05: a Pauli operator must reject a negative qubit at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        PauliX(-1)


def test_simulator_rejects_out_of_range_qubit_from_unvalidated_gate():
    """QSDK-05 (keystone): even when the Python guard is bypassed, as happens on
    deserialization (ruamel reconstructs gates without re-running ``__init__``),
    the C++ parser must reject an out-of-range target before it reaches the
    matrix-free kernels, raising instead of reading out of bounds."""
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(X(1))
    backend = QiliSim()
    readout = Readout().with_sampling(nshots=100)

    # Sanity: the valid circuit runs.
    backend.execute(DigitalPropagation(circuit), readout)

    # Bypass validation the way deserialization does: an out-of-range target on
    # an already-added gate, with no __init__ re-check.
    gate = circuit.gates[1]
    inner = getattr(gate, "_basic_gate", gate)
    inner._target_qubits = (5,)

    with pytest.raises(ValueError, match="out of range"):
        backend.execute(DigitalPropagation(circuit), readout)
