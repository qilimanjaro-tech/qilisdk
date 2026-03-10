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
from numpy import pi

from qilisdk.analog.hamiltonian import Hamiltonian, PauliI, PauliX, PauliY, PauliZ
from qilisdk.digital.gates import CNOT, RX, RZ, H
from qilisdk.utils.trotterization import trotter_evolution
from qilisdk.utils.trotterization.trotterization import _commuting_trotter_evolution


def test_trotter_evolution_commuting_parts_splits_coefficients():
    parts = [{(PauliZ(0),): 1.0, (PauliZ(1),): 2.0}]
    gates = list(_commuting_trotter_evolution(parts, time=0.5, trotter_steps=2))

    assert len(gates) == 4
    assert all(isinstance(gate, RZ) for gate in gates)
    phis = [gate.phi for gate in gates if isinstance(gate, RZ)]
    assert phis == pytest.approx([0.5, 1.0, 0.5, 1.0])


def test_trotter_evolution_handles_y_basis_change():
    parts = [{(PauliY(0),): 1.0}]
    gates = list(_commuting_trotter_evolution(parts, time=1.0, trotter_steps=1))

    assert len(gates) == 3
    assert isinstance(gates[0], RX)
    assert isinstance(gates[1], RZ)
    assert isinstance(gates[2], RX)
    assert gates[0].theta == pytest.approx(pi / 2)
    assert gates[1].phi == pytest.approx(2.0)
    assert gates[2].theta == pytest.approx(-pi / 2)


def test_trotter_evolution_ignores_identity_terms():
    parts = [{(PauliI(0),): 1.0}]
    gates = list(_commuting_trotter_evolution(parts, time=1.0, trotter_steps=3))
    assert gates == []


def test_trotter_evolution_accepts_hamiltonian_input():
    hamiltonian = Hamiltonian({(PauliX(0),): 1.0, (PauliZ(0),): 2.0})
    gates = list(trotter_evolution(hamiltonian, time=0.5, trotter_steps=2))

    assert len(gates) == 8
    assert [type(gate) for gate in gates] == [H, RZ, H, RZ, H, RZ, H, RZ]
    phis = [gate.phi for gate in gates if isinstance(gate, RZ)]
    assert phis == pytest.approx([0.5, 1.0, 0.5, 1.0])


def test_trotter_evolution_imaginary_time():
    hamiltonian = Hamiltonian({(PauliX(0),): 1.0, (PauliZ(0),): 2.0})
    gates = list(trotter_evolution(hamiltonian, time=0.5 + 0.5j, trotter_steps=2))

    assert len(gates) == 8
    assert [type(gate) for gate in gates] == [H, RZ, H, RZ, H, RZ, H, RZ]
    phis = [gate.phi for gate in gates if isinstance(gate, RZ)]
    assert phis == pytest.approx([0.5, 1.0, 0.5, 1.0])


def test_multi_qubit_trotter_evolution():
    hamiltonian = Hamiltonian(
        {
            (PauliX(0), PauliX(1)): 1.0,
        }
    )
    gates = list(trotter_evolution(hamiltonian, time=1.0, trotter_steps=1))

    assert len(gates) == 7
    expected_types = [
        H,
        H,
        CNOT,
        RZ,
        CNOT,
        H,
        H,
    ]
    assert [type(gate) for gate in gates] == expected_types
    phis = [gate.phi for gate in gates if isinstance(gate, RZ)]
    assert phis == pytest.approx([2.0])
