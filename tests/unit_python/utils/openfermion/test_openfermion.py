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

import numpy as np
import pytest

pytest.importorskip(
    "openfermion",
    reason="OpenFermion integration tests require the 'openfermion' optional dependency",
    exc_type=ImportError,
)

from openfermion.hamiltonians import jellium_model
from openfermion.transforms import fourier_transform, jordan_wigner
from openfermion.utils import Grid

from qilisdk.analog import I, X, Y, Z
from qilisdk.utils.openfermion import openfermion_to_qilisdk, qilisdk_to_openfermion


def test_translation_from_open_fermion_to_qilisdk_and_back():
    # Let's look at a very small model of jellium in 1D.
    grid = Grid(dimensions=1, length=3, scale=1.0)
    spinless = True

    # Get the momentum Hamiltonian.
    momentum_hamiltonian = jellium_model(grid, spinless)
    momentum_qubit_operator = jordan_wigner(momentum_hamiltonian)
    momentum_qubit_operator.compress()

    # Fourier transform the Hamiltonian to the position basis.
    position_hamiltonian = fourier_transform(momentum_hamiltonian, grid, spinless)
    position_qubit_operator = jordan_wigner(position_hamiltonian)
    position_qubit_operator.compress()

    qili_ham = openfermion_to_qilisdk(position_qubit_operator)

    of_ham = qilisdk_to_openfermion(qili_ham)

    assert of_ham == position_qubit_operator


def test_translation_from_qilisdk_to_openfermion_and_back():
    paulis = [X, Y, Z, I]
    nqubits = 3
    gen = np.random.Generator(np.random.PCG64(132))
    J = gen.random((nqubits, nqubits))
    qili_ham = sum(
        sum(gen.choice(paulis)(i) * gen.choice(paulis)(j) * J[i][j] for i in range(nqubits)) for j in range(nqubits)
    )

    of_ham = qilisdk_to_openfermion(qili_ham)

    qili_ham2 = openfermion_to_qilisdk(of_ham)

    assert qili_ham == qili_ham2
