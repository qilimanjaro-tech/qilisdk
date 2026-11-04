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

from qilisdk.analog.hamiltonian import X, Z
from tests.timing.utils import best_time


def _all_to_all_ising(nqubits: int):
    couplings = sum(Z(i) * Z(j) for i in range(nqubits) for j in range(i + 1, nqubits))
    return couplings + sum(X(i) for i in range(nqubits))


def test_build_hamiltonian():
    assert best_time(lambda: _all_to_all_ising(40)) < 0.8


def test_hamiltonian_product():
    hamiltonian = _all_to_all_ising(20)
    assert best_time(lambda: hamiltonian * hamiltonian) < 1.4


def test_hamiltonian_to_matrix():
    hamiltonian = _all_to_all_ising(14)
    assert best_time(hamiltonian.to_matrix) < 1.8


def test_hamiltonian_exponential():
    tensor = _all_to_all_ising(9).to_qtensor()
    assert best_time(tensor.expm) < 4.0
