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

from qilisdk.core import Model
from qilisdk.digital.ansatz import QAOA
from tests.timing.utils import best_time


def test_travelling_salesman_to_qubo():
    model = Model.random_travelling_salesman(6)
    assert best_time(model.to_qubo) < 0.14


def test_knapsack_to_qubo():
    model = Model.random_knapsack(60)
    assert best_time(model.to_qubo) < 1.0


def test_qubo_to_hamiltonian():
    qubo = Model.random_travelling_salesman(8).to_qubo()
    assert best_time(qubo.to_hamiltonian) < 0.4


def test_ising_model_to_hamiltonian():
    model = Model.random_ising(60)
    assert best_time(lambda: model.to_qubo().to_hamiltonian()) < 2.8


def test_qaoa_from_qubo():
    hamiltonian = Model.random_travelling_salesman(6).to_qubo().to_hamiltonian()
    assert best_time(lambda: QAOA(hamiltonian, layers=3)) < 0.6
