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
from qilisdk.utils.classical_solvers import BruteForceSolver, SimulatedAnnealingSolver
from tests.timing.utils import best_time

# These take their thread count from here rather than from OMP_NUM_THREADS, so pin it to one too.
annealer = SimulatedAnnealingSolver(seed=1, num_threads=1)


def test_simulated_annealing_knapsack():
    qubo = Model.random_knapsack(40).to_qubo()
    assert best_time(lambda: annealer.solve(qubo)) < 0.5


def test_simulated_annealing_ising():
    qubo = Model.random_ising(100).to_qubo()
    assert best_time(lambda: annealer.solve(qubo)) < 1.2


def test_brute_force():
    model = Model.random_ising(12)
    assert best_time(lambda: BruteForceSolver().solve(model)) < 2.1
