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

from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.digital.circuit_transpiler import CircuitTranspiler


def test_transpile_without_topology(best_time):
    circuit = HardwareEfficientAnsatz(30, layers=20)
    transpiler = CircuitTranspiler.default()
    assert best_time(lambda: transpiler.transpile(circuit)) < 1.5


def test_transpile_with_routing(best_time):
    circuit = HardwareEfficientAnsatz(12, layers=6, connectivity="full")
    transpiler = CircuitTranspiler.default(topology=[(i, i + 1) for i in range(11)])
    assert best_time(lambda: transpiler.transpile(circuit)) < 3.0
