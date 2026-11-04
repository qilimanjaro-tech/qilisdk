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
from qilisdk.analog.schedule import Schedule
from qilisdk.digital.ansatz import HardwareEfficientAnsatz, TrotterizedSchedule


def test_circuit_to_matrix(best_time):
    circuit = HardwareEfficientAnsatz(9, layers=5)
    assert best_time(circuit.to_matrix) < 2.0


def test_trotterized_schedule(best_time):
    nqubits = 10
    schedule = Schedule(
        hamiltonians={
            "driver": sum(X(i) for i in range(nqubits)),
            "problem": sum(Z(i) * Z(i + 1) for i in range(nqubits - 1)),
        },
        coefficients={"driver": {(0, 10): lambda t: 1 - t / 10}, "problem": {(0, 10): lambda t: t / 10}},
        dt=0.1,
    )
    assert best_time(lambda: TrotterizedSchedule(schedule, trotter_steps=2)) < 1.0
