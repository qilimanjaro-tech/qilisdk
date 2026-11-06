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

from qilisdk.digital.ansatz import HardwareEfficientAnsatz, TrotterizedSchedule
from tests.timing.utils import best_time, linear_ramp_schedule


def test_circuit_to_matrix():
    circuit = HardwareEfficientAnsatz(9, layers=5)
    assert best_time(circuit.to_matrix) < 1.6


def test_trotterized_schedule():
    schedule = linear_ramp_schedule(10, dt=0.1)
    assert best_time(lambda: TrotterizedSchedule(schedule, trotter_steps=2)) < 0.8
