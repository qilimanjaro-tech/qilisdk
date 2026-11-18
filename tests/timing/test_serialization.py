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
from qilisdk.digital.ansatz import HardwareEfficientAnsatz
from qilisdk.utils.serialization import deserialize, serialize
from tests.timing.utils import best_time


def test_circuit_round_trip():
    circuit = HardwareEfficientAnsatz(10, layers=10)
    assert best_time(lambda: deserialize(serialize(circuit))) < 1.8


def test_model_round_trip():
    model = Model.random_travelling_salesman(6)
    assert best_time(lambda: deserialize(serialize(model))) < 0.6
