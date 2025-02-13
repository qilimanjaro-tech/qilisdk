# Copyright 2025 Qilimanjaro Quantum Tech
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

from .gates import Gate


class QubitOutOfRangeError(Exception): ...


class Circuit:
    def __init__(self, nqubits: int) -> None:
        self._nqubits = nqubits
        self._gates: list[Gate] = []
        self._parameterized_gates: list[Gate] = []
        self._init_state: np.ndarray = np.zeros(nqubits)

    @property
    def nqubits(self) -> int:
        return self._nqubits

    def add(self, gate: Gate) -> None:
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError
        if gate.is_parameterized():
            self._parameterized_gates.append(gate)
        self._gates.append(gate)
