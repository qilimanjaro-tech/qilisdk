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

from .exceptions import ParametersNotEqualError, QubitOutOfRangeError
from .gates import Gate


class Circuit:
    def __init__(self, nqubits: int) -> None:
        self._nqubits: int = nqubits
        self._gates: list[Gate] = []
        self._init_state: np.ndarray = np.zeros(nqubits)
        self._parameterized_gates: list[Gate] = []

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @property
    def nparameters(self) -> int:
        return len([value for gate in self._gates if gate.is_parameterized for value in gate.parameter_values])

    @property
    def gates(self) -> list[Gate]:
        return self._gates

    def get_parameter_values(self) -> list[float]:
        return [value for gate in self._gates if gate.is_parameterized for value in gate.parameter_values]

    def set_parameter_values(self, values: list[float]) -> None:
        """_summary_

        Args:
            values (list[float]): _description_

        Raises:
            ParametersNotEqualError: _description_
        """
        if len(values) != self.nparameters:
            raise ParametersNotEqualError
        k = 0
        for i, gate in enumerate(self._parameterized_gates):
            gate.set_parameter_values(values[i + k : i + k + gate.nparameters])
            k += gate.nparameters - 1

    def add(self, gate: Gate) -> None:
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError
        if gate.is_parameterized:
            self._parameterized_gates.append(gate)
        self._gates.append(gate)
