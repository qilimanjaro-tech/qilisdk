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

from qilisdk.yaml import yaml

from .exceptions import ParametersNotEqualError, QubitOutOfRangeError
from .gates import Gate


@yaml.register_class
class Circuit:
    def __init__(self, nqubits: int) -> None:
        """
        Initialize a Circuit instance with a specified number of qubits.

        Args:
            nqubits (int): The number of qubits in the circuit.
        """
        self._nqubits: int = nqubits
        self._gates: list[Gate] = []
        self._init_state: np.ndarray = np.zeros(nqubits)
        self._parameterized_gates: list[Gate] = []

    @property
    def nqubits(self) -> int:
        """
        Retrieve the number of qubits in the circuit.

        Returns:
            int: The total number of qubits.
        """
        return self._nqubits

    @property
    def nparameters(self) -> int:
        """
        Retrieve the total number of parameters required by all parameterized gates in the circuit.

        Returns:
            int: The total count of parameters from all parameterized gates.
        """
        return len([value for gate in self._gates if gate.is_parameterized for value in gate.parameter_values])

    @property
    def gates(self) -> list[Gate]:
        """
        Retrieve the list of gates in the circuit.

        Returns:
            list[Gate]: A list of gates that have been added to the circuit.
        """
        return self._gates

    def get_parameter_values(self) -> list[float]:
        """
        Retrieve the parameter values from all parameterized gates in the circuit.

        Returns:
            list[float]: A list of parameter values from each parameterized gate.
        """
        return [value for gate in self._gates if gate.is_parameterized for value in gate.parameter_values]

    def set_parameter_values(self, values: list[float]) -> None:
        """
        Set new parameter values for all parameterized gates in the circuit.

        Args:
            values (list[float]): A list containing new parameter values to assign to the parameterized gates.

        Raises:
            ParametersNotEqualError: If the number of provided values does not match the expected number of parameters.
        """
        if len(values) != self.nparameters:
            raise ParametersNotEqualError
        k = 0
        for i, gate in enumerate(self._parameterized_gates):
            gate.set_parameter_values(values[i + k : i + k + gate.nparameters])
            k += gate.nparameters - 1

    def add(self, gate: Gate) -> None:
        """
        Add a quantum gate to the circuit.

        Args:
            gate (Gate): The quantum gate to add to the circuit.

        Raises:
            QubitOutOfRangeError: If any qubit index used by the gate is not within the circuit's qubit range.
        """
        if any(qubit >= self.nqubits for qubit in gate.qubits):
            raise QubitOutOfRangeError
        if gate.is_parameterized:
            self._parameterized_gates.append(gate)
        self._gates.append(gate)
