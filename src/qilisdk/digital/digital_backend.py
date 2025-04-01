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
from abc import ABC, abstractmethod
from enum import Enum

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.digital_result import DigitalResult


class DigitalSimulationMethod(str, Enum):
    """
    Enumeration of available simulation methods for the CUDA backend.
    """

    STATE_VECTOR = "state_vector"
    TENSOR_NETWORK = "tensor_network"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


class DigitalBackend(ABC):
    """
    Abstract base class for digital quantum circuit backends.

    This abstract class defines the interface for a digital backend capable of executing a
    quantum circuit. Subclasses must implement the execute method to run the circuit with a
    specified number of measurement shots and return a DigitalResult encapsulating the measurement
    outcomes.
    """

    def __init__(
        self, digital_simulation_method: DigitalSimulationMethod = DigitalSimulationMethod.STATE_VECTOR
    ) -> None:
        """
        Initialize the DigitalBackend.

        Args:
            simulation_method (DigitalSimulationMethod, optional): The simulation method to use.
                Options include STATE_VECTOR, TENSOR_NETWORK, or MATRIX_PRODUCT_STATE.
                Defaults to STATE_VECTOR.
        """
        self._digital_simulation_method = digital_simulation_method

    @property
    def digital_simulation_method(self) -> DigitalSimulationMethod:
        """
        Get the simulation method currently configured for the backend.

        Returns:
            SimulationMethod: The simulation method to be used for circuit execution.
        """
        return self._digital_simulation_method

    @digital_simulation_method.setter
    def digital_simulation_method(self, value: DigitalSimulationMethod) -> None:
        """
        Set the simulation method for the backend.

        Args:
            value (SimulationMethod): The simulation method to set. Options include
                STATE_VECTOR, TENSOR_NETWORK, or MATRIX_PRODUCT_STATE.
        """
        self._digital_simulation_method = value

    @abstractmethod
    def execute(self, circuit: Circuit, nshots: int = 1000) -> DigitalResult:
        """
        Execute the provided quantum circuit and return the measurement results.

        This method should run the given circuit for the specified number of measurement shots and
        produce a DigitalResult instance containing the raw measurement samples and any computed
        probabilities.

        Args:
            circuit (Circuit): The quantum circuit to be executed.
            nshots (int, optional): The number of measurement shots to perform. Defaults to 1000.

        Returns:
            DigitalResult: The result of executing the circuit, including measurement samples and probabilities.
        """
