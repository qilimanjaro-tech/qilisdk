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
import heapq
import operator
from pprint import pformat

import numpy as np

from qilisdk.common.qtensor import QTensor
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class StateTomographyResult(FunctionalResult):
    """
    Class representing the result of the state after a  quantum circuit execution.

    StateTomographyResult encapsulates the outcome of a quantum circuit.
    This includes the final state and the probabilities of the various states.
    """

    def __init__(self, state: QTensor) -> None:
        """
        Args:
            state (QTensor): the state at the end of the execution.

        Raises:
            ValueError: if the final state is not a ket of a density matrix
        """
        self._final_state = state

        # Assign nqubits to attribute
        self._nqubits = self._final_state.nqubits

        # Calculate probabilities

        probs = np.abs(state.dense) ** 2
        if state.is_ket():
            probs = np.abs(state.dense.flatten()) ** 2
        elif state.is_density_matrix():
            probs = np.real(np.diag(state.dense))
        else:
            raise ValueError("State must be a ket or density matrix")

        bitstrings = [format(i, f"0{self._nqubits}b") for i in range(len(state.dense))]

        self._probabilities = dict(zip(bitstrings, probs))

    @property
    def nqubits(self) -> int:
        """
        Gets the number of qubits involved in the measurement.

        Returns:
            int: The number of qubits measured.
        """
        return self._nqubits

    @property
    def final_state(self) -> QTensor:
        """
        Get the state vector at the end of the circuit.

        Returns:
            QTensor: The final state after evolution.
        """
        return self._final_state

    @property
    def probabilities(self) -> dict[str, float]:
        """
        Gets the probabilities for each measurement outcome.

        Returns:
            dict[str, float]: A dictionary mapping each bitstring outcome to its corresponding probability.
        """
        return dict(self._probabilities)

    def get_probability(self, bitstring: str) -> float:
        """
        Computes the probability of a specific measurement outcome.

        Args:
            bitstring (str): The bitstring representing the measurement outcome of interest.

        Returns:
            float: The probability of the specified bitstring occurring.
        """
        return self._probabilities.get(bitstring, 0.0)

    def get_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """
        Returns the n most probable bitstrings along with their probabilities.

        Parameters:
            n (int): The number of most probable bitstrings to return.

        Returns:
            list[tuple[str, float]]: A list of tuples (bitstring, probability) sorted in descending order by probability.
        """
        if n is None:
            n = len(self._probabilities)
        return heapq.nlargest(n, self._probabilities.items(), key=operator.itemgetter(1))

    def __repr__(self) -> str:
        """
        Returns a string representation of the DigitalResult instance for debugging purposes.

        The representation includes the class name, the number of measurement shots, and a formatted
        display of the measurement samples.

        Returns:
            str: A string representation of the DigitalResult instance for debugging.
        """
        class_name = self.__class__.__name__
        return f"{class_name}(\n  state=\n{self.final_state},\n  probabilities={pformat(self.probabilities)}\n)"
