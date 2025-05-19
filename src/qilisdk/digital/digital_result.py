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

from qilisdk.common.result import Result
from qilisdk.yaml import yaml


@yaml.register_class
class DigitalResult(Result):
    """
    Class representing the result of a quantum circuit measurement.

    DigitalResult encapsulates the outcome of a digital measurement performed on a quantum circuit.
    It includes the total number of measurement shots, the measurement samples and measurement probabilities.

    Attributes:
        nshots (int): The number of measurement shots performed in the experiment.
        nqubits (int): The number of qubits measured.
        samples (dict[str, int]): A dictionary where keys are bitstrings representing measurement outcomes and values are the number of times each outcome was observed.
    """

    def __init__(self, nshots: int, samples: dict[str, int]) -> None:
        """
        Initializes a DigitalResult instance.

        Args:
            nshots (int): The total number of measurement shots performed.
            samples (dict[str, int]): A dictionary mapping bitstring outcomes to their occurrence counts.
                All keys (bitstrings) must have the same length, which determines the number of qubits.

        Raises:
            ValueError: If the samples dictionary is empty or if not all bitstring keys have the same length.
        """
        self._nshots = nshots
        self._samples = samples

        # Ensure samples is not empty and is correct.
        if not samples:
            raise ValueError("The samples dictionary is empty.")
        bitstrings = list(samples.keys())
        nqubits = len(bitstrings[0])
        if not all(len(bitstring) == nqubits for bitstring in bitstrings):
            raise ValueError("Not all bitstring keys have the same length.")

        # Assign nqubits to attribute
        self._nqubits = nqubits

        # Calculate probabilities
        self._probabilities = {
            bitstring: counts / self._nshots if self._nshots > 0 else 0.0 for bitstring, counts in self._samples.items()
        }

    @property
    def nshots(self) -> int:
        """
        Gets the number of measurement shots.

        Returns:
            int: The total number of measurement shots performed.
        """
        return self._nshots

    @property
    def nqubits(self) -> int:
        """
        Gets the number of qubits involved in the measurement.

        Returns:
            int: The number of qubits measured.
        """
        return self._nqubits

    @property
    def samples(self) -> dict[str, int]:
        """
        Gets the raw measurement samples.

        Returns:
            dict[str, int]: A dictionary where keys are bitstrings representing measurement outcomes
            and values are the number of times each outcome was observed.
        """
        return dict(self._samples)

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
        return f"{class_name}(\n  nshots={self.nshots},\n  samples={pformat(self.samples)}\n)"
