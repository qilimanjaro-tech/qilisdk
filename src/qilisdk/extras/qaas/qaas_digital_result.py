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
from itertools import product

from qilisdk.digital.digital_result import DigitalResult


class QaaSDigitalResult(DigitalResult):
    def __init__(self, nshots: int, samples: dict[str, int]) -> None:
        """
        Initialize a QililabDigitalResult instance.

        Args:
            nshots (int): The total number of measurement shots performed.
            samples (dict[str, int]): A dictionary mapping bitstring outcomes to their occurrence counts.
                All keys (bitstrings) must have the same length, which determines the number of qubits.

        Raises:
            ValueError: If the samples dictionary is empty or if not all bitstring keys have the same length.
        """
        super().__init__(nshots=nshots)
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

    @property
    def nqubits(self) -> int:
        return self._nqubits

    @property
    def nshots(self) -> int:
        return self._nshots

    @property
    def samples(self) -> dict[str, int]:
        return dict(self._samples)

    def get_probabilities(self) -> dict[str, float]:
        probabilities = {}

        # Generate all possible bitstrings of the inferred length.
        for bits in product("01", repeat=self.nqubits):
            bitstring = "".join(bits)
            count = self.samples.get(bitstring, 0)
            probabilities[bitstring] = count / self.nshots if self.nshots > 0 else 0.0

        return probabilities

    def get_probability(self, bitstring: str) -> float:
        if bitstring not in self._samples:
            return 0.0
        return self._samples[bitstring] / self.nshots
