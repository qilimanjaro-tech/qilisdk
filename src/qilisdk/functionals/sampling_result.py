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

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class SamplingResult(FunctionalResult):
    """Store shot counts and derived probabilities for a sampling experiment."""

    def __init__(self, nshots: int, samples: dict[str, int]) -> None:
        """
        Args:
            nshots (int): Total number of circuit evaluations.
            samples (dict[str, int]): Mapping from bitstring to observed counts.

        Raises:
            ValueError: If ``samples`` is empty or contains bitstrings with different length.
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
        """Total number of repetitions used to gather samples."""
        return self._nshots

    @property
    def nqubits(self) -> int:
        """Number of qubits inferred from the sample bitstrings."""
        return self._nqubits

    @property
    def samples(self) -> dict[str, int]:
        """Return a copy of the raw sample counts."""
        return dict(self._samples)

    @property
    def probabilities(self) -> dict[str, float]:
        """Return a copy of the estimated probability distribution."""
        return dict(self._probabilities)

    def get_probability(self, bitstring: str) -> float:
        """Return the probability associated with ``bitstring`` (0.0 if unseen)."""
        return self._probabilities.get(bitstring, 0.0)

    def get_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """
        Args:
            n (int | None): Maximum number of items to return. Defaults to all outcomes.

        Returns:
            list[tuple[str, float]]: the ``n`` most probable bitstrings in descending probability order.
        """
        if n is None:
            n = len(self._probabilities)
        return heapq.nlargest(n, self._probabilities.items(), key=operator.itemgetter(1))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(\n  nshots={self.nshots},\n  samples={pformat(self.samples)}\n)"
