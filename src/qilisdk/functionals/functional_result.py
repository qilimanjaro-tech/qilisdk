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

from qilisdk.core import QTensor
from qilisdk.core.qtensor import probabilities_from_state
from qilisdk.core.result import Result
from qilisdk.core.types import Number

from .functional import ReadoutMethod


class ReadoutResults(Result):

    def __init__(
        self,
        readout: ReadoutMethod,
        samples: dict[str, int] | None = None,
        probabilities: dict[str, float] | None = None,
        final_state: QTensor | None = None,
        expected_values: list[Number] | None = None,
    ) -> None:
        self._readout = readout
        self._final_state = final_state
        self._expected_values = expected_values
        self._samples: dict[str, int] = samples
        self._probabilities: dict[str, float] | None = probabilities

        # Ensure samples is not empty and is correct.
        if not self._probabilities:
            if self._final_state:
                self._probabilities = probabilities_from_state(self._final_state)
            if self._samples and self._readout.is_sample():
                bitstrings = list(self._samples.keys())
                nqubits = len(bitstrings[0])
                if not all(len(bitstring) == nqubits for bitstring in bitstrings):
                    raise ValueError("Not all bitstring keys have the same length.")

                # Calculate probabilities
                self._probabilities = {
                    bitstring: (
                        counts / self._readout.nshots if self._readout.nshots and self._readout.nshots > 0 else 0.0
                    )
                    for bitstring, counts in self._samples.items()
                }

    @property
    def readout(self) -> ReadoutMethod:
        return self._readout

    @property
    def samples(self) -> dict[str, int] | None:
        return self._samples

    @property
    def final_state(self) -> QTensor | None:
        return self._final_state

    @property
    def expected_values(self) -> list[Number] | None:
        return self._expected_values

    @property
    def probabilities(self) -> dict[str, float] | None:
        """
        Returns:
            dict[str,float]: a copy of the estimated probability distribution.
        """
        return self._probabilities

    def get_probability(self, bitstring: str) -> float:
        """
        Returns:
            float: the probability associated with ``bitstring`` (0.0 if unseen).

        Raises:
            ValueError: if the ReadoutResult object doesn't contain probabilities.
        """
        if self._probabilities:
            return self._probabilities.get(bitstring, 0.0)
        raise ValueError("This result object doesn't contain probabilities")

    def get_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """
        Args:
            n (int | None): Maximum number of items to return. Defaults to all outcomes.

        Returns:
            list[tuple[str, float]]: the ``n`` most probable bitstrings in descending probability order.

        Raises:
            ValueError: if the ReadoutResult object doesn't contain probabilities.
        """
        if self._probabilities:
            if n is None:
                n = len(self._probabilities)
            return heapq.nlargest(n, self._probabilities.items(), key=operator.itemgetter(1))
        raise ValueError("This result object doesn't contain probabilities.")

    def __repr__(self) -> str:
        if self.readout.is_sample():
            return f"Sampling Results: (\n  nshots={self._readout.nshots},\n  samples={pformat(self.samples)}\n)\n\n"
        if self.readout.is_expectation_values():
            return (
                "Expectation Value Results: (\n"
                + (f"\tnshots = {self._readout.nshots},\n" if self._readout.nshots > 0 else "")
                + f"\texpected_values={pformat(self._expected_values)},\n"
                + (f"\tfinal_state={pformat(self.final_state)}\n" if self.final_state is not None else "")
                + ")\n\n"
            )
        if self.readout.is_state_tomography():
            return "State Tomography Results: (\n" + (f"\tfinal_state={pformat(self.final_state)}\n") + ")\n\n"
        return ""

    __str__ = __repr__


class FunctionalResult(Result):
    def __init__(
        self, readout_results: list[ReadoutResults], intermediate_results: list[list[ReadoutMethod]] | None = None
    ) -> None:
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results

    @property
    def readout_results(self) -> list[ReadoutMethod]:
        return self._readout_results

    @property
    def intermediate_results(self) -> list[list[ReadoutMethod]] | None:
        return self._intermediate_results

    @property
    def samples(self) -> list[dict[str, int] | None]:
        result = []
        for readout in self._readout_results:
            result.append(readout.samples())
        return result

    @property
    def final_state(self) -> list[QTensor | None]:
        result = []
        for readout in self._readout_results:
            result.append(readout.final_state())
        return result

    @property
    def expected_values(self) -> list[list[float] | None]:
        result = []
        for readout in self._readout_results:
            result.append(readout.expected_values())
        return result

    @property
    def probabilities(self) -> list[dict[str, float] | None]:
        result = []
        for readout in self._readout_results:
            result.append(readout.probabilities())
        return result

    def get_probability(self, bitstring: str) -> list[float]:
        result = []
        for readout in self._readout_results:
            result.append(readout.get_probability(bitstring))
        return result

    def get_probabilities(self, n: int | None = None) -> list[list[tuple[str, float]]]:
        result = []
        for readout in self._readout_results:
            result.append(readout.get_probabilities(n))
        return result

    def __repr__(self) -> str:
        out = ""
        for readout in self._readout_results:
            out += str(readout)
        return out
