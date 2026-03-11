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
from abc import abstractmethod
from pprint import pformat
from typing import Iterator

from qilisdk.core import QTensor
from qilisdk.core.qtensor import probabilities_from_state
from qilisdk.core.result import Result
from qilisdk.core.types import Number

from .functional import ReadoutBase, ReadoutMethod, SamplingReadout


class ReadoutResult(Result):
    @property
    @abstractmethod
    def readout(self) -> ReadoutBase: ...


class SamplingReadoutResults(ReadoutResult):
    def __init__(
        self, readout: SamplingReadout, samples: dict[str, int], probabilities: dict[str, float] | None = None
    ) -> None:
        self._readout = readout
        self._samples: dict[str, int] = samples

        if not probabilities:
            bitstrings = list(self._samples.keys())
            nqubits = len(bitstrings[0])
            if not all(len(bitstring) == nqubits for bitstring in bitstrings):
                raise ValueError("Not all bitstring keys have the same length.")

            # Calculate probabilities
            self._probabilities = {
                bitstring: (counts / self._readout.nshots if self._readout.nshots and self._readout.nshots > 0 else 0.0)
                for bitstring, counts in self._samples.items()
            }
        else:
            self._probabilities = probabilities

    @property
    def readout(self) -> ReadoutBase:
        return self._readout

    @property
    def samples(self) -> dict[str, int]:
        return self._samples

    @property
    def probabilities(self) -> dict[str, float]:
        """
        Returns:
            dict[str,float]: a copy of the estimated probability distribution.
        """
        return self._probabilities

    def get_probability(self, bitstring: str) -> float:
        """
        Returns:
            float: the probability associated with ``bitstring`` (0.0 if unseen).
        """
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

        return f"Sampling Results: (\n\tnshots={self._readout.nshots},\n\tsamples={pformat(self.samples)}\n)\n\n"

    __str__ = __repr__


class ExpectationReadoutResults(ReadoutResult):

    def __init__(
        self,
        readout: ReadoutMethod,
        expected_values: list[Number],
    ) -> None:
        self._readout = readout
        self._expected_values = expected_values

    @property
    def readout(self) -> ReadoutBase:
        return self._readout

    @property
    def expected_values(self) -> list[Number]:
        return self._expected_values

    def __repr__(self) -> str:
        return (
            "Expectation Value Results: (\n"
            + (f"\tnshots = {self._readout.nshots},\n" if self._readout.nshots and self._readout.nshots > 0 else "")
            + f"\texpected_values={pformat(self._expected_values)},\n"
            + ")\n\n"
        )

    __str__ = __repr__


class StateTomographyReadoutResults(ReadoutResult):

    def __init__(
        self,
        readout: ReadoutMethod,
        final_state: QTensor,
    ) -> None:
        self._readout = readout
        self._final_state = final_state
        self._probabilities = probabilities_from_state(self._final_state)

    @property
    def readout(self) -> ReadoutBase:
        return self._readout

    @property
    def final_state(self) -> QTensor:
        return self._final_state

    @property
    def probabilities(self) -> dict[str, float]:
        """
        Returns:
            dict[str,float]: a copy of the estimated probability distribution.
        """
        return self._probabilities

    def get_probability(self, bitstring: str) -> float:
        """
        Returns:
            float: the probability associated with ``bitstring`` (0.0 if unseen).
        """
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
        return "State Tomography Results: (\n" + (f"\tfinal_state={pformat(self.final_state)}\n") + ")\n\n"

    __str__ = __repr__


class FunctionalResult(Result):

    def __init__(
        self, readout_results: list[ReadoutResult], intermediate_results: list[list[ReadoutResult]] | None = None
    ) -> None:
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results

    @property
    def readout_results(self) -> list[ReadoutResult]:
        return self._readout_results

    @property
    def intermediate_results(self) -> list[list[ReadoutResult]] | None:
        return self._intermediate_results

    @property
    def samples(self) -> list[dict[str, int]]:
        samples_list: list[dict[str, int]] = []
        for result in self:
            if isinstance(result, SamplingReadoutResults):
                samples_list.append(result.samples)
        return samples_list

    @property
    def probabilities(self) -> list[dict[str, float]]:
        probabilities_list: list[dict[str, float]] = []
        for result in self:
            if isinstance(result, (StateTomographyReadoutResults, SamplingReadoutResults)):
                probabilities_list.append(result.probabilities)
        return probabilities_list

    @property
    def final_states(self) -> list[QTensor]:
        final_state_list: list[QTensor] = []
        for result in self:
            if isinstance(result, StateTomographyReadoutResults):
                final_state_list.append(result.final_state)
        return final_state_list

    @property
    def expected_values(self) -> list[list[Number]]:
        expected_values_list: list[list[Number]] = []
        for result in self:
            if isinstance(result, ExpectationReadoutResults):
                expected_values_list.append(result.expected_values)
        return expected_values_list

    def has_final_state(self) -> bool:
        return any(isinstance(res, StateTomographyReadoutResults) for res in self)

    def has_samples(self) -> bool:
        return any(isinstance(res, SamplingReadoutResults) for res in self)

    def has_probabilities(self) -> bool:
        return any(isinstance(res, (SamplingReadoutResults, StateTomographyReadoutResults)) for res in self)

    def has_expectation_values(self) -> bool:
        return any(isinstance(res, (ExpectationReadoutResults)) for res in self)

    def __len__(self) -> int:
        """
        Get the number of final readout results in the functional results.

        Returns:
            int: The number of final readout results in the functional results.
        """
        return len(self._readout_results)

    def __iter__(self) -> Iterator[ReadoutResult]:
        """
        Return an iterator over the readout results in the functional result object.

        Yields:
            Iterator[ReadoutResult]: The readout results in the functional result object.
        """
        yield from self._readout_results

    def __repr__(self) -> str:
        out = ""
        for readout in self._readout_results:
            out += str(readout)
        return out
