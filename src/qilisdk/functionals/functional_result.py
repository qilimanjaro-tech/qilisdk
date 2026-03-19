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
from collections.abc import Iterator

from qilisdk.core import QTensor
from qilisdk.core.result import Result
from qilisdk.core.types import Number
from qilisdk.readout import ExpectationReadoutResult, ReadoutResult, SamplingReadoutResult, StateTomographyReadoutResult
from qilisdk.readout.readout_result import ReadoutCompositeResults


class FunctionalResult(Result):

    def __init__(
        self, readout_results: list[ReadoutResult], intermediate_results: list[list[ReadoutResult]] | None = None
    ) -> None:

        if len({ro.__class__ for ro in readout_results}) != len(readout_results):
            raise ValueError(
                f"Each type of readout is allowed to be specified once.\nprovided a list with the following types {readout_results}"
            )
        self._readout_results = ReadoutCompositeResults(readout_results)
        self._intermediate_results = (
            [ReadoutCompositeResults(res) for res in intermediate_results] if intermediate_results else []
        )

    @property
    def readout_results(self) -> ReadoutCompositeResults:
        return self._readout_results

    @property
    def intermediate_results(self) -> list[ReadoutCompositeResults]:
        return self._intermediate_results

    @property
    def final_samples(self) -> dict[str, int]:
        return self._readout_results.samples

    @property
    def final_probabilities(self) -> dict[str, float]:
        return self._readout_results.probabilities

    @property
    def final_state(self) -> QTensor:
        return self._readout_results.final_state

    @property
    def final_expected_values(self) -> list[Number]:
        return self._readout_results.expected_values

    @property
    def samples(self) -> list[dict[str, int]]:
        if self._intermediate_results:
            if self.has_samples():
                results = []
                for res in self:
                    results.append(res.samples)
                return results
            raise ValueError("Can't find samples in results, because no Sampling readout was provided.")
        raise ValueError("Intermediate Results were not stored.")

    @property
    def probabilities(self) -> list[dict[str, float]]:
        if self._intermediate_results:
            if self.has_probabilities():
                results = []
                for res in self:
                    results.append(res.probabilities)
                return results
            raise ValueError(
                "Can't find probabilities in results, because no Sampling/State Tomography readout was provided."
            )
        raise ValueError("Intermediate Results were not stored.")

    @property
    def states(self) -> list[QTensor]:
        if self._intermediate_results:
            if self.has_final_state():
                results = []
                for res in self:
                    results.append(res.final_state)
                return results
            raise ValueError("Can't find final state in results, because no State Tomography readout was provided.")
        raise ValueError("Intermediate Results were not stored.")

    @property
    def expected_values(self) -> list[list[Number]]:
        if self._intermediate_results:
            if self.has_expectation_values():
                results = []
                for res in self:
                    results.append(res.expected_values)
                return results
            raise ValueError("Can't find expected values in results, because no Expectation readout was provided.")
        raise ValueError("Intermediate Results were not stored.")

    def has_final_state(self) -> bool:
        return self._readout_results.has_final_state()

    def has_samples(self) -> bool:
        return self._readout_results.has_samples()

    def has_probabilities(self) -> bool:
        return self._readout_results.has_probabilities()

    def has_expectation_values(self) -> bool:
        return self._readout_results.has_expectation_values()

    def __len__(self) -> int:
        """
        Get the number of final readout results in the functional results.

        Returns:
            int: The number of final readout results in the functional results.
        """
        return len(self._intermediate_results) + 1 if self._intermediate_results else 1

    def __iter__(self) -> Iterator[ReadoutCompositeResults]:
        """
        Return an iterator over the readout results in the functional result object.

        Yields:
            Iterator[ReadoutResult]: The readout results in the functional result object.
        """
        if self._intermediate_results:
            yield from self._intermediate_results
        yield self._readout_results

    def __repr__(self) -> str:
        out = "Functional Results: (\n"
        for readout in self._readout_results:
            out += str(readout)
        out += "\n)"
        return out
