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


class FunctionalResult(Result):

    def __init__(
        self, readout_results: list[ReadoutResult], intermediate_results: list[list[ReadoutResult]] | None = None
    ) -> None:

        if len({ro.__class__ for ro in readout_results}) != len(readout_results):
            raise ValueError(
                f"Each type of readout is allowed to be specified once.\nprovided a list with the following types {readout_results}"
            )
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results

    @property
    def readout_results(self) -> list[ReadoutResult]:
        return self._readout_results

    @property
    def intermediate_results(self) -> list[list[ReadoutResult]] | None:
        return self._intermediate_results

    @property
    def samples(self) -> dict[str, int]:
        for ro in self._readout_results:
            if isinstance(ro, SamplingReadoutResult):
                return ro.samples
        raise ValueError("Can't find samples in results, because no Sampling readout was provided.")

    @property
    def probabilities(self) -> dict[str, float]:
        for ro in self._readout_results:
            if isinstance(ro, SamplingReadoutResult):
                return ro.probabilities
            if isinstance(ro, StateTomographyReadoutResult) and ro.readout.compute_probabilities:
                return ro.probabilities  # ty:ignore[invalid-return-type]
        raise ValueError(
            "Can't find probabilities in results, because no Sampling/State Tomography readout was provided."
        )

    @property
    def final_state(self) -> QTensor:
        for ro in self._readout_results:
            if isinstance(ro, StateTomographyReadoutResult):
                return ro.final_state
        raise ValueError("Can't find final state in results, because no State Tomography readout was provided.")

    @property
    def expected_values(self) -> list[Number]:
        for ro in self._readout_results:
            if isinstance(ro, ExpectationReadoutResult):
                return ro.expected_values
        raise ValueError("Can't find expected values in results, because no Expectation readout was provided.")

    def has_final_state(self) -> bool:
        return any(isinstance(res, StateTomographyReadoutResult) for res in self)

    def has_samples(self) -> bool:
        return any(isinstance(res, SamplingReadoutResult) for res in self)

    def has_probabilities(self) -> bool:
        return any(
            isinstance(res, (SamplingReadoutResult))
            or (isinstance(res, StateTomographyReadoutResult) and res.readout.compute_probabilities)
            for res in self
        )

    def has_expectation_values(self) -> bool:
        return any(isinstance(res, (ExpectationReadoutResult)) for res in self)

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
