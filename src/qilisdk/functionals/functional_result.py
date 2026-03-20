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
from qilisdk.readout import ReadoutResult
from qilisdk.readout.readout_result import ReadoutCompositeResults


class FunctionalResult(Result):
    """Container for the outputs produced by executing a functional on a backend.

    A ``FunctionalResult`` wraps one or more :class:`~qilisdk.readout.ReadoutResult`
    objects and exposes convenience accessors for samples, probabilities,
    final states, and expectation values.  When intermediate results are
    stored, the object is iterable over all time-steps (intermediates
    followed by the final readout).
    """

    def __init__(
        self, readout_results: list[ReadoutResult], intermediate_results: list[list[ReadoutResult]] | None = None
    ) -> None:
        """Initialise a functional result from readout outputs.

        Args:
            readout_results (list[ReadoutResult]): Final readout results.
                Each type of readout may appear at most once.
            intermediate_results (list[list[ReadoutResult]] | None): Optional
                per-step intermediate readout results. Defaults to None.

        Raises:
            ValueError: If ``readout_results`` contains duplicate readout types.
        """

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
        """Composite readout results from the final execution step."""
        return self._readout_results

    @property
    def intermediate_results(self) -> list[ReadoutCompositeResults]:
        """Intermediate readout results for each time-step, if stored."""
        return self._intermediate_results

    @property
    def final_samples(self) -> dict[str, int]:
        """Measurement samples from the final execution step."""
        return self._readout_results.samples

    @property
    def final_probabilities(self) -> dict[str, float]:
        """Outcome probabilities from the final execution step."""
        return self._readout_results.probabilities

    @property
    def final_state(self) -> QTensor:
        """Quantum state vector from the final execution step."""
        return self._readout_results.final_state

    @property
    def final_expected_values(self) -> list[Number]:
        """Expectation values from the final execution step."""
        return self._readout_results.expected_values

    @property
    def samples(self) -> list[dict[str, int]]:
        """Measurement samples for every time-step (intermediate + final).

        Returns:
            list[dict[str, int]]: Per-step sample dictionaries.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``SamplingReadout`` was provided.
        """
        if self._intermediate_results:
            if self.has_samples():
                results = []
                for res in self:
                    results.append(res.samples)
                return results
            raise ValueError("Can't find samples in results, because no Sampling readout was provided.")
        raise ValueError("Can't find intermediate samples because intermediate Results were not stored.")

    @property
    def probabilities(self) -> list[dict[str, float]]:
        """Outcome probabilities for every time-step (intermediate + final).

        Returns:
            list[dict[str, float]]: Per-step probability dictionaries.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``SamplingReadout`` / ``StateTomographyReadout`` was provided.
        """
        if self._intermediate_results:
            if self.has_probabilities():
                results = []
                for res in self:
                    results.append(res.probabilities)
                return results
            raise ValueError(
                "Can't find probabilities in results, because no Sampling/State Tomography readout was provided."
            )
        raise ValueError("Can't find intermediate probabilities because intermediate Results were not stored.")

    @property
    def states(self) -> list[QTensor]:
        """Quantum state vectors for every time-step (intermediate + final).

        Returns:
            list[QTensor]: Per-step state vectors.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``StateTomographyReadout`` was provided.
        """
        if self._intermediate_results:
            if self.has_final_state():
                results = []
                for res in self:
                    results.append(res.final_state)
                return results
            raise ValueError("Can't find final state in results, because no State Tomography readout was provided.")
        raise ValueError("Can't find intermediate states because intermediate Results were not stored.")

    @property
    def expected_values(self) -> list[list[Number]]:
        """Expectation values for every time-step (intermediate + final).

        Returns:
            list[list[Number]]: Per-step expectation value lists.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``ExpectationReadout`` was provided.
        """
        if self._intermediate_results:
            if self.has_expectation_values():
                results = []
                for res in self:
                    results.append(res.expected_values)
                return results
            raise ValueError("Can't find expected values in results, because no Expectation readout was provided.")
        raise ValueError("Can't find intermediate expected values because intermediate Results were not stored.")

    def has_final_state(self) -> bool:
        """Check whether the result contains a final quantum state.

        Returns:
            bool: ``True`` if a ``StateTomographyReadout`` result is present.
        """
        return self._readout_results.has_final_state()

    def has_samples(self) -> bool:
        """Check whether the result contains measurement samples.

        Returns:
            bool: ``True`` if a ``SamplingReadout`` result is present.
        """
        return self._readout_results.has_samples()

    def has_probabilities(self) -> bool:
        """Check whether the result contains outcome probabilities.

        Returns:
            bool: ``True`` if a sampling or state-tomography readout result
                is present.
        """
        return self._readout_results.has_probabilities()

    def has_expectation_values(self) -> bool:
        """Check whether the result contains expectation values.

        Returns:
            bool: ``True`` if an ``ExpectationReadout`` result is present.
        """
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

    def __getitem__(self, index: int) -> ReadoutCompositeResults:
        """Return the readout composite results at the given time-step index.

        Args:
            index (int): Zero-based index into the sequence of intermediate
                results followed by the final result.

        Returns:
            ReadoutCompositeResults: The composite readout result at
                ``index``.

        Raises:
            ValueError: If ``index`` exceeds the number of stored results.
        """
        if index > len(self):
            raise ValueError("Invalid Index")
        if index < len(self._intermediate_results):
            return self._intermediate_results[index]
        return self._readout_results
