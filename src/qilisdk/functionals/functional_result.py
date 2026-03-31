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
from qilisdk.readout.readout_result import (
    ReadoutCompositeResults,
    has_expectation_values,
    has_sampling,
    has_state_tomography,
)


class FunctionalResult(Result):
    """Container for the outputs produced by executing a functional on a backend.

    A ``FunctionalResult`` wraps one or more :class:`~qilisdk.readout.ReadoutResult`
    objects and exposes convenience accessors for samples, probabilities,
    final states, and expectation values.  When intermediate results are
    stored, the object is iterable over all time-steps (intermediates
    followed by the final readout).
    """

    def __init__(
        self,
        readout_results: ReadoutCompositeResults,
        intermediate_results: list[ReadoutCompositeResults] | None = None,
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
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results or []

    @property
    def readout_results(self) -> ReadoutCompositeResults:
        """Composite readout results from the final execution step."""
        return self._readout_results

    @property
    def intermediate_results(self) -> list[ReadoutCompositeResults]:
        """Intermediate readout results for each time-step, if stored."""
        return self._intermediate_results

    @property
    def samples(self) -> dict[str, int]:
        """
        Measurement samples from the final execution step.

        Raises:
            ValueError: If samples are queried but sample results were not specified in the experiment.
        """
        if has_sampling(self._readout_results):
            return self._readout_results.sampling.samples
        raise ValueError("Sampling Readout was not provided.")

    @property
    def probabilities(self) -> dict[str, float]:
        """
        Outcome probabilities from the final execution step.

        Raises:
            ValueError: if sampling or state tomography readouts are not specified.
        """
        if has_sampling(self._readout_results):
            return self._readout_results.sampling.probabilities
        if has_state_tomography(self._readout_results):
            return self._readout_results.state_tomography.probabilities
        raise ValueError("Can't compute probabilities if Sampling or State tomography readouts are not specified.")

    @property
    def state(self) -> QTensor:
        """
        Quantum state vector from the final execution step.

        Raises:
            ValueError: if state tomography readout is not specified.
        """
        if has_state_tomography(self._readout_results):
            return self._readout_results.state_tomography.state
        raise ValueError("Can't obtain the final state if State Tomography readout is not specified.")

    @property
    def expected_values(self) -> list[float]:
        """
        Expectation values from the final execution step.

        Raises:
            ValueError: if ExpectationReadout is not specified.
        """
        if has_expectation_values(self._readout_results):
            return self._readout_results.expectation_values.expected_values
        raise ValueError("Can't Compute Expectations because Expectation readout was not specified.")

    @property
    def intermediate_samples(self) -> list[dict[str, int]]:
        """Measurement samples for every time-step (intermediate + final).

        Returns:
            list[dict[str, int]]: Per-step sample dictionaries.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``SamplingReadout`` was provided.
        """
        if self._intermediate_results:
            results = []
            for res in self:
                if has_sampling(res):
                    results.append(res.sampling.samples)
            return results
        raise ValueError("Can't find intermediate samples because intermediate Results were not stored.")

    @property
    def intermediate_probabilities(self) -> list[dict[str, float]]:
        """Outcome probabilities for every time-step (intermediate + final).

        Returns:
            list[dict[str, float]]: Per-step probability dictionaries.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``SamplingReadout`` / ``StateTomographyReadout`` was provided.
        """
        if self._intermediate_results:
            results = []
            for res in self:
                if has_state_tomography(res):
                    results.append(res.state_tomography.probabilities)
                elif has_sampling(res):
                    results.append(res.sampling.probabilities)
            return results
        raise ValueError("Can't find intermediate probabilities because intermediate Results were not stored.")

    @property
    def intermediate_states(self) -> list[QTensor]:
        """Quantum state vectors for every time-step (intermediate + final).

        Returns:
            list[QTensor]: Per-step state vectors.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``StateTomographyReadout`` was provided.
        """
        if self._intermediate_results:
            results = []
            for res in self:
                if has_state_tomography(res):
                    results.append(res.state_tomography.state)
            return results
        raise ValueError("Can't find intermediate states because intermediate Results were not stored.")

    @property
    def intermediate_expected_values(self) -> list[list[float]]:
        """Expectation values for every time-step (intermediate + final).

        Returns:
            list[list[Number]]: Per-step expectation value lists.

        Raises:
            ValueError: If no intermediate results were stored or no
                ``ExpectationReadout`` was provided.
        """
        if self._intermediate_results:
            results = []
            for res in self:
                if has_expectation_values(res):
                    results.append(res.expectation_values.expected_values)
            return results
        raise ValueError("Can't find intermediate expected values because intermediate Results were not stored.")

    def has_state(self) -> bool:
        """Check whether the result contains a final quantum state.

        Returns:
            bool: ``True`` if a ``StateTomographyReadout`` result is present.
        """
        return has_state_tomography(self._readout_results)

    def has_samples(self) -> bool:
        """Check whether the result contains measurement samples.

        Returns:
            bool: ``True`` if a ``SamplingReadout`` result is present.
        """
        return has_sampling(self._readout_results)

    def has_probabilities(self) -> bool:
        """Check whether the result contains outcome probabilities.

        Returns:
            bool: ``True`` if a sampling or state-tomography readout result
                is present.
        """
        return has_state_tomography(self._readout_results) or has_sampling(self._readout_results)

    def has_expectation_values(self) -> bool:
        """Check whether the result contains expectation values.

        Returns:
            bool: ``True`` if an ``ExpectationReadout`` result is present.
        """
        return has_expectation_values(self._readout_results)

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
        LIMIT = 10
        out = "- Functional Results: [\n\n"
        out += str(self._readout_results)
        out += "]"
        if self._intermediate_results:
            out += "\n\n\n- Intermediate Results: [\n\n"
            for i, res in enumerate(self._intermediate_results[:LIMIT]):
                out += str(res)
                if i < LIMIT - 1:
                    out += "\n" + "-" * 20 + "\n"
            if len(self._intermediate_results) > LIMIT:
                out += "\n...\n\n"
            out += "]"
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
