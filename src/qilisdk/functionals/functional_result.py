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
from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from qilisdk.core.result import Result
from qilisdk.readout.readout_result import (
    E,
    ReadoutCompositeResults,
    S,
    T,
    has_expectation_values,
    has_sampling,
    has_state_tomography,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qilisdk.core import QTensor


class FunctionalResult(Result, Generic[S, E, T]):
    """Container for the outputs produced by executing a functional on a backend.

    The three type parameters encode which readout results are present:

    * ``S``: :class:`SamplingReadoutResult` or ``None``
    * ``E``: :class:`ExpectationReadoutResult` or ``None``
    * ``T``: :class:`StateTomographyReadoutResult` or ``None``

    **Typed access** — use the forwarding properties that return the readout
    result objects directly.  The type checker knows whether each is ``None``
    or populated based on the type parameters::

        result.sampling.samples  # ✅ when S = SamplingReadoutResult
        result.state_tomography.state  # ✅ when T = StateTomographyReadoutResult
        result.expectation  # None when E = None → .expected_values is a type error

    **Convenience shortcuts** — ``result.samples``, ``result.state``, etc.
    remain available for interactive / notebook use.  They raise
    ``ValueError`` at runtime when the corresponding readout is absent.
    """

    def __init__(
        self,
        readout_results: ReadoutCompositeResults[S, E, T],
        intermediate_results: list[ReadoutCompositeResults[S, E, T]] | None = None,
    ) -> None:
        """Initialise a functional result from readout outputs.

        Args:
            readout_results: Final readout results.
            intermediate_results: Optional per-step intermediate readout
                results. Defaults to ``None``.
        """
        self._readout_results = readout_results
        self._intermediate_results = intermediate_results or []

    # -- forwarding properties (typed safe path) --------------------------
    # These return the generic type parameters S, E, T directly, so the
    # type checker knows the exact type from the FunctionalResult's
    # parameterisation.  No descriptors or overloads needed.

    @property
    def sampling(self) -> S:
        """The sampling readout result, or ``None`` if not requested.

        When ``S = SamplingReadoutResult``, this returns the result object
        with typed access to ``.samples``, ``.probabilities``, etc.
        When ``S = None``, the type checker sees ``None``.
        """
        return self._readout_results.sampling

    @property
    def expectation(self) -> E:
        """The expectation-value readout result, or ``None`` if not requested.

        When ``E = ExpectationReadoutResult``, this returns the result object
        with typed access to ``.expected_values``.
        When ``E = None``, the type checker sees ``None``.
        """
        return self._readout_results.expectation_values

    @property
    def state_tomography(self) -> T:
        """The state-tomography readout result, or ``None`` if not requested.

        When ``T = StateTomographyReadoutResult``, this returns the result
        object with typed access to ``.state``, ``.probabilities``, etc.
        When ``T = None``, the type checker sees ``None``.
        """
        return self._readout_results.state_tomography

    @property
    def readout_results(self) -> ReadoutCompositeResults[S, E, T]:
        """Composite readout results from the final execution step."""
        return self._readout_results

    @property
    def intermediate_results(self) -> list[ReadoutCompositeResults[S, E, T]]:
        """Intermediate readout results for each time-step, if stored."""
        return self._intermediate_results

    # -- convenience shortcuts (runtime-checked, not type-narrowed) -------
    # These provide quick access for REPL / notebook use.  They always
    # return the concrete type or raise ValueError.  The typed safe path
    # above is preferred for production code.

    @property
    def samples(self) -> dict[str, int]:
        """Measurement samples from the final execution step.

        Raises:
            ValueError: If sampling readout was not provided.
        """
        if has_sampling(self._readout_results):
            return self._readout_results.sampling.samples
        raise ValueError("Sampling readout was not provided.")

    @property
    def probabilities(self) -> dict[str, float]:
        """Outcome probabilities from the final execution step.

        Raises:
            ValueError: If neither sampling nor state-tomography readout was provided.
        """
        if has_sampling(self._readout_results):
            return self._readout_results.sampling.probabilities
        if has_state_tomography(self._readout_results):
            return self._readout_results.state_tomography.probabilities
        raise ValueError("Can't compute probabilities if Sampling or State tomography readouts are not specified.")

    @property
    def state(self) -> QTensor:
        """Quantum state vector from the final execution step.

        Raises:
            ValueError: If state-tomography readout was not provided.
        """
        if has_state_tomography(self._readout_results):
            return self._readout_results.state_tomography.state
        raise ValueError("Can't obtain the final state if State Tomography readout is not specified.")

    @property
    def expected_values(self) -> list[float]:
        """Expectation values from the final execution step.

        Raises:
            ValueError: If expectation readout was not provided.
        """
        if has_expectation_values(self._readout_results):
            return self._readout_results.expectation_values.expected_values
        raise ValueError("Can't compute expectations because Expectation readout was not specified.")

    # -- intermediate accessors (runtime-checked) -------------------------

    @property
    def intermediate_samples(self) -> list[dict[str, int]]:
        """Measurement samples for every time-step (intermediate + final).

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

    # -- has_* helpers (useful for runtime checks) ------------------------

    def has_state(self) -> bool:
        """Check whether the result contains a final quantum state.
        Returns:
            bool: True if the Functional result has the state. False Otherwise.
        """
        return has_state_tomography(self._readout_results)

    def has_samples(self) -> bool:
        """Check whether the result contains measurement samples.
        Returns:
            bool: True if the Functional result has samples. False Otherwise.
        """
        return has_sampling(self._readout_results)

    def has_probabilities(self) -> bool:
        """Check whether the result contains outcome probabilities.
        Returns:
            bool: True if the Functional result has probabilities. False Otherwise.
        """
        return has_state_tomography(self._readout_results) or has_sampling(self._readout_results)

    def has_expectation_values(self) -> bool:
        """Check whether the result contains expectation values.
        Returns:
            bool: True if the Functional result has expectation  values. False Otherwise.
        """
        return has_expectation_values(self._readout_results)

    # -- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self._intermediate_results) + 1 if self._intermediate_results else 1

    def __iter__(self) -> Iterator[ReadoutCompositeResults[S, E, T]]:
        if self._intermediate_results:
            yield from self._intermediate_results
        yield self._readout_results

    def __getitem__(self, index: int) -> ReadoutCompositeResults[S, E, T]:
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for FunctionalResult of length {len(self)}")
        if index < len(self._intermediate_results):
            return self._intermediate_results[index]
        return self._readout_results

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
