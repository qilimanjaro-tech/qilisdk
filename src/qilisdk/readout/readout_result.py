# Copyright 2026 Qilimanjaro Quantum Tech
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
"""Result containers returned by the various readout methods.

Each concrete :class:`ReadoutMethod` has a corresponding result class defined
here:

* :class:`SamplingReadoutResult` -- bitstring counts and probabilities.
* :class:`ExpectationReadoutResult` -- expectation values of observables.
* :class:`StateTomographyReadoutResult` -- full quantum state with optional
  probability extraction.

:class:`ReadoutCompositeResults` aggregates several result objects when
multiple readout methods are requested in a single execution.
"""

from __future__ import annotations

import heapq
import operator
from dataclasses import dataclass
from pprint import pformat
from typing import TYPE_CHECKING, Protocol, Self, TypeGuard, TypeVar

import numpy as np
from loguru import logger

from qilisdk.core import QTensor, expect_val
from qilisdk.core.result import Result
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core.types import Number

    from .readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout


NORMALIZATION_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_if_close(number: Number) -> Number:
    if isinstance(number, complex) and abs(number.imag) < get_settings().atol:
        return number.real
    return number


def _assert_real(number: Number) -> float:
    if isinstance(number, complex):
        if abs(number.imag) < get_settings().atol:
            return number.real
        raise ValueError("Complex Number encountered when expecting only real values to be present.")
    return number


# ---------------------------------------------------------------------------
# ReadoutResult base class
# ---------------------------------------------------------------------------

C = TypeVar("C", bound="ReadoutMethod")


class ReadoutResult(Result, Generic[C]):
    """Abstract base class for a single readout result.

    Every concrete subclass must expose a :attr:`readout` property that returns the :class:`~qilisdk.readout.ReadoutMethod`
    configuration used to produce the result.
    """



# ---------------------------------------------------------------------------
# Concrete result classes
# ---------------------------------------------------------------------------


class SamplingReadoutResult(ReadoutResult[SamplingReadout]):
    """Result produced by a :class:`~qilisdk.readout.SamplingReadout`.

    Holds bitstring measurement counts and the corresponding probability
    distribution.  The object can be constructed in two ways:

    1. From explicit ``samples`` (and optionally ``probabilities``).
    2. From a quantum ``state``, in which case samples are drawn stochastically according to the state amplitudes.

    Args:
        readout (SamplingReadout): The readout configuration that produced this result.
        samples (dict[str, int] | None): Mapping of bitstring to measurement count.  Mutually exclusive with ``state``.
        probabilities (dict[str, float] | None): Pre-computed probability distribution.  If omitted when ``samples``
            is given, it is derived from the counts.
        state (QTensor | None): Quantum state from which to derive samples and probabilities.  Mutually exclusive
            with ``samples``.

    Raises:
        ValueError: If neither ``samples``/``probabilities`` nor ``state``
            is provided.
    """

    @classmethod
    def from_samples(cls, samples: dict[str, int]) -> Self:
        if not samples:
            raise ValueError("can't initialize Sampling Results if samples are not provided.")

        nshots = sum(samples.values())
        bitstrings = list(samples.keys())
        nqubits = len(bitstrings[0])
        if not all(len(bitstring) == nqubits for bitstring in bitstrings):
            raise ValueError("Not all bitstring keys have the same length.")

        # Calculate probabilities
        probabilities: dict[str, int | float] = {
            bitstring: (counts / nshots if nshots and nshots > 0 else 0.0) for bitstring, counts in samples.items()
        }
        return cls(samples=samples, probabilities=probabilities)

    @classmethod
    def from_state(
        cls,
        sampling_readout: SamplingReadout,
        state: QTensor,
    ) -> Self:
        f_string = "{:0" + str(state.nqubits) + "b}"
        probabilities: dict[str, int | float] = {(f_string).format(i): p for i, p in enumerate(state.probabilities())}
        samples: dict[str, int] = _samples_from_probabilities(probabilities, nshots=sampling_readout.nshots)
        return cls(samples=samples, probabilities=probabilities)

    def __init__(self, samples: dict[str, int] | None, probabilities: dict[str, float] | None = None) -> None:
        if samples is None:
            raise ValueError("Can't construct the Sampling results if samples are not provided.")
        self._samples: dict[str, int] = samples or {}
        self._probabilities: dict[str, int | float] = probabilities or {}

    @property
    def samples(self) -> dict[str, int]:
        """dict[str, int]: Mapping of measured bitstring to count."""
        return self._samples

    @property
    def probabilities(self) -> dict[str, float]:
        """Estimated probability distribution over bitstrings.

        Returns:
            dict[str, float]: Mapping of bitstring to probability.
        """
        return self._probabilities

    def get_probability(self, bitstring: str) -> float:
        """Return the probability for a single bitstring.

        Args:
            bitstring (str): The bitstring to look up (e.g. ``"010"``).

        Returns:
            float: The probability associated with ``bitstring``, or ``0.0``
            if it was not observed.
        """
        return self._probabilities.get(bitstring, 0.0)

    def get_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """Return the most probable bitstrings in descending order.

        Args:
            n (int | None): Maximum number of entries to return.  ``None``
                (the default) returns all outcomes.

        Returns:
            list[tuple[str, float]]: Up to ``n`` ``(bitstring, probability)``
            pairs sorted by probability in descending order.
        """
        if n is None:
            n = len(self._probabilities)
        return heapq.nlargest(n, self._probabilities.items(), key=operator.itemgetter(1))

    def __repr__(self) -> str:

        return f"Sampling Results: (\n\tnshots={sum(self.samples.values())},\n\tsamples={pformat(self.samples)}\n)\n\n"

    __str__ = __repr__


class ExpectationReadoutResult(ReadoutResult[ExpectationReadout]):
    """Result produced by an :class:`~qilisdk.readout.ExpectationReadout`.

    Contains the computed expectation values for each observable specified in
    the readout configuration.  The object can be constructed in two ways:

    1. From pre-computed ``expected_values``.
    2. From a quantum ``state``, in which case the expectation values are
       derived via :func:`~qilisdk.core.expect_val`.

    Args:
        readout (ExpectationReadout): The readout configuration that
            produced this result.
        expected_values (list[Number] | None): Pre-computed expectation
            values, one per observable.  Mutually exclusive with ``state``.
        state (QTensor | None): Quantum state used to compute expectation
            values on-the-fly.  Mutually exclusive with
            ``expected_values``.

    Raises:
        ValueError: If neither ``expected_values`` nor ``state`` is
            provided.
    """

    @classmethod
    def from_expectations(cls, expected_values: list[float], nshots: int | None = None) -> Self:
        return cls(expected_values=expected_values, nshots=nshots)

    @classmethod
    def from_state(cls, expectation_readout: ExpectationReadout, state: QTensor) -> Self:
        expectation_readout.expand_observables(nqubits=state.nqubits)
        try:
            expected_values: list[int | float] = [
                _assert_real((expect_val(o, state))) for o in expectation_readout.qtensor_observables
            ]
        except ValueError:
            raise ValueError(
                "Encountered an imaginary expected value while computing the expectation values, try reducing the total tolerance or improving simulation precision."
            )
        return cls(expected_values=expected_values, nshots=expectation_readout.nshots)

    def __init__(self, expected_values: list[float], nshots: int | None = None) -> None:
        if expected_values is None:
            raise ValueError("Can't initialize Expectation Readout if the expected values are not provided.")
        self._expected_values: list[int | float] = expected_values
        self._nshots: int | None = nshots

    @property
    def expected_values(self) -> list[float]:
        """list[Number]: Expectation values, one per observable, in the same order as specified in the readout."""
        return self._expected_values

    def __repr__(self) -> str:
        return (
            "Expectation Value Results: (\n"
            + (f"\tnshots = {self._nshots},\n" if self._nshots and self._nshots > 0 else "")
            + f"\texpected_values={pformat(self._expected_values)},\n"
            + ")\n\n"
        )

    __str__ = __repr__


class StateTomographyReadoutResult(ReadoutResult[StateTomographyReadout]):
    """Result produced by a :class:`~qilisdk.readout.StateTomographyReadout`.

    Contains the full quantum state after execution and, optionally, the
    computational-basis probability distribution derived from it.

    Args:
        readout (StateTomographyReadout): The readout configuration that
            produced this result.
        state (QTensor): The reconstructed quantum state (ket or
            density matrix).
    """

    @classmethod
    def from_state(cls, state: QTensor) -> Self:
        return cls(state=state)

    def __init__(
        self,
        state: QTensor,
    ) -> None:
        self._state: QTensor = state

    @property
    def state(self) -> QTensor:
        """QTensor: The reconstructed quantum state (ket or density matrix)."""
        return self._state

    @property
    def probabilities(self) -> dict[str, float]:
        """Computational-basis probability distribution derived from the state.

        Returns:
            dict[str, float]: Mapping of bitstring to probability.
        """
        f_string = "{:0" + str(self.state.nqubits) + "b}"
        return {(f_string).format(i): p for i, p in enumerate(self.state.probabilities())}

    def get_probability(self, bitstring: str) -> float:
        """Return the probability for a single bitstring.

        Args:
            bitstring (str): The bitstring to look up (e.g. ``"010"``).

        Returns:
            float: The probability associated with ``bitstring``, or ``0.0``
            if it was not observed.
        """
        return self.probabilities.get(bitstring, 0.0)

    def get_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """Return the most probable bitstrings in descending order.

        Args:
            n (int | None): Maximum number of entries to return.  ``None``
                (the default) returns all outcomes.

        Returns:
            list[tuple[str, float]]: Up to ``n`` ``(bitstring, probability)``
            pairs sorted by probability in descending order.
        """
        probs = self.probabilities
        if n is None:
            n = len(probs)
        return heapq.nlargest(n, probs.items(), key=operator.itemgetter(1))

    def __repr__(self) -> str:
        return "State Tomography Results: (\n" + (f"\tfinal_state={pformat(self.state)}\n") + ")\n\n"

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Type variables for generic readout result containers
#
# Defined AFTER the concrete result classes so the constraints reference
# real types, not string literals.  TypeVar arguments are evaluated at
# runtime (they are function parameters, not annotations), so forward
# references via strings would silently pass as literal strings.
# ---------------------------------------------------------------------------

#: Type variable tracking whether sampling results are present.
S = TypeVar("S", SamplingReadoutResult, None)

#: Type variable tracking whether expectation-value results are present.
E = TypeVar("E", ExpectationReadoutResult, None)

#: Type variable tracking whether state-tomography results are present.
T = TypeVar("T", StateTomographyReadoutResult, None)


# ---------------------------------------------------------------------------
# ReadoutCompositeResults — generic aggregate container
# ---------------------------------------------------------------------------


class _HasSampling(Protocol):
    sampling: SamplingReadoutResult


def has_sampling(obj: ReadoutCompositeResults) -> TypeGuard[_HasSampling]:  # type: ignore[type-arg]
    """Return ``True`` if the composite contains a sampling result."""
    return obj.sampling is not None


class _HasExpectation(Protocol):
    expectation_values: ExpectationReadoutResult


def has_expectation_values(obj: ReadoutCompositeResults) -> TypeGuard[_HasExpectation]:  # type: ignore[type-arg]
    """Return ``True`` if the composite contains an expectation-value result."""
    return obj.expectation_values is not None


class _HasStateTomography(Protocol):
    state_tomography: StateTomographyReadoutResult


def has_state_tomography(obj: ReadoutCompositeResults) -> TypeGuard[_HasStateTomography]:  # type: ignore[type-arg]
    """Return ``True`` if the composite contains a state-tomography result."""
    return obj.state_tomography is not None


@dataclass(frozen=True)
class ReadoutCompositeResults(Result, Generic[S, E, T]):
    """Aggregated container for readout results from a single execution step.

    The three type parameters ``S``, ``E``, ``T`` encode at the *type level*
    which readout results are present:

    * ``S`` is :class:`SamplingReadoutResult` or ``None``
    * ``E`` is :class:`ExpectationReadoutResult` or ``None``
    * ``T`` is :class:`StateTomographyReadoutResult` or ``None``

    When the concrete type parameter is the result class (not ``None``), the
    corresponding field is guaranteed to be populated and the type checker
    can verify access without runtime guards.

    All three fields must be passed explicitly (no defaults) so that the
    type checker can always infer the type parameters correctly.
    """

    sampling: S
    expectation_values: E
    state_tomography: T

    def __repr__(self) -> str:
        out = ""
        if self.sampling:
            out += str(self.sampling)
        if self.expectation_values:
            out += str(self.expectation_values)
        if self.state_tomography:
            out += str(self.state_tomography)
        return out or f"{type(self).__name__}(empty)"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _samples_from_state(state: QTensor, nshots: int = 100, seed: int | None = None) -> dict[str, int]:
    f_string = "{:0" + str(state.nqubits) + "b}"
    probabilities: dict[str, int | float] = {(f_string).format(i): p for i, p in enumerate(state.probabilities())}
    return _samples_from_probabilities(probabilities=probabilities, nshots=nshots, seed=seed)


def _samples_from_probabilities(
    probabilities: dict[str, float], nshots: int = 100, seed: int | None = None
) -> dict[str, int]:
    states = np.array(list(probabilities.keys()))
    probs = np.array(list(probabilities.values()), dtype=np.float64)
    if not np.isclose(probs.sum(), 1):
        logger.warning("Renormalizing probabilities obtained as they don't sum up to 1.")
        probs /= probs.sum()

    rng = np.random.default_rng(seed)
    draws = rng.choice(len(states), size=nshots, p=probs)

    counts = np.bincount(draws, minlength=len(states))
    return {str(states[i]): int(counts[i]) for i in range(len(states)) if counts[i] > 0}
