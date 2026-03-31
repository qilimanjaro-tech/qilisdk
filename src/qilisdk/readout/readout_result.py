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
from abc import abstractmethod
from dataclasses import dataclass
from pprint import pformat
from typing import TYPE_CHECKING, Generic, Protocol, Self, TypeGuard, TypeVar

import numpy as np
from loguru import logger

from qilisdk.core import QTensor, expect_val
from qilisdk.core.result import Result
from qilisdk.settings import get_settings

from .readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout

if TYPE_CHECKING:
    from qilisdk.core.types import Number


NORMALIZATION_TOLERANCE = 1e-8


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


class _HasSampling(Protocol):
    sampling: SamplingReadoutResult


def has_sampling(obj: ReadoutCompositeResults) -> TypeGuard[_HasSampling]:
    return obj.sampling is not None


class _HasExpectation(Protocol):
    expectation_values: ExpectationReadoutResult


def has_expectation_values(obj: ReadoutCompositeResults) -> TypeGuard[_HasExpectation]:
    return obj.expectation_values is not None


class _HasStateTomography(Protocol):
    state_tomography: StateTomographyReadoutResult


def has_state_tomography(obj: ReadoutCompositeResults) -> TypeGuard[_HasStateTomography]:
    return obj.state_tomography is not None


@dataclass(frozen=True)
class ReadoutCompositeResults(Result):
    """Aggregated container for multiple :class:`ReadoutResult` objects.

    When a backend execution is configured with more than one readout method,
    the results are collected into a single :class:`ReadoutCompositeResults`
    instance.  Convenience properties (:attr:`samples`, :attr:`probabilities`,
    :attr:`state`, :attr:`expected_values`) provide direct access to the
    first matching result of each kind.

    The container is iterable and supports ``len()``.

    Args:
        readout_results (list[ReadoutResult]): Individual readout results
            from the execution.
    """

    sampling: SamplingReadoutResult | None = None
    expectation_values: ExpectationReadoutResult | None = None
    state_tomography: StateTomographyReadoutResult | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ReadoutCompositeResults:
        if "sampling" in data and not isinstance(data.get("sampling"), SamplingReadoutResult):
            raise TypeError("sampling must be SamplingReadoutResult")
        if "expectation_values" in data and not isinstance(data.get("expectation_values"), ExpectationReadoutResult):
            raise TypeError("expectation_values must be ExpectationReadoutResult")
        if "state_tomography" in data and not isinstance(data.get("state_tomography"), StateTomographyReadoutResult):
            raise TypeError("state_tomography must be ExpectationReadoutResult")
        return cls(
            sampling=data.get("sampling"),
            expectation_values=data.get("expectation_values"),
            state_tomography=data.get("state_tomography"),
        )

    @classmethod
    def from_list(cls, list_data: list) -> ReadoutCompositeResults:
        data = {}
        for element in list_data:
            if isinstance(element, SamplingReadoutResult):
                data["sampling"] = element
            if isinstance(element, ExpectationReadoutResult):
                data["expectation_values"] = element
            if isinstance(element, StateTomographyReadoutResult):
                data["state_tomography"] = element
        return cls(
            sampling=data.get("sampling"),
            expectation_values=data.get("expectation_values"),
            state_tomography=data.get("state_tomography"),
        )

    def __repr__(self) -> str:
        out = ""
        if self.sampling:
            out += str(self.sampling)
        if self.expectation_values:
            out += str(self.expectation_values)
        if self.state_tomography:
            out += str(self.state_tomography)
        return out


C = TypeVar("C", bound="ReadoutMethod")


class ReadoutResult(Result, Generic[C]):
    """Abstract base class for a single readout result.

    Every concrete subclass must expose a :attr:`readout` property that
    returns the :class:`~qilisdk.readout.ReadoutMethod` configuration used
    to produce the result.
    """

    @property
    @abstractmethod
    def readout(self) -> C:
        """ReadoutMethod: The readout configuration that produced this result."""
        ...


class SamplingReadoutResult(ReadoutResult[SamplingReadout]):
    """Result produced by a :class:`~qilisdk.readout.SamplingReadout`.

    Holds bitstring measurement counts and the corresponding probability
    distribution.  The object can be constructed in two ways:

    1. From explicit ``samples`` (and optionally ``probabilities``).
    2. From a quantum ``state``, in which case samples are drawn
       stochastically according to the state amplitudes.

    Args:
        readout (SamplingReadout): The readout configuration that produced
            this result.
        samples (dict[str, int] | None): Mapping of bitstring to
            measurement count.  Mutually exclusive with ``state``.
        probabilities (dict[str, float] | None): Pre-computed probability
            distribution.  If omitted when ``samples`` is given, it is
            derived from the counts.
        state (QTensor | None): Quantum state from which to derive
            samples and probabilities.  Mutually exclusive with
            ``samples``.

    Raises:
        ValueError: If neither ``samples``/``probabilities`` nor ``state``
            is provided.
    """

    @classmethod
    def from_samples(
        cls, readout: SamplingReadout, samples: dict[str, int], probabilities: dict[str, float] | None = None
    ) -> Self:
        if not samples and not probabilities:
            raise ValueError("can't initialize Sampling Results if both samples and probabilities are not provided.")
        if not probabilities:
            bitstrings = list(samples.keys())
            nqubits = len(bitstrings[0])
            if not all(len(bitstring) == nqubits for bitstring in bitstrings):
                raise ValueError("Not all bitstring keys have the same length.")

            # Calculate probabilities
            probabilities: dict[str, int | float] = {
                bitstring: (counts / readout.nshots if readout.nshots and readout.nshots > 0 else 0.0)
                for bitstring, counts in samples.items()
            }
        return cls(readout=readout, samples=samples, probabilities=probabilities)

    @classmethod
    def from_state(cls, readout: SamplingReadout, state: QTensor) -> Self:
        f_string = "{:0" + str(state.nqubits) + "b}"
        probabilities: dict[str, int | float] = {(f_string).format(i): p for i, p in enumerate(state.probabilities())}
        samples: dict[str, int] = _samples_from_probabilities(probabilities, nshots=readout.nshots)
        return cls(readout=readout, samples=samples, probabilities=probabilities)

    def __init__(
        self, readout: SamplingReadout, *, samples: dict[str, int] | None, probabilities: dict[str, float] | None = None
    ) -> None:
        if not isinstance(readout, SamplingReadout):
            raise TypeError(f"Expected SamplingReadout, got {type(readout).__name__}")
        if samples is None:
            raise ValueError("Can't construct the Sampling results if samples are not provided.")
        self._samples: dict[str, int] = samples or {}
        self._probabilities: dict[str, int | float] = probabilities or {}
        self._readout: SamplingReadout = readout

    @property
    def readout(self) -> SamplingReadout:
        """SamplingReadout: The readout configuration that produced this result."""
        return self._readout

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

        return f"Sampling Results: (\n\tnshots={self._readout.nshots},\n\tsamples={pformat(self.samples)}\n)\n\n"

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
    def from_expectations(cls, readout: ExpectationReadout, expected_values: list[float]) -> Self:
        return cls(readout=readout, expected_values=expected_values)

    @classmethod
    def from_state(cls, readout: ExpectationReadout, state: QTensor) -> Self:
        readout.expand_observables(nqubits=state.nqubits)
        try:
            expected_values: list[int | float] = [
                _assert_real((expect_val(o, state))) for o in readout.qtensor_observables
            ]
        except ValueError:
            raise ValueError(
                "Encountered an imaginary expected value while computing the expectation values, try reducing the total tolerance or improving simulation precision."
            )
        return cls(readout=readout, expected_values=expected_values)

    def __init__(self, readout: ExpectationReadout, expected_values: list[float]) -> None:
        if not isinstance(readout, ExpectationReadout):
            raise TypeError(f"Expected ExpectationReadout, got {type(readout).__name__}")
        if expected_values is None:
            raise ValueError("Can't initialize Expectation Readout if the expected values are not provided.")
        self._expected_values: list[int | float] = expected_values
        self._readout: ExpectationReadout = readout

    @property
    def readout(self) -> ExpectationReadout:
        """ExpectationReadout: The readout configuration that produced this result."""
        return self._readout

    @property
    def expected_values(self) -> list[float]:
        """list[Number]: Expectation values, one per observable, in the same order as specified in the readout."""
        return self._expected_values

    def __repr__(self) -> str:
        return (
            "Expectation Value Results: (\n"
            + (f"\tnshots = {self._readout.nshots},\n" if self._readout.nshots and self._readout.nshots > 0 else "")
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

    def __init__(
        self,
        readout: StateTomographyReadout,
        state: QTensor,
    ) -> None:
        if not isinstance(readout, StateTomographyReadout):
            raise TypeError(f"Expected StateTomographyReadout, got {type(readout).__name__}")
        self._readout: StateTomographyReadout = readout
        self._state: QTensor = state

    @property
    def readout(self) -> StateTomographyReadout:
        """StateTomographyReadout: The readout configuration that produced this result."""
        return self._readout

    @property
    def state(self) -> QTensor:
        """QTensor: The reconstructed quantum state (ket or density matrix)."""
        return self._state

    @property
    def probabilities(self) -> dict[str, float]:
        """Computational-basis probability distribution derived from the state.

        Returns:
            dict[str, float] | None: Mapping of bitstring to probability, or
            ``None`` if ``compute_probabilities`` was disabled in the readout.
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

        Raises:
            ValueError: If ``compute_probabilities`` was set to ``False``
                in the readout configuration.
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

        Raises:
            ValueError: If ``compute_probabilities`` was set to ``False``
                in the readout configuration.
        """
        probs = self.probabilities
        if n is None:
            n = len(probs)
        return heapq.nlargest(n, probs.items(), key=operator.itemgetter(1))

    def __repr__(self) -> str:
        return "State Tomography Results: (\n" + (f"\tfinal_state={pformat(self.state)}\n") + ")\n\n"

    __str__ = __repr__


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
