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
from __future__ import annotations

import heapq
import operator
from abc import abstractmethod
from pprint import pformat
from typing import TYPE_CHECKING, Iterator, overload

import numpy as np

from qilisdk.core import QTensor, expect_val
from qilisdk.core.result import Result
from qilisdk.settings import get_settings

if TYPE_CHECKING:
    from qilisdk.core.types import Number

    from .readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout


NORMALIZATION_TOLERANCE = 1e-8


def _real_if_close(number: Number) -> Number:
    if isinstance(number, complex) and number.imag < get_settings().atol:
        return number.real
    return number


class ReadoutCompositeResults(Result):
    def __init__(self, readout_results: list[ReadoutResult]) -> None:
        self._readout_results = readout_results

    @property
    def readout_results(self) -> list[ReadoutResult]:
        return self._readout_results

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


class ReadoutResult(Result):
    @property
    @abstractmethod
    def readout(self) -> ReadoutMethod: ...


class SamplingReadoutResult(ReadoutResult):
    @overload
    def __init__(
        self,
        readout: SamplingReadout,
        *,
        samples: dict[str, int] | None,
        probabilities: dict[str, float] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        readout: SamplingReadout,
        *,
        state: QTensor | None = None,
    ) -> None: ...

    def __init__(
        self,
        readout: SamplingReadout,
        *,
        samples: dict[str, int] | None = None,
        probabilities: dict[str, float] | None = None,
        state: QTensor | None = None,
    ) -> None:

        if samples is not None or probabilities is not None:
            self._init_from_samples(readout, samples, probabilities)
        elif state is not None:
            self._init_from_state(readout, state)
        else:
            raise ValueError("Can't construct the Sampling results if samples and state are not provided.")

    def _init_from_samples(
        self, readout: SamplingReadout, samples: dict[str, int] | None, probabilities: dict[str, float] | None = None
    ) -> None:
        self._readout: SamplingReadout = readout
        if samples:
            self._samples: dict[str, int] = samples
        elif probabilities:
            self._samples = _samples_from_probabilities(probabilities)
        else:
            raise ValueError("can't initialize Sampling Results if both samples and probabilities are not provided.")

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

    def _init_from_state(self, readout: SamplingReadout, state: QTensor) -> None:
        self._readout: SamplingReadout = readout
        self._probabilities = _probabilities_from_state(state)
        self._samples: dict[str, int] = _samples_from_probabilities(self._probabilities, nshots=readout.nshots)

    @property
    def readout(self) -> SamplingReadout:
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


class ExpectationReadoutResult(ReadoutResult):
    @overload
    def __init__(self, readout: ExpectationReadout, *, state: QTensor) -> None: ...

    @overload
    def __init__(
        self,
        readout: ExpectationReadout,
        *,
        expected_values: list[Number],
    ) -> None: ...

    def __init__(
        self, readout: ExpectationReadout, *, expected_values: list[Number] | None = None, state: QTensor | None = None
    ) -> None:
        if expected_values is not None:
            self._init_from_expected_values(readout, expected_values)
        elif state is not None:
            self._init_from_state(readout, state)
        else:
            raise ValueError(
                "Can't initialize Expectation Readout if the expected values and the state are not provided."
            )

    def _init_from_state(self, readout: ExpectationReadout, state: QTensor) -> None:
        self._readout = readout
        readout.scale_observables(nqubits=state.nqubits)
        self._expected_values = [_real_if_close((expect_val(o, state))) for o in readout.qtensor_observables]

    def _init_from_expected_values(
        self,
        readout: ExpectationReadout,
        expected_values: list[Number],
    ) -> None:
        self._readout: ExpectationReadout = readout
        self._expected_values = expected_values

    @property
    def readout(self) -> ExpectationReadout:
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


class StateTomographyReadoutResult(ReadoutResult):
    def __init__(
        self,
        readout: StateTomographyReadout,
        final_state: QTensor,
    ) -> None:
        self._readout: StateTomographyReadout = readout
        self._final_state: QTensor = final_state
        self._probabilities: dict[str, float] | None = (
            _probabilities_from_state(self._final_state) if self._readout.compute_probabilities else None
        )

    @property
    def readout(self) -> StateTomographyReadout:
        return self._readout

    @property
    def final_state(self) -> QTensor:
        return self._final_state

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
            ValueError: if the readout set `compute_probabilities` to False,
        """
        if not self._probabilities:
            raise ValueError(
                "Probabilities where not computed because `compute_probabilities` was set to False in the readout."
            )
        return self._probabilities.get(bitstring, 0.0)

    def get_probabilities(self, n: int | None = None) -> list[tuple[str, float]]:
        """
        Args:
            n (int | None): Maximum number of items to return. Defaults to all outcomes.

        Returns:
            list[tuple[str, float]]: the ``n`` most probable bitstrings in descending probability order.

        Raises:
            ValueError: if the readout set `compute_probabilities` to False,
        """
        if not self._probabilities:
            raise ValueError(
                "Probabilities where not computed because `compute_probabilities` was set to False in the readout."
            )
        if n is None:
            n = len(self._probabilities)
        return heapq.nlargest(n, self._probabilities.items(), key=operator.itemgetter(1))

    def __repr__(self) -> str:
        return "State Tomography Results: (\n" + (f"\tfinal_state={pformat(self.final_state)}\n") + ")\n\n"

    __str__ = __repr__


def _probabilities_from_state(state: QTensor, *, check_normalization: bool = False) -> dict[str, float]:
    """
    Return computational-basis probabilities for a QTensor state.

    Returns:
        dict[str, float]: a dictionary mapping every possible outcome to the probability of it appearing.

    Raises:
        TypeError: if the state provided is not a valid state.
        ValueError: if the probabilities are not normalized and check_normalization is set to True.
    """
    atol = get_settings().atol

    if state.is_ket():
        psi = state.dense().reshape(-1)
        probs = np.abs(psi) ** 2

    elif state.is_bra():
        psi = state.adjoint().dense().reshape(-1)
        probs = np.abs(psi) ** 2

    elif state.is_density_matrix():
        rho = state.dense()
        probs = np.real(np.diag(rho))

        # Optional sanity: tiny negative values can happen from numerical noise
        probs = np.where(np.abs(probs) < atol, 0.0, probs)
        if np.any(probs < -atol):
            raise ValueError(f"Density matrix has significantly negative diagonal entries (min={probs.min()}).")

    else:
        raise TypeError(
            "QTensor must represent a ket, bra, or density matrix "
            "(use state.is_ket(), state.is_bra(), state.is_density_matrix())."
        )

    if check_normalization:
        s = float(np.sum(probs))
        if not np.isfinite(s) or abs(s - 1.0) > NORMALIZATION_TOLERANCE:
            # Up to you whether to error or renormalize; here we error by default.
            raise ValueError(
                f"Probabilities not normalized: sum={s}. "
                "If expected (e.g. unnormalized state), set check_normalization=False."
            )
    n_qubits = state.nqubits
    return {format(idx, f"0{n_qubits}b"): float(p) for idx, p in enumerate(probs)}


def _samples_from_state(state: QTensor, nshots: int = 100, seed: int | None = None) -> dict[str, int]:
    probabilities = _probabilities_from_state(state)
    return _samples_from_probabilities(probabilities=probabilities, nshots=nshots, seed=seed)


def _samples_from_probabilities(
    probabilities: dict[str, float], nshots: int = 100, seed: int | None = None
) -> dict[str, int]:
    states = np.array(list(probabilities.keys()))
    probs = np.array(list(probabilities.values()), dtype=np.float64)
    probs /= probs.sum()

    rng = np.random.default_rng(seed)
    draws = rng.choice(len(states), size=nshots, p=probs)

    counts = np.bincount(draws, minlength=len(states))
    return {str(states[i]): int(counts[i]) for i in range(len(states)) if counts[i] > 0}
