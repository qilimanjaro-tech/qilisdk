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

from copy import copy
from typing import TYPE_CHECKING, Generic, Literal

from .readout import ExpectationReadout, SamplingReadout, StateTomographyReadout
from .readout_result import E, ExpectationReadoutResult, S, SamplingReadoutResult, StateTomographyReadoutResult, T

if TYPE_CHECKING:
    from qilisdk.analog import Hamiltonian
    from qilisdk.core import QTensor

    from .readout import ReadoutMethod


class ReadoutSpec(Generic[S, E, T]):
    """Type-safe specification of which readout methods to apply.

    Construct via chaining::

        spec = ReadoutSpec().with_sampling(SamplingReadout(nshots=1000))

    Each ``with_*`` method returns a **new** :class:`ReadoutSpec` whose
    type parameters reflect the added readout.  The original instance is
    not mutated.
    """

    def __init__(self: ReadoutSpec[None, None, None]) -> None:
        self._sampling: SamplingReadout | None = None
        self._expectation: ExpectationReadout | None = None
        self._state_tomography: StateTomographyReadout | None = None

    # -- builder methods -- each one transforms one type parameter --------

    def with_sampling(self, nshots: int = 100) -> ReadoutSpec[SamplingReadoutResult, E, T]:
        """Add a sampling readout to the specification.

        Args:
            nshots (int): Number of measurement shots.  Must be >= 0.

        Returns:
            A new :class:`ReadoutSpec` with the sampling slot populated.

        Raises:
            ValueError: If a sampling readout was already set.
        """
        if self._sampling is not None:
            raise ValueError("Sampling readout already set in this specification.")
        new: ReadoutSpec = copy(self)
        new._sampling = SamplingReadout(nshots=nshots)
        return new  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

    def with_expectation(
        self,
        observables: list[Hamiltonian | QTensor],
        nshots: int = 0,
    ) -> ReadoutSpec[S, ExpectationReadoutResult, T]:
        """Add an expectation-value readout to the specification.

        Args:
            observables (list[Hamiltonian | QTensor]): Observables whose expectation values will be evaluated.
            nshots (int): Number of measurement shots.  Use ``0`` for exact (state-vector-based) evaluation.

        Returns:
            A new :class:`ReadoutSpec` with the expectation slot populated.

        Raises:
            ValueError: If an expectation readout was already set.
        """
        if self._expectation is not None:
            raise ValueError("Expectation readout already set in this specification.")
        new: ReadoutSpec = copy(self)
        new._expectation = ExpectationReadout(observables=observables, nshots=nshots)
        return new  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

    def with_state_tomography(
        self, method: Literal["exact"] = "exact"
    ) -> ReadoutSpec[S, E, StateTomographyReadoutResult]:
        """Add a state-tomography readout to the specification.

        Args:
            method (Literal["exact"]): Tomography method identifier. Currently only ``"exact"`` is supported.

        Returns:
            A new :class:`ReadoutSpec` with the state-tomography slot populated.

        Raises:
            ValueError: If a state-tomography readout was already set.
        """
        if self._state_tomography is not None:
            raise ValueError("State-tomography readout already set in this specification.")
        new: ReadoutSpec = copy(self)
        new._state_tomography = StateTomographyReadout(state_tomography_method=method)
        return new  # type: ignore[return-value]  # ty:ignore[invalid-return-type]

    # -- accessors --------------------------------------------------------

    @property
    def sampling(self) -> SamplingReadout | None:
        """The sampling readout, or ``None`` if not set."""
        return self._sampling

    @property
    def expectation(self) -> ExpectationReadout | None:
        """The expectation-value readout, or ``None`` if not set."""
        return self._expectation

    @property
    def state_tomography(self) -> StateTomographyReadout | None:
        """The state-tomography readout, or ``None`` if not set."""
        return self._state_tomography

    def to_list(self) -> list[ReadoutMethod]:
        """Return the readout methods as a flat list.

        Useful for backward compatibility with APIs that accept
        ``list[ReadoutMethod]``.
        """
        return [ro for ro in (self._sampling, self._expectation, self._state_tomography) if ro is not None]

    def __repr__(self) -> str:
        parts: list[str] = []
        if self._sampling is not None:
            parts.append(f"sampling={self._sampling!r}")
        if self._expectation is not None:
            parts.append(f"expectation={self._expectation!r}")
        if self._state_tomography is not None:
            parts.append(f"state_tomography={self._state_tomography!r}")
        return f"ReadoutSpec({', '.join(parts)})"

    def __bool__(self) -> bool:
        return self._sampling is not None or self._expectation is not None or self._state_tomography is not None
