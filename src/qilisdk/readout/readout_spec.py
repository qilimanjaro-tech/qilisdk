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
"""Type-safe builder for composing readout method specifications.

:class:`ReadoutSpec` tracks which readout methods are requested at the
*type level* via three generic parameters ``S``, ``E``, ``T``.  Each
``with_*`` method transforms the corresponding type parameter from
``None`` to the concrete result type, so the type checker always knows
exactly which results will be available after execution.

Example::

    from qilisdk.readout import ReadoutSpec, SamplingReadout, StateTomographyReadout

    spec = (
        ReadoutSpec()
        .with_sampling(SamplingReadout(nshots=1000))
        .with_state_tomography(StateTomographyReadout())
    )
    # Inferred type: ReadoutSpec[SamplingReadoutResult, None, StateTomographyReadoutResult]

    result = backend.execute(evolution, spec)
    result.sampling.samples            # statically typed as dict[str, int]
    result.state_tomography.state      # statically typed as QTensor
    result.expectation                 # statically typed as None
"""
from __future__ import annotations

from copy import copy
from typing import Generic

from .readout import ExpectationReadout, ReadoutMethod, SamplingReadout, StateTomographyReadout
from .readout_result import (
    E,
    ExpectationReadoutResult,
    S,
    SamplingReadoutResult,
    StateTomographyReadoutResult,
    T,
)


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

    def with_sampling(self, readout: SamplingReadout) -> ReadoutSpec[SamplingReadoutResult, E, T]:
        """Add a sampling readout to the specification.

        Args:
            readout: The sampling readout configuration.

        Returns:
            A new :class:`ReadoutSpec` with the sampling slot populated.

        Raises:
            ValueError: If a sampling readout was already set.
        """
        if self._sampling is not None:
            raise ValueError("Sampling readout already set in this specification.")
        new = copy(self)
        new._sampling = readout
        return new  # type: ignore[return-value]

    def with_expectation(self, readout: ExpectationReadout) -> ReadoutSpec[S, ExpectationReadoutResult, T]:
        """Add an expectation-value readout to the specification.

        Args:
            readout: The expectation-value readout configuration.

        Returns:
            A new :class:`ReadoutSpec` with the expectation slot populated.

        Raises:
            ValueError: If an expectation readout was already set.
        """
        if self._expectation is not None:
            raise ValueError("Expectation readout already set in this specification.")
        new = copy(self)
        new._expectation = readout
        return new  # type: ignore[return-value]

    def with_state_tomography(self, readout: StateTomographyReadout) -> ReadoutSpec[S, E, StateTomographyReadoutResult]:
        """Add a state-tomography readout to the specification.

        Args:
            readout: The state-tomography readout configuration.

        Returns:
            A new :class:`ReadoutSpec` with the state-tomography slot populated.

        Raises:
            ValueError: If a state-tomography readout was already set.
        """
        if self._state_tomography is not None:
            raise ValueError("State-tomography readout already set in this specification.")
        new = copy(self)
        new._state_tomography = readout
        return new  # type: ignore[return-value]

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
