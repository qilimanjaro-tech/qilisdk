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
    """Type-safe specification of which readout methods to apply during execution.

    :class:`ReadoutSpec` is the **primary interface** for declaring what information to extract from the quantum
    backend after execution.  Build a specification by chaining ``with_*`` builder methods, then pass it to
    :meth:`~qilisdk.backends.Backend.execute`.

    Each ``with_*`` method returns a **new** :class:`ReadoutSpec` whose type parameters encode which readout
    slots are populated, allowing the type checker to verify result access at compile time.  The original
    instance is not mutated.

    Type parameters:
        S: :class:`~qilisdk.readout.SamplingReadoutResult` when :meth:`with_sampling` was called, ``None``
            otherwise.
        E: :class:`~qilisdk.readout.ExpectationReadoutResult` when :meth:`with_expectation` was called,
            ``None`` otherwise.
        T: :class:`~qilisdk.readout.StateTomographyReadoutResult` when :meth:`with_state_tomography` was
            called, ``None`` otherwise.

    Examples:
        Sampling — measure in the computational basis and collect bitstring counts::

            spec = ReadoutSpec().with_sampling(nshots=1000)
            result = backend.execute(functional, readout=spec)
            counts = result.samples        # dict[str, int]
            probs  = result.probabilities  # dict[str, float]

        Expectation values — compute ``<psi|O|psi>`` for one or more observables::

            from qilisdk.analog import Z

            spec = ReadoutSpec().with_expectation(observables=[Z(0)], nshots=0)
            result = backend.execute(functional, readout=spec)
            ev = result.expectation_values  # list[float], one entry per observable

        State tomography — retrieve the full quantum state vector after execution::

            spec = ReadoutSpec().with_state_tomography()
            result = backend.execute(functional, readout=spec)
            state = result.state  # QTensor

        Multiple readout types can be combined in a single specification::

            from qilisdk.analog import Z

            spec = (
                ReadoutSpec()
                .with_sampling(nshots=500)
                .with_expectation(observables=[Z(0)])
            )
            result = backend.execute(functional, readout=spec)
            counts = result.samples           # dict[str, int]
            ev     = result.expectation_values  # list[float]
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
        new._state_tomography = StateTomographyReadout(method=method)
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
