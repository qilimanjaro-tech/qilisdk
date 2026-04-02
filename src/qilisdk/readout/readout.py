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
"""Readout method definitions for quantum circuit execution.

This module provides the :class:`ReadoutMethod` base class and its concrete
subclasses -- :class:`SamplingReadout`, :class:`ExpectationReadout`, and
:class:`StateTomographyReadout` -- that specify how results are extracted from
a quantum backend after execution.
"""

from __future__ import annotations

from typing import ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from qilisdk.analog import Hamiltonian  # noqa: TC001
from qilisdk.core import QTensor

from .readout_result import ExpectationReadoutResult, ReadoutResult, SamplingReadoutResult, StateTomographyReadoutResult

R = TypeVar("R", bound="ReadoutResult")


class ReadoutMethod(BaseModel, Generic[R]):
    """Base type for readout configurations.

    :class:`ReadoutMethod` is not meant to be instantiated directly.  Use one
    of the concrete subclasses (:class:`SamplingReadout`,
    :class:`ExpectationReadout`, :class:`StateTomographyReadout`) or the
    convenience factory methods defined on this class:

    * :meth:`sample` -- creates a :class:`SamplingReadout`
    * :meth:`expectation_values` -- creates an :class:`ExpectationReadout`
    * :meth:`state_tomography` -- creates a :class:`StateTomographyReadout`
    """

    results_type: ClassVar[type[ReadoutResult]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _require_factory_called_on_base(cls, factory_name: str) -> None:
        """Ensure factory helpers are only called on ``ReadoutMethod``.

        Raises:
            TypeError: if it's called on a child rather than the base class.
        """
        if cls is not ReadoutMethod:
            raise TypeError(
                f"{factory_name}() is only available on ReadoutMethod; use ReadoutMethod.{factory_name}(...)"
            )

    @classmethod
    def sampling(cls, nshots: int = 100) -> SamplingReadout:
        """Create a :class:`SamplingReadout`.

        This factory must be called on :class:`ReadoutMethod` itself, not on a
        subclass.

        Args:
            nshots (int): Number of measurement shots.  Must be >= 0.

        Returns:
            SamplingReadout: A configured sampling readout instance.

        Raises:
            AttributeError: If called on a subclass instead of
                :class:`ReadoutMethod`.
        """
        cls._require_factory_called_on_base("sampling")
        return SamplingReadout(nshots=nshots)

    def is_sampling_readout(self) -> bool:
        """Check whether this readout is a :class:`SamplingReadout`.

        Returns:
            bool: ``True`` if this instance is a :class:`SamplingReadout`,
            ``False`` otherwise.
        """
        return isinstance(self, SamplingReadout)

    @classmethod
    def expectation(cls, observables: list[Hamiltonian | QTensor], nshots: int = 0) -> ExpectationReadout:
        """Create an :class:`ExpectationReadout`.

        This factory must be called on :class:`ReadoutMethod` itself, not on a
        subclass.

        Args:
            observables (list[Hamiltonian | QTensor]): Observables whose
                expectation values will be evaluated.
            nshots (int): Number of measurement shots.  Use ``0`` for
                exact (state-vector-based) evaluation.

        Returns:
            ExpectationReadout: A configured expectation-value readout instance.

        Raises:
            AttributeError: If called on a subclass instead of
                :class:`ReadoutMethod`.
        """
        cls._require_factory_called_on_base("expectation")
        return ExpectationReadout(observables=observables, nshots=nshots)

    def is_expectation_readout(self) -> bool:
        """Check whether this readout is an :class:`ExpectationReadout`.

        Returns:
            bool: ``True`` if this instance is an :class:`ExpectationReadout`,
            ``False`` otherwise.
        """
        return isinstance(self, ExpectationReadout)

    @classmethod
    def state_tomography(cls, method: Literal["exact"] = "exact") -> StateTomographyReadout:
        """Create a :class:`StateTomographyReadout`.

        This factory must be called on :class:`ReadoutMethod` itself, not on a
        subclass.

        Args:
            method (Literal["exact"]): Tomography method identifier.
                Currently only ``"exact"`` is supported.

        Returns:
            StateTomographyReadout: A configured state-tomography readout
            instance.

        Raises:
            AttributeError: If called on a subclass instead of
                :class:`ReadoutMethod`.
        """
        cls._require_factory_called_on_base("state_tomography")
        return StateTomographyReadout(state_tomography_method=method)

    def is_state_tomography_readout(self) -> bool:
        """Check whether this readout is a :class:`StateTomographyReadout`.

        Returns:
            bool: ``True`` if this instance is a
            :class:`StateTomographyReadout`, ``False`` otherwise.
        """
        return isinstance(self, StateTomographyReadout)

    @staticmethod
    def _merge_positional_args(
        *,
        class_name: str,
        positional_args: tuple[object, ...],
        positional_names: tuple[str, ...],
        keyword_args: dict[str, object],
    ) -> dict[str, object]:
        if len(positional_args) > len(positional_names):
            expected = ", ".join(positional_names)
            raise TypeError(
                f"{class_name} expects at most {len(positional_names)} positional argument(s): ({expected})."
            )
        merged = dict(keyword_args)
        for name, value in zip(positional_names, positional_args):
            if name in merged:
                raise TypeError(f"{class_name} got multiple values for '{name}' (positional and keyword).")
            merged[name] = value
        return merged


class SamplingReadout(ReadoutMethod[SamplingReadoutResult]):
    """Sampling readout configuration.

    Instructs the backend to perform repeated measurement shots and return
    bitstring counts.  Can also be created via
    :meth:`ReadoutMethod.sample`.

    Args:
        nshots (int): Number of measurement shots (must be >= 0).  Accepted
            as a keyword argument or as the first positional argument.

    Examples:
        >>> SamplingReadout(nshots=1000)
    """

    results_type: ClassVar[type[SamplingReadoutResult]] = SamplingReadoutResult
    nshots: int = Field(ge=0)


class ExpectationReadout(ReadoutMethod[ExpectationReadoutResult]):
    """Expectation-value readout configuration.

    Instructs the backend to compute expectation values
    ``<observable>`` for one or more observables.  Can also be created via
    :meth:`ReadoutMethod.expectation_values`.

    Args:
        observables (list[Hamiltonian | QTensor]): Observables whose
            expectation values are requested.  Accepted as a keyword
            argument or as the first positional argument.
        nshots (int): Number of measurement shots.  Use ``0`` (the default)
            for exact state-vector evaluation.  Accepted as a keyword
            argument or as the second positional argument.

    Attributes:
        qtensor_observables (list[QTensor]): The ``observables`` converted
            to :class:`~qilisdk.core.QTensor` form.  Populated
            automatically by a model validator; not intended to be set
            manually.

    Examples:
        >>> ExpectationReadout(observables=[hamiltonian], nshots=0)
    """

    results_type: ClassVar[type[ExpectationReadoutResult]] = ExpectationReadoutResult
    nshots: int = Field(default=0, ge=0)
    observables: list[Hamiltonian | QTensor]
    qtensor_observables: list[QTensor] = Field(default=[], init=False)

    _scaled_nqubits: int | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_qtensor_observables(self) -> ExpectationReadout:
        self.qtensor_observables = [(o if isinstance(o, QTensor) else o.to_qtensor()) for o in self.observables]
        self._scaled_nqubits = None
        return self

    def expand_observables(self, nqubits: int) -> None:
        """Scale each observable to match a given number of qubits.

        The conversion is cached: calling this method again with the same
        ``nqubits`` value is a no-op.

        Args:
            nqubits (int): Target qubit count to scale the observables to.
        """
        if self._scaled_nqubits == nqubits:
            return
        self.qtensor_observables = [
            (o.expand(nqubits) if isinstance(o, QTensor) else o.to_qtensor(nqubits)) for o in self.observables
        ]
        self._scaled_nqubits = nqubits


class StateTomographyReadout(ReadoutMethod[StateTomographyReadoutResult]):
    """State-tomography readout configuration.

    Instructs the backend to return the full quantum state (ket or density
    matrix) after execution.  Can also be created via
    :meth:`ReadoutMethod.state_tomography`.

    Args:
        state_tomography_method (Literal["exact"]): Tomography method
            identifier.  Currently only ``"exact"`` is supported.  Accepted
            as a keyword argument or as the first positional argument.

    Examples:
        >>> StateTomographyReadout(state_tomography_method="exact")
    """

    results_type: ClassVar[type[StateTomographyReadoutResult]] = StateTomographyReadoutResult
    state_tomography_method: Literal["exact"] = Field(default="exact")
