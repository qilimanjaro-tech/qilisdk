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

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qilisdk.analog import Hamiltonian  # noqa: TC001
from qilisdk.core import QTensor  # noqa: TC001


class ReadoutMethod(BaseModel):
    """Base type for readout configurations.

    Use the subclasses directly or build them with the factory helpers defined
    in this class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _require_factory_called_on_base(cls, factory_name: str) -> None:
        """Ensure factory helpers are only called on ``ReadoutMethod``.

        Raises:
            AttributeError: if it's called on a child rather than the base class.
        """
        if cls is not ReadoutMethod:
            raise AttributeError(
                f"{factory_name}() is only available on ReadoutMethod; use ReadoutMethod.{factory_name}(...)"
            )

    @classmethod
    def sample(cls, nshots: int = 100) -> ReadoutMethod:
        """Create a sampling readout.

        Args:
            nshots (int): Number of measurement shots.

        Returns:
            SamplingReadout: a constructed sampling readout.
        """
        cls._require_factory_called_on_base("sample")
        return SamplingReadout(nshots=nshots)

    def is_sample(self) -> bool:
        """Return ``True`` when this readout is :class:`SamplingReadout`."""
        return isinstance(self, SamplingReadout)

    @classmethod
    def expectation_values(cls, observables: list[Hamiltonian | QTensor], nshots: int = 0) -> ExpectationReadout:
        """Create an expectation-value readout.

        Args:
            observables (list[Hamiltonian | QTensor]): Observables to evaluate.
            nshots (int): Number of shots. Use 0 for exact/state-based evaluation.

        Returns:
            ExpectationReadout: The constructed expectation readout.
        """
        cls._require_factory_called_on_base("expectation_values")
        return ExpectationReadout(observables=observables, nshots=nshots)

    def is_expectation_values(self) -> bool:
        """Return ``True`` when this readout is :class:`ExpectationReadout`."""
        return isinstance(self, ExpectationReadout)

    @classmethod
    def state_tomography(cls, method: Literal["exact"] = "exact") -> StateTomographyReadout:
        """Create a state-tomography readout.

        Args:
            method (Literal["exact"]): Tomography method identifier.

        Returns:
            StateTomographyReadout: The constructed state tomography readout.
        """
        cls._require_factory_called_on_base("state_tomography")
        return StateTomographyReadout(state_tomography_method=method)

    def is_state_tomography(self) -> bool:
        """Return ``True`` when this readout is :class:`StateTomographyReadout`."""
        return isinstance(self, StateTomographyReadout)

    def is_valid(self) -> bool:
        """Return whether this readout is one of the supported concrete kinds."""
        return self.is_state_tomography() or self.is_expectation_values() or self.is_sample()

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


class SamplingReadout(ReadoutMethod):
    """Sampling readout configuration.

    Constructor examples:
        ``SamplingReadout(nshots=1000)``
        ``SamplingReadout(1000)``
    """

    nshots: int = Field(ge=0)

    def __init__(self, *args: object, **data: object) -> None:
        """Create a sampling readout.

        Args:
            nshots (int | None): Number of shots (>= 0), by keyword or first positional arg.
        """
        payload = self._merge_positional_args(
            class_name=type(self).__name__,
            positional_args=args,
            positional_names=("nshots",),
            keyword_args=data,
        )
        super().__init__(**payload)


class ExpectationReadout(ReadoutMethod):
    """Expectation-value readout configuration.

    Constructor examples:
        ``ExpectationReadout(observables=[...], nshots=0)``
        ``ExpectationReadout([...], 0)``
    """

    nshots: int = Field(default=0, ge=0)
    observables: list[Hamiltonian | QTensor]
    qtensor_observables: list[QTensor] = Field(default=[], init=False)

    def __init__(self, *args: object, **data: object) -> None:
        """Create an expectation-value readout.

        Args:
            observables (list[Hamiltonian | QTensor] | None): Observables to measure.
            nshots (int | None): Number of shots (>= 0), by keyword or second positional arg.
        """
        payload = self._merge_positional_args(
            class_name=type(self).__name__,
            positional_args=args,
            positional_names=("observables", "nshots"),
            keyword_args=data,
        )
        super().__init__(**payload)

    @model_validator(mode="after")
    def set_qtensor_observables(self) -> ExpectationReadout:
        self.qtensor_observables = [(o if isinstance(o, QTensor) else o.to_qtensor()) for o in self.observables]
        return self

    def scale_observables(self, nqubits: int) -> None:
        self.qtensor_observables = [o.scale_qtensor(nqubits) for o in self.qtensor_observables]


class StateTomographyReadout(ReadoutMethod):
    """State-tomography readout configuration.

    Constructor examples:
        ``StateTomographyReadout(state_tomography_method="exact")``
        ``StateTomographyReadout("exact")``
    """

    state_tomography_method: Literal["exact"] = Field(default="exact")
    compute_probabilities: bool = Field(default=True)

    def __init__(self, *args: object, **data: object) -> None:
        """Create a state-tomography readout.

        Args:
            state_tomography_method (Literal["exact"] | None): Method by keyword or first positional arg.
        """
        payload = self._merge_positional_args(
            class_name=type(self).__name__,
            positional_args=args,
            positional_names=("state_tomography_method",),
            keyword_args=data,
        )
        super().__init__(**payload)
