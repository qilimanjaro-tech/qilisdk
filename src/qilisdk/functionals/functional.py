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

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qilisdk.analog import Hamiltonian  # noqa: TC001
from qilisdk.core import QTensor  # noqa: TC001
from qilisdk.core.parameterizable import Parameterizable


class ReadoutMethod(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nshots: int | None = Field(default=None, ge=0)
    observables: list[Hamiltonian | QTensor] | None = Field(default=None)
    state_tomography_method: Literal["exact"] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_mutual_exclusion(self) -> ReadoutMethod:
        tomography_set = self.state_tomography_method is not None
        expect_or_sampling_set = self.nshots is not None or self.observables is not None

        if tomography_set and expect_or_sampling_set:
            raise ValueError("state_tomography_method is mutually exclusive with " "nshots and observables.")

        return self

    @classmethod
    def sample(cls, *, nshots: int = 100) -> ReadoutMethod:
        return cls(nshots=nshots)

    def is_sample(self) -> bool:
        return self.nshots is not None and self.observables is None and self.state_tomography_method is None

    @classmethod
    def expectation_values(cls, *, observables: list[Hamiltonian | QTensor], nshots: int = 0) -> ReadoutMethod:
        return cls(observables=observables, nshots=nshots)

    def is_expectation_values(self) -> bool:
        return self.observables is not None and self.state_tomography_method is None

    @classmethod
    def state_tomography(cls, *, method: Literal["exact"] = "exact") -> ReadoutMethod:
        return cls(state_tomography_method=method)

    def is_state_tomography(self) -> bool:
        return self.observables is None and self.nshots is None and self.state_tomography_method is not None


class Functional(ABC):
    """
    Abstract interface for executable routines that return a :class:`FunctionalResult`.

    Subclasses detail the concrete `result_type` they generate.
    """


class PrimitiveFunctional(Parameterizable, Functional, ABC):
    """
    Base class for functionals backed by a :class:`~qilisdk.core.parameterizable.Parameterizable` object.
    """

    @abstractmethod
    def __init__(self, readout: ReadoutMethod | list[ReadoutMethod]) -> None:
        if not isinstance(readout, (list, ReadoutMethod)):
            raise ValueError("Invalid Readout method provided.")
        self.readout = readout if isinstance(readout, list) else [readout]
