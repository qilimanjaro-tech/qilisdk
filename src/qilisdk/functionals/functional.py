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

from pydantic import BaseModel, ConfigDict, Field

from qilisdk.analog import Hamiltonian  # noqa: TC001
from qilisdk.core import QTensor  # noqa: TC001
from qilisdk.core.parameterizable import Parameterizable


class ReadoutBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SamplingReadout(ReadoutBase):
    nshots: int | None = Field(default=None, ge=0)


class ExpectationReadout(ReadoutBase):
    nshots: int | None = Field(default=None, ge=0)
    observables: list[Hamiltonian | QTensor] | None = Field(default=None)


class StateTomographyReadout(ReadoutBase):
    state_tomography_method: Literal["exact"] | None = Field(default=None)


class ReadoutMethod(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    readout_method: ReadoutBase

    @classmethod
    def sample(cls, *, nshots: int = 100) -> ReadoutMethod:
        return cls(readout_method=SamplingReadout(nshots=nshots))

    def is_sample(self) -> bool:
        return isinstance(self.readout_method, SamplingReadout)

    @classmethod
    def expectation_values(cls, *, observables: list[Hamiltonian | QTensor], nshots: int = 0) -> ReadoutMethod:
        return cls(readout_method=ExpectationReadout(observables=observables, nshots=nshots))

    def is_expectation_values(self) -> bool:
        return isinstance(self.readout_method, ExpectationReadout)

    @classmethod
    def state_tomography(cls, *, method: Literal["exact"] = "exact") -> ReadoutMethod:
        return cls(readout_method=StateTomographyReadout(state_tomography_method=method))

    def is_state_tomography(self) -> bool:
        return isinstance(self.readout_method, StateTomographyReadout)


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
        super().__init__()
        if not isinstance(readout, (list, ReadoutMethod)):
            raise ValueError("Invalid Readout method provided.")
        self._readout = readout if isinstance(readout, list) else [readout]

    @property
    def readout(self) -> list[ReadoutMethod]:
        return self._readout

    def has_sampling_readout(self) -> bool:
        return any(isinstance(ro.readout_method, SamplingReadout) for ro in self._readout)

    def has_state_tomography_readout(self) -> bool:
        return any(isinstance(ro.readout_method, StateTomographyReadout) for ro in self._readout)

    def has_expectation_readout(self) -> bool:
        return any(isinstance(ro.readout_method, ExpectationReadout) for ro in self._readout)
