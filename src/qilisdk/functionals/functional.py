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

from abc import ABC
from typing import ClassVar, Generic, TypeVar

from qilisdk.core.parameterizable import Parameterizable
from qilisdk.functionals.functional_result import FunctionalResult

TResult_co = TypeVar("TResult_co", bound=FunctionalResult, covariant=True)


class Functional(ABC):
    """
    Abstract interface for executable routines that return a :class:`FunctionalResult`.

    Subclasses detail the concrete `result_type` they generate.
    """

    result_type: ClassVar[type[FunctionalResult]]
    """Concrete :class:`~qilisdk.functionals.functional_result.FunctionalResult` subclass returned."""


class PrimitiveFunctional(Parameterizable, Functional, ABC, Generic[TResult_co]):
    """
    Base class for functionals backed by a :class:`~qilisdk.core.parameterizable.Parameterizable` object.
    """
