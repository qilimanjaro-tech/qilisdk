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

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Type, TypeVar

from qilisdk.common.result import FunctionalResult, Result

if TYPE_CHECKING:
    from qilisdk.common.variables import Number

TResult_co = TypeVar("TResult_co", bound=FunctionalResult, covariant=True)
TGenericResult_co = TypeVar("TGenericResult_co", bound=Result, covariant=True)


class Functional(Protocol[TGenericResult_co]): ...


class PrimitiveFunctional(Functional, Protocol[TResult_co]):
    @property
    def result_type(self) -> Type[TResult_co]: ...

    @abstractmethod
    def set_parameters(self, parameters: dict[str, Number]) -> None:
        """Sets the parameters of the functional.

        Args:
            parameters (dict[str, Number]): a dictionary with the parameter label and new value.
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, Number]:
        """Gets the values of the parameters of the functional.

        Returns:
            dict[str, Number]: a dictionary with the parameter label and its current value.
        """

    @abstractmethod
    def get_parameter_names(self) -> list[str]:
        """Gets the names of the parameters of the functional.

        Returns:
            list[str]: a list of parameter names.
        """

    @abstractmethod
    def get_parameter_values(self) -> list[Number]:
        """Gets the values of the parameters of the functional.

        Returns:
            list[Number]: a list of parameter values.
        """
