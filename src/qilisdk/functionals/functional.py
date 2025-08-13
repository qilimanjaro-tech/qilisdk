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
from typing import TYPE_CHECKING, List, Protocol, Type, TypeVar

from qilisdk.common.result import Result

if TYPE_CHECKING:
    from qilisdk.common.model import Model
    from qilisdk.common.variables import Number

TResult = TypeVar("TResult", bound=Result, covariant=False)


class Functional(Protocol[TResult]):
    @property
    def result_type(self) -> Type[TResult]: ...

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
    def get_parameter_names(self) -> List[str]:
        """Gets the names of the parameters of the functional.

        Returns:
            list[str]: a list of parameter names.
        """

    @abstractmethod
    def get_parameter_values(self) -> List[Number]:
        """Gets the values of the parameters of the functional.

        Returns:
            list[Number]: a list of parameter values.
        """

    @abstractmethod
    def compute_cost(self, results: TResult, cost_model: Model) -> float:
        """Compute the cost of the functional given a cost model.

        Args:
            results (Result): The functional results
            cost_model (Model): The Model object used to represent the cost of different states.

        Returns:
            float: the cost of the results.
        """
