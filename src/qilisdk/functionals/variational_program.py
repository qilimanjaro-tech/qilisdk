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

from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from qilisdk.functionals.functional import Functional, PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from qilisdk.cost_functions.cost_function import CostFunction
    from qilisdk.optimizers.optimizer import Optimizer

TFunctional = TypeVar("TFunctional", bound=PrimitiveFunctional[FunctionalResult])


@yaml.register_class
class VariationalProgram(Functional, Generic[TFunctional]):
    """
    Bundle a parameterized functional, optimizer, and cost function into a variational loop.

    Example:
        .. code-block:: python

            program = VariationalProgram(functional, optimizer, cost_function)
    """

    result_type: ClassVar[type[FunctionalResult]] = VariationalProgramResult

    def __init__(
        self,
        functional: TFunctional,
        optimizer: Optimizer,
        cost_function: CostFunction,
        store_intermediate_results: bool = False,
    ) -> None:
        """
        Args:
            functional (PrimitiveFunctional): Parameterized functional to optimize.
            optimizer (Optimizer): Optimization routine controlling parameter updates.
            cost_function (CostFunction): Metric used to evaluate functional executions.
            store_intermediate_results (bool, optional): Persist intermediate executions if requested by the optimizer.
        """
        self._functional = functional
        self._optimizer = optimizer
        self._cost_function = cost_function
        self._store_intermediate_results = store_intermediate_results

    @property
    def functional(self) -> TFunctional:
        """Return the wrapped functional that will be optimised."""
        return self._functional

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer responsible for parameter updates."""
        return self._optimizer

    @property
    def cost_function(self) -> CostFunction:
        """Return the cost function applied to functional results."""
        return self._cost_function

    @property
    def store_intermediate_results(self) -> bool:
        """Indicate whether intermediate execution data should be stored."""
        return self._store_intermediate_results
