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

from typing import ClassVar, Generic, TypeVar

from qilisdk.cost_functions.cost_function import CostFunction
from qilisdk.functionals.functional import Functional, PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.optimizers.optimizer import Optimizer
from qilisdk.yaml import yaml

TFunctional = TypeVar("TFunctional", bound=PrimitiveFunctional[FunctionalResult])


@yaml.register_class
class VariationalProgram(Functional, Generic[TFunctional]):
    result_type: ClassVar[type[FunctionalResult]] = VariationalProgramResult

    def __init__(
        self,
        functional: TFunctional,
        optimizer: Optimizer,
        cost_function: CostFunction,
        store_intermediate_results: bool = False,
    ) -> None:
        """The Parameterized Program is a data class that gathers the necessary parameters to optimize a parameterized
        functional.

        Args:
            functional (Functional): The parameterized Functional to be optimized.
            optimizer (Optimizer): The optimizer to be used in optimizing the Functional's parameters.
            cost_function (CostFunction): A CostFunction object that defines how the cost is being computed.
            store_intermediate_results (bool, optional): If True, stores a list of intermediate results.
        """
        self._functional = functional
        self._optimizer = optimizer
        self._cost_function = cost_function
        self._store_intermediate_results = store_intermediate_results

    @property
    def functional(self) -> TFunctional:
        return self._functional

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_function

    @property
    def store_intermediate_results(self) -> bool:
        return self._store_intermediate_results
