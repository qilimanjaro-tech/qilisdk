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

from typing import Generic, Type, TypeVar, cast

from qilisdk.common.model import Model
from qilisdk.functionals.functional import Functional, PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.optimizers.optimizer import Optimizer
from qilisdk.yaml import yaml

R = TypeVar("R", bound=FunctionalResult)


@yaml.register_class
class VariationalProgram(Functional[VariationalProgramResult[R]], Generic[R]):
    def __init__(
        self,
        functional: PrimitiveFunctional[R],
        optimizer: Optimizer,
        cost_model: Model,
        store_intermediate_results: bool = False,
    ) -> None:
        """The Parameterized Program is a data class that gathers the necessary parameters to optimize a parameterized
        functional.

        Args:
            functional (Functional): A parameterized Functional to be optimized.
            optimizer (Optimizer): A QiliSDK optimizer, to be used in optimizing the Functional's parameters.
            cost_model (Model): A Model object to evaluate the cost of a given set of parameters. This model
                                is the cost function used by the optimizer.
        """
        self._functional = functional
        self._optimizer = optimizer
        self._cost_model = cost_model
        self._store_intermediate_results = store_intermediate_results

    @property
    def result_type(self) -> Type[VariationalProgramResult[R]]:
        # Generics are erased at runtime; cast helps static analyzers.
        return cast("Type[VariationalProgramResult[R]]", VariationalProgramResult)

    @property
    def functional(self) -> PrimitiveFunctional[R]:
        return self._functional

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def cost_model(self) -> Model:
        return self._cost_model

    @property
    def store_intermediate_results(self) -> bool:
        return self._store_intermediate_results
