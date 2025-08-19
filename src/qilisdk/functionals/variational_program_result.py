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


from pprint import pformat
from typing import Generic, TypeVar

from qilisdk.common.model import Model
from qilisdk.common.variables import Number
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.optimizers.optimizer_result import OptimizerIntermediateResult, OptimizerResult
from qilisdk.yaml import yaml

TResult_co = TypeVar("TResult_co", bound=FunctionalResult, covariant=True)


@yaml.register_class
class VariationalProgramResult(FunctionalResult, Generic[TResult_co]):
    """
    Represents the result of a Parameterized Program calculation.
    """

    def __init__(self, optimizer_result: OptimizerResult, result: TResult_co) -> None:
        """
        Args:
            optimizer_result (OptimizerResult): The optimizer's final results. (depends on the optimizer used)
            result (TResult_co): The results of executing the Functional with the optimal parameters. (depends on the Functional)
        """
        super().__init__()
        self._optimizer_result = optimizer_result
        self._result = result

    @property
    def optimal_cost(self) -> float:
        """
        Get the optimal cost (estimated ground state energy).

        Returns:
            float: The optimal cost.
        """
        return self._optimizer_result.optimal_cost

    @property
    def optimal_execution_results(self) -> TResult_co:
        """The results of executing the Functional with the optimal parameters found by the optimizer.

        Returns:
            TResult_co: The results object corresponding to the Functional.
        """
        return self._result

    @property
    def optimal_parameters(self) -> list[float]:
        """
        Get the optimal ansatz parameters.

        Returns:
            list[float]: The optimal parameters.
        """
        return self._optimizer_result.optimal_parameters

    @property
    def intermediate_results(self) -> list[OptimizerIntermediateResult]:
        """
        Get the intermediate results.

        Returns:
            list[OptimizerResult]: The intermediate results.
        """
        return self._optimizer_result.intermediate_results

    def __repr__(self) -> str:
        """
        Return a string representation of the Parameterized Program Results for debugging.

        Returns:
            str: A formatted string detailing the optimal cost and parameters.
        """
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n  Optimal Cost = {self.optimal_cost},"
            + f"\n  Optimal Parameters={pformat(self.optimal_parameters)},"
            + f"\n  Intermediate Results={pformat(self.intermediate_results)})"
            + f"\n  Optimal results={pformat(self.optimal_execution_results)})"
        )

    def compute_cost(self, cost_model: Model) -> Number:
        return self.optimal_execution_results.compute_cost(cost_model)
