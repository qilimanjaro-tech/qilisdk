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

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.optimizers.optimizer_result import OptimizerIntermediateResult, OptimizerResult
from qilisdk.yaml import yaml

TResult_co = TypeVar("TResult_co", bound=FunctionalResult, covariant=True)


@yaml.register_class
class VariationalProgramResult(FunctionalResult, Generic[TResult_co]):
    """Aggregate the optimizer summary and best functional result from a variational run."""

    def __init__(self, optimizer_result: OptimizerResult, result: TResult_co) -> None:
        """
        Args:
            optimizer_result (OptimizerResult): Summary produced by the optimiser.
            result (TResult_co): Functional result evaluated at the final parameters.
        """
        super().__init__()
        self._optimizer_result = optimizer_result
        self._result = result

    @property
    def optimal_cost(self) -> float:
        """Best cost reported by the optimiser."""
        return self._optimizer_result.optimal_cost

    @property
    def optimal_execution_results(self) -> TResult_co:
        """Return the functional result evaluated at the optimal parameters."""
        return self._result

    @property
    def optimal_parameters(self) -> list[float]:
        """Optimised parameter values."""
        return self._optimizer_result.optimal_parameters

    @property
    def intermediate_results(self) -> list[OptimizerIntermediateResult]:
        """Sequence of intermediate optimiser snapshots, if recorded."""
        return self._optimizer_result.intermediate_results

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(\n"
            f"  Optimal Cost={self.optimal_cost},\n"
            f"  Optimal Parameters={pformat(self.optimal_parameters)},\n"
            f"  Intermediate Results={pformat(self.intermediate_results)},\n"
            f"  Optimal Results={pformat(self.optimal_execution_results)}\n"
            ")"
        )
