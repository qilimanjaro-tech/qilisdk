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
from typing import TYPE_CHECKING, Any, Callable, List, TypeVar, cast, overload

from qilisdk.common.result import Result
from qilisdk.functionals.parameterized_program_results import ParameterizedProgramResults
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution

if TYPE_CHECKING:
    from qilisdk.functionals.functional import Functional
    from qilisdk.functionals.parameterized_program import ParameterizedProgram
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

TResult = TypeVar("TResult", bound=Result)


class Backend(ABC):
    def __init__(self) -> None:
        self._handlers: dict[type[Functional[Any]], Callable[[Functional[Any]], Result]] = {
            Sampling: lambda f: self._execute_sampling(cast("Sampling", f)),
            TimeEvolution: lambda f: self._execute_time_evolution(cast("TimeEvolution", f)),
        }

    @overload
    def execute(self, functional: Sampling) -> SamplingResult: ...

    @overload
    def execute(self, functional: TimeEvolution) -> TimeEvolutionResult: ...

    @overload
    def execute(self, functional: Functional[TResult]) -> TResult: ...

    def execute(self, functional: Functional[Any]) -> Any:
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc

        return handler(functional)

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no Sampling implementation")

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no TimeEvolution implementation")

    def optimize(
        self, parameterized_program: ParameterizedProgram, store_intermediate_results: bool = False
    ) -> ParameterizedProgramResults:
        def evaluate_sample(parameters: List[float]) -> float:
            param_names = parameterized_program.functional.get_parameter_names()
            parameterized_program.functional.set_parameters(
                {param_names[i]: param for i, param in enumerate(parameters)}
            )
            results = self.execute(parameterized_program.functional)
            return parameterized_program.functional.compute_cost(results, parameterized_program.cost_model)

        optimizer_result = parameterized_program.optimizer.optimize(
            cost_function=evaluate_sample,
            init_parameters=list(parameterized_program.functional.get_parameters().values()),
            store_intermediate_results=store_intermediate_results,
        )

        param_names = parameterized_program.functional.get_parameter_names()
        parameterized_program.functional.set_parameters(
            {param_names[i]: param for i, param in enumerate(optimizer_result.optimal_parameters)}
        )
        optimal_results = self.execute(parameterized_program.functional)

        return ParameterizedProgramResults(optimizer_result=optimizer_result, result=optimal_results)
