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
from typing import TYPE_CHECKING, Callable, TypeVar, cast, overload

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.sampling import Sampling
from qilisdk.functionals.time_evolution import TimeEvolution
from qilisdk.functionals.variational_program import VariationalProgram
from qilisdk.functionals.variational_program_result import VariationalProgramResult

if TYPE_CHECKING:
    from qilisdk.functionals.functional import Functional, PrimitiveFunctional
    from qilisdk.functionals.sampling_result import SamplingResult
    from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

TResult = TypeVar("TResult", bound=FunctionalResult)


class Backend(ABC):
    def __init__(self) -> None:
        self._handlers: dict[type[Functional], Callable[[Functional], FunctionalResult]] = {
            Sampling: lambda f: self._execute_sampling(cast("Sampling", f)),
            TimeEvolution: lambda f: self._execute_time_evolution(cast("TimeEvolution", f)),
            VariationalProgram: lambda f: self._execute_variational_program(cast("VariationalProgram", f)),
        }

    @overload
    def execute(self, functional: Sampling) -> SamplingResult: ...

    @overload
    def execute(self, functional: TimeEvolution) -> TimeEvolutionResult: ...

    @overload
    def execute(self, functional: VariationalProgram[Sampling]) -> VariationalProgramResult[SamplingResult]: ...

    @overload
    def execute(
        self, functional: VariationalProgram[TimeEvolution]
    ) -> VariationalProgramResult[TimeEvolutionResult]: ...

    @overload
    def execute(self, functional: PrimitiveFunctional[TResult]) -> TResult: ...

    def execute(self, functional: Functional) -> FunctionalResult:
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

    def _execute_variational_program(
        self, functional: VariationalProgram[PrimitiveFunctional[TResult]]
    ) -> VariationalProgramResult[TResult]:
        """Optimize a Parameterized Program (:class:`~qilisdk.functionals.variational_program.VariationalProgram`)
            and returns the optimal parameters and results.

        Args:
            functional (VariationalProgram): The variational program to be optimized.

        Returns:
            ParameterizedProgramResults: The final optimizer and functional results.

        Raises:
            ValueError: If the functional is not parameterized.
        """

        def evaluate_sample(parameters: list[float]) -> float:
            param_names = functional.functional.get_parameter_names()
            param_dict = {param_names[i]: param for i, param in enumerate(parameters)}
            err = functional.check_parameter_constraints(param_dict)
            if err > 0:
                return err
            functional.functional.set_parameters(param_dict)
            results = self.execute(functional.functional)
            final_results = functional.cost_function.compute_cost(results)
            if isinstance(final_results, float):
                return final_results
            if isinstance(final_results, complex) and final_results.imag == 0:
                return final_results.real
            raise ValueError(f"Unsupported result type {type(final_results)}.")

        if len(functional.functional.get_parameters()) == 0:
            raise ValueError("Functional provided is not parameterized.")

        optimizer_result = functional.optimizer.optimize(
            cost_function=evaluate_sample,
            init_parameters=list(functional.functional.get_parameters().values()),
            bounds=list(functional.functional.get_parameter_bounds().values()),
            store_intermediate_results=functional.store_intermediate_results,
        )

        param_names = functional.functional.get_parameter_names()
        functional.functional.set_parameters(
            {param_names[i]: param for i, param in enumerate(optimizer_result.optimal_parameters)}
        )
        optimal_results: TResult = cast("TResult", self.execute(functional.functional))

        return VariationalProgramResult(optimizer_result=optimizer_result, result=optimal_results)
