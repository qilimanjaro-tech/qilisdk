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
from qilisdk.functionals.sampling_result import SamplingResult
from qilisdk.functionals.time_evolution_result import TimeEvolutionResult

if TYPE_CHECKING:
    from qilisdk.core.variables import Number

TResult = TypeVar("TResult", bound=FunctionalResult)


class CostFunction(ABC):
    """
    Base class that maps functional results into scalar costs.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[FunctionalResult], Callable[[FunctionalResult], Number]] = {
            SamplingResult: lambda f: self._compute_cost_sampling(cast("SamplingResult", f)),
            TimeEvolutionResult: lambda f: self._compute_cost_time_evolution(cast("TimeEvolutionResult", f)),
        }

    @overload
    def compute_cost(self, results: SamplingResult) -> Number: ...

    @overload
    def compute_cost(self, results: TimeEvolutionResult) -> Number: ...

    @overload
    def compute_cost(self, results: FunctionalResult) -> Number: ...

    def compute_cost(self, results: TResult) -> Number:
        """
        Dispatch to the appropriate cost implementation based on the result type.

        Args:
            results (FunctionalResult): Output of a functional execution.

        Returns:
            Number: Scalar cost extracted from the results.

        Raises:
            NotImplementedError: If the concrete cost function does not support the given result type.
        """
        try:
            handler = self._handlers[type(results)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(results).__qualname__}"
            ) from exc

        return handler(results)

    def _compute_cost_sampling(self, results: SamplingResult) -> Number:
        """Compute the cost associated with a :class:`SamplingResult`."""
        raise NotImplementedError

    def _compute_cost_time_evolution(self, results: TimeEvolutionResult) -> Number:
        """Compute the cost associated with a :class:`TimeEvolutionResult`."""
        raise NotImplementedError
