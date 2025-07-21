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
from typing import TYPE_CHECKING, Callable, TypeVar, cast

from qilisdk.analog.time_evolution import TimeEvolution
from qilisdk.common.result import Result
from qilisdk.digital.sampling import Sampling

if TYPE_CHECKING:
    from qilisdk.analog.time_evolution_result import TimeEvolutionResult
    from qilisdk.common.functional import Functional
    from qilisdk.digital.sampling_result import SamplingResult

TResult = TypeVar("TResult", bound=Result)


class Backend(ABC):
    def __init__(self) -> None:
        self._handlers: dict[type[Functional], Callable[[Functional], Result]] = {
            Sampling: self._execute_sampling,
            TimeEvolution: self._execute_time_evolution,
        }

    def execute(self, functional: Functional[TResult]) -> TResult:
        try:
            handler = self._handlers[type(functional)]
        except KeyError as exc:
            raise NotImplementedError(
                f"{type(self).__qualname__} does not support {type(functional).__qualname__}"
            ) from exc

        # mypy cannot follow the per-class mapping, so we cast.
        return cast("TResult", handler(functional))

    def _execute_sampling(self, functional: Sampling) -> SamplingResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no Sampling implementation")

    def _execute_time_evolution(self, functional: TimeEvolution) -> TimeEvolutionResult:
        raise NotImplementedError(f"{type(self).__qualname__} has no TimeEvolution implementation")
