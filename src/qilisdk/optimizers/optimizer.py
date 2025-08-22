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

from abc import ABC, abstractmethod
from typing import Callable

from .optimizer_result import OptimizerResult


class Optimizer(ABC):
    @abstractmethod
    def optimize(
        self,
        cost_function: Callable[[list[float]], float],
        init_parameters: list[float],
        bounds: list[tuple[float, float]],
        store_intermediate_results: bool = False,
    ) -> OptimizerResult:
        """optimize the cost function and return the optimal parameters.

        Args:
            cost_function (Callable[[list[float]], float]): a function that takes in a list of parameters and returns the cost.
            init_parameters (list[float]): the list of initial parameters. Note: the length of this list determines the number of parameters the optimizer will consider.
            bounds (list[float, float]): a list of the variable value bounds.

        Returns:
            list[float]: the optimal set of parameters that minimize the cost function.
        """
