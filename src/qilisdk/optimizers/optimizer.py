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
        store_intermediate_results: bool = False,
    ) -> OptimizerResult:
        """
        Optimize the cost function and return an OptimizerResult.

        Args:
            cost_function (Callable[[List[float]], float]): A function that takes a list of parameters and returns the cost.
            init_parameters (List[float]): The initial parameters for the optimization.
            store_intermediate_results (bool, optional): If True, stores a list of intermediate optimization results.
                Each intermediate result is recorded as an OptimizerResult containing the parameters and cost at that iteration.
                Defaults to False.

        Returns:
            OptimizerResult: An object containing the optimal cost, optimal parameters, and, if requested, the intermediate results.
        """
