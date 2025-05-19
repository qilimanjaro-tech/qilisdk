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

from qilisdk.yaml import yaml


@yaml.register_class
class OptimizerIntermediateResult:
    """
    Represents an intermediate result.

    Attributes:
        cost (float): The optimal cost value (e.g., minimum energy) found.
        parameters (List[float]): The parameters that yield the optimal cost.
    """

    def __init__(
        self,
        cost: float,
        parameters: list[float],
    ) -> None:
        self._cost = cost
        self._parameters = parameters

    @property
    def cost(self) -> float:
        """Return the optimal cost value."""
        return self._cost

    @property
    def parameters(self) -> list[float]:
        """Return the optimal parameters as a list of floats."""
        return list(self._parameters)

    def __repr__(self) -> str:
        """Return a formatted string representation for debugging."""
        return f"OptimizerIntermediateResult(cost={self._cost}, parameters={self._parameters})"


@yaml.register_class
class OptimizerResult:
    """
    Represents the result of an optimization run.

    Attributes:
        optimal_cost (float): The optimal cost value (e.g., minimum energy) found.
        optimal_parameters (List[float]): The parameters that yield the optimal cost.
        intermediate_results (List[OptimizerResult]): A list of intermediate optimization results.
            Each intermediate result is an instance of OptimizerResult containing the current cost and parameters.
    """

    def __init__(
        self,
        optimal_cost: float,
        optimal_parameters: list[float],
        intermediate_results: list[OptimizerIntermediateResult] | None = None,
    ) -> None:
        """
        Initialize an OptimizerResult.

        Args:
            optimal_cost (float): The optimal cost value.
            optimal_parameters (List[float]): The parameters corresponding to the optimal cost.
            intermediate_results (Optional[List[OptimizerResult]]): (Optional) A list of intermediate results recorded during optimization.
                Each intermediate result is an OptimizerResult. Defaults to an empty list if not provided.
        """
        self._optimal_cost = optimal_cost
        self._optimal_parameters = optimal_parameters
        self._intermediate_results = intermediate_results or []

    @property
    def optimal_cost(self) -> float:
        """Return the optimal cost value."""
        return self._optimal_cost

    @property
    def optimal_parameters(self) -> list[float]:
        """Return the optimal parameters as a list of floats."""
        return list(self._optimal_parameters)

    @property
    def intermediate_results(self) -> list[OptimizerIntermediateResult]:
        """
        Return the list of intermediate results.

        Each intermediate result is an instance of OptimizerResult containing:
            - optimal_cost: The cost computed at that iteration.
            - optimal_parameters: The parameters corresponding to that iteration.
        """
        return list(self._intermediate_results)

    def __repr__(self) -> str:
        """Return a formatted string representation for debugging."""
        return (
            f"OptimizerResult(optimal_cost={self._optimal_cost}, "
            f"optimal_parameters={self._optimal_parameters}, "
            f"intermediate_results={self._intermediate_results})"
        )
