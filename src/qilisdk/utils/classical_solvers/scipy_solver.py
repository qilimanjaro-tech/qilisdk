# Copyright 2026 Qilimanjaro Quantum Tech
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

from typing import Any, Callable

from qilisdk.core import Model
from qilisdk.core.variables import BaseVariable, BinaryVariable, Domain, Number, RealNumber, Variable
from qilisdk.optimizers import SciPyOptimizer

from .base_solver import ClassicalSolver, _assert_real, _variable_bounds


def _decode_value(variable: BaseVariable, parameter: float) -> RealNumber:
    """
    Decode a SciPy parameter back to the corresponding QiliSDK variable value.

    Args:
        variable: The QiliSDK variable.
        parameter: The SciPy parameter value.

    Returns:
        The corresponding QiliSDK variable value.
    """
    lower, upper = _variable_bounds(variable)
    parameter = min(max(parameter, lower), upper)
    if variable.domain in {Domain.INTEGER, Domain.POSITIVE_INTEGER, Domain.BINARY}:
        return round(parameter)
    if variable.domain is Domain.SPIN:
        return 1 if parameter >= 0 else -1
    return parameter


class ScipySolver(ClassicalSolver):
    """Classical solver that uses SciPy to minimize the model's objective.

    This uses the existing QiliSDK SciPyOptimizer (the one used for variational algorithms)
    to optimize the model function.

    Example:
        .. code-block:: python

            from qilisdk.core import Model
            from qilisdk.utils.classical_solvers import ScipySolver

            model = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
            results, sample = ScipySolver().solve(model, method="l-bfgs-b")
    """

    def solve(  # noqa: PLR6301
        self,
        model: Model,
        method: str | Callable | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[dict[str, Number], dict[BaseVariable, RealNumber]]:
        """Solve the given model by minimizing its objective with SciPy.

        Args:
            model: The ``Model`` instance to solve.
            method (str | Callable | None, optional): The SciPy optimizer to use. See
                :class:`SciPyOptimizer` for the full list of supported methods. If not given, SciPy
                chooses a default local minimizer.

        Extra Args:
            Any argument supported by ``scipy.optimize.minimize`` (or the corresponding global
            optimizer) can be passed and is forwarded to the underlying :class:`SciPyOptimizer`.

        Returns:
            tuple[dict[str, Number], dict[BaseVariable, RealNumber]]: a tuple of
            (results dict mapping objective/constraint labels to their evaluated values,
            sample dict mapping each variable to its value in the best solution found).

        Raises:
            ValueError: if the model contains a variable that is neither a BinaryVariable nor a
                Variable.
        """

        # Get the list of variables from the model
        variables = model.variables()

        # Make sure all variables are either BinaryVariable or Variable
        for v in variables:
            if not isinstance(v, (BinaryVariable, Variable)):
                raise ValueError(f"SciPy solving is not supported for variable {v} of domain {v.domain}.")

        # Get the bounds for each variable
        bounds = [_variable_bounds(v) for v in variables]

        # Convert from SciPy parameters to a sample dict mapping variables to their values
        def build_sample(parameters: list[float]) -> dict[BaseVariable, RealNumber]:
            return {v: _decode_value(v, p) for v, p in zip(variables, parameters)}

        # Evaluate the model for a given set of SciPy parameters
        def cost_function(parameters: list[float]) -> float:
            results = model.evaluate(build_sample(parameters))
            objective_value = _assert_real(results[model.objective.label])
            penalty = sum(_assert_real(results[c.label]) for c in model.constraints)
            return objective_value + penalty

        # Run the optimizer
        optimizer = SciPyOptimizer(method=method, **kwargs)
        result = optimizer.optimize(
            cost_function=cost_function,
            init_parameters=[(lower + upper) / 2 for lower, upper in bounds],
            bounds=bounds,
        )

        # Return the best results
        best_sample = build_sample(result.optimal_parameters)
        return model.evaluate(best_sample), best_sample
