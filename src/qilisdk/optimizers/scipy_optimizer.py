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

from typing import TYPE_CHECKING, Any, Callable

from scipy import optimize as scipy_optimize

from qilisdk.yaml import yaml

from .optimizer import Optimizer
from .optimizer_result import OptimizerIntermediateResult, OptimizerResult

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult


@yaml.register_class
class SciPyOptimizer(Optimizer):
    def __init__(
        self,
        method: str | Callable | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Create a new Gradient Based optimizer instance.

        Args:
            method (str | Callable | None, optional):Type of solver.  Should be one of
                    - 'Nelder-Mead
                    - 'Powell'
                    - 'CG'
                    - 'BFGS'
                    - 'Newton-CG'
                    - 'L-BFGS-B'
                    - 'TNC'
                    - 'COBYLA'
                    - 'COBYQA'
                    - 'SLSQP'
                    - 'trust-constr
                    - 'dogleg'
                    - 'trust-ncg'
                    - 'trust-exact
                    - 'trust-krylov
                    - custom - a callable object, see `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__ for description.

                    If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
                    depending on whether or not the problem has constraints or bounds.
            bounds (list[tuple[int, int]] | None, optional):
                    Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
                    trust-constr, COBYLA, and COBYQA methods. To specify it you can provide a sequence of ``(min, max)`` pairs
                    for each element in parameter list.

        Extra Args:
            Any argument supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>` can be passed.
            Note: the parameters, cost function and the ``args`` that are passed to this function will be specified in the optimize method. Moreover, callbacks are not supported for the moment.
        """
        super().__init__()
        self.method = method
        self.extra_arguments = kwargs

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
        intermediate_results: list[OptimizerIntermediateResult] = []

        def callback_fun(intermediate_result: OptimizeResult) -> None:
            # Create an OptimizerResult for this intermediate iteration.
            intermediate_results.append(
                OptimizerIntermediateResult(cost=intermediate_result.fun, parameters=intermediate_result.x.tolist())
            )

        # Only pass the callback if we want to store intermediate results.
        callback = callback_fun if store_intermediate_results else None

        res = scipy_optimize.minimize(
            cost_function,
            x0=init_parameters,
            method=self.method,
            bounds=bounds,
            jac=self.extra_arguments.get("jac", None),
            hess=self.extra_arguments.get("hess", None),
            hessp=self.extra_arguments.get("hessp", None),
            constraints=self.extra_arguments.get("constraints", ()),
            tol=self.extra_arguments.get("tol", None),
            options=self.extra_arguments.get("options", None),
            callback=callback,
        )

        return OptimizerResult(
            optimal_cost=res.fun,
            optimal_parameters=res.x.tolist(),
            intermediate_results=intermediate_results,
        )
