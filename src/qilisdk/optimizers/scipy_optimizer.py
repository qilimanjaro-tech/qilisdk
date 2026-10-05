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

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from loguru import logger
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
                    - 'trust-exact'
                    - 'trust-krylov'
                    - 'basinhopping' (global)
                    - 'direct' (global)
                    - 'dual_annealing' (global)
                    - 'differential_evolution' (global)
                    - 'shgo' (global)
                    - 'brute' (global)
                    - custom - a callable object, see `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__ for description.

                    If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
                    depending on whether or not the problem has constraints or bounds.
            bounds (list[tuple[int, int]] | None, optional):
                    Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
                    trust-constr, COBYLA, and COBYQA methods. To specify it you can provide a sequence of ``(min, max)`` pairs
                    for each element in parameter list.

        Extra Args:
            Any argument supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>` can be passed.
            For the global methods, any argument of the corresponding ``scipy.optimize`` function can be passed instead.
            Note: the parameters, cost function and the ``args`` that are passed to this function will be specified in the optimize method. Moreover, callbacks are not supported for the moment.
        """
        super().__init__()
        self.method = method
        self.extra_arguments = kwargs
        logger.debug("[SciPyOptimizer] Created optimizer with method {}", method)

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
            store_intermediate_results (bool, optional): whether to record the parameters and cost reported by scipy at
                each iteration. ``brute`` has no iterations, so nothing is recorded for it. Defaults to False.

        Returns:
            list[float]: the optimal set of parameters that minimize the cost function.
        """
        logger.debug(
            "[SciPyOptimizer] Starting optimization with method {} and {} parameters",
            self.method,
            len(init_parameters),
        )
        intermediate_results: list[OptimizerIntermediateResult] = []

        def store_intermediate_result(parameters: list[float], cost: float) -> None:
            logger.trace("[SciPyOptimizer] Intermediate iteration with cost {}", cost)
            intermediate_results.append(
                OptimizerIntermediateResult(cost=float(cost), parameters=np.asarray(parameters, dtype=float).tolist())
            )

        # scipy uses a different callback signature depending on the method
        def result_callback(intermediate_result: OptimizeResult) -> None:
            store_intermediate_result(intermediate_result.x, intermediate_result.fun)

        def value_callback(parameters: list[float], cost: float, *_: object) -> None:
            store_intermediate_result(parameters, cost)

        def parameter_callback(parameters: list[float]) -> None:
            store_intermediate_result(parameters, cost_function(parameters))

        callback: Callable | None = None
        if store_intermediate_results and self.method in {"basinhopping", "dual_annealing"}:
            callback = value_callback
        elif store_intermediate_results and str(self.method).lower() in {"direct", "tnc"}:
            callback = parameter_callback
        elif store_intermediate_results:
            callback = result_callback

        # Global optimizer have a different interface, like `scipy.optimize.shgo` rather than `scipy.optimize.minimize`
        if self.method in {"direct", "dual_annealing", "differential_evolution", "shgo"} and isinstance(
            self.method, str
        ):
            logger.debug("[SciPyOptimizer] Using global optimizer interface {}", self.method)
            res = getattr(scipy_optimize, self.method)(
                cost_function,
                bounds=bounds,
                callback=callback,
                **self.extra_arguments,
            )
        # basinhopping doesn't take bounds itself, so they are passed to its local minimizer
        elif self.method == "basinhopping":
            logger.debug("[SciPyOptimizer] Using global optimizer interface {}", self.method)
            minimizer_kwargs = {"bounds": bounds, **self.extra_arguments.get("minimizer_kwargs", {})}
            res = scipy_optimize.basinhopping(
                cost_function,
                x0=init_parameters,
                callback=callback,
                **{**self.extra_arguments, "minimizer_kwargs": minimizer_kwargs},
            )
        # brute has a different syntax, it evaluates a grid over the bounds and then polishes the best point within the bounds
        elif self.method == "brute":
            logger.debug("[SciPyOptimizer] Using global optimizer interface {}", self.method)
            if store_intermediate_results:
                logger.warning(
                    "[SciPyOptimizer] Intermediate results are not supported for method brute, none will be stored"
                )
            optimal_parameters, optimal_cost, *_ = scipy_optimize.brute(
                cost_function,
                ranges=bounds,
                full_output=True,
                **{"finish": partial(scipy_optimize.minimize, bounds=bounds), **self.extra_arguments},
            )
            res = scipy_optimize.OptimizeResult(x=np.atleast_1d(optimal_parameters), fun=optimal_cost)
        # the more general local minimizer interface
        else:
            logger.debug("[SciPyOptimizer] Using local minimizer interface with method {}", self.method)
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

        logger.debug(
            "[SciPyOptimizer] Optimization finished with optimal cost {} and {} intermediate results",
            res.fun,
            len(intermediate_results),
        )
        return OptimizerResult(
            optimal_cost=res.fun,
            optimal_parameters=res.x.tolist(),
            intermediate_results=intermediate_results,
        )

    def __repr__(self) -> str:
        extra_args_str = ", ".join(f"{key}={value!r}" for key, value in self.extra_arguments.items())
        return f"SciPyOptimizer(method={self.method!r}, {extra_args_str})"
