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
from typing import Any, Callable

from scipy import optimize as scipy_optimize


class Optimizer(ABC):
    def __init__(self) -> None:
        self._optimal_parameters: list[float] = []

    @property
    def optimal_parameters(self) -> list[float]:
        if len(self._optimal_parameters) == 0:
            raise ValueError("No function has been optimized yet")
        return self._optimal_parameters

    @abstractmethod
    def optimize(self, cost_function: Callable[[list[float]], float], init_parameters: list[float]) -> list[float]:
        """Optimize the cost function and return the optimal parameters.

        Args:
            cost_function (Callable[[list[float]], float]): a function that takes in a list of parameters and returns the cost.
            init_parameters (list[float]): the list of initial parameters. Note: the length of this list determines the number of parameters the optimizer will consider.

        Returns:
            list[float]: the optimal set of parameters that minimize the cost function.
        """


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
            Any argument supported by `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__ can be passed.
            Note: the parameters, cost function and the ``args``that are passed to this function will be specified in the optimize method. Moreover, callbacks are not supported for the moment.
        """
        super().__init__()
        self.method = method
        self.extra_arguments = kwargs

    def optimize(self, cost_function: Callable[[list[float]], float], initial_parameters: list[float]) -> list[float]:
        """optimize the cost function and return the optimal parameters.

        Args:
            cost_function (Callable[[list[float]], float]): a function that takes in a list of parameters and returns the cost.
            init_parameters (list[float]): the list of initial parameters. Note: the length of this list determines the number of parameters the optimizer will consider.

        Returns:
            list[float]: the optimal set of parameters that minimize the cost function.
        """
        res = scipy_optimize.minimize(
            cost_function,
            x0=initial_parameters,
            method=self.method,
            jac=self.extra_arguments.get("jac", None),
            hess=self.extra_arguments.get("hess", None),
            hessp=self.extra_arguments.get("hessp", None),
            bounds=self.extra_arguments.get("bounds", None),
            constraints=self.extra_arguments.get("constraints", ()),
            tol=self.extra_arguments.get("tol", None),
            options=self.extra_arguments.get("options", None),
        )
        self._optimal_parameters = res.x

        return res
