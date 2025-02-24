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


from typing import Any, Callable

from scipy import optimize as sci_opt


class Optimizer:
    def __init__(self) -> None:
        self._optimal_parameters: list[float] = []

    @property
    def optimal_parameters(self) -> list[float]:
        if len(self._optimal_parameters) == 0:
            raise ValueError("No function has been optimized yet")
        return self._optimal_parameters

    def optimize(
        self, cost_function: Callable[[list[float]], float], init_parameters: list[float], **kwargs: dict[str, Any]
    ) -> list[float]:
        """optimize the cost function and return the optimal parameters.

        Args:
            cost_function (Callable[[list[float]], float]): a function that takes in a list of parameters and returns
            the cost.
            TODO: change the cost function for a model instance.
            init_parameters (list[float]): the list of initial parameters. Note: the length of this list determines
            the number of parameters the optimizer will consider.

        Raises:
            NotImplementedError: because this is an abstract class.

        Returns:
            list[float]: the optimal set of parameters that minimize the cost function.
        """
        raise NotImplementedError


class GradientDecent(Optimizer):

    def optimize(
        self,
        cost_function: Callable[[list[float]], float],
        init_parameters: list[float],
        **kwargs: dict[str, Any],
    ) -> list[float]:
        """optimize the cost function and return the optimal parameters.

        Args:
            cost_function (Callable[[list[float]], float]): a function that takes in a list of parameters and returns
            the cost.
            TODO: change the cost function for a model instance.
            init_parameters (list[float]): the list of initial parameters. Note: the length of this list determines
            the number of parameters the optimizer will consider.

        Optional Args:
            method (str | Callable): one of the methods offered by scipy optimize
            cost_function_params (tuple(Any, ...)): a tuple of extra parameters to be passed to the cost function.
            Note: any other parameter specified in scipy optimize can be used in this method.
            (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

        Returns:
            list[float]: the optimal set of parameters that minimize the cost function.
        """
        method = kwargs.pop("method", "Powell")
        cost_function_params = kwargs.pop("cost_function_params", ())

        res = sci_opt.minimize(cost_function, args=cost_function_params, method=method, x0=init_parameters, **kwargs)
        self._optimal_parameters = res.x

        return res
