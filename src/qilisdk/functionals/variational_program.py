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

import functools
import operator
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from qilisdk.core.variables import BaseVariable, Parameter
from qilisdk.functionals.functional import Functional, PrimitiveFunctional
from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.functionals.variational_program_result import VariationalProgramResult
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from qilisdk.core.variables import ComparisonTerm
    from qilisdk.cost_functions.cost_function import CostFunction
    from qilisdk.optimizers.optimizer import Optimizer

TFunctional = TypeVar("TFunctional", bound=PrimitiveFunctional[FunctionalResult])


@yaml.register_class
class VariationalProgram(Functional, Generic[TFunctional]):
    """
    Bundle a parameterized functional, optimizer, and cost function into a variational loop.

    Example:
        .. code-block:: python

            program = VariationalProgram(functional, optimizer, cost_function)
    """

    result_type: ClassVar[type[FunctionalResult]] = VariationalProgramResult

    def __init__(
        self,
        functional: TFunctional,
        optimizer: Optimizer,
        cost_function: CostFunction,
        store_intermediate_results: bool = False,
        parameter_constraints: list[ComparisonTerm] | None = None,
    ) -> None:
        """
        Args:
            functional (PrimitiveFunctional): Parameterized functional to optimize.
            optimizer (Optimizer): Optimization routine controlling parameter updates.
            cost_function (CostFunction): Metric used to evaluate functional executions.
            store_intermediate_results (bool, optional): Persist intermediate executions if requested by the optimizer.
            parameter_constraints (list[ComparisonTerm] | None): Optional constraints on parameter values that are
                enforced before optimizer updates are applied.

        Raises:
            ValueError: if the user applies constraints on parameters that are not present in the variational program.
                        Or the constraints contain Objects that are not parameters.
        """
        self._functional = functional
        self._optimizer = optimizer
        self._cost_function = cost_function
        self._store_intermediate_results = store_intermediate_results
        parameter_constraints = parameter_constraints or []
        functional_params = self._functional.get_parameters()
        for p in parameter_constraints:
            if not p.lhs.is_parameterized_term() or not p.rhs.is_parameterized_term():
                raise ValueError("Only parameters are allowed to be constrained.")
            variables = p.variables()
            for v in variables:
                if v.label not in functional_params:
                    raise ValueError(
                        f"Writing a constraint on the parameter ({v}) that is not present in the variational program "
                    )
        self._parameter_constraints = parameter_constraints

    @property
    def functional(self) -> TFunctional:
        """Return the wrapped functional that will be optimised."""
        return self._functional

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer responsible for parameter updates."""
        return self._optimizer

    @property
    def cost_function(self) -> CostFunction:
        """Return the cost function applied to functional results."""
        return self._cost_function

    @property
    def store_intermediate_results(self) -> bool:
        """Indicate whether intermediate execution data should be stored."""
        return self._store_intermediate_results

    def get_constraints(self) -> list[ComparisonTerm]:
        """Return variational-program-level constraints plus those from the underlying functional."""
        return self._parameter_constraints + self._functional.get_constraints()

    def _check_constraints(self, parameters: dict[str, float]) -> list[bool]:
        """Evaluate each constraint with a proposed parameter set.

        Returns:
            list[bool]: list of booleans that correspond to whether each constraint is satisfied or not.

        Raises:
            ValueError: if the parameter is not defined in the underlying functional.
        """
        params: list[BaseVariable] = functools.reduce(
            operator.iadd, (con.variables() for con in self.get_constraints()), []
        )
        params = list(set(params))
        if any(not isinstance(p, Parameter) for p in params):
            raise ValueError("Only Parameters are allowed.")
        params_dict = {p.label: p for p in params}
        evaluate_dict: dict[BaseVariable, float] = {}
        functional_params = self._functional.get_parameters()
        for label, value in parameters.items():
            if label not in functional_params:
                raise ValueError(f"Parameter {label} is not defined in the functional.")
            if label in params_dict:
                evaluate_dict[params_dict[label]] = value
        constraints = self.get_constraints()
        return [con.evaluate(evaluate_dict) for con in constraints]

    def check_parameter_constraints(self, parameters: dict[str, float]) -> int:
        """Return a penalty-like score (0 if valid) indicating how many constraints are violated."""
        const_list = self._check_constraints(parameters)
        return sum((100 if not con else 0) for con in const_list)
