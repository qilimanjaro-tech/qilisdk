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
from typing import TYPE_CHECKING, Any

from loguru import logger

from qilisdk.algorithms.algorithm import Algorithm
from qilisdk.algorithms.variational_program_result import VariationalProgramResult
from qilisdk.core.variables import BaseVariable, Parameter
from qilisdk.optimizers.optimizer_result import OptimizerResult
from qilisdk.settings import get_settings
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from qilisdk.algorithms.quantum_reservoirs import QuantumReservoir
    from qilisdk.backends.backend import Backend
    from qilisdk.core.comparison import Comparison
    from qilisdk.cost_functions.cost_function import CostFunction
    from qilisdk.functionals.functional import PrimitiveFunctional
    from qilisdk.functionals.functional_result import FunctionalResult
    from qilisdk.optimizers.optimizer import Optimizer
    from qilisdk.readout import Readout


@yaml.register_class
class VariationalProgram(Algorithm):
    """
    Bundle a parameterized functional, optimizer, and cost function into a variational loop.

    Example:
        .. code-block:: python

            from qilisdk.analog import Z
            from qilisdk.backends import QiliSim
            from qilisdk.cost_functions import ObservableCostFunction
            from qilisdk.digital import QAOA
            from qilisdk.algorithms import VariationalProgram
            from qilisdk.functionals import DigitalPropagation
            from qilisdk.optimizers import SciPyOptimizer
            from qilisdk.readout import Readout

            hamiltonian = Z(0) * Z(1) + Z(0)
            functional = DigitalPropagation(QAOA(problem_hamiltonian=hamiltonian, layers=1))
            optimizer = SciPyOptimizer(method="cobyla")
            cost_function = ObservableCostFunction(hamiltonian)

            program = VariationalProgram(functional, optimizer, cost_function)
            result = QiliSim().execute(program, readout=Readout().with_sampling(nshots=1000))
    """

    def __init__(
        self,
        functional: PrimitiveFunctional | QuantumReservoir,
        optimizer: Optimizer,
        cost_function: CostFunction,
        store_intermediate_results: bool = False,
        parameter_constraints: list[Comparison] | None = None,
    ) -> None:
        """
        Args:
            functional (PrimitiveFunctional | QuantumReservoir): Parameterized functional to optimize.
            optimizer (Optimizer): Optimization routine controlling parameter updates.
            cost_function (CostFunction): Metric used to evaluate functional executions.
            store_intermediate_results (bool): Persist intermediate executions
                if requested by the optimizer. Defaults to False.
            parameter_constraints (list[Comparison] | None): Optional
                constraints on parameter values that are enforced before
                optimizer updates are applied. Defaults to None.

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
            if not p.lhs.is_parameterized() or not p.rhs.is_parameterized():
                raise ValueError("Only parameters are allowed to be constrained.")
            variables = p.variables()
            for v in variables:
                if v.label not in functional_params:
                    raise ValueError(
                        f"Writing a constraint on the parameter ({v}) that is not present in the variational program "
                    )
        self._parameter_constraints = parameter_constraints
        logger.debug(
            "[VariationalProgram] Created VariationalProgram (functional={}, optimizer={}, cost_function={}, {} constraints)",
            type(functional).__name__,
            type(optimizer).__name__,
            type(cost_function).__name__,
            len(parameter_constraints),
        )

    @property
    def functional(self) -> PrimitiveFunctional | QuantumReservoir:
        """Parameterized functional that will be optimised."""
        return self._functional

    @property
    def optimizer(self) -> Optimizer:
        """Optimizer responsible for parameter updates."""
        return self._optimizer

    @property
    def cost_function(self) -> CostFunction:
        """Cost function applied to functional results."""
        return self._cost_function

    @property
    def store_intermediate_results(self) -> bool:
        """Whether intermediate execution data should be stored."""
        return self._store_intermediate_results

    def get_constraints(self) -> list[Comparison]:
        """Return variational-program-level constraints plus those from the underlying functional.

        Returns:
            list[Comparison]: Combined list of constraints from the
                program and the wrapped functional.
        """
        return self._parameter_constraints + self._functional.get_constraints()

    def _check_constraints(self, parameters: dict[str, float]) -> list[bool]:
        """Evaluate each constraint with a proposed parameter set.

        Args:
            parameters (dict[str, float]): Proposed parameter values keyed
                by label.

        Returns:
            list[bool]: List of booleans indicating whether each constraint
                is satisfied.

        Raises:
            ValueError: If a parameter label is not defined in the underlying
                functional or if a constraint references a non-parameter
                variable.
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
        """Return a penalty score indicating how many constraints are violated.

        Each violated constraint adds 100 to the returned score; a fully
        valid parameter set returns 0.

        Args:
            parameters (dict[str, float]): Proposed parameter values keyed
                by label.

        Returns:
            int: Cumulative penalty (0 when all constraints are satisfied).
        """
        const_list = self._check_constraints(parameters)
        return sum((100 if not con else 0) for con in const_list)

    def _execute(self, backend: Backend, readout: Readout) -> VariationalProgramResult[FunctionalResult[Any, Any, Any]]:
        """Optimize the trainable parameters of the wrapped functional by executing it on ``backend``.

        Args:
            backend (Backend): The backend used to execute the wrapped functional at each optimizer step.
            readout (Readout): The readout used for every execution of the wrapped functional.

        Returns:
            VariationalProgramResult: The optimizer summary and the functional result at the optimal parameters.

        Raises:
            ValueError: If the wrapped functional has no trainable parameters, the cost function returns an
                unsupported type, or the optimal parameters violate the parameter constraints.
        """
        logger.info("[VariationalProgram] Executing Variational Program")

        def evaluate_sample(parameters: list[float]) -> float:
            logger.trace("[VariationalProgram] Evaluating cost sample at {} parameters", len(parameters))
            param_names = self._functional.get_parameter_names(where=lambda param: param.is_trainable)
            param_bounds = self._functional.get_parameter_bounds()
            new_param_dict = {}
            for i, param in enumerate(parameters):
                name = param_names[i]
                lower_bound, upper_bound = param_bounds[name]
                if lower_bound != upper_bound:
                    # Ensure we clamp the parameter within its bounds, since some methods might ignore bounds
                    new_param_dict[name] = min(max(param, lower_bound), upper_bound)
            err = self.check_parameter_constraints(new_param_dict)
            if err > 0:
                return err
            self._functional.set_parameters(new_param_dict)
            results = backend.execute(self._functional, readout)
            final_results = self._cost_function.compute_cost(results)
            if isinstance(final_results, float):
                return final_results
            if isinstance(final_results, complex) and abs(final_results.imag) < get_settings().atol:
                return final_results.real
            raise ValueError(f"Unsupported result type {type(final_results)}.")

        if len(self._functional.get_parameters(where=lambda param: param.is_trainable)) == 0:
            raise ValueError("Functional provided does not contain trainable parameters.")

        logger.debug("[VariationalProgram] Starting optimization with {}", type(self._optimizer).__name__)
        optimizer_result = self._optimizer.optimize(
            cost_function=evaluate_sample,
            init_parameters=list(self._functional.get_parameters(where=lambda param: param.is_trainable).values()),
            bounds=list(self._functional.get_parameter_bounds(where=lambda param: param.is_trainable).values()),
            store_intermediate_results=self._store_intermediate_results,
        )

        param_bounds = self._functional.get_parameter_bounds(where=lambda param: param.is_trainable)
        optimal_parameter_dict = {}
        for name, param in zip(param_bounds, optimizer_result.optimal_parameters, strict=False):
            lower_bound, upper_bound = param_bounds[name]
            optimal_parameter_dict[name] = min(max(param, lower_bound), upper_bound)
            if optimal_parameter_dict[name] != param:
                logger.warning(
                    "[VariationalProgram] Optimal value {} for parameter '{}' is outside its bounds {}, clamping to {}. "
                    "The chosen optimizer method may not honour bounds.",
                    param,
                    name,
                    (lower_bound, upper_bound),
                    optimal_parameter_dict[name],
                )
        err = self.check_parameter_constraints(optimal_parameter_dict)
        if err > 0:
            raise ValueError(
                "Optimizer Failed at finding an optimal solution. Check the parameter constraints or try with a different optimization method."
            )
        self._functional.set_parameters(optimal_parameter_dict)
        optimal_results = backend.execute(self._functional, readout)

        # Report the clamped parameters, since those are the ones the functional was left with
        optimizer_result = OptimizerResult(
            optimal_cost=optimizer_result.optimal_cost,
            optimal_parameters=list(optimal_parameter_dict.values()),
            intermediate_results=optimizer_result.intermediate_results,
        )

        logger.info("[VariationalProgram] Variational Program finished")
        return VariationalProgramResult(optimizer_result=optimizer_result, result=optimal_results)
