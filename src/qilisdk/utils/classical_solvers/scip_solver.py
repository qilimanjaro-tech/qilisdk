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

from typing import Any, TypeAlias

from pyscipopt import Expr
from pyscipopt import Model as ScipModel
from pyscipopt.recipes.nonlinear import set_nonlinear_objective

from qilisdk.core import Model
from qilisdk.core.comparison import ComparisonOperation
from qilisdk.core.expression import Add, Constant, Expression, Mul, Pow
from qilisdk.core.model import ObjectiveSense
from qilisdk.core.variables import BaseVariable, BinaryVariable, Domain, Number, RealNumber, SpinVariable, Variable

from .base_solver import ClassicalSolver, _assert_real, _variable_bounds

# What SCIP accepts as an operand: one of its expressions (``Variable`` is an ``Expr``) or a plain number.
ScipExpr: TypeAlias = Expr | float


def _expression_to_scip_expr(expression: Expression, var_exprs: dict[BaseVariable, ScipExpr]) -> ScipExpr:
    """
    Convert a QiliSDK :class:`~qilisdk.core.Expression` to a SCIP expression, using the provided
    mapping of model variables to SCIP expressions.

    Args:
        expression: The QiliSDK ``Expression`` to convert.
        var_exprs: A dictionary mapping each model variable to its corresponding SCIP expression.

    Returns:
        The corresponding SCIP expression.

    Raises:
        ValueError: if the expression contains a node SCIP cannot represent here.
    """
    if isinstance(expression, Constant):
        return _assert_real(expression.value)
    if isinstance(expression, BaseVariable):
        return var_exprs[expression]
    if isinstance(expression, Add):
        total: ScipExpr = 0.0
        for arg in expression.args:
            total += _expression_to_scip_expr(arg, var_exprs)
        return total
    if isinstance(expression, Mul):
        product: ScipExpr = 1.0
        for arg in expression.args:
            product *= _expression_to_scip_expr(arg, var_exprs)
        return product
    if isinstance(expression, Pow):
        if not isinstance(expression.exp, Constant):
            raise ValueError(f"The symbolic exponent in {expression} is not supported by the SCIP solver.")
        return _expression_to_scip_expr(expression.base, var_exprs) ** _assert_real(expression.exp.value)
    raise ValueError(f"The expression {expression} is not supported by the SCIP solver.")


def _decode_scip_value(variable: BaseVariable, value: float) -> RealNumber:
    """
    Decode a SCIP variable value back to the corresponding QiliSDK variable value.

    Args:
        variable: The QiliSDK variable.
        value: The SCIP variable value.

    Returns:
        The corresponding QiliSDK variable value.
    """
    if isinstance(variable, SpinVariable):
        return 1 if round(value) >= 1 else -1
    if variable.domain is Domain.REAL:
        return value
    return round(value)


class ScipSolver(ClassicalSolver):
    """Classical solver that uses `SCIP <https://www.scipopt.org/>`__ (via ``pyscipopt``).

    This requires the optional ``pyscipopt`` dependency (``pip install qilisdk[scip]``).

    Example:
        .. code-block:: python

            from qilisdk.core import Model
            from qilisdk.utils.classical_solvers import ScipSolver

            model = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
            results, sample = ScipSolver().solve(model)
    """

    def solve(  # ruff: ignore[no-self-use]
        self,
        model: Model,
        verbose: bool = False,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Number], dict[BaseVariable, RealNumber]]:
        """Solve the given model to global optimality with SCIP.

        Args:
            model: The ``Model`` instance to solve.
            verbose (bool, optional): If ``False`` (the default) SCIP's solver output is hidden.
            params (dict[str, Any] | None, optional): SCIP parameters forwarded to
                ``pyscipopt.Model.setParams`` (e.g. ``{"limits/time": 60}``).

        Returns:
            tuple[dict[str, Number], dict[BaseVariable, RealNumber]]: a tuple of
            (results dict mapping objective/constraint labels to their evaluated values,
            sample dict mapping each variable to its value in the optimal solution).

        Raises:
            ValueError: if the model contains an unsupported variable, uses an unsupported
                operation, or if SCIP finds no feasible solution.
        """
        scip_model = ScipModel(model.label)

        # Build a SCIP variable (and its algebraic expression) for every model variable.
        scip_vars: dict[BaseVariable, Any] = {}
        var_exprs: dict[BaseVariable, ScipExpr] = {}
        for v in model.variables():
            if isinstance(v, BinaryVariable):
                scip_var = scip_model.addVar(name=v.label, vtype="B")
                var_exprs[v] = scip_var
            elif isinstance(v, SpinVariable):
                scip_var = scip_model.addVar(name=v.label, vtype="B")
                var_exprs[v] = 2 * scip_var - 1
            elif isinstance(v, Variable):
                lower, upper = _variable_bounds(v)
                vtype = "C" if v.domain is Domain.REAL else "I"
                scip_var = scip_model.addVar(name=v.label, vtype=vtype, lb=lower, ub=upper)
                var_exprs[v] = scip_var
            else:
                raise ValueError(f"SCIP solving is not supported for variable {v} of domain {v.domain}.")
            scip_vars[v] = scip_var

        # Translate the objective, keeping the model's optimization sense
        sense = "maximize" if model.objective.sense is ObjectiveSense.MAXIMIZE else "minimize"
        objective_expr = _expression_to_scip_expr(model.objective.term, var_exprs)
        try:
            scip_model.setObjective(objective_expr, sense=sense)
        except ValueError:
            set_nonlinear_objective(scip_model, objective_expr, sense=sense)

        # Add each constraint as a hard constraint.
        for constraint in model.constraints:
            lhs = _expression_to_scip_expr(constraint.term.lhs, var_exprs)
            rhs = _expression_to_scip_expr(constraint.term.rhs, var_exprs)
            operation = constraint.term.operation
            if operation in {ComparisonOperation.LEQ, ComparisonOperation.LT}:
                scip_model.addCons(lhs <= rhs)
            elif operation in {ComparisonOperation.GEQ, ComparisonOperation.GT}:
                scip_model.addCons(lhs >= rhs)
            elif operation is ComparisonOperation.EQ:
                scip_model.addCons(lhs == rhs)
            else:
                raise ValueError(f"Constraint operation {operation.value} is not supported by the SCIP solver.")

        # Pass settings to SCIP
        if not verbose:
            scip_model.hideOutput()
        if params:
            scip_model.setParams(params)

        # Solve the model
        scip_model.optimize()
        if scip_model.getNSols() == 0:
            raise ValueError(f"SCIP found no feasible solution (status: {scip_model.getStatus()}).")

        # Extract the best solution and return it
        solution = scip_model.getBestSol()
        best_sample = {v: _decode_scip_value(v, solution[scip_var]) for v, scip_var in scip_vars.items()}
        return model.evaluate(best_sample), best_sample
