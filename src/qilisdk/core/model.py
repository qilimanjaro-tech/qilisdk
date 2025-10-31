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

# import numpy as np
import copy
from enum import Enum
from typing import TYPE_CHECKING, Literal, Mapping, Type

# import cupy as np
import numpy as np
from loguru import logger

from qilisdk.yaml import yaml

from .variables import (
    GEQ,
    LEQ,
    BaseVariable,
    Bitwise,
    ComparisonOperation,
    ComparisonTerm,
    Domain,
    Number,
    Operation,
    RealNumber,
    Term,
    Variable,
)

if TYPE_CHECKING:
    from ruamel.yaml.nodes import ScalarNode
    from ruamel.yaml.representer import RoundTripRepresenter

    from qilisdk.analog.hamiltonian import Hamiltonian


class SlackCounter:
    """A singleton class to generate a slack counter id that increments continuously within the user's active session."""

    _instance: SlackCounter | None = None
    _count: int = 0

    def __new__(cls: Type[SlackCounter]) -> SlackCounter:  # noqa: PYI034
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def next(self) -> int:
        """Return the next counter value and increment the counter."""
        value = self._count
        self._count += 1
        return value

    def reset_counter(self) -> None:
        self._count = 0


@yaml.register_class
class ObjectiveSense(str, Enum):
    """An Enumeration of the Objective sense options."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @classmethod
    def to_yaml(cls, representer: RoundTripRepresenter, node: ObjectiveSense) -> ScalarNode:
        """
        Method to be called automatically during YAML serialization.

        Returns:
            ScalarNode: The YAML scalar node representing the ObjectiveSense.
        """
        return representer.represent_scalar("!ObjectiveSense", f"{node.value}")

    @classmethod
    def from_yaml(cls, _, node: ScalarNode) -> ObjectiveSense:
        """
        Method to be called automatically during YAML deserialization.

        Returns:
            ObjectiveSense: The ObjectiveSense instance created from the YAML node value.
        """
        return cls(node.value)


@yaml.register_class
class Constraint:
    """
    Represent a symbolic constraint inside a ``Model``.

    Example:
        .. code-block:: python

            from qilisdk.core.model import Constraint
            from qilisdk.core.variables import BinaryVariable, LEQ

            x = BinaryVariable("x")
            constraint = Constraint("limit", LEQ(x, 1))
    """

    def __init__(self, label: str, term: ComparisonTerm) -> None:
        """
        Build a constraint defined by a comparison term such as ``x + y <= 2``.

        Args:
            label (str): The constraint's label.
            term (ComparisonTerm): The comparison term that defines the constraint.

        Raises:
            ValueError: if the term provided is not a ConstraintTerm.
        """
        self._label = label
        if not isinstance(term, ComparisonTerm):
            raise ValueError(f"the parameter term is expecting a {ComparisonTerm} but received {term.__class__}")

        self._term = term

    @property
    def label(self) -> str:
        """
        Returns:
            str: The label of the constraint object.
        """
        return self._label

    @property
    def term(self) -> ComparisonTerm:
        """
        Returns:
            ComparisonTerm: The comparison term of the constraint object.
        """
        return self._term

    def variables(self) -> list[BaseVariable]:
        """
        Returns the list of variables in the constraint term.

        :rtype: list[BaseVariable]
        Returns:
            list[BaseVariable]: the list of variables in the constraint term.
        """
        return self._term.variables()

    @property
    def lhs(self) -> Term:
        """
        Returns:
            Term: The left hand side of the constraint term.
        """
        return self.term.lhs

    @property
    def rhs(self) -> Term:
        """
        Returns:
            Term: The right hand side of the constraint term.
        """
        return self.term.rhs

    @property
    def degree(self) -> int:
        """
        Returns:
            int: The degree of the constraint term.
        """
        return max(self.lhs.degree, self.rhs.degree)

    def __copy__(self) -> Constraint:
        return Constraint(label=self.label, term=copy.copy(self.term))

    def __repr__(self) -> str:
        return f"{self.label}: {self.term}"

    def __str__(self) -> str:
        return f"{self.label}: {self.term}"


@yaml.register_class
class Objective:
    """
    Represent the scalar objective function optimized by a ``Model``.

    Example:
        .. code-block:: python

            from qilisdk.core.model import Objective, ObjectiveSense
            from qilisdk.core.variables import BinaryVariable

            x = BinaryVariable("x")
            obj = Objective("profit", 3 * x, sense=ObjectiveSense.MAXIMIZE)
    """

    def __init__(self, label: str, term: BaseVariable | Term, sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        """
        Build a new objective function.

        Args:
            label (str): Objective label.
            term (BaseVariable | Term): Expression to minimize or maximize.
            sense (ObjectiveSense, optional): Optimization sense. Defaults to ``ObjectiveSense.MINIMIZE``.

        Raises:
            ValueError: if the term provided is not a Term Object.
            ValueError: if the optimization sense provided is not one that is defined by the ObjectiveSense Enum.
        """
        if isinstance(term, Variable):
            term = Term(elements=[term], operation=Operation.ADD)
        if not isinstance(term, Term):
            raise ValueError(f"the parameter term is expecting a {Term} but received {term.__class__}")
        if not isinstance(sense, ObjectiveSense):
            raise ValueError(f"the objective sense is expecting a {ObjectiveSense} but received {sense.__class__}")
        self._term = term
        self._label = label
        self._sense = sense

    @property
    def label(self) -> str:
        """
        Returns:
            str: the label of the objective.
        """
        return self._label

    @property
    def term(self) -> Term:
        """
        Returns:
            Term: the objective term.
        """
        return self._term

    @property
    def sense(self) -> ObjectiveSense:
        """
        Returns:
            ObjectiveSense: the objective optimization sense.
        """
        return self._sense

    def variables(self) -> list[BaseVariable]:
        """Gathers a list of all the variables in the objective term.

        Returns:
            list[BaseVariable]: the list of variables in the objective term.
        """
        return self._term.variables()

    def __repr__(self) -> str:
        return f"{self.label}: {self.term}"

    def __str__(self) -> str:
        return f"{self.label}: {self.term}"

    def __copy__(self) -> Objective:
        return Objective(label=self.label, term=copy.copy(self.term), sense=self.sense)


@yaml.register_class
class Model:
    """
    Aggregate an objective and constraints into an optimization problem.

    Example:
        .. code-block:: python

            from qilisdk.core import BinaryVariable, LEQ, Model

            num_items = 4
            values = [1, 3, 5, 2]
            weights = [3, 2, 4, 5]
            max_weight = 6
            bin_vars = [BinaryVariable(f"b{i}") for i in range(num_items)]
            model = Model("Knapsack")
            objective = sum(values[i] * bin_vars[i] for i in range(num_items))
            model.set_objective(objective)
            constraint = LEQ(sum(weights[i] * bin_vars[i] for i in range(num_items)), max_weight)
            model.add_constraint("maximum weight", constraint)

            print(model)
    """

    def __init__(self, label: str) -> None:
        """
        Args:
            label (str): Model label.
        """
        self._constraints: dict[str, Constraint] = {}
        self._encoding_constraints: dict[str, Constraint] = {}
        self._lagrange_multipliers: dict[str, float] = {}
        self._objective = Objective("objective", Term([0], Operation.ADD))
        self._label = label

    @property
    def lagrange_multipliers(self) -> dict[str, float]:
        return self._lagrange_multipliers

    def set_lagrange_multiplier(self, constraint_label: str, lagrange_multiplier: float) -> None:
        """Sets the lagrange multiplier value for a given constraint.

        Args:
            constraint_label (str): the constraint to which the lagrange multiplier value corresponds.
            lagrange_multiplier (float): the lagrange multiplier value.

        Raises:
            ValueError: if the constraint provided is not in the model.
        """
        if constraint_label not in self._lagrange_multipliers:
            raise ValueError(f'constraint "{constraint_label}" not in model.')
        self.lagrange_multipliers[constraint_label] = lagrange_multiplier

    @property
    def label(self) -> str:
        """
        Returns:
            str: The model label.
        """
        return self._label

    @property
    def constraints(self) -> list[Constraint]:
        """
        Returns:
            list[Constraint]: a list of all the constraints in the model.
        """
        return list(self._constraints.values())

    @property
    def encoding_constraints(self) -> list[Constraint]:
        """
        Returns:
            list[Constraint]: a list of all variable encoding constraints in the model.
        """
        return list(self._encoding_constraints.values())

    @property
    def objective(self) -> Objective:
        """
        Returns:
            Objective: The objective of the model.
        """
        return self._objective

    def variables(self) -> list[BaseVariable]:
        """
        Returns:
            list[BaseVariable]: a list of variables that are used in the model whether that is in the constraints
            or the objective.
        """
        var = set()

        for c in self.constraints:
            var.update(c.variables())

        var.update(self.objective.variables())

        return sorted(var, key=lambda x: x.label)

    def _generate_encoding_constraints(
        self,
        lagrange_multiplier: float = 100,
    ) -> None:
        for var in self.variables():
            if not isinstance(var, Variable) or var.domain in {Domain.BINARY, Domain.SPIN}:
                continue
            ub_encoding_name = f"{var}_upper_bound_constraint"
            lb_encoding_name = f"{var}_lower_bound_constraint"
            if ub_encoding_name not in self._encoding_constraints:
                self._encoding_constraints[ub_encoding_name] = Constraint(
                    label=ub_encoding_name, term=LEQ(var, var.upper_bound)
                )
                self._lagrange_multipliers[ub_encoding_name] = lagrange_multiplier
            if lb_encoding_name not in self._encoding_constraints:
                self._encoding_constraints[lb_encoding_name] = Constraint(
                    label=lb_encoding_name, term=GEQ(var, var.lower_bound)
                )
                self._lagrange_multipliers[lb_encoding_name] = lagrange_multiplier

    def __str__(self) -> str:
        output = f"Model name: {self.label} \n"
        if self.objective is not None:
            output += (
                f"objective ({self.objective.label}):"
                + f" \n\t {self.objective.sense.value} : \n\t {self.objective.term} \n\n"
            )
        if len(self.constraints) > 0:
            output += "subject to the constraint/s: \n"
            for c in self.constraints:
                output += f"\t {c} \n"
            output += "\n"

        if len(self.encoding_constraints) > 0:
            output += "subject to the encoding constraint/s: \n"
            for c in self.encoding_constraints:
                output += f"\t {c} \n"
            output += "\n"

        if len(self.lagrange_multipliers) > 0:
            output += "With Lagrange Multiplier/s: \n"
            for key, value in self.lagrange_multipliers.items():
                output += f"\t {key} : {value} \n"
        return output

    def __repr__(self) -> str:
        return self.label

    def __copy__(self) -> Model:
        out = Model(label=self.label)
        obj = copy.copy(self.objective)
        out.set_objective(term=obj.term, label=obj.label, sense=obj.sense)
        for c in self.constraints:
            out.add_constraint(label=c.label, term=copy.copy(c.term))
        return out

    def add_constraint(
        self,
        label: str,
        term: ComparisonTerm,
        lagrange_multiplier: float = 100,
    ) -> None:
        """Add a constraint to the model.

        Args:
            label (str): constraint label.
            term (ComparisonTerm): The constraint's comparison term.

        Raises:
            ValueError: if the constraint label is already used in the model.
        """
        if label in self._constraints:
            raise ValueError((f'Constraint "{label}" already exists:\n \t\t{self._constraints[label]}'))
        c = Constraint(label=label, term=copy.copy(term))
        self._constraints[label] = c
        self._lagrange_multipliers[label] = lagrange_multiplier
        self._generate_encoding_constraints(lagrange_multiplier=lagrange_multiplier)

    def set_objective(self, term: Term, label: str = "obj", sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        """Sets the model's objective.

        Args:
            term (Term): the objective term.
            label (str, optional): the objective's label. Defaults to "obj".
            sense (ObjectiveSense, optional): The optimization sense of the model's objective.
                                                Defaults to ObjectiveSense.MINIMIZE.
        """
        self._objective = Objective(label=label, term=copy.copy(term), sense=sense)
        self._generate_encoding_constraints()

    def evaluate(self, sample: Mapping[BaseVariable, RealNumber | list[int]]) -> dict[str, Number]:
        """Evaluates the objective and the constraints of the model given a set of values for the variables.

        Args:
            sample (Mapping[BaseVariable, Number  |  list[int]]): The dictionary maps the variable to the value to be
                                                                used during the evaluation. In case the variable is
                                                                continuous (Not Binary or Spin) then the value could
                                                                either be a number or a list of binary bits that
                                                                correspond to the encoding of the variable.
                                                                Note: All the model's variables must be provided for
                                                                the model to be evaluated.

        Returns:
            dict[str, float]: a dictionary that maps the name of the objective/constraint to it's evaluated value.
                            Note: For constraints, the value is equal to lagrange multiplier of that constraint if
                            the constraint is not satisfied or 0 otherwise.
        """
        results = {}

        results[self.objective.label] = self.objective.term.evaluate(sample)
        results[self.objective.label] *= -1 if self.objective.sense is ObjectiveSense.MAXIMIZE else 1

        for c in self.constraints:
            results[c.label] = float(not c.term.evaluate(sample)) * self.lagrange_multipliers[c.label]
        return results

    def to_qubo(
        self,
        lagrange_multiplier_dict: dict[str, float] | None = None,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
    ) -> QUBO:
        """Export the model to a qubo model.
        Args:
            lagrange_multiplier_dict (dict[str, float] | None, optional): A dictionary with lagrange multiplier values to scale the model's constraints. Defaults to None.
            penalization (Literal[&quot;unbalanced&quot;, &quot;slack&quot;], optional): the penalization used to handel inequality constraints. Defaults to "slack".
            parameters (list[float] | None, optional): the parameters used for the unbalanced penalization method. Defaults to None.

        Note:
            this exportation only works if the model doesn't violate the QUBO format.
            Automatic constraint and objective linearization will be added in the future.
        Returns:
            QUBO: A QUBO model that is generate from the model object.
        """
        if lagrange_multiplier_dict is None:
            lagrange_multiplier_dict = {}
        for lm in self.lagrange_multipliers:
            if lm not in lagrange_multiplier_dict:
                lagrange_multiplier_dict[lm] = self.lagrange_multipliers[lm]
        return QUBO.from_model(self, lagrange_multiplier_dict, penalization, parameters)


@yaml.register_class
class QUBO(Model):
    """
    Specialized ``Model`` constrained to Quadratic Unconstrained Binary Optimization form.

    Example:
        .. code-block:: python

            from qilisdk.core.model import QUBO
            from qilisdk.core.variables import BinaryVariable

            x0, x1 = BinaryVariable("x0"), BinaryVariable("x1")
            qubo = QUBO("Example")
            qubo.set_objective((x0 + x1) ** 2)
    """

    def __init__(self, label: str) -> None:
        """
        Args:
            label (str): QUBO model label.
        """
        super().__init__(label)
        self.continuous_vars: dict[str, Variable] = {}
        self.__qubo_objective: Objective | None = None

    @property
    def qubo_objective(self) -> Objective | None:
        """
        Returns:
            Objective | None: The QUBO objective (factoring in the constraints and objective of the model). If the objective and constraints are not defined in the model, this property returns None.
        """
        self.__qubo_objective = None
        if self.objective is not None:
            self._build_qubo_objective(self.objective.term, self.objective.label, self.objective.sense)
        for constraint in self.constraints:
            if constraint.label in self.lagrange_multipliers:
                self._build_qubo_objective(
                    constraint.term.lhs * self.lagrange_multipliers[constraint.label]
                    - constraint.term.rhs * self.lagrange_multipliers[constraint.label]
                )
            else:
                self._build_qubo_objective(
                    constraint.term.lhs - constraint.term.rhs
                )  # I don't think this line can be reached.
        return self.__qubo_objective

    def __repr__(self) -> str:
        return self.label

    def _parse_term(self, term: Term) -> tuple[Number, list[tuple[Number, BaseVariable]]]:
        """parses a Term object into a list of variables and coefficients.

        Args:
            term (Term): The term to be parsed.

        Raises:
            ValueError: if the degree of the term is higher than 1.

        Returns:
            tuple[Number, list[tuple[Number, BaseVariable]]]:
                The first return value is the constant value in the term.
                The second return value is a list of variables and their respective coefficients.
        """
        const = term.get_constant()
        terms: list[tuple[Number, BaseVariable]] = []

        if term.degree > 1:
            raise ValueError(f'QUBO constraints only allow linear terms but received "{term}" of degree {term.degree}')

        if term.operation is Operation.ADD:
            for element in term:
                if isinstance(element, Term):  # I don't think this will ever be true for a QUBO model.
                    _, aux_terms = self._parse_term(element)
                    terms.extend(aux_terms)
                elif element != Term.CONST:
                    terms.append((term[element], element))
        if term.operation is Operation.MUL:
            for element in term:
                if not isinstance(element, Term) and element != Term.CONST:
                    terms.append((1, element))
        return const, terms

    def _check_valid_constraint(self, label: str, term: Term, operation: ComparisonOperation) -> int | None:
        """Checks if a given constraint is valid. Assumes that the right hand side of the constraint is set to zero.

        Args:
            label (str): the label of the constraint.
            term (Term): the left hand side of the constraint term.
            operation (ComparisonOperation): the comparison operation between the left and right hand sides.

        Raises:
            ValueError: if the constraint is never feasible given the variable ranges.

        Returns:
            int | None: the upper bound of the continuous slack variable needed for this given constraint.
                        None in case the constraint is always feasible.
        """
        ub = np.iinfo(np.int64).max if operation in {ComparisonOperation.GEQ, ComparisonOperation.GT} else 0
        lb = np.iinfo(np.int64).min if operation in {ComparisonOperation.LEQ, ComparisonOperation.LT} else 0
        _const, terms = self._parse_term(term)

        def to_real(num: Number) -> RealNumber:
            if isinstance(num, RealNumber):
                return num
            if isinstance(num, complex) and num.imag == 0:
                return num.real
            raise ValueError("Complex values encountered in the constraint.")

        const = to_real(_const)
        term_upper_limit: RealNumber = sum(to_real(coeff) for coeff, _ in terms if to_real(coeff) > 0)
        term_lower_limit: RealNumber = sum(to_real(coeff) for coeff, _ in terms if to_real(coeff) < 0)

        if operation == ComparisonOperation.GT and term_upper_limit + const <= 0:
            raise ValueError(f"Constraint {label} is unsatisfiable.")
        if operation == ComparisonOperation.LT and term_lower_limit + const >= 0:
            raise ValueError(f"Constraint {label} is unsatisfiable.")

        upper_cut = min(term_upper_limit, ub - const)
        lower_cut = max(term_lower_limit, lb - const)

        if term_upper_limit <= upper_cut and term_lower_limit >= lower_cut:
            logger.warning(
                f'constraint "{label}" was not added to model "{self.label}" because it is always feasible.',
            )
            return None

        ub_slack = int(upper_cut - lower_cut)

        if upper_cut < lower_cut:
            raise ValueError(f"Constraint {label} is unsatisfiable.")

        return ub_slack

    def _transform_constraint(
        self,
        label: str,
        term: ComparisonTerm,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
    ) -> Term | None:
        """Transforms a constraint into QUBO format.

        Args:
            label (str): the constraint's label.
            term (ComparisonTerm): the constraint term.
            penalization (Literal[&quot;unbalanced&quot;, &quot;slack&quot;], optional): The penalization used to
                            handel inequality constraints. Defaults to "slack".
            parameters (list[float] | None, optional): the parameters used for the unbalanced penalization method.
                            Defaults to None.

        Raises:
            ValueError: if a penalization method is provided that is not (&quot;unbalanced&quot;, &quot;slack&quot;)
            ValueError: if unbalanced penalization method is used and not enough parameters are provided.

        Returns:
            Term | None: A transformed term that is in QUBO format.
                        None if the constraint is always feasible.
        """

        lower_penalization = penalization.lower()

        if lower_penalization not in {"unbalanced", "slack"}:
            raise ValueError('Only penalization of type "unbalanced" or "slack" is supported.')

        if parameters is None:
            parameters = []

        if term.operation is ComparisonOperation.EQ:
            h = term.lhs - term.rhs
            ub_slack = self._check_valid_constraint(label, h, term.operation)
            if ub_slack is None:
                return None
            return h**2

        if term.operation in {
            ComparisonOperation.GEQ,
            ComparisonOperation.GT,
        }:
            # assuming the operation is h >= 0 or h > 0
            h = term.lhs - term.rhs
            if lower_penalization == "unbalanced":
                if len(parameters) < 2:  # noqa: PLR2004
                    raise ValueError("using unbalanced penalization requires at least 2 parameters.")
                return -parameters[0] * h + parameters[1] * (h**2)

            if lower_penalization == "slack":
                ub_slack = self._check_valid_constraint(label, h, term.operation)

                if ub_slack is None:
                    return None
                if ub_slack == 0:
                    return h**2

                slack = Variable(
                    f"{label}_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=Bitwise
                )
                slack_terms = slack.to_binary()
                out = h + slack_terms
                return (out) ** 2

        if term.operation in {
            ComparisonOperation.LEQ,
            ComparisonOperation.LT,
        }:
            if lower_penalization == "unbalanced":
                # assuming the operation is -> 0 < h  or 0 <= h
                h = term.rhs - term.lhs
                if len(parameters) < 2:  # noqa: PLR2004
                    raise ValueError("using unbalanced penalization requires at least 2 parameters.")
                return -parameters[0] * h + parameters[1] * (h**2)
            if lower_penalization == "slack":
                # assuming the operation is h <= 0 or h < 0
                h = term.lhs - term.rhs
                ub_slack = self._check_valid_constraint(label, h, term.operation)

                if ub_slack is None:
                    return None
                if ub_slack == 0:
                    return h**2

                slack = Variable(
                    f"{label}_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=Bitwise
                )

                slack_terms = slack.to_binary()
                out = h + slack_terms
                return (out) ** 2
        return None

    def add_constraint(
        self,
        label: str,
        term: ComparisonTerm,
        lagrange_multiplier: float = 100,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
        transform_to_qubo: bool = True,
    ) -> None:
        """Adds a constraint to the QUBO model.

        Args:
            label (str): the constraint label.
            term (ComparisonTerm): the constraint's comparison term.
            lagrange_multiplier (float, optional): the lagrange multiplier used to scale this constraint.
                                                    Defaults to 100.
            penalization (Literal[&quot;unbalanced&quot;, &quot;slack&quot;], optional): the penalization used to
                            handel inequality constraints. Defaults to "slack".
            parameters (list[float] | None, optional): the parameters used for the unbalanced penalization method.
                            Defaults to None.
            transform_to_qubo (bool, optional): Automatically transform a given constraint to QUBO format.
                                                Defaults to True.

        Raises:
            ValueError: if constraint label already exists in the model.
            ValueError: if a penalization method is provided that is not (&quot;unbalanced&quot;, &quot;slack&quot;)
            ValueError: if unbalanced penalization method is used and not enough parameters are provided.
            ValueError: if the degree of the provided term is larger than 2.
            ValueError: if the constraint term contains variables that are not from Positive Integers or Binary domains.
            ValueError: if the constraint term contains variable that do not have 0 as their lower bound.
        """
        if label in self._constraints:
            raise ValueError((f'Constraint "{label}" already exists:\n \t\t{self._constraints[label]}'))

        lower_penalization = penalization.lower()

        if lower_penalization not in {"unbalanced", "slack"}:
            raise ValueError(
                'Only penalization of type "unbalanced" or "slack" is supported for inequality constraints.'
            )

        if parameters is None:
            parameters = [1, 1] if lower_penalization == "unbalanced" else []

        if term.operation in {ComparisonOperation.GEQ, ComparisonOperation.GT}:
            c = ComparisonTerm(lhs=(term.lhs - term.rhs), rhs=0, operation=term.operation)
        elif term.operation in {ComparisonOperation.LEQ, ComparisonOperation.LT}:
            c = ComparisonTerm(lhs=0, rhs=(term.rhs - term.lhs), operation=term.operation)
        else:
            c = copy.copy(term)

        if c.degree > 2:  # noqa: PLR2004
            raise ValueError(
                f"QUBO models can not contain terms of order 2 or higher but received terms with degree {c.degree}."
            )

        self._check_variables(c, lagrange_multiplier=lagrange_multiplier)

        if transform_to_qubo:
            c = c.to_binary()
            transformed_c = self._transform_constraint(label, c, penalization=penalization, parameters=parameters)
            if transformed_c is None:
                return
            if lower_penalization == "unbalanced" and lagrange_multiplier != 1:
                self.lagrange_multipliers[label] = 1
                logger.warning(
                    "add_constraint() in QUBO model:"
                    + f' The Lagrange Multiplier for the constraint "{label}" in the QUBO model ({self.label})'
                    + " has been set to 1 because the constraint uses unbalanced"
                    + " penalization method."
                    + ' To customize the penalization coefficient, please use the "parameters" field.',
                )
            else:
                self.lagrange_multipliers[label] = lagrange_multiplier
            self._constraints[label] = Constraint(label, term=ComparisonTerm(transformed_c, 0, ComparisonOperation.EQ))

        else:
            self.lagrange_multipliers[label] = lagrange_multiplier
            self._constraints[label] = Constraint(label, term=c)

    def set_objective(self, term: Term, label: str = "obj", sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        """Set the QUBO objective.

        Args:
            term (Term): The objective's term.
            label (str, optional): the objective's label. Defaults to "obj".
            sense (ObjectiveSense, optional): The optimization sense of the model's objective.
                                                Defaults to ObjectiveSense.MINIMIZE.

        """

        self._check_variables(term)

        term = term.to_binary()
        self._objective = Objective(label=label, term=term, sense=sense)

    def _check_variables(self, term: Term | ComparisonTerm, lagrange_multiplier: RealNumber = 100) -> None:
        """checks if the variables in the provided term are valid to be used in a QUBO model. Moreover, we add all the
        encoding constraint for supported continuous variables.

        Args:
            term (Term): the term to be checked.

        Raises:
            ValueError: if the constraint term contains variables that are not from Positive Integers or Binary domains.
            ValueError: if the constraint term contains variable that do not have 0 as their lower bound.
        """
        for v in term.variables():
            if v.domain not in {Domain.POSITIVE_INTEGER, Domain.BINARY}:
                raise ValueError(
                    "QUBO models are not supported for variables that are not in the positive integers or binary domains."
                )
            if v.lower_bound != 0:
                raise ValueError(
                    f"All variables must have a lower bound of 0. But variable {v} has a lower bound of {v.lower_bound}"
                )
            if isinstance(v, Variable) and v.domain is Domain.POSITIVE_INTEGER and v.label not in self.continuous_vars:
                self.continuous_vars[v.label] = v
                encoding_constraint = v.encoding_constraint()
                if encoding_constraint is not None:
                    enc_label = f"{v.label}_encoding_constraint"
                    self.add_constraint(
                        label=enc_label, term=encoding_constraint, lagrange_multiplier=lagrange_multiplier
                    )

    def _build_qubo_objective(
        self, term: Term, label: str | None = None, sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    ) -> None:
        """updates the internal qubo objective term.

        Args:
            term (Term): A term to be added to the qubo objective.
            label (str | None, optional): the label of the objective (if None then the current label is maintained).
                                            Defaults to None.
            sense (ObjectiveSense, optional): The optimization sense of the model's objective.
                                                Defaults to ObjectiveSense.MINIMIZE.
        """
        term = copy.copy(term.to_binary())
        if self.__qubo_objective is None:
            self.__qubo_objective = Objective(
                label=label if label is not None else "obj",
                term=-term if sense == ObjectiveSense.MAXIMIZE else term,
                sense=ObjectiveSense.MINIMIZE,
            )
        else:
            self.__qubo_objective = Objective(
                label=label if label is not None else self.__qubo_objective.label,
                term=(
                    copy.copy(self.__qubo_objective.term) - term
                    if sense == ObjectiveSense.MAXIMIZE
                    else copy.copy(self.__qubo_objective.term) + term
                ),
                sense=ObjectiveSense.MINIMIZE,
            )

    def evaluate(self, sample: Mapping[BaseVariable, RealNumber | list[int]]) -> dict[str, Number]:
        """Evaluates the objective and the constraints of the model given a set of values for the variables.

        Args:
            sample (Mapping[BaseVariable, RealNumber  |  list[int]]): The dictionary maps the variable to the value to be
                                                                used during the evaluation. In case the variable is
                                                                continuous (Not Binary or Spin) then the value could
                                                                either be a number or a list of binary bits that
                                                                correspond to the encoding of the variable.
                                                                Note: All the model's variables must be provided for
                                                                the model to be evaluated.

        Returns:
            dict[str, float]: a dictionary that maps the name of the objective/constraint to it's evaluated value.
                            Note: For constraints, the value is equal to the value of the evaluated constraint term
                            multiplied by the lagrange multiplier of that constraint.
        """
        results = {}

        results[self.objective.label] = self.objective.term.evaluate(sample)
        results[self.objective.label] *= -1 if self.objective.sense is ObjectiveSense.MAXIMIZE else 1

        for c in self.constraints:
            results[c.label] = c.term.lhs.evaluate(sample) - c.term.rhs.evaluate(sample)
            results[c.label] *= self.lagrange_multipliers[c.label]
        return results

    @classmethod
    def from_model(
        cls,
        model: Model,
        lagrange_multiplier_dict: dict[str, float] | None = None,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
    ) -> QUBO:
        """A class method that constructs a QUBO model from a regular model if possible.

        Args:
            model (Model): the model to be used to construct the QUBO model.
            lagrange_multiplier_dict (dict[str, float] | None, optional): A dictionary with lagrange multiplier values
                                    to scale the model's constraints. Defaults to None.
            penalization (Literal[&quot;unbalanced&quot;, &quot;slack&quot;], optional): the penalization used to
                            handel inequality constraints. Defaults to "slack".
            parameters (list[float] | None, optional): the parameters used for the unbalanced penalization method.
                            Defaults to None.
        Returns:
            QUBO: _description_
        """
        instance = QUBO(label="QUBO_" + model.label)
        instance.set_objective(term=model.objective.term, label=model.objective.label, sense=model.objective.sense)
        for constraint in model.constraints:
            if lagrange_multiplier_dict is not None and constraint.label in lagrange_multiplier_dict:
                lagrange_multiplier = lagrange_multiplier_dict[constraint.label]

            else:
                lagrange_multiplier = 100

            instance.add_constraint(
                label=constraint.label,
                term=constraint.term,
                lagrange_multiplier=lagrange_multiplier,
                penalization=penalization,
                parameters=parameters,
            )
        return instance

    def to_hamiltonian(self) -> Hamiltonian:
        """Construct an ising hamiltonian from the current QUBO model.

        Raises:
            ValueError: if the QUBO model is empty (doesn't have an objective nor constraints.)
            ValueError: if the QUBO model uses operations that are not addition or multiplications.

        Returns:
            Hamiltonian: An ising hamiltonian that represents the QUBO model.
        """
        from qilisdk.analog.hamiltonian import Hamiltonian, Z  # noqa: PLC0415

        spins: dict[BaseVariable, Hamiltonian] = {}
        obj = self.qubo_objective

        if obj is None:
            raise ValueError("Can't transform empty QUBO model to a Hamiltonian.")

        for i, v in enumerate(obj.variables()):
            spins[v] = (1 - Z(i)) / 2

        def _parse_term(term: Term) -> Hamiltonian:
            ham = Hamiltonian()
            terms = term.to_list()
            operation = term.operation
            default = 0.0 if operation is Operation.ADD else 1.0
            aux_term: Number | Hamiltonian = copy.copy(default)
            for t in terms:
                aux: Number | Hamiltonian = copy.copy(default)
                if isinstance(t, Term):
                    aux = _parse_term(t)
                elif isinstance(t, Number):
                    aux = t
                elif isinstance(t, BaseVariable):
                    aux = spins[t]

                if operation is Operation.ADD:
                    aux_term += aux
                elif operation is Operation.MUL:
                    aux_term *= aux
                else:  # I don't think this can be reached.
                    raise ValueError(f"operation {operation} is not supported")
            ham += aux_term
            return ham

        ham = _parse_term(obj.term)

        return ham

    def to_qubo(
        self,
        lagrange_multiplier_dict: dict[str, float] | None = None,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
    ) -> QUBO:
        logger.warning(
            f"Running `to_qubo()` on the model {self.label} that is already in QUBO format.",
        )
        return copy.copy(self)

    def __copy__(self) -> QUBO:
        out = QUBO(label=self.label)
        obj = copy.copy(self.objective)
        out.set_objective(term=obj.term, label=obj.label, sense=obj.sense)
        for c in self.constraints:
            # THIS DOESN'T COPY ANY PARAMETERS ATTACHED TO A CONSTRAINT
            out.add_constraint(
                label=c.label,
                term=copy.copy(c.term),
                lagrange_multiplier=self.lagrange_multipliers[c.label],
                transform_to_qubo=False,
            )
        return out
