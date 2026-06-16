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

import copy
import itertools
from typing import TYPE_CHECKING, Literal, Mapping, Type

import numpy as np
from loguru import logger

from qilisdk.settings import get_settings
from qilisdk.yaml import yaml

from .types import QiliEnum
from .variables import (
    EQ,
    GEQ,
    LEQ,
    BaseVariable,
    BinaryVariable,
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
    from qilisdk.analog.hamiltonian import Hamiltonian


_EMPTY_GRAPH_MSG = "The graph must have at least one edge."


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
class ObjectiveSense(QiliEnum):
    """An Enumeration of the Objective sense options."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


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
        linearize: bool = True,
        linearization_lagrange_multiplier: float = 100,
    ) -> QUBO:
        """Export the model to a qubo model.

        When ``linearize`` is ``True``, any pseudo-Boolean monomial of degree greater than two, coming either from the
        objective or from the squared/slack penalty of a constraint, is automatically rewritten to quadratic form by
        introducing auxiliary binary variables and corresponding Rosenberg penalty constraints.
        See :meth:`QUBO.from_model` for details on the reduction scheme.

        Args:
            lagrange_multiplier_dict (dict[str, float] | None, optional): A dictionary with lagrange multiplier values
                            to scale the model's constraints. Defaults to None.
            penalization (Literal[&quot;unbalanced&quot;, &quot;slack&quot;], optional): the penalization used to handle
                            inequality constraints. Defaults to "slack".
            parameters (list[float] | None, optional): the parameters used for the unbalanced penalization method.
                            Defaults to None.
            linearize (bool, optional): Automatically reduce high-degree pseudo-Boolean monomials to quadratic form by
                            introducing auxiliary binary variables. When ``False``, exporting a model whose objective or
                            constraints contain terms of degree three or higher raises a ``ValueError``.
                            Defaults to ``True``.
            linearization_lagrange_multiplier (float, optional): The Lagrange multiplier applied to each Rosenberg
                            penalty constraint added during linearization. Defaults to 100.

        Returns:
            QUBO: A QUBO model that is generated from the model object.
        """
        if lagrange_multiplier_dict is None:
            lagrange_multiplier_dict = {}
        for lm in self.lagrange_multipliers:
            if lm not in lagrange_multiplier_dict:
                lagrange_multiplier_dict[lm] = self.lagrange_multipliers[lm]
        return QUBO.from_model(
            self,
            lagrange_multiplier_dict,
            penalization,
            parameters,
            linearize=linearize,
            linearization_lagrange_multiplier=linearization_lagrange_multiplier,
        )

    @classmethod
    def knapsack(
        cls,
        values: list[float],
        weights: list[float],
        max_weight: float,
        label: str = "Knapsack",
        lagrange_multiplier: float = 100,
    ) -> Model:
        """Factory method to generate a knapsack model.

        Binary variable ``b_i = 1`` if item *i* is selected. The objective
        maximises total value subject to a single weight inequality:

        .. math::

            \\text{maximise} \\quad \\sum_i v_i b_i

            \\text{subject to} \\quad \\sum_i w_i b_i \\leq W

        Args:
            values (list[float]): the value of each item.
            weights (list[float]): the non-negative weight of each item.
            max_weight (float): the maximum weight the knapsack can carry.
            label (str, optional): the model label. Defaults to "Knapsack".
            lagrange_multiplier (float, optional): penalty scale for the weight
                constraint when converting to QUBO. Defaults to 100.

        Returns:
            Model: a model of the knapsack problem with the given parameters.

        Raises:
            ValueError: if the number of values and weights differ.
            ValueError: if the number of items is zero.
        """
        num_items = len(values)
        if len(weights) != num_items:
            raise ValueError("The number of weights must be equal to the number of values.")
        if num_items == 0:
            raise ValueError("The number of items must be greater than zero.")

        x = [BinaryVariable(f"b{i}") for i in range(num_items)]
        model = cls(label)
        obj = sum(values[i] * x[i] for i in range(num_items))
        if not isinstance(obj, Term):
            raise ValueError("The objective term is empty. Please provide a non-empty list of values.")
        model.set_objective(
            obj,
            sense=ObjectiveSense.MAXIMIZE,
        )
        model.add_constraint(
            "weight",
            LEQ(sum(weights[i] * x[i] for i in range(num_items)), max_weight),
            lagrange_multiplier=lagrange_multiplier,
        )
        return model

    @classmethod
    def random_ising(
        cls,
        num_variables: int,
        coefficient_range: tuple[float, float] = (-1, 1),
        label: str = "Random Ising",
        seed: int = 1,
    ) -> Model:
        """Factory method to generate a random Ising model.

        Args:
            num_variables (int): the number of variables in the Ising model.
            coefficient_range (tuple[float, float], optional): the range from which the coefficients of the Ising model are drawn uniformly at random. Defaults to (-1, 1).
            label (str, optional): the model label. Defaults to "Random Ising".
            seed (int, optional): the seed for the random number generator. Defaults to 1.

        Returns:
            Model: a model of a random Ising problem with the given parameters.
        """
        model = cls(label)
        variables = [BinaryVariable(f"x{i}") for i in range(num_variables)]
        generator = np.random.default_rng(seed)
        term = Term([0], Operation.ADD)
        for i in range(num_variables):
            term += generator.uniform(low=coefficient_range[0], high=coefficient_range[1]) * variables[i]
            for j in range(i + 1, num_variables):
                term += (
                    generator.uniform(low=coefficient_range[0], high=coefficient_range[1]) * variables[i] * variables[j]
                )
        model.set_objective(term)
        return model

    @classmethod
    def factoring(
        cls,
        number: int,
        label: str = "Factoring",
        lagrange_multiplier: float = 100,
    ) -> Model:
        """Factory method to generate a factoring model.

        Binary variables ``x_i`` and ``y_j`` encode two factors whose product
        must equal *number*. The problem is a constraint satisfaction instance
        with no objective:

        .. math::

            \\text{subject to} \\quad \\sum_{i,j} 2^{i+j}\\, x_i y_j = N

        Args:
            number (int): the number to factor.
            label (str, optional): the model label. Defaults to "Factoring".
            lagrange_multiplier (float, optional): penalty scale for the
                factoring constraint when converting to QUBO. Defaults to 100.

        Returns:
            Model: a model of the factoring problem for the given number.
        """
        model = cls(label)
        num_bits = (number // 2).bit_length()
        x = [BinaryVariable(f"x{i}") for i in range(num_bits)]
        y = [BinaryVariable(f"y{i}") for i in range(num_bits)]
        product = Term([0], Operation.ADD)
        for i in range(num_bits):
            for j in range(num_bits):
                product += (2 ** (i + j)) * x[i] * y[j]
        model.add_constraint("factoring", EQ(product, number), lagrange_multiplier=lagrange_multiplier)
        return model

    @classmethod
    def max_cut(
        cls,
        edges: list[tuple[int, int]],
        weights: list[float] | None = None,
        label: str = "Max-Cut",
    ) -> Model:
        """Factory method to generate a max-cut model.

        Args:
            edges (list[tuple[int, int]]): the edges of the graph as ``(u, v)`` pairs.
            weights (list[float] | None, optional): a weight for each edge. Defaults to 1 for all edges.
            label (str, optional): the model label. Defaults to "Max-Cut".

        Returns:
            Model: a model of the max-cut problem for the given graph.

        Raises:
            ValueError: if weights are provided and their number is different from the number of edges.
        """
        if weights is not None and len(weights) != len(edges):
            raise ValueError("the number of weights must be equal to the number of edges.")
        nodes = sorted({n for u, v in edges for n in (u, v)})
        x = {n: BinaryVariable(f"x{n}") for n in nodes}
        model = cls(label)
        objective = sum(
            (1 if weights is None else weights[i]) * (x[u] + x[v] - 2 * x[u] * x[v]) for i, (u, v) in enumerate(edges)
        )
        if isinstance(objective, Number):
            raise ValueError(_EMPTY_GRAPH_MSG)
        model.set_objective(objective, sense=ObjectiveSense.MAXIMIZE)
        return model

    @classmethod
    def graph_coloring(
        cls,
        edges: list[tuple[int, int]],
        num_colors: int,
        label: str = "Graph Coloring",
        lagrange_multiplier: float = 100,
    ) -> Model:
        """Factory method to generate a graph coloring model.

        Binary variable ``x_{v,k} = 1`` if vertex *v* has color *k*. Each vertex
        must have exactly one color (an equality constraint), while the objective
        minimises the number of edges whose endpoints share a color:

        .. math::

            \\text{minimise} \\quad \\sum_{(u,v)\\in E} \\sum_k x_{u,k}\\, x_{v,k}

            \\text{subject to} \\quad \\sum_k x_{v,k} = 1 \\quad \\forall v

        A valid ``num_colors``-coloring exists if and only if the optimal
        objective value is zero.

        Args:
            edges (list[tuple[int, int]]): the edges of the graph as ``(u, v)`` pairs.
            num_colors (int): the number of colors available.
            label (str, optional): the model label. Defaults to "Graph Coloring".
            lagrange_multiplier (float, optional): penalty scale for the one-color
                constraints when converting to QUBO. Defaults to 100.

        Returns:
            Model: a model of the graph coloring problem for the given graph.
        """
        nodes = sorted({n for u, v in edges for n in (u, v)})
        x = {(n, k): BinaryVariable(f"x{n}_{k}") for n in nodes for k in range(num_colors)}
        model = cls(label)

        for v in nodes:
            model.add_constraint(
                f"vertex_{v}_one_color",
                EQ(sum(x[v, k] for k in range(num_colors)), 1),
                lagrange_multiplier=lagrange_multiplier,
            )

        conflict_terms = [x[u, k] * x[v, k] for u, v in edges for k in range(num_colors)]
        if conflict_terms:
            objective = sum(conflict_terms)
            if not isinstance(objective, Number):
                model.set_objective(objective, sense=ObjectiveSense.MINIMIZE)

        return model

    @classmethod
    def travelling_salesman(
        cls,
        edges: list[tuple[int, int]],
        distances: list[float],
        label: str = "Travelling Salesman",
        lagrange_multiplier: float = 100,
    ) -> Model:
        """Factory method to generate a travelling salesman model.

        Binary variable ``x_{i,t} = 1`` if city *i* is at tour position *t*.
        The objective minimises total travel distance; two sets of equality
        constraints enforce a valid tour:

        .. math::

            \\text{minimise} \\quad
            \\sum_{(u,v)\\in E} W_{uv} \\sum_t \\bigl(x_{u,t}\\,x_{v,t+1} + x_{v,t}\\,x_{u,t+1}\\bigr)

            \\text{subject to} \\quad \\sum_t x_{i,t} = 1 \\quad \\forall i

            \\sum_i x_{i,t} = 1 \\quad \\forall t

        where index arithmetic on positions is modulo *n*.

        Args:
            edges (list[tuple[int, int]]): list of undirected edges as ``(city_i, city_j)`` pairs.
            distances (list[float]): travel cost for each edge, parallel to ``edges``.
            label (str, optional): the model label. Defaults to "Travelling Salesman".
            lagrange_multiplier (float, optional): penalty scale for the tour
                validity constraints when converting to QUBO. Defaults to 100.

        Returns:
            Model: a model of the travelling salesman problem for the given graph.

        Raises:
            ValueError: if ``edges`` and ``distances`` have different lengths.
            ValueError: if the graph has no edges.
        """
        if len(edges) != len(distances):
            raise ValueError("edges and distances must have the same length.")
        if not edges:
            raise ValueError(_EMPTY_GRAPH_MSG)

        n = max(node for edge in edges for node in edge) + 1
        x = [[BinaryVariable(f"x{i}_{t}") for t in range(n)] for i in range(n)]
        model = cls(label)

        for i in range(n):
            model.add_constraint(
                f"city_{i}_once",
                EQ(sum(x[i][t] for t in range(n)), 1),
                lagrange_multiplier=lagrange_multiplier,
            )
        for t in range(n):
            model.add_constraint(
                f"position_{t}_once",
                EQ(sum(x[i][t] for i in range(n)), 1),
                lagrange_multiplier=lagrange_multiplier,
            )

        dist_terms = []
        for k, (u, v) in enumerate(edges):
            w = distances[k]
            for t in range(n):
                dist_terms.extend({w * x[u][t] * x[v][(t + 1) % n], w * x[v][t] * x[u][(t + 1) % n]})

        if dist_terms:
            objective = sum(dist_terms)
            if not isinstance(objective, Number):
                model.set_objective(objective, sense=ObjectiveSense.MINIMIZE)

        return model


class _Linearizer:
    """Degree-reduction helper that rewrites binary polynomials as quadratic expressions.

    Given a pseudo-Boolean term (i.e. a polynomial in ``BinaryVariable``'s obtained via
    :meth:`~qilisdk.core.variables.Term.to_binary`), :meth:`reduce` iteratively replaces each monomial of degree greater
    than two with an auxiliary binary variable that represents the product of two of its factors. Let's say the pair
    ``a`` and ``b`` are two binary variables contributing in a non-linear term, we can add an auxiliary binary
    variable ``w`` to substitute the pair, and the correctness of the substitution is enforced by the **Rosenberg** penalty:

    .. math::

        P(a, b, w) = a \\cdot b - 2 \\cdot a \\cdot w - 2 \\cdot b \\cdot w + 3 \\cdot w,

    which is quadratic, non-negative for ``a``, ``b``, ``w`` binary, and equal to zero if and only
    if ``w = a * b``.

    Auxiliary variables are cached per (unordered) pair of factors so that the same product shared across several
    monomials — for example ``x*y*z`` and ``x*y*w`` both reusing ``x*y`` — introduces a single auxiliary and a single
    Rosenberg penalty. Variables participating in a monomial are sorted deterministically by label so generated
    auxiliary names are reproducible across runs.

    Example:
        .. code-block:: python

            from qilisdk.core.model import _Linearizer
            from qilisdk.core.variables import BinaryVariable

            x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
            linearizer = _Linearizer()
            reduced = linearizer.reduce(x * y * z)
            # reduced has degree 2, auxiliary variables are registered on the linearizer
            penalties = linearizer.rosenberg_constraints()
    """

    _AUX_PREFIX = "_linearization_aux"

    def __init__(self) -> None:
        self._substitutions: dict[tuple[str, str], tuple[BaseVariable, BaseVariable, BinaryVariable]] = {}
        self._counter = 0

    @property
    def substitutions(self) -> dict[tuple[str, str], tuple[BaseVariable, BaseVariable, BinaryVariable]]:
        """Mapping from the ordered labels of a substituted pair ``(a_label, b_label)`` to the
        triple ``(a, b, aux)`` where ``aux`` is the binary variable that stands in for ``a * b``.

        Returns:
            dict: a dictionary of the currently registered pair-to-auxiliary substitutions.
        """
        return self._substitutions

    def reduce(self, term: Term) -> Term:
        """Rewrite ``term`` so that every monomial has degree at most two.

        The input is expected to be in binary-encoded form (i.e. the output of
        :meth:`~qilisdk.core.variables.Term.to_binary`). Terms that are already quadratic are returned unchanged
        (up to a structural copy).

        Args:
            term (Term): the polynomial expression to reduce.

        Returns:
            Term: an expression whose monomials all have degree at most two, with new auxiliary
            binary variables standing in for higher-degree sub-products.
        """
        if not isinstance(term, Term):
            return term

        if term.operation == Operation.MUL:
            return self._reduce_monomial(term)

        if term.operation == Operation.ADD:
            new_elements: list[BaseVariable | Term | Number] = []
            for element in term:
                coeff = term[element]
                if isinstance(element, Term) and element.operation == Operation.MUL:
                    new_elements.append(coeff * self._reduce_monomial(element))
                elif isinstance(element, Term):
                    new_elements.append(coeff * self.reduce(element))
                elif isinstance(element, BaseVariable) and element == Term.CONST:
                    new_elements.append(coeff)
                else:
                    new_elements.append(coeff * element)
            return Term(new_elements, Operation.ADD)

        return term

    def rosenberg_constraints(self) -> list[tuple[str, ComparisonTerm]]:
        """Materialize the Rosenberg penalty constraints that pin each auxiliary to its product.

        One equality constraint ``P(a, b, w) = 0`` is returned per registered pair. Each penalty is
        already quadratic in binary variables and can be added to a QUBO model verbatim, i.e. with
        ``transform_to_qubo=False``.

        Returns:
            list[tuple[str, ComparisonTerm]]: pairs of ``(label, penalty_constraint)``.
        """
        out: list[tuple[str, ComparisonTerm]] = []
        for _, (a, b, w) in self._substitutions.items():
            penalty = a * b - 2 * a * w - 2 * b * w + 3 * w
            out.append((f"linearization_{w.label}", EQ(penalty, 0)))
        return out

    def _get_or_create_aux(self, a: BaseVariable, b: BaseVariable) -> BinaryVariable:
        a_sorted, b_sorted = sorted([a, b], key=lambda v: v.label)
        key = (a_sorted.label, b_sorted.label)
        if key not in self._substitutions:
            aux = BinaryVariable(f"{self._AUX_PREFIX}({self._counter})")
            self._counter += 1
            self._substitutions[key] = (a_sorted, b_sorted, aux)
        return self._substitutions[key][2]

    def _reduce_monomial(self, monomial: Term) -> Term:
        if monomial.operation != Operation.MUL or monomial.degree <= 2:  # noqa: PLR2004
            return monomial

        coeff: Number = 1
        variables: list[BaseVariable] = []
        for elem in monomial:
            if isinstance(elem, BaseVariable) and elem == Term.CONST:
                coeff *= monomial[elem]
            elif isinstance(elem, BinaryVariable):
                variables.append(elem)
            elif isinstance(elem, BaseVariable):
                raise ValueError(
                    f"_Linearizer only operates on binary-encoded terms but received variable {elem}"
                    f" of domain {elem.domain}. Call `to_binary()` before linearizing."
                )
            else:
                raise ValueError(f"_Linearizer does not support nested sub-term {elem} inside a term.")

        variables.sort(key=lambda v: v.label)

        while len(variables) > 2:  # noqa: PLR2004
            a, b = self._pick_pair(variables)
            variables.remove(a)
            variables.remove(b)
            variables.insert(0, self._get_or_create_aux(a, b))

        result: Number | BaseVariable | Term = coeff
        for v in variables:
            result *= v
        if not isinstance(result, Term):
            result = Term([result], Operation.MUL)
        return result

    def _pick_pair(self, variables: list[BaseVariable]) -> tuple[BaseVariable, BaseVariable]:
        """Select which pair of factors to collapse into an auxiliary next.

        Preference order:

        1. A pair that is already registered with an auxiliary, so the existing aux is reused and
           no new Rosenberg penalty is introduced. This maximizes aux sharing across monomials
           (e.g. ``x*y*z`` and ``x*y*w`` both end up using the same ``w_{xy}``).
        2. Otherwise, the lexicographically smallest pair, which keeps aux generation
           deterministic and independent of the monomial ordering in the source term.

        Returns:
            tuple[BaseVariable, BaseVariable]: the pair of factors to replace with an auxiliary.
        """
        for a, b in itertools.combinations(variables, 2):
            a_sorted, b_sorted = sorted([a, b], key=lambda v: v.label)
            if (a_sorted.label, b_sorted.label) in self._substitutions:
                return a, b
        return variables[0], variables[1]


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
        self._linearizer: _Linearizer | None = None

    def _reduce(self, term: Term) -> Term:
        """Reduce the degree of ``term`` if a :class:`_Linearizer` is attached.

        The reduction introduces auxiliary binary variables that stand in for products of existing factors. Those
        auxiliaries are registered on the linearizer, and the corresponding Rosenberg penalty constraints are
        added in :meth:`from_model` once all objective and constraint terms have been processed.

        Args:
            term (Term): the term to potentially reduce.

        Returns:
            Term: the (possibly degree-reduced) term.
        """
        if self._linearizer is None:
            return term
        return self._linearizer.reduce(term)

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

    def _compute_lower_and_upper_limits(  # noqa: PLR6301
        self,
        term: Term,
    ) -> tuple[RealNumber, RealNumber, RealNumber]:
        """Computes the lower and upper bounds of a term.

        Args:
            term (Term): The term to compute the lower and upper limits for.

        Returns:
            tuple[RealNumber, RealNumber, RealNumber]: The Constant terms, lower limit, upper limit in this order.

        Raises:
            ValueError: if the operation the term uses is not addition or multiplication.
        """

        def to_real(num: Number) -> RealNumber:
            if isinstance(num, RealNumber):
                return num
            if isinstance(num, complex) and abs(num.imag) < get_settings().atol:
                return num.real
            raise ValueError("Complex values encountered in the constraint.")

        const: RealNumber = 0
        term_upper_limit: RealNumber = 0
        term_lower_limit: RealNumber = 0
        if term.operation is Operation.ADD:
            for element in term:
                if isinstance(element, BaseVariable) and element == Term.CONST:
                    const = to_real(term[element])
                else:
                    coeff_value = to_real(term[element])
                    if coeff_value > 0:
                        term_upper_limit += coeff_value
                    elif coeff_value < 0:
                        term_lower_limit += coeff_value
        elif term.operation is Operation.MUL:
            coeff_value = to_real(term.get_constant())
            if coeff_value > 0:
                term_upper_limit = coeff_value
            elif coeff_value < 0:
                term_lower_limit = coeff_value
        else:
            raise ValueError(f"Operation {term.operation.value} in constraint is not supported.")

        return const, term_lower_limit, term_upper_limit

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

        const, term_lower_limit, term_upper_limit = self._compute_lower_and_upper_limits(term)

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
        linearize: bool = True,
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
            linearize (bool, optional): linearize the constraints if they are above degree 2.

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

        if linearize and self._linearizer is None:
            self._linearizer = _Linearizer()

        if self._linearizer is None and c.degree > 2:  # noqa: PLR2004
            raise ValueError(
                f"QUBO constraints can not contain terms of order 2 or higher but received terms with degree {c.degree}. Set linearize=True to allow linearization."
            )

        self._check_variables(c, lagrange_multiplier=lagrange_multiplier)

        if transform_to_qubo:
            c = c.to_binary()
            if self._linearizer:
                c = ComparisonTerm(
                    lhs=self._linearizer.reduce(c.lhs),
                    rhs=self._linearizer.reduce(c.rhs),
                    operation=c.operation,
                )
            transformed_c = self._transform_constraint(label, c, penalization=penalization, parameters=parameters)
            if transformed_c is None:
                return
            transformed_c = self._reduce(transformed_c)
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

        If a :class:`_Linearizer` has been attached to this QUBO instance (via :meth:`from_model`
        with ``linearize=True``), the binary-encoded objective is additionally rewritten so that
        every monomial has degree at most two. Auxiliary variables introduced by the rewrite are
        registered on the linearizer.

        Args:
            term (Term): The objective's term.
            label (str, optional): the objective's label. Defaults to "obj".
            sense (ObjectiveSense, optional): The optimization sense of the model's objective.
                                                Defaults to ObjectiveSense.MINIMIZE.

        Raises:
            ValueError: if the degree of the provided term is larger than 2.

        """

        self._check_variables(term)

        if self._linearizer is None and term.degree > 2:  # noqa: PLR2004
            raise ValueError(
                f"QUBO objective can not contain terms of order higher than 2 but received terms with degree {term.degree}. Set linearize=True to enable linearization."
            )

        term = term.to_binary()
        term = self._reduce(term)
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
        linearize: bool = True,
        linearization_lagrange_multiplier: float = 100,
    ) -> QUBO:
        """A class method that constructs a QUBO model from a regular model if possible.

        When ``linearize`` is ``True`` (default), any pseudo-Boolean monomial of degree greater
        than two that appears in the objective or in a transformed constraint penalty is rewritten
        using auxiliary binary variables via the **Rosenberg** penalty

        .. math::

            P(a, b, w) = a \\cdot b - 2 \\cdot a \\cdot w - 2 \\cdot b \\cdot w + 3 \\cdot w,

        so ``w`` is forced to equal the product ``a * b`` at the optimum. One such penalty is
        added as an equality QUBO constraint for every unique pair substitution. Setting
        ``linearize=False`` restores the previous behaviour, where a ``ValueError`` is raised if
        the model contains terms of degree three or higher.

        Args:
            model (Model): the model to be used to construct the QUBO model.
            lagrange_multiplier_dict (dict[str, float] | None, optional): A dictionary with lagrange multiplier values
                                    to scale the model's constraints. Defaults to None.
            penalization (Literal[&quot;unbalanced&quot;, &quot;slack&quot;], optional): the penalization used to
                            handel inequality constraints. Defaults to "slack".
            parameters (list[float] | None, optional): the parameters used for the unbalanced penalization method.
                            Defaults to None.
            linearize (bool, optional): Automatically reduce high-degree pseudo-Boolean monomials to
                            quadratic form by introducing auxiliary binary variables. Defaults to ``True``.
            linearization_lagrange_multiplier (float, optional): The Lagrange multiplier applied to
                            each Rosenberg penalty constraint added as part of the linearization.
                            Must be large enough to dominate any incentive to violate the
                            auxiliary equalities ``w = a * b``. Defaults to 100.
        Returns:
            QUBO: a QUBO model equivalent to the input model, with any high-degree terms rewritten
            via auxiliary binary variables when ``linearize`` is enabled.
        """
        instance = QUBO(label="QUBO_" + model.label)
        if linearize:
            instance._linearizer = _Linearizer()
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
                linearize=linearize,
            )
        if instance._linearizer is not None:
            for pen_label, pen_term in instance._linearizer.rosenberg_constraints():
                instance.add_constraint(
                    label=pen_label,
                    term=pen_term,
                    lagrange_multiplier=linearization_lagrange_multiplier,
                    transform_to_qubo=False,
                    linearize=linearize,
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
        linearize: bool = True,
        linearization_lagrange_multiplier: float = 100,
    ) -> QUBO:
        """Return a copy of this QUBO model.

        QUBO models are already in quadratic form, so the linearization arguments are accepted for
        signature compatibility with :meth:`Model.to_qubo` but have no effect. A warning is
        emitted noting that no conversion was performed.
        """
        logger.warning(
            f"Running `to_qubo()` on the model {self.label} that is already in QUBO format.",
        )
        return copy.copy(self)

    def __copy__(self) -> QUBO:
        out = QUBO(label=self.label)
        obj = copy.copy(self.objective)
        out.set_objective(term=obj.term, label=obj.label, sense=obj.sense)

        for label, constraint in self._constraints.items():
            out._constraints[copy.copy(label)] = Constraint(
                label=copy.copy(constraint.label), term=copy.copy(constraint.term)
            )

        out._lagrange_multipliers = copy.copy(self._lagrange_multipliers)

        out.continuous_vars = copy.copy(self.continuous_vars)
        out.__qubo_objective = copy.copy(self.__qubo_objective)
        out._linearizer = copy.copy(self._linearizer)

        for label, constraint in self._encoding_constraints.items():
            out._encoding_constraints[copy.copy(label)] = Constraint(
                label=copy.copy(constraint.label), term=copy.copy(constraint.term)
            )

        return out
