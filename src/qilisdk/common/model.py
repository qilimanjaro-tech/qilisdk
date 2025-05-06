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
import enum
import warnings
from typing import Literal, Type

# import cupy as np
import numpy as np

from qilisdk.analog.hamiltonian import Hamiltonian, Z

from .variables import (
    HOBO,
    BaseVariable,
    ComparisonOperation,
    ComparisonTerm,
    Domain,
    Number,
    Operation,
    Term,
    Variable,
)

# Utils ###


class SlackCounter:
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


def cast_to_list(terms: ComparisonTerm | list[ComparisonTerm]) -> list[ComparisonTerm]:
    if isinstance(terms, list):
        return terms
    if isinstance(terms, ComparisonTerm):
        return [terms]
    raise ValueError(f"terms provided to the constraints should be of type {ComparisonTerm}")


class ObjectiveSense(enum.Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class Constraint:
    def __init__(self, label: str, term: ComparisonTerm) -> None:
        self._label = label
        if not isinstance(term, ComparisonTerm):
            raise ValueError(f"the parameter term is expecting a {ComparisonTerm} but received {term.__class__}")

        self._term = term

    @property
    def label(self) -> str:
        return self._label

    @property
    def term(self) -> ComparisonTerm:
        return self._term

    def variables(self) -> list[BaseVariable]:
        return self._term.variables()

    @property
    def lhs(self) -> Term:
        return self.term.lhs

    @property
    def rhs(self) -> Term:
        return self.term.rhs

    @property
    def degree(self) -> int:
        return max(self.lhs.degree, self.rhs.degree)

    def __copy__(self) -> Constraint:
        return Constraint(label=self.label, term=copy.copy(self.term))

    def __repr__(self) -> str:
        return f"{self.label}: {self.term}"

    def __str__(self) -> str:
        return f"{self.label}: {self.term}"


class Objective:
    def __init__(self, label: str, term: Term, sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        if isinstance(term, BaseVariable):
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
        return self._label

    @property
    def term(self) -> Term:
        return self._term

    @property
    def sense(self) -> ObjectiveSense:
        return self._sense

    def variables(self) -> list[BaseVariable]:
        return self._term.variables()

    def __repr__(self) -> str:
        return f"{self.label}: {self.term}"

    def __str__(self) -> str:
        return f"{self.label}: {self.term}"

    def __copy__(self) -> Objective:
        return Objective(label=self.label, term=copy.copy(self.term), sense=self.sense)


class Model:
    def __init__(self, label: str) -> None:
        self._constraints: dict[str, Constraint] = {}
        self._encoding_constraints: dict[str, Constraint] = {}
        self._objective = Objective("objective", Term([0], Operation.ADD))
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @property
    def constraints(self) -> list[Constraint]:
        return list(self._constraints.values())

    @property
    def encoding_constraints(self) -> dict[str, Constraint]:
        return self._encoding_constraints

    @property
    def objective(self) -> Objective:
        return self._objective

    @property
    def variables(self) -> list[BaseVariable]:
        var = set()

        for c in self.constraints:
            var.update(c.variables())

        var.update(self.objective.variables())

        return list(var)

    def __repr__(self) -> str:
        output = f"Model name: {self.label} \n"
        output += (
            f"objective ({self.objective.label}): \n\t {self.objective.sense.value} : \n\t {self.objective.term} \n\n"
        )
        if len(self.constraints) > 0:
            output += "subject to the constraint/s: \n"
            for c in self.constraints:
                output += f"\t {c} \n"
            for label, value in self.encoding_constraints.items():
                output += f"\t {label}: {value} \n"
        return output

    def __str__(self) -> str:
        output = f"Model name: {self.label} \n"
        output += (
            f"objective ({self.objective.label}): \n\t {self.objective.sense.value} : \n\t {self.objective.term} \n\n"
        )
        if len(self.constraints) > 0:
            output += "subject to the constraint/s: \n"
            for c in self.constraints:
                output += f"\t {c} \n"
            for label, value in self.encoding_constraints.items():
                output += f"\t {label}: {value} \n"
        return output

    def __copy__(self) -> Model:
        out = Model(label=self.label)
        obj = copy.copy(self.objective)
        out.set_objective(term=obj.term, label=obj.label, sense=obj.sense)
        for c in self.constraints:
            out.add_constraint(label=c.label, term=copy.copy(c.term))
        return out

    def add_constraint(self, label: str, term: ComparisonTerm) -> None:
        if label in self._constraints:
            raise ValueError((f'Constraint "{label}" already exists:\n \t\t{self._constraints[label]}'))
        c = Constraint(label=label, term=copy.copy(term))
        self._constraints[label] = c

    def set_objective(self, term: Term, label: str = "", sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
        self._objective = Objective(label=label, term=copy.copy(term), sense=sense)

    def to_qubo(self) -> QUBO:
        return QUBO.from_model(self)

    def to_ham(self) -> Hamiltonian:
        return self.to_qubo().to_ham()


class QUBO(Model):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.continuous_vars: dict[str, Variable] = {}
        self.lagrange_multipliers: dict[str, float] = {}
        self.__qubo_objective: Objective | None = None

    @property
    def qubo_objective(self) -> Objective | None:
        self.__qubo_objective = None
        if self.objective is not None:
            self._set_qubo_objective(self.objective.term, self.objective.label, self.objective.sense)
        for constraint in self.constraints:
            if constraint.label in self.lagrange_multipliers:
                self._set_qubo_objective(
                    constraint.term.lhs * self.lagrange_multipliers[constraint.label]
                    - constraint.term.rhs * self.lagrange_multipliers[constraint.label]
                )
            else:
                self._set_qubo_objective(constraint.term.lhs - constraint.term.rhs)
        return self.__qubo_objective

    def __repr__(self) -> str:
        return self.label

    def __str__(self) -> str:
        output = f"Model name: {self.label} \n"
        if self.objective is not None:
            output += f"objective ({self.objective.label}): \n\t {self.objective.sense.value} : \n\t {self.objective.term} \n\n"
        if len(self.constraints) > 0:
            output += "subject to the constraint/s: \n"
            for c in self.constraints:
                output += f"\t {c} \n"
        if len(self.lagrange_multipliers) > 0:
            output += "\nWith Lagrange Multiplier/s: \n"
            for key, value in self.lagrange_multipliers.items():
                output += f"\t {key} : {value} \n"
        return output

    def _parse_term(self, term: Term) -> tuple[Number, list[tuple[Number, BaseVariable]]]:
        const = term.get_constant()
        terms: list[tuple[Number, BaseVariable]] = []

        if term.degree > 1:
            raise ValueError(f'QUBO constraints only allow linear terms but received "{term}" of degree {term.degree}')

        if term.operation is Operation.ADD:
            for element in term:
                if isinstance(element, Term):
                    _, aux_terms = self._parse_term(element)
                    terms.extend(aux_terms)
                elif not term.is_constant(element):
                    terms.append((term[element], element))
        if term.operation is Operation.MUL:
            for element in term:
                if not isinstance(element, Term) and not term.is_constant(element):
                    terms.append((1, element))
        return const, terms

    def _check_valid_constraint(self, label: str, term: Term, operation: ComparisonOperation) -> int | None:
        ub = np.iinfo(np.int64).max if operation in {ComparisonOperation.GE, ComparisonOperation.GT} else 0
        lb = np.iinfo(np.int64).min if operation in {ComparisonOperation.LE, ComparisonOperation.LT} else 0
        const, terms = self._parse_term(term)
        term_upper_limit = sum(coeff for coeff, _ in terms if coeff > 0)
        term_lower_limit = sum(coeff for coeff, _ in terms if coeff < 0)

        upper_cut = min(term_upper_limit, ub - const)
        lower_cut = max(term_lower_limit, lb - const)

        if term_upper_limit <= upper_cut and term_lower_limit >= lower_cut:
            warnings.warn(f"constraint ({label}) not added because it is always feasible.")
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
            ComparisonOperation.GE,
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

                slack = Variable(f"{label}_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=HOBO)
                slack_terms = slack.encode()
                out = h + slack_terms
                return (out) ** 2

        if term.operation in {
            ComparisonOperation.LE,
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

                slack = Variable(f"{label}_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=HOBO)

                slack_terms = slack.encode()
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
        if label in self._constraints:
            raise ValueError((f'Constraint "{label}" already exists:\n \t\t{self._constraints[label]}'))

        lower_penalization = penalization.lower()

        if lower_penalization not in {"unbalanced", "slack"}:
            raise ValueError(
                'Only penalization of type "unbalanced" or "slack" is supported for inequality constraints.'
            )

        if parameters is None:
            parameters = [1, 1] if lower_penalization == "unbalanced" else []

        if term.operation in {ComparisonOperation.GE, ComparisonOperation.GT}:
            c = ComparisonTerm(lhs=(term.lhs - term.rhs), rhs=0, operation=term.operation)
        elif term.operation in {ComparisonOperation.LE, ComparisonOperation.LT}:
            c = ComparisonTerm(lhs=0, rhs=(term.rhs - term.lhs), operation=term.operation)
        else:
            c = copy.copy(term)

        if c.degree() > 2:  # noqa: PLR2004
            raise ValueError(
                f"QUBO models can not contain terms of order 2 or higher but received terms with degree {c.degree()}."
            )

        for v in c.variables():
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
                    self.add_constraint(label=f"{self.label}_var_encoding_{v.label}", term=encoding_constraint)

        if transform_to_qubo:
            c = c.to_binary()
            transformed_c = self._transform_constraint(label, c, penalization=penalization, parameters=parameters)
            if transformed_c is None:
                return

            self.lagrange_multipliers[label] = lagrange_multiplier
            self._constraints[label] = Constraint(label, term=ComparisonTerm(transformed_c, 0, ComparisonOperation.EQ))

        else:
            self.lagrange_multipliers[label] = lagrange_multiplier
            self._constraints[label] = Constraint(label, term=c)

    def set_objective(self, term: Term, label: str = "obj", sense: ObjectiveSense = ObjectiveSense.MINIMIZE) -> None:
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
                    label = f"{self.label}_var_encoding_{v.label}"
                    self._constraints[label] = Constraint(label=label, term=encoding_constraint)

        term = term.to_binary()
        self._objective = Objective(label=label, term=term, sense=sense)

    def _set_qubo_objective(
        self, term: Term, label: str | None = None, sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    ) -> None:
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

    def set_lagrange_multiplier(self, constraint_label: str, lagrange_multiplier: float) -> None:
        self.lagrange_multipliers[constraint_label] = lagrange_multiplier

    @classmethod
    def from_model(
        cls,
        model: Model,
        lagrange_multiplier_dict: dict[str, float] | None = None,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
    ) -> QUBO:
        instance = QUBO(label=model.label)
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

    def to_ham(self) -> Hamiltonian:
        spins: dict[BaseVariable, Hamiltonian] = {}
        obj = self.qubo_objective

        if obj is None:
            raise ValueError("Can't transform empty QUBO model to a Hamiltonian.")

        for i, v in enumerate(obj.variables()):
            spins[v] = (1 - Z(i)) / 2

        ham = Hamiltonian()
        aux_term: Number | Hamiltonian = 0.0

        for terms in obj.term.to_list():
            if isinstance(terms, Operation):
                if terms is Operation.ADD:
                    ham += aux_term
                else:
                    raise ValueError(f"operation {terms} is not supported")

            elif isinstance(terms, list):
                aux_term = 1
                for term in terms:
                    if isinstance(term, Operation):
                        if term is not Operation.MUL:
                            raise ValueError(f"operation {term} is not supported")
                    elif isinstance(term, Number):
                        aux_term *= term.value
                    elif isinstance(term, BaseVariable):
                        aux_term *= spins[term.label]
            else:
                aux_term = 1
                if isinstance(terms, Number):
                    aux_term *= terms
                elif isinstance(terms, BaseVariable):
                    aux_term *= spins[terms]

        ham += aux_term
        return ham

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
