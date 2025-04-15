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
from typing import Literal

# import cupy as np
import numpy as np

from qilisdk.analog.hamiltonian import Hamiltonian, Z

from .variables import (
    HOBO,
    BinaryVar,
    ComparisonOperation,
    ComparisonTerm,
    ContinuousVar,
    Domain,
    Operation,
    Term,
    Variable,
)

# Utils ###

slack_count = 0


def cast_to_list(terms):
    if isinstance(terms, list):
        return terms
    if isinstance(terms, ConstraintTerm):
        return [terms]
    # if isinstance(terms, np.array):
    #     return terms.to_list()
    raise ValueError(f"terms provided to the constraints should be of type {ConstraintTerm}")


class ObjectiveSense(enum.Enum):
    Minimize = "minimize"
    Maximize = "maximize"


class Constraint:
    def __init__(self, label: str, term: ConstraintTerm = None) -> None:
        self._label = label
        if not isinstance(term, ConstraintTerm):
            raise ValueError(f"the parameter term is expecting a {ConstraintTerm} but received {term.__class__}")

        self._term = term
        self._variables = {}
        for v in term.variables():
            if v.label in self._variables and not compare_vars(v, self._variables[v.label]):
                raise ValueError(
                    f"Error in ({self._label}: {self._term}): you can not include two different variables ({v}, {self.variables[v.label]}) with the same name in the model."
                )
            self._variables[v.label] = v

    @property
    def label(self) -> str:
        return self._label

    @property
    def term(self) -> ConstraintTerm:
        return self._term

    @property
    def variables(self):
        return self._variables

    def linearize(self):

        for v in self.variables.values():
            if v.domain is Domain.REAL:
                raise ValueError("Can not linearize models with real variables.")
            if v.domain is Domain.INTEGER and v.lower_bound < 0:
                raise ValueError("Can not linearize model with variables with negative bounds.")

        lhs = copy.copy(self.term.lhs)
        slack_constraints = []
        if isinstance(lhs, Term) and lhs.operation is Operation.ADD:
            for j, e in enumerate(lhs.elements):
                if isinstance(e, Term) and e.operation is Operation.MUL:
                    _, coeff, var_list = e.create_hashable_term_name()
                    if len(var_list) > 2:
                        raise ValueError(f"cannot linearize constraints with degrees higher than quadratic.\n {self}")
                    if len(var_list) == 2:
                        # quadratic term
                        x = var_list[0]
                        y = var_list[1]
                        if x.domain is Domain.BINARY and y.domain is Domain.BINARY:
                            z = BinaryVar(f"linearization_slack_{slack_count}")
                            slack_count += 1
                            slack_constraints.extend(
                                (
                                    (f"{self.label}_{z}_{x}", z - x <= 0),
                                    (f"{self.label}_{z}_{y}", z - y <= 0),
                                    (f"{self.label}_{z}_{x}_{y}", z - x - y + 1 >= 0),
                                )
                            )

                            lhs.elements[j] = coeff * z
                        elif (
                            (x.domain != y.domain)
                            and x.domain in {Domain.BINARY, Domain.INTEGER}
                            and y.domain in {Domain.BINARY, Domain.INTEGER}
                        ):
                            x = var_list[0] if var_list[0].domain is Domain.BINARY else var_list[1]
                            y = var_list[1] if var_list[0].domain is Domain.BINARY else var_list[0]
                            z = ContinuousVar(
                                f"linearization_slack_{slack_count}", domain=Domain.INTEGER, bounds=y.bounds
                            )
                            slack_count += 1
                            slack_constraints.extend(
                                (
                                    (f"{self.label}_{z}_{y}", z - y <= 0),
                                    (f"{self.label}_{z}_{x}", z - y.bounds[1] * x <= 0),
                                    (f"{self.label}_{z}_{x}_{y}", z - y - y.bounds[1] * (x - 1) >= 0),
                                    (f"{self.label}_{z}", z >= 0),
                                )
                            )

                            lhs.elements[j] = coeff * z

                        elif x.domain is Domain.INTEGER and y.domain is Domain.INTEGER:
                            x_ub = x.bounds[1]
                            x_lb = x.bounds[0]

                            y_ub = y.bounds[1]
                            y_lb = y.bounds[0]

                            z1 = 1 / 2 * (x + y)
                            z1_ub = int(1 / 2 * (x_ub + y_ub))
                            z1_lb = int(1 / 2 * (x_lb + y_lb))

                            z2 = 1 / 2 * (x - y)
                            z2_ub = int(1 / 2 * (x_ub - y_ub))
                            z2_lb = int(1 / 2 * (x_lb - y_lb))

                            n = int(z1_ub - z1_lb)
                            a = list(range(z1_lb, z1_ub))
                            h = int(np.ceil(np.log2(n)))

                            z = []
                            lam = []
                            for ni in range(n):
                                aux = []
                                for hi in range(h):
                                    aux.append(BinaryVar(f"linearization_z1_{ni}_{hi}"))

                                z.append(aux)
                                if ni == 0:
                                    z.append(aux)

                                lam.append(ContinuousVar(f"lambda1_{ni}", Domain.INTEGER, (0, None)))
                            lam.append(ContinuousVar(f"lambda1_{n+1}", Domain.INTEGER, (0, None)))

                            l_z1 = sum(a[i] ** 2 * lam[i] for i in range(n))

                            s_p = []
                            s_m = []

                            for k in range(h):
                                aux_s_p = []
                                aux_s_m = []

                                for i in range(n):
                                    if i in {n, 0}:
                                        if z[i][k] == 1:
                                            if i not in aux_s_p:
                                                aux_s_p.append(i)
                                        elif z[i][k] == 0 and i not in aux_s_m:
                                            aux_s_m.append(i)

                                    elif z[i][k] == z[i + 1][k] == 1:
                                        if i not in aux_s_p:
                                            aux_s_p.append(i)
                                    elif z[i][k] == z[i + 1][k] == 0:
                                        if i not in aux_s_m:
                                            aux_s_m.append(i)

                                s_p.append(aux_s_p)
                                s_m.append(aux_s_m)

                            slack_constraints.extend(
                                (
                                    (f"{self.label}_linearization_{z1}", z1 == sum(a[i] * lam[i] for i in range(n))),
                                    (f"{self.label}_linearization_{lam}", sum(lam[i] for i in range(n)) == 1),
                                )
                            )

                            for k in range(h):
                                if len(s_p[k]) > 0:
                                    slack_constraints.append(
                                        (
                                            f"{self.label}_linearization_{lam}_{z[i]}",
                                            sum(lam[i] for i in s_p[k]) <= sum(sum(z[i]) for i in s_p[k]),
                                        )
                                    )
                                if len(s_m[k]) > 0:
                                    slack_constraints.append(
                                        (
                                            f"{self.label}_linearization_{lam}_{z[i]}_2",
                                            sum(lam[i] for i in s_m[k]) <= sum(1 - z[i] for i in s_m[k]),
                                        )
                                    )

                            # Y2

                            n = int(z2_ub - z2_lb)
                            if n != 0:
                                a = list(range(z2_lb, z2_ub))
                                h = int(np.ceil(np.log2(n)))

                                z = []
                                lam = []
                                for ni in range(n):
                                    aux = []
                                    for hi in range(h):
                                        aux.append(BinaryVar(f"linearization_z2_{ni}_{hi}"))

                                    z.append(aux)
                                    if ni == 0:
                                        z.append(aux)

                                    lam.append(ContinuousVar(f"lambda2_{ni}", Domain.INTEGER, (0, None)))
                                lam.append(ContinuousVar(f"lambda2_{n+1}", Domain.INTEGER, (0, None)))

                                l_z2 = sum(a[i] ** 2 * lam[i] for i in range(n))

                                s_p = []
                                s_m = []

                                for k in range(h):
                                    aux_s_p = []
                                    aux_s_m = []

                                    for i in range(n):
                                        if i in {n, 0}:
                                            if z[i][k] == 1:
                                                if i not in aux_s_p:
                                                    aux_s_p.append(i)
                                            elif z[i][k] == 0:
                                                if i not in aux_s_m:
                                                    aux_s_m.append(i)

                                        elif z[i][k] == z[i + 1][k] == 1:
                                            if i not in aux_s_p:
                                                aux_s_p.append(i)
                                        elif z[i][k] == z[i + 1][k] == 0:
                                            if i not in aux_s_m:
                                                aux_s_m.append(i)

                                    s_p.append(aux_s_p)
                                    s_m.append(aux_s_m)

                                slack_constraints.extend(
                                    (
                                        (
                                            f"{self.label}_linearization2_{z2}",
                                            z2 == sum(a[i] * lam[i] for i in range(n)),
                                        ),
                                        (f"{self.label}_linearization2_{lam}", sum(lam[i] for i in range(n)) == 1),
                                    )
                                )

                                for k in range(h):
                                    if len(s_p[k]) > 0:
                                        slack_constraints.append(
                                            (
                                                f"{self.label}_linearization2_{lam}_{z[i]}",
                                                sum(lam[i] for i in s_p[k]) <= sum(z[i] for i in s_p[k]),
                                            )
                                        )
                                    if len(s_m[k]) > 0:
                                        slack_constraints.append(
                                            (
                                                f"{self.label}_linearization2_{lam}_{z[i]}_2",
                                                sum(lam[i] for i in s_m[k]) <= sum(1 - z[i] for i in s_m[k]),
                                            )
                                        )

                                lhs.elements[j] = coeff * (l_z1 - l_z2)
                            else:
                                lhs.elements[j] = coeff * (l_z1)
            slack_constraints.append(
                (
                    self.label,
                    ConstraintTerm(rhs=copy.copy(self.term.rhs), lhs=lhs, operation=self.term.operation),
                )
            )

        return slack_constraints

    def __copy__(self) -> Constraint:
        return Constraint(label=self.label, term=copy.copy(self.term))

    def __repr__(self) -> str:
        return f"{self.label}: {self.term}"

    def __str__(self) -> str:
        return f"{self.label}: {self.term}"

    def simplify_terms(self) -> None:
        if isinstance(self.term, Term):
            self._term = ConstraintTerm(
                lhs=self.term.lhs.simplify_constants(), rhs=self.term.rhs, operation=self.term.operation
            )
        if isinstance(self.term.lhs, Term) and self.term.lhs.operation is Operation.ADD:
            self.term.lhs.simplify_variable_coefficients()


class Objective:
    def __init__(self, label: str, term: Term, sense: ObjectiveSense = ObjectiveSense.Minimize) -> None:
        if isinstance(term, Variable):
            term = Term(elements=[term], operation=Operation.ADD)
        if not isinstance(term, Term):
            raise ValueError(f"the parameter term is expecting a {Term} but received {term.__class__}")
        if not isinstance(sense, ObjectiveSense):
            raise ValueError(f"the objective sense is expecting a {ObjectiveSense} but received {sense.__class__}")
        self._term = term
        self._label = label
        self._sense = sense
        self._variables = {}
        for v in term.variables():
            if v.label in self._variables and not compare_vars(v, self._variables[v.label]):
                raise ValueError(
                    f"Error in ({self._label}: {self._term}): you can not include two different variables ({v}, {self.variables[v.label]}) with the same name in the model."
                )
            self._variables[v.label] = v

    @property
    def label(self) -> str:
        return self._label

    @property
    def term(self) -> ConstraintTerm:
        return self._term

    @property
    def sense(self) -> ObjectiveSense:
        return self._sense

    @property
    def variables(self):
        return self._variables

    def __repr__(self) -> str:
        return f"{self.label}: {self.term}"

    def __str__(self) -> str:
        return f"{self.label}: {self.term}"

    def __copy__(self) -> Objective:
        return Objective(label=self.label, term=copy.copy(self.term), sense=self.sense)

    def simplify_term(self) -> None:
        self._term = self.term.simplify_constants()
        if isinstance(self.term, Term) and self.term.operation is Operation.ADD:
            self.term.simplify_variable_coefficients()


class Model:
    def __init__(self, label: str) -> None:
        self._constraints = []
        self._encoding_constraints = {}
        self._objective = None
        self._label = label
        self._variables: dict[str, Variable] = {}  # var_label : var
        self._real_variables = {}  # var_label : {'var' : var, 'precision': precision}
        self._recentered = {}  # var_label : ('var' : var, 'precision': precision, 'shift': lower_bound)

    @property
    def label(self) -> str:
        return self._label

    @property
    def constraints(self) -> list[Constraint]:
        return self._constraints

    @property
    def encoding_constraints(self):
        return self._encoding_constraints

    @property
    def objective(self):
        return self._objective

    @property
    def variables(self):
        return self._variables

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
        for v in self._variables.values():
            if isinstance(v, ContinuousVar):
                out.add_variable(v)
        return out

    def add_variable(self, var: Variable) -> None:
        self._variables[var.label] = var
        if isinstance(var, ContinuousVar):
            self._encoding_constraints[f"encoding_{var.label}"] = var.encoding_constraint()

    def add_constraint(self, label: str, term: ConstraintTerm) -> None:
        c = Constraint(label=label, term=copy.copy(term))
        self._constraints.append(c)

        for k, v in c.variables.items():
            if k in self._variables and not compare_vars(v, self._variables[k]):
                raise ValueError("you can not include two different variables with the same name in the model.")
            self._variables[k] = v

    def set_objective(self, term: Term, label: str = "", sense: ObjectiveSense = ObjectiveSense.Minimize) -> None:
        self._objective = Objective(label=label, term=copy.copy(term), sense=sense)

        for k, v in self._objective.variables.items():
            if k in self._variables and not compare_vars(v, self._variables[k]):
                raise ValueError("you can not include two different variables with the same name in the model.")
            self._variables[k] = v

    def real_to_int(self, precision: int = 100) -> None:
        maxint = 2**53

        for k, v in self.variables.items():
            if v.domain is Domain.REAL:
                lower_bound = int(v.lower_bound * precision)
                upper_bound = int(v.upper_bound * precision)

                lower_bound = max(-maxint, lower_bound)
                upper_bound = min(maxint, upper_bound)

                self._real_variables[k] = {"var": v, "precision": precision}

                new_var = ContinuousVar(
                    v.label, domain=Domain.INTEGER, bounds=(lower_bound, upper_bound), encoding=v.encoding
                )
                self.variables[k] = new_var

        self.objective.term.replace_variables(self.variables)
        self.objective.term.update_variables_precision(self._real_variables)
        self.objective.simplify_term()
        for i, _ in enumerate(self.constraints):
            self.constraints[i].term.replace_variables(self.variables)
            self.constraints[i].term.update_variables_precision(self._real_variables)
            self.constraints[i].simplify_terms()

    def real_to_pos_int(self, precision: int = 100):
        maxint = 2**53

        for k, v in self.variables.items():
            if v.domain is Domain.REAL:
                lower_bound = int(v.lower_bound * precision)
                upper_bound = int(v.upper_bound * precision)

                if lower_bound < 0:
                    upper_bound -= lower_bound
                    self._recentered[k] = {"var": v, "precision": lower_bound / precision, "original_bounds": v.bounds}

                lower_bound = max(0, lower_bound)
                upper_bound = min(maxint, upper_bound)

                self._real_variables[k] = {"var": v, "precision": precision}

                v = ContinuousVar(
                    v.label,
                    domain=Domain.POSITIVE_INTEGER,
                    bounds=(lower_bound, upper_bound),
                    encoding=v.encoding,
                )
                self.variables[k] = v
            elif v.domain is Domain.INTEGER:
                lower_bound = int(v.lower_bound)
                upper_bound = int(v.upper_bound)

                if lower_bound < 0:
                    upper_bound -= lower_bound
                    self._recentered[k] = {"var": v, "precision": lower_bound, "original_bounds": v.bounds}

                lower_bound = max(0, lower_bound)
                upper_bound = min(maxint, upper_bound)

                v = ContinuousVar(
                    v.label, domain=Domain.POSITIVE_INTEGER, bounds=(lower_bound, upper_bound), encoding=v.encoding
                )
                self.variables[k] = v

        self.objective.term.replace_variables(self.variables)
        self.objective.term.update_variables_precision(self._real_variables)
        self.objective.term.update_negative_variables_range(self._recentered)
        self.objective.simplify_term()
        for i, _ in enumerate(self.constraints):
            self.constraints[i].term.replace_variables(self.variables)
            self.constraints[i].term.update_variables_precision(self._real_variables)
            self.constraints[i].term.update_negative_variables_range(self._recentered)
            self.constraints[i].simplify_terms()

    def resize_samples(self, sample):
        for k, v in sample.items():
            value = v
            if k in self._recentered:
                if k in self._real_variables:
                    value /= self._real_variables[k]["precision"]
                value += self._recentered[k]["precision"]

            elif k in self._real_variables:
                value /= self._real_variables[k]["precision"]
            sample[k] = value

        return sample

    def constraint_linearization(self) -> None:
        for v in self.variables.values():
            if v.domain is Domain.REAL:
                raise ValueError("Can not linearize models with real variables.")
            if v.domain is Domain.INTEGER and v.lower_bound < 0:
                raise ValueError("Can not linearize model with variables with negative bounds.")
        pop_list = []
        slack_constraints = []
        for constraint_i, constraint in enumerate(self.constraints):
            lhs = constraint.term.lhs
            if isinstance(lhs, Term) and lhs.operation is Operation.ADD:
                slack_constraints.extend(constraint.linearize())
                pop_list.append(constraint_i)

        pop_list = sorted(pop_list, reverse=True)
        for i in pop_list:
            self.constraints.pop(i)

        for sc in slack_constraints:
            self.add_constraint(sc[0], sc[1])

    def to_ham(self) -> QUBO:
        return QUBO.from_model(self)


class QUBO(Model):
    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.continuous_vars = {}
        self.lagrange_multipliers = {}
        self.__qubo_objective = None

    @property
    def qubo_objective(self):
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

    def parse_term(self, term: Term):
        const = ConstantVar(0)
        terms = []

        if term.operation is Operation.ADD:
            for e in term.elements:
                if isinstance(e, ConstantVar):
                    const += e
                elif isinstance(e, Term):
                    c, t = self.parse_term(e)
                    terms.extend(t)
                    const += c
                elif isinstance(e, Variable):
                    terms.append((ConstantVar(1), e))
        elif term.operation is Operation.MUL:
            if len(term.elements) > 2:  # noqa: PLR2004
                raise ValueError(f"QUBO constraints only allow linear terms but received {term}.")
            if isinstance(term.elements[0], ConstantVar):
                return 0, [(term.elements[0], term.elements[1])]
            if isinstance(term.elements[1], ConstantVar):
                return 0, [(term.elements[1], term.elements[0])]
            raise ValueError(f"QUBO constraints only allow linear terms but received {term}.")
        return const, terms

    def _transform_constraint(
        self,
        label: str,
        term: ConstraintTerm,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
    ):
        if parameters is None:
            parameters = []
        if term.operation is ComparisonOperation.EQ:
            return (term.lhs - term.rhs) ** 2

        if term.operation in {
            ComparisonOperation.GE,
            ComparisonOperation.GT,
        }:
            # assuming the operation is h >= 0 or h > 0 because of the way inequality constraints are coded.
            h = term.lhs - term.rhs
            if penalization == "unbalanced":
                if len(parameters) < 2:
                    raise ValueError("using unbalanced penalization requires at least 2 parameters.")
                return -parameters[0] * h + parameters[1] * (h**2)
            ub = np.iinfo(np.int64).max
            lb = 0

            const, terms = self.parse_term(h)
            term_upper_limit = sum(coeff.value for coeff, _ in terms if coeff.value > 0)
            term_lower_limit = sum(coeff.value for coeff, _ in terms if coeff.value < 0)

            upper_cut = min(term_upper_limit, ub - const.value)
            lower_cut = max(term_lower_limit, lb - const.value)

            if term_upper_limit <= upper_cut and term_lower_limit >= lower_cut:
                warnings.warn(f"constraint ({label}) not added because it is always feasible.")
                return None

            ub_slack = int(upper_cut - lower_cut)

            if upper_cut < lower_cut:
                raise ValueError(f"Constraint {label} is unsatisfiable.")

            if ub_slack == 0:
                return h**2

            slack = ContinuousVar(f"{label}_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=HOBO)

            slack_terms, _ = slack.encode()
            out = h + slack_terms
            return (out) ** 2

        if term.operation in {
            ComparisonOperation.LE,
            ComparisonOperation.LT,
        }:
            if penalization == "unbalanced":
                # assumption: input has the following structure -> 0 < h  or 0 <= h
                h = term.rhs - term.lhs
                if len(parameters) < 2:
                    raise ValueError("using unbalanced penalization requires at least 2 parameters.")
                return -parameters[0] * h + parameters[1] * (h**2)
            # assuming the operation is h <= 0 or h < 0 because of the way inequality constraints are coded.
            h = term.lhs - term.rhs
            lb = np.iinfo(np.int64).min
            ub = 0
            const, terms = self.parse_term(h)
            term_upper_limit = sum(coeff.value for coeff, _ in terms if coeff.value > 0)
            term_lower_limit = sum(coeff.value for coeff, _ in terms if coeff.value < 0)

            upper_cut = min(term_upper_limit, ub - const.value)
            lower_cut = max(term_lower_limit, lb - const.value)

            if term_upper_limit <= upper_cut and term_lower_limit >= lower_cut:
                warnings.warn(f"constraint ({label}) not added because it is always feasible.")
                return None

            ub_slack = int(upper_cut - lower_cut)

            if upper_cut < lower_cut:
                raise ValueError(f"Constraint {label} is unsatisfiable.")

            if ub_slack == 0:
                return h**2

            slack = ContinuousVar(f"{label}_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=HOBO)

            slack_terms, _ = slack.encode()
            out = h + slack_terms
            return (out) ** 2
        return None

    def add_constraint(
        self,
        label: str,
        term: ConstraintTerm,
        lagrange_multiplier: float = 100,
        penalization: Literal["unbalanced", "slack"] = "slack",
        parameters: list[float] | None = None,
        transform_to_qubo: bool = True,
    ) -> None:
        if label in self.lagrange_multipliers:
            raise ValueError(f"A constraint with the label {label} already exists.")

        if parameters is None:
            parameters = [1, 1] if penalization == "unbalanced" else []

        if penalization not in {"unbalanced", "slack"}:
            raise ValueError("The type of penalization for inequality constraints can only be unbalanced or slack.")

        if term.operation in {ComparisonOperation.GE, ComparisonOperation.GT}:
            c = ConstraintTerm(lhs=(term.lhs - term.rhs), rhs=0, operation=term.operation)
        elif term.operation in {ComparisonOperation.LE, ComparisonOperation.LT}:
            c = ConstraintTerm(lhs=0, rhs=(term.rhs - term.lhs), operation=term.operation)
        else:
            c = copy.copy(term)

        if c.degree() > 2:
            raise ValueError("QUBO models can not contain terms of order 2 or higher.")

        for v in c.variables():
            if v.domain not in {Domain.POSITIVE_INTEGER, Domain.BINARY}:
                # NOTE: is this needed ???
                raise ValueError(
                    "QUBO models are not supported for variables that are not in the positive integers or binary domains."
                )
            if v.domain is Domain.POSITIVE_INTEGER and v.label not in self.continuous_vars:
                self.continuous_vars[v.label] = v
                encoding_constraint = v.encoding_constraint()
                if encoding_constraint is not None:
                    self.add_constraint(label=f"{self.label}_var_encoding_{v.label}", term=encoding_constraint)

            if v.lower_bound != 0:
                raise ValueError("All variables must have a lower bound of 0.")
        if transform_to_qubo:
            c = c.to_binary()

            c = self._transform_constraint(label, c, penalization=penalization, parameters=parameters)
            if c is None:
                return

            self.lagrange_multipliers[label] = lagrange_multiplier
            self.constraints.append(Constraint(label, term=c == 0))
        else:

            self.lagrange_multipliers[label] = lagrange_multiplier
            self.constraints.append(Constraint(label, term=c))

        # self.__set_qubo_objective(term=copy.copy(lagrange_multiplier * c))

        for v in c.variables():
            k = v.label
            if k in self._variables and not compare_vars(v, self._variables[k]):
                raise ValueError("you can not include two different variables with the same name in the model.")
            self._variables[k] = v

    def set_objective(self, term: Term, label: str = "obj", sense: ObjectiveSense = ObjectiveSense.Minimize) -> None:
        for v in term.variables():
            if v.domain not in {Domain.POSITIVE_INTEGER, Domain.BINARY}:
                # NOTE: is this needed ???
                raise ValueError(
                    "QUBO models are not supported for variables that are not in the positive integers or binary domains."
                )
            if v.domain is Domain.POSITIVE_INTEGER and v.label not in self.continuous_vars:
                self.continuous_vars[v.label] = v
                encoding_constraint = v.encoding_constraint()
                if encoding_constraint is not None:
                    self.constraints.append(
                        Constraint(label=f"{self.label}_var_encoding_{v.label}", term=encoding_constraint)
                    )

        term = term.to_binary()
        self._objective = Objective(label=label, term=term, sense=sense)

        # self.__set_qubo_objective(label=label, term=term, sense=sense)

        for k, v in self._objective.variables.items():
            if k in self._variables and not compare_vars(v, self._variables[k]):
                raise ValueError("you can not include two different variables with the same name in the model.")
            self._variables[k] = v

    def _set_qubo_objective(self, term: Term, label: str | None = None, sense: ObjectiveSense = None) -> None:
        term = copy.copy(term.to_binary())
        if self.__qubo_objective is None:
            self.__qubo_objective = Objective(
                label=label if label is not None else "obj",
                term=term,
                sense=sense if sense is not None else ObjectiveSense.Minimize,
            )
        else:
            self.__qubo_objective = Objective(
                label=label if label is not None else self.__qubo_objective.label,
                term=copy.copy(self.__qubo_objective.term) + term,
                sense=sense if sense is not None else self.__qubo_objective.sense,
            )

        for k, v in self.__qubo_objective.variables.items():
            if k in self._variables and not compare_vars(v, self._variables[k]):
                raise ValueError("you can not include two different variables with the same name in the model.")
            self._variables[k] = v

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

    def to_ham(self):
        spins = {}
        obj = self.qubo_objective
        for i, v in enumerate(obj.variables.keys()):
            spins[v] = (1 - Z(i)) / 2

        ham = 0
        aux_term = 0

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
                    elif isinstance(term, ConstantVar):
                        aux_term *= term.value
                    elif isinstance(term, Variable):
                        aux_term *= spins[term.label]
            else:
                aux_term = 1
                if isinstance(terms, ConstantVar):
                    aux_term *= terms.value
                elif isinstance(terms, Variable):
                    aux_term *= spins[terms.label]

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
        for v in self._variables.values():
            if isinstance(v, ContinuousVar):
                out.add_variable(v)
        return out
