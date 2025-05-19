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


import copy

import pytest

from qilisdk.common.model import QUBO, Constraint, Model, Objective, ObjectiveSense, SlackCounter, cast_to_list
from qilisdk.common.variables import (
    BinaryVar,
    ComparisonOperation,
    ComparisonTerm,
    Domain,
    Operation,
    Term,
    Variable,
)


# ---------- SlackCounter ----------
def test_slackcounter_singleton_and_increment():
    c1 = SlackCounter()
    c2 = SlackCounter()
    assert c1 is c2
    # reset counter for test
    c1._count = 0
    assert c1.next() == 0
    assert c2.next() == 1


# ---------- cast_to_list ----------
def test_cast_to_list_single_and_list():
    term = ComparisonTerm(lhs=1, rhs=0, operation=ComparisonOperation.EQ)
    assert cast_to_list(term) == [term]
    lst = [term, term]
    assert cast_to_list(lst) == lst
    with pytest.raises(ValueError):  # noqa: PT011
        cast_to_list(123)


# ---------- Constraint ----------
def test_constraint_init_and_repr():
    var = Variable("x", Domain.BINARY)
    ct = ComparisonTerm(lhs=var, rhs=0, operation=ComparisonOperation.GE)
    cons = Constraint(label="c1", term=ct)
    assert cons.label == "c1"
    assert cons.term is ct
    assert "c1" in repr(cons)
    assert cons.lhs == ct.lhs
    assert cons.rhs == ct.rhs
    assert cons.degree == max(ct.lhs.degree, ct.rhs.degree)


# ---------- Objective ----------
def test_objective_init_and_copy_and_errors():
    var = Variable("b", Domain.BINARY)
    t = Term(elements=[var], operation=Operation.ADD)
    obj = Objective(label="o1", term=t, sense=ObjectiveSense.MAXIMIZE)
    assert obj.label == "o1"
    assert obj.term == t
    assert obj.sense == ObjectiveSense.MAXIMIZE
    # copy
    obj2 = copy.copy(obj)
    assert obj2.label == obj.label
    assert obj2.term == obj.term
    assert obj2.sense == obj.sense
    # errors
    with pytest.raises(ValueError):  # noqa: PT011
        Objective(label="bad", term=123, sense=ObjectiveSense.MINIMIZE)
    with pytest.raises(ValueError):  # noqa: PT011
        Objective(label="bad", term=t, sense="wrong")


# ---------- Model ----------
@pytest.fixture
def simple_model():
    return Model(label="m1")


def test_model_add_duplicate_constraint(simple_model):
    m = simple_model
    var = Variable("x", Domain.BINARY)
    ct = ComparisonTerm(lhs=var, rhs=0, operation=ComparisonOperation.EQ)
    m.add_constraint("c", ct)
    with pytest.raises(ValueError):  # noqa: PT011
        m.add_constraint("c", ct)


def test_model_set_objective_and_repr(simple_model):
    m = simple_model
    var = Variable("y", Domain.BINARY)
    t = Term(elements=[var], operation=Operation.ADD)
    m.set_objective(term=t, label="obj1", sense=ObjectiveSense.MINIMIZE)
    assert m.objective.label == "obj1"
    assert m.objective.term == t
    s = repr(m)
    assert "Model name: m1" in s
    assert "objective (obj1)" in s


# ---------- QUBO ----------
def test_qubo_parse_term_add_and_mul():
    q = QUBO(label="q1")
    v = BinaryVar("b")
    # ADD term: 2*b + 3
    t = Term(elements=[2, v, 1, v], operation=Operation.ADD)
    const, terms = q._parse_term(t)
    # terms should contain b twice coefficient sum= v:2?
    assert const == 3
    assert any(coeff == 2 and var == v for coeff, var in terms)
    # MUL term: b * 1
    t2 = v * 1
    const2, terms2 = q._parse_term(t2)
    assert const2 == 1
    assert any(var == v for _, var in terms2)


def test_qubo_check_valid_constraint_always_feasible_and_unsat():
    q = QUBO(label="q2")
    v = BinaryVar("b2")
    # always feasible: term 0 >= 0
    h = ComparisonTerm(lhs=v, rhs=v, operation=ComparisonOperation.GE)
    slack = q._check_valid_constraint("c1", h.lhs - h.rhs, h.operation)
    assert slack is None
    # unsatisfiable: v > 2
    h2 = ComparisonTerm(lhs=v, rhs=2, operation=ComparisonOperation.GT)
    with pytest.raises(ValueError):  # noqa: PT011
        q._check_valid_constraint("c2", h2.lhs - h2.rhs, h2.operation)


def test_qubo_add_constraint_and_objective_errors():
    q = QUBO(label="q3")
    x = BinaryVar("x")
    term = ComparisonTerm(lhs=x, rhs=0, operation=ComparisonOperation.EQ)
    # invalid penalization
    with pytest.raises(ValueError):  # noqa: PT011
        q.add_constraint("c", term, penalization="bad")
    # add valid
    q.add_constraint("c", term)
    assert "c" in q.lagrange_multipliers
    # non-binary domain var
    y = Variable("y", Domain.INTEGER)
    t2 = ComparisonTerm(lhs=y, rhs=0, operation=ComparisonOperation.EQ)
    q2 = QUBO(label="q4")
    with pytest.raises(ValueError):  # noqa: PT011
        q2.add_constraint("c2", t2)


# ---------- set_objective QUBO ----------
def test_qubo_set_objective_errors():
    q = QUBO(label="q5")
    # non-binary domain
    y = Variable("y", Domain.REAL, bounds=(0, 1))
    t = Term(elements=[y], operation=Operation.ADD)
    with pytest.raises(ValueError):  # noqa: PT011
        q.set_objective(term=t)
    # valid binary
    b = BinaryVar("b3")
    t2 = Term(elements=[b], operation=Operation.ADD)
    q.set_objective(term=t2, label="o2", sense=ObjectiveSense.MAXIMIZE)
    assert q.objective.label == "o2"
    assert q.qubo_objective.sense.value == ObjectiveSense.MINIMIZE.value  # always stored as minimize
