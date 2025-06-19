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

from qilisdk.common.model import QUBO, Constraint, Model, Objective, ObjectiveSense, SlackCounter
from qilisdk.common.variables import (
    EQ,
    BinaryVariable,
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
    c1.reset_counter()
    assert c1.next() == 0
    assert c2.next() == 1
    c1.reset_counter()
    assert c1.next() == 0


# ---------- Constraint ----------
def test_constraint_init_and_repr():
    var = Variable("x", Domain.BINARY)
    ct = ComparisonTerm(lhs=var, rhs=0, operation=ComparisonOperation.GE)
    cons = Constraint(label="c1", term=ct)
    assert cons.label == "c1"
    assert cons.term is ct
    assert "c1" in repr(cons)
    assert hash(cons.lhs) == hash(ct.lhs)
    assert hash(cons.rhs) == hash(ct.rhs)
    assert cons.degree == max(ct.lhs.degree, ct.rhs.degree)
    # errors
    with pytest.raises(ValueError):  # noqa: PT011
        Constraint(label="bad", term=Term([0], Operation.ADD))


def test_constraint_variables():
    var = Variable("x", Domain.INTEGER)
    ct = ComparisonTerm(lhs=var + 1, rhs=2, operation=ComparisonOperation.EQ)
    cons = Constraint(label="c1", term=ct)
    var_list = cons.variables()
    assert len(var_list) == 1
    assert var in var_list

    var2 = Variable("x2", Domain.INTEGER)
    ct = ComparisonTerm(lhs=var + var2, rhs=2 - var, operation=ComparisonOperation.EQ)
    cons2 = Constraint(label="c1", term=ct)
    var_list = cons2.variables()
    assert len(var_list) == 2


def test_constraint_copy():
    var = Variable("x", Domain.INTEGER)
    ct = ComparisonTerm(lhs=var + 1, rhs=2, operation=ComparisonOperation.EQ)
    cons = Constraint(label="c1", term=ct)
    cons2 = copy.copy(cons)

    assert hash(cons2.lhs) == hash(cons.lhs)
    assert cons2.term.operation == cons.term.operation
    assert hash(cons2.rhs) == hash(cons.rhs)
    assert cons2.label == cons.label


# ---------- Objective ----------
def test_objective_init_and_copy_and_errors():
    var = Variable("b", Domain.BINARY)
    t = Term(elements=[var], operation=Operation.ADD)
    obj = Objective(label="o1", term=t, sense=ObjectiveSense.MAXIMIZE)
    obj2 = Objective(label="o2", term=var, sense=ObjectiveSense.MAXIMIZE)
    assert obj.label == "o1"
    assert hash(obj.term) == hash(t)
    assert obj.sense == ObjectiveSense.MAXIMIZE
    assert obj2.label == "o2"
    assert hash(obj2.term) == hash(t)
    assert obj2.sense == ObjectiveSense.MAXIMIZE
    assert "o1" in str(obj)
    assert f"{t}" in str(obj)
    assert "o2" in repr(obj2)
    assert f"{t}" in repr(obj2)
    # copy
    obj3 = copy.copy(obj)
    assert obj3.label == obj.label
    assert hash(obj3.term) == hash(obj.term)
    assert obj3.sense == obj.sense
    # errors
    with pytest.raises(ValueError):  # noqa: PT011
        Objective(label="bad", term=123, sense=ObjectiveSense.MINIMIZE)
    with pytest.raises(ValueError):  # noqa: PT011
        Objective(label="bad", term=t, sense="wrong")


def test_objective_variables():
    var = Variable("x", Domain.INTEGER)
    t = var + 1
    obj = Objective(label="o1", term=t)
    var_list = obj.variables()
    assert len(var_list) == 1
    assert var in var_list

    var2 = Variable("x2", Domain.INTEGER)
    t = var + var2 + var
    obj2 = Objective(label="o2", term=t)
    var_list = obj2.variables()
    assert len(var_list) == 2


# ---------- Model ----------
@pytest.fixture
def simple_model():
    return Model(label="m1")


def test_model_add_duplicate_constraint(simple_model):
    m = simple_model
    var = Variable("x", Domain.BINARY)
    ct = ComparisonTerm(lhs=var, rhs=0, operation=ComparisonOperation.EQ)
    m.add_constraint("c", ct)
    assert len(m.encoding_constraints) == 0
    with pytest.raises(ValueError):  # noqa: PT011
        m.add_constraint("c", ct)


def test_model_set_objective_and_repr(simple_model):
    m = simple_model
    var = Variable("y", Domain.BINARY)
    t = Term(elements=[var], operation=Operation.ADD)
    m.set_objective(term=t, label="obj1", sense=ObjectiveSense.MINIMIZE)
    assert m.objective.label == "obj1"
    assert m.objective.term == t
    s = str(m)
    assert "Model name: m1" in s
    assert "objective (obj1)" in s
    assert "subject to the constraint" not in s

    var2 = Variable("x2", Domain.INTEGER)
    ct = ComparisonTerm(lhs=var + var2, rhs=2 - var, operation=ComparisonOperation.EQ)
    cons1 = Constraint(label="cons1", term=ct)

    m.add_constraint("cons1", ct)
    s = str(m)
    assert "subject to the constraint" in s
    assert str(cons1) in s
    assert m.label == repr(m)


def test_model_variables(simple_model):
    m = simple_model
    var = Variable("x", Domain.INTEGER)
    t = var + 1

    var2 = Variable("x2", Domain.INTEGER)
    ct = ComparisonTerm(lhs=var + var2, rhs=2 - var, operation=ComparisonOperation.EQ)

    m.set_objective(t)
    assert len(m.variables()) == 1
    m.add_constraint("cons1", ct)
    assert len(m.variables()) == 2


def test_model_copy(simple_model):
    m = simple_model
    var = Variable("x", Domain.INTEGER)
    t = var + 1

    var2 = Variable("x2", Domain.INTEGER)
    ct = ComparisonTerm(lhs=var + var2, rhs=2 - var, operation=ComparisonOperation.EQ)

    m.set_objective(t)
    m.add_constraint("cons1", ct)

    m2 = copy.copy(m)
    assert m2.label == m.label
    assert hash(m2.objective.term) == hash(m.objective.term)
    assert len(m2.constraints) == len(m.constraints)
    for i, c in enumerate(m.constraints):
        assert m2.constraints[i].label == c.label
        assert hash(m2.constraints[i].term.lhs) == hash(c.term.lhs)
        assert hash(m2.constraints[i].term.rhs) == hash(c.term.rhs)
        assert m2.constraints[i].term.operation == c.term.operation


# ---------- QUBO ----------
def test_qubo_parse_term_add_and_mul():
    q = QUBO(label="q1")
    v = BinaryVariable("b")
    v2 = Variable("v2", Domain.BINARY)
    # ADD term: 2*b + 3
    t = Term(elements=[2, v, 1, v], operation=Operation.ADD)
    const, terms = q._parse_term(t)
    assert any(coeff == 2 and hash(var) == hash(v) for coeff, var in terms)
    assert const == 3
    # MUL term: b * 1
    t2 = v * 1
    const2, terms2 = q._parse_term(t2)
    assert const2 == 1
    assert any(hash(var) == hash(v) for _, var in terms2)

    # MUL term: b + q1 * v + 2
    t2 = v + v2 * 2 + 2
    const2, terms2 = q._parse_term(t2)
    assert const2 == 2

    t3 = v2**2
    with pytest.raises(ValueError):  # noqa: PT011
        q._parse_term(t3)


def test_qubo_print_and_str():
    q = QUBO(label="q1")
    v = BinaryVariable("b")

    t = 2 * v
    q.set_objective(t)
    s = str(q)
    assert "Model name: q1" in s

    ct = EQ(v, 1)
    q.add_constraint("con1", ct, lagrange_multiplier=2)
    s = str(q)
    assert "subject to the constraint/s:" in s
    assert "\nWith Lagrange Multiplier/s: \n" in s


def test_qubo_transform_constraint():
    q = QUBO(label="q1")
    v = BinaryVariable("b")

    t = 2 * v
    q.set_objective(t)

    ct = EQ(v, 1)
    q.add_constraint("con1", ct, lagrange_multiplier=2)
    # TODO (ameer): finish this test


def test_qubo_check_valid_constraint_always_feasible_and_unsat():
    q = QUBO(label="q2")
    v = BinaryVariable("b2")
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
    x = BinaryVariable("x")
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
    b = BinaryVariable("b3")
    t2 = Term(elements=[b], operation=Operation.ADD)
    q.set_objective(term=t2, label="o2", sense=ObjectiveSense.MAXIMIZE)
    assert q.objective.label == "o2"
    assert q.qubo_objective.sense.value == ObjectiveSense.MINIMIZE.value  # always stored as minimize
