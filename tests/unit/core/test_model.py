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
from unittest.mock import MagicMock

import pytest

from qilisdk.analog.hamiltonian import Z
from qilisdk.core.model import QUBO, Constraint, Model, Objective, ObjectiveSense, SlackCounter
from qilisdk.core.variables import (
    EQ,
    GT,
    LEQ,
    LT,
    NEQ,
    BinaryVariable,
    Bitwise,
    ComparisonOperation,
    ComparisonTerm,
    Domain,
    OneHot,
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
    ct = ComparisonTerm(lhs=var, rhs=0, operation=ComparisonOperation.GEQ)
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


def test_model_lagrange_multipliers(simple_model):
    m = simple_model
    var = Variable("x", Domain.BINARY)
    ct = ComparisonTerm(lhs=var, rhs=0, operation=ComparisonOperation.EQ)
    m.add_constraint("c", ct, lagrange_multiplier=10)
    assert m.lagrange_multipliers["c"] == 10
    m.set_lagrange_multiplier("c", 20)
    assert m.lagrange_multipliers["c"] == 20
    with pytest.raises(ValueError, match=r'constraint "c2" not in model.'):
        m.set_lagrange_multiplier("c2", 20)


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


def test_model_evaluation():
    m = Model("test")
    v = Variable("v", Domain.INTEGER, (-10, 10))

    m.set_objective(v * 2 + 3)
    m.add_constraint("c", LT(v * 2, 15), lagrange_multiplier=20)
    # TODO (Ameer): Objective contains an Integer variable, so results["obj"] should be 13.
    # Now, it is 13.0, a float.

    # results = m.evaluate({v: 5})
    # assert results["obj"] == -(5 * 2 + 3)
    # assert results["c"] == 0
    # results = m.evaluate({v: 10})
    # assert results["obj"] == -(10 * 2 + 3)
    # assert results["c"] == 20


def test_model_to_ham():
    m = Model("test")
    b = [BinaryVariable(f"b{i}") for i in range(3)]

    term = b[0] + 2 * b[1] + 3 * b[2]
    m.set_objective(term)
    assert m.to_qubo().to_hamiltonian() == 3 - Z(1) - 0.5 * Z(0) - 1.5 * Z(2)
    m.add_constraint("c", EQ(b[0], 0), lagrange_multiplier=10)
    assert m.to_qubo().to_hamiltonian() == (
        (1 - Z(0)) / 2 + 2 * (1 - Z(1)) / 2 + 3 * (1 - Z(2)) / 2 + 10 * ((1 - Z(0)) / 2) * ((1 - Z(0)) / 2)
    )


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

    t3 = v + v * v * 2 + 2
    const3, term3 = q._parse_term(t3)
    assert const3 == 2
    assert term3[0] == (3, v)

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

    assert repr(q) == q.label


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
    h = ComparisonTerm(lhs=v, rhs=v, operation=ComparisonOperation.GEQ)
    slack = q._check_valid_constraint("c1", h.lhs - h.rhs, h.operation)
    assert slack is None
    # unsatisfiable: v > 2
    h2 = GT(v, 2)
    with pytest.raises(ValueError):  # noqa: PT011
        q._check_valid_constraint("c2", h2.lhs - h2.rhs, h2.operation)

    h2 = LT(v, -1)
    with pytest.raises(ValueError):  # noqa: PT011
        q._check_valid_constraint("c2", h2.lhs - h2.rhs, h2.operation)

    h2 = LT(v, -1)
    with pytest.raises(ValueError):  # noqa: PT011
        q._check_valid_constraint("c2", h2.lhs - h2.rhs, h2.operation)

    h2 = LEQ(v, -10)
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


def test_qubo_transform_unsupported_penalization_error():
    q = QUBO(label="test")
    y = Variable("y", Domain.REAL, bounds=(0, 1))
    ct = GT(y, 0)
    q._transform_constraint("c", term=ct, penalization="slack", parameters=None)
    with pytest.raises(ValueError, match=r"Only penalization of type \"unbalanced\" or \"slack\" is supported."):
        q._transform_constraint("c2", term=ct, penalization="test", parameters=None)

    with pytest.raises(ValueError, match=r"using unbalanced penalization requires at least 2 parameters."):
        q._transform_constraint("c2", term=ct, penalization="unbalanced", parameters=None)
    ct = LT(y, 1)
    with pytest.raises(ValueError, match=r"using unbalanced penalization requires at least 2 parameters."):
        q._transform_constraint("c2", term=ct, penalization="unbalanced", parameters=None)


def test_qubo_transform_unbalanced_penalization():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 10))

    ct = GT(x, 5)

    q.add_constraint("c", ct, penalization="unbalanced", parameters=(2, 1))

    h = ct.lhs - ct.rhs
    assert q._constraints["c"].lhs - q._constraints["c"].rhs == (-2 * h + h**2).to_binary()

    ct = LT(x, 5)

    q.add_constraint("c2", ct, penalization="unbalanced", parameters=(2, 1))

    h = ct.rhs - ct.lhs
    assert q._constraints["c2"].lhs - q._constraints["c2"].rhs == (-2 * h + h**2).to_binary()


def test_qubo_transform_slack_penalization():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = GT(x, 1)

    ub_slack = q._check_valid_constraint("c", (ct.lhs - ct.rhs).to_binary(), ComparisonOperation.GT)
    slack = Variable("c_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=Bitwise)
    assert ub_slack == 2
    q.add_constraint("c", ct, penalization="slack")

    c = q._constraints["c"].term.lhs - q._constraints["c"].term.rhs
    assert slack[0] in c
    assert slack[1] in c

    ct = LT(x, 2)

    ub_slack = q._check_valid_constraint("c2", (ct.lhs - ct.rhs).to_binary(), ComparisonOperation.GT)
    slack = Variable("c2_slack", domain=Domain.POSITIVE_INTEGER, bounds=(0, ub_slack), encoding=Bitwise)
    assert ub_slack == 1
    q.add_constraint("c2", ct, penalization="slack")

    c = q._constraints["c2"].term.lhs - q._constraints["c2"].term.rhs
    assert slack[0] in c


def test_qubo_transform_eq_always_feasible():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 0))

    ct = EQ(x, 0)

    q.add_constraint("c", ct, penalization="slack")
    assert len(q._constraints) == 1


def test_qubo_transform_gt_always_feasible():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = GT(x, 0)

    q.add_constraint("c", ct, penalization="slack")
    assert len(q._constraints) == 1


def test_qubo_transform_lt_always_feasible():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 1))

    ct = LT(x, 2)

    q.add_constraint("c", ct, penalization="slack")
    assert len(q._constraints) == 1


def test_qubo_transform_gt_zero_slack(monkeypatch):
    # Pretend that we are somehow able to add an inequality constraint that requires no slack but isn't always feasible
    check_valid_mock = MagicMock(return_value=0)

    monkeypatch.setattr(QUBO, "_check_valid_constraint", check_valid_mock)
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = GT(x, 1)
    q.add_constraint("c", ct, penalization="slack")
    assert len(q._constraints) == 2
    assert "c_slack" not in q._constraints


def test_qubo_transform_lt_zero_slack(monkeypatch):
    # Pretend that we are somehow able to add an inequality constraint that requires no slack but isn't always feasible
    check_valid_mock = MagicMock(return_value=0)

    monkeypatch.setattr(QUBO, "_check_valid_constraint", check_valid_mock)
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = LT(x, 1)
    q.add_constraint("c", ct, penalization="slack")
    assert len(q._constraints) == 2
    assert "c_slack" not in q._constraints


def test_add_constraint_unknown_type():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = NEQ(x, 1)

    q.add_constraint("c", ct, penalization="slack")
    assert len(q._constraints) == 1


def test_add_constraint_error_constraint_already_xists():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = GT(x, 1)
    q.add_constraint("c", ct, penalization="slack")

    with pytest.raises(ValueError, match=r'Constraint "c" already exists'):
        q.add_constraint("c", ct, penalization="slack")


def test_add_constraint_error_degree_greater_than_two():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = GT(x**3, 1)

    with pytest.raises(
        ValueError, match=r"QUBO models can not contain terms of order 2 or higher but received terms with degree 3"
    ):
        q.add_constraint("c", ct, penalization="slack")


def test_add_constraint_without_transform_to_qubo():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = EQ(x**2, 1)
    q.add_constraint("c", ct, transform_to_qubo=False)

    assert q._constraints["c"].term.lhs == ct.lhs
    assert q._constraints["c"].term.rhs == ct.rhs


def test_check_variables():
    q = QUBO(label="test")
    x = Variable("x", Domain.INTEGER, encoding=OneHot, bounds=(0, 2))

    ct = EQ(x**2, 1)
    with pytest.raises(
        ValueError,
        match=r"QUBO models are not supported for variables that are not in the positive integers or binary domains.",
    ):
        q._check_variables(ct)

    x = Variable("x", Domain.REAL, encoding=OneHot, bounds=(0, 2))

    ct = EQ(x**2, 1)
    with pytest.raises(
        ValueError,
        match=r"QUBO models are not supported for variables that are not in the positive integers or binary domains.",
    ):
        q._check_variables(ct)

    x = Variable("x", Domain.SPIN, encoding=OneHot, bounds=(-1, 1))

    ct = EQ(x**2, 1)
    with pytest.raises(
        ValueError,
        match=r"QUBO models are not supported for variables that are not in the positive integers or binary domains.",
    ):
        q._check_variables(ct)

    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(1, 2))

    ct = EQ(x**2, 1)
    with pytest.raises(
        ValueError,
        match=r"All variables must have a lower bound of 0. But variable x has a lower bound of 1",
    ):
        q._check_variables(ct)


def test_qubo_model_to_qubo():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))

    q.set_objective(x + x**2)
    ct = EQ(x, 1)
    q.add_constraint("c", ct)

    assert q.qubo_objective.term == q.to_qubo().qubo_objective.term


def test_qubo_model_evaluation():
    m = QUBO("test")
    v = Variable("v", Domain.POSITIVE_INTEGER, (0, 10), encoding=OneHot)

    m.set_objective(v * 2 + 3, sense=ObjectiveSense.MAXIMIZE)
    m.add_constraint("c", LT(v * 2, 15), lagrange_multiplier=20)
    variables = m.variables()
    values = dict.fromkeys(m.variables(), 0)
    values[v[5]] = 1
    values[variables[0]] = 1
    values[variables[1]] = 0
    values[variables[2]] = 1
    values[variables[3]] = 0
    results = m.evaluate(values)
    assert results["obj"] == -(5 * 2 + 3)
    assert results["c"] == 0
    values = dict.fromkeys(m.variables(), 0)
    values[v[7]] = 1
    values[variables[0]] = 1
    values[variables[1]] = 0
    values[variables[2]] = 0
    values[variables[3]] = 0

    results = m.evaluate(values)
    assert results["obj"] == -(7 * 2 + 3)
    assert results["c"] == 0


def test_to_hamiltonian_with_empty_qubo():
    q = QUBO("test")
    q._objective = None
    with pytest.raises(ValueError, match=r"Can't transform empty QUBO model to a Hamiltonian."):
        q.to_hamiltonian()


def test_qubo_constraint_name_mismatch():
    """
    Test what happens when lagrange multipliers are set wrong (somehow)
    """
    q = QUBO(label="q1")
    v = BinaryVariable("b")

    t = 2 * v
    q.set_objective(t)
    assert q.lagrange_multipliers == {}
    assert q.qubo_objective.sense.value == ObjectiveSense.MINIMIZE.value

    # add a constraint
    ct = EQ(v, 1)
    q.add_constraint("con1", ct)
    q._constraints["con1"]._label = "con2"  # force a mismatch
    assert "con1" in q.lagrange_multipliers
    assert q.qubo_objective.sense.value == ObjectiveSense.MINIMIZE.value


# def test_qubo_constraint_evaluate():
#     q = QUBO(label="q1")
#     v = BinaryVariable("b")

#     t = 2 * v
#     q.set_objective(t)

#     ct = EQ(v, 1)
#     q.add_constraint("con1", ct, lagrange_multiplier=5)

#     values = {v: 1}
#     results = q.evaluate(values)
#     assert results["obj"] == 2
#     assert results["con1"] == 0

#     values = {v: 0}
#     results = q.evaluate(values)
#     assert results["obj"] == 0
#     assert results["con1"] == 5


def test_model_evaluate():
    m = Model("test")
    v = Variable("v", Domain.INTEGER, (-10, 10))

    m.set_objective(v * 2 + 3)
    m.add_constraint("c", LT(v * 2, 15), lagrange_multiplier=20)

    results = m.evaluate({v: 5})
    assert results["obj"] == 13
    assert results["c"] == 0
    results = m.evaluate({v: 10})
    assert results["obj"] == 23
    assert results["c"] == 20


def test_add_constraint_with_terms():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))
    term = 2 * x + 1
    term_of_terms = 2 * term + 1
    term_of_terms._elements = {term: 2, 1: 2}
    assert q._parse_term(term_of_terms) == (0, [(2, x), (2, 1)])


def test_complex_constraint_raises():
    q = QUBO(label="test")
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))
    complex_constraint = ComparisonTerm(lhs=2 * x + 1j, rhs=1, operation=ComparisonOperation.EQ)
    with pytest.raises(ValueError, match=r"Complex values"):
        q.add_constraint("c", complex_constraint)


def test_qubo_from_model():
    m = Model("test")
    b = [BinaryVariable(f"b{i}") for i in range(3)]

    term = b[0] + 2 * b[1] + 3 * b[2]
    m.set_objective(term)
    m.add_constraint("c", EQ(b[0], 0), lagrange_multiplier=10)

    q = QUBO.from_model(m)
    assert q.label == "QUBO_" + m.label
    assert q.objective.term == m.objective.term
    assert q.lagrange_multipliers == {"c": 100}
    assert len(q._constraints) == len(m.constraints)


def test_parse_term_unsupported_element(monkeypatch):
    x = Variable("x", Domain.POSITIVE_INTEGER, encoding=OneHot, bounds=(0, 2))
    bad_objective = MagicMock()
    bad_objective.term = Term(elements=[x, 3], operation=Operation.SUB)
    bad_objective.variables = MagicMock(return_value=[x])
    monkeypatch.setattr(QUBO, "qubo_objective", bad_objective)
    q = QUBO(label="test")
    Term(elements=[x, 3], operation=Operation.SUB)
    with pytest.raises(ValueError, match=r"is not supported"):
        q.to_hamiltonian()
