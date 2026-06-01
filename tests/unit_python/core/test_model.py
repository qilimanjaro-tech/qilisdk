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
from qilisdk.core.model import QUBO, Constraint, Model, Objective, ObjectiveSense, SlackCounter, _Linearizer
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
    SpinVariable,
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


def test_model_to_ham():
    m = Model("test")
    b = [BinaryVariable(f"b{i}") for i in range(3)]

    term = b[0] + 2 * b[1] + 3 * b[2]
    m.set_objective(term)
    assert m.to_qubo().to_hamiltonian() == (3 - Z(1) - 0.5 * Z(0) - 1.5 * Z(2))
    m.add_constraint("c", EQ(b[0], 0), lagrange_multiplier=10)
    assert m.to_qubo().to_hamiltonian() == (
        (1 - Z(0)) / 2 + 2 * (1 - Z(1)) / 2 + 3 * (1 - Z(2)) / 2 + 10 * ((1 - Z(0)) / 2) * ((1 - Z(0)) / 2)
    )


# ---------- Model constructors ----------


def test_model_knapsack_basic():
    m = Model.knapsack(values=[3, 2, 5], weights=[2, 1, 3], max_weight=4)
    assert m.label == "Knapsack"
    assert len(m.variables()) == 3
    assert len(m.constraints) == 1
    assert m.constraints[0].label == "maximum weight"
    assert m.objective.sense == ObjectiveSense.MINIMIZE  # set_objective default is MINIMIZE


def test_model_knapsack_custom_label():
    m = Model.knapsack(values=[1, 2], weights=[1, 1], max_weight=1, label="MyKnapsack")
    assert m.label == "MyKnapsack"


def test_model_knapsack_mismatched_lengths():
    with pytest.raises(ValueError, match=r"number of weights must be equal"):
        Model.knapsack(values=[1, 2, 3], weights=[1, 2], max_weight=5)


def test_model_knapsack_brute_force_solution():
    # items: value=5, weight=3 and value=4, weight=2; max_weight=3
    # optimal: take only item 0 (value 5) or only item 1 (value 4) but not both (weight 5 > 3)
    # objective is to minimize -sum(values * vars) so the QUBO finds the set that maximises value
    # knapsack uses MINIMIZE on the positive sum, so brute_force should pick the feasible assignment
    m = Model.knapsack(values=[5, 4], weights=[3, 2], max_weight=3)
    _, sample = m.brute_force()
    # constraint must be satisfied: 3*b0 + 2*b1 <= 3
    b0, b1 = list(sample.keys())
    assert 3 * sample[b0] + 2 * sample[b1] <= 3


def test_model_random_ising_structure():
    m = Model.random_ising(num_variables=3)
    assert m.label == "Random Ising"
    assert len(m.variables()) == 3
    assert len(m.constraints) == 0


def test_model_random_ising_custom_label_and_seed():
    m1 = Model.random_ising(num_variables=4, seed=42, label="RandIsing")
    m2 = Model.random_ising(num_variables=4, seed=42)
    assert m1.label == "RandIsing"
    # Same seed produces identical objective terms
    assert hash(m1.objective.term) == hash(m2.objective.term)


def test_model_random_ising_different_seeds_differ():
    m1 = Model.random_ising(num_variables=4, seed=1)
    m2 = Model.random_ising(num_variables=4, seed=2)
    assert hash(m1.objective.term) != hash(m2.objective.term)


def test_model_factoring_basic():
    m = Model.factoring(6)
    assert m.label == "Factoring"
    assert len(m.constraints) == 0
    # brute-force should find factors: 6 = 2*3 → x0=0,x1=1 and y0=1,y1=0 (or permutations)
    results, _ = m.brute_force()
    obj_val = next(iter(results.values()))
    assert obj_val == 0


def test_model_factoring_custom_label():
    m = Model.factoring(4, label="Factor4")
    assert m.label == "Factor4"


def test_model_max_cut_basic():
    # Triangle graph: 3 nodes, 3 edges
    m = Model.max_cut(edges=[(0, 1), (1, 2), (0, 2)])
    assert m.label == "Max-Cut"
    assert m.objective.sense == ObjectiveSense.MAXIMIZE
    assert len(m.variables()) == 3
    assert len(m.constraints) == 0


def test_model_max_cut_custom_label_and_weights():
    m = Model.max_cut(edges=[(0, 1), (1, 2)], weights=[2.0, 3.0], label="WMax-Cut")
    assert m.label == "WMax-Cut"
    assert len(m.variables()) == 3


def test_model_max_cut_mismatched_weights():
    with pytest.raises(ValueError, match=r"number of weights must be equal"):
        Model.max_cut(edges=[(0, 1), (1, 2)], weights=[1.0])


def test_model_max_cut_brute_force_solution():
    # Path graph 0-1-2: max cut = 2 (put nodes 0,2 on one side, node 1 on the other).
    # evaluate() negates MAXIMIZE objectives, so brute_force returns the negated value (-2).
    m = Model.max_cut(edges=[(0, 1), (1, 2)])
    results, _ = m.brute_force()
    obj_val = results[m.objective.label]
    assert obj_val == -2


def test_model_graph_coloring_basic():
    # Triangle graph needs 3 colors
    m = Model.graph_coloring(edges=[(0, 1), (1, 2), (0, 2)], num_colors=3)
    assert m.label == "Graph Coloring"
    # one_color constraint per node (3) + conflict constraint per (edge, color) (3*3=9) = 12
    assert len(m.constraints) == 12
    assert len(m.variables()) == 9  # 3 nodes * 3 colors


def test_model_graph_coloring_custom_label():
    m = Model.graph_coloring(edges=[(0, 1)], num_colors=2, label="MyColoring")
    assert m.label == "MyColoring"


def test_model_graph_coloring_constraint_labels():
    m = Model.graph_coloring(edges=[(0, 1)], num_colors=2)
    labels = {c.label for c in m.constraints}
    assert "one_color_0" in labels
    assert "one_color_1" in labels
    assert "conflict_0_1_0" in labels
    assert "conflict_0_1_1" in labels


# ---------- Model.brute_force ----------


def test_brute_force_minimize():
    m = Model("bf_min")
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m.set_objective(x + 2 * y, sense=ObjectiveSense.MINIMIZE)
    results, sample = m.brute_force()
    assert sample[x] == 0
    assert sample[y] == 0
    assert results[m.objective.label] == 0


def test_brute_force_maximize():
    m = Model("bf_max")
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m.set_objective(x + 2 * y, sense=ObjectiveSense.MAXIMIZE)
    results, sample = m.brute_force()
    assert sample[x] == 1
    assert sample[y] == 1
    # evaluate() negates MAXIMIZE objectives, so the returned value is -(x + 2y) = -3.
    assert results[m.objective.label] == -3


def test_brute_force_returns_evaluate_dict():
    m = Model("bf_eval")
    x = BinaryVariable("x")
    m.set_objective(x + 0, sense=ObjectiveSense.MINIMIZE)
    m.add_constraint("must_be_zero", EQ(x, 0))
    results, _ = m.brute_force()
    # results should contain both the objective label and the constraint label
    assert m.objective.label in results
    assert "must_be_zero" in results


def test_brute_force_with_bounded_variable():
    m = Model("bf_bounded")
    v = Variable("v", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    m.set_objective(v + 0, sense=ObjectiveSense.MINIMIZE)
    _, sample = m.brute_force()
    assert sample[v] == 0


def test_brute_force_raises_for_unsupported_variable():
    # SpinVariable is not BinaryVariable or Variable, so brute_force raises ValueError.
    m = Model("bf_spin")
    s = SpinVariable("s")
    m.set_objective(s + 0)
    with pytest.raises(ValueError):  # noqa: PT011
        m.brute_force()


# ---------- QUBO ----------
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
        ValueError,
        match=r"QUBO constraints can not contain terms of order 2 or higher but received terms with degree 3",
    ):
        q.add_constraint("c", ct, penalization="slack", linearize=False)


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


# ---------- Linearization ----------


def _qubo_minimum(qubo: QUBO) -> tuple[dict, float]:
    """Brute-force: minimise the QUBO objective over all binary assignments."""
    obj = qubo.qubo_objective.term
    var_list = qubo.variables()
    best_val = float("inf")
    best_args: dict = {}
    for mask in range(2 ** len(var_list)):
        assignment = {v: (mask >> i) & 1 for i, v in enumerate(var_list)}
        value = obj.evaluate(assignment)
        if value < best_val:
            best_val = value
            best_args = assignment
    return best_args, best_val


def test_linearizer_reduce_cubic_monomial():
    linearizer = _Linearizer()
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    reduced = linearizer.reduce(x * y * z)
    assert reduced.degree == 2
    assert len(linearizer.substitutions) == 1


def test_linearizer_quadratic_passthrough():
    linearizer = _Linearizer()
    x, y = BinaryVariable("x"), BinaryVariable("y")
    reduced = linearizer.reduce(x * y + 2 * x + 3)
    assert reduced.degree == 2
    # No aux should be introduced when the input is already quadratic.
    assert linearizer.substitutions == {}


def test_linearizer_rejects_non_binary():
    linearizer = _Linearizer()
    x = Variable("x", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    y = Variable("y", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    z = Variable("z", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    # x*y*z involves non-binary Variable objects directly — should fail.
    with pytest.raises(ValueError, match=r"binary-encoded"):
        linearizer.reduce(x * y * z)


def test_linearizer_reuses_auxiliary_across_monomials():
    linearizer = _Linearizer()
    x, y, z, w = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z"), BinaryVariable("w")
    linearizer.reduce(x * y * z + x * y * w)
    # Both monomials share the x*y pair, so only one aux should be registered.
    assert len(linearizer.substitutions) == 1
    assert ("x", "y") in linearizer.substitutions


def test_to_qubo_linearises_cubic_objective():
    m = Model("cubic_obj")
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    m.set_objective(-(x * y * z), sense=ObjectiveSense.MINIMIZE)  # forces xyz=1 at optimum
    qubo = m.to_qubo(linearization_lagrange_multiplier=50)
    aux_labels = [v.label for v in qubo.variables() if v.label.startswith("_linearization_aux")]
    assert len(aux_labels) == 1
    # Exactly one Rosenberg constraint.
    rosenberg = [c for c in qubo.constraints if c.label.startswith("linearization_")]
    assert len(rosenberg) == 1
    # Brute force: the QUBO optimum should have xyz = 1 and aux = x*y.
    best_args, _ = _qubo_minimum(qubo)
    assert best_args[x] * best_args[y] * best_args[z] == 1


def test_to_qubo_linearises_cubic_equality_constraint():
    m = Model("cubic_eq")
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    m.set_objective(-x - y - z)  # maximise population, unconstrained optimum is (1,1,1)
    m.add_constraint("forbid_triple", EQ(x * y * z, 0), lagrange_multiplier=20)
    qubo = m.to_qubo(linearization_lagrange_multiplier=50)
    best_args, _ = _qubo_minimum(qubo)
    # The constraint forbids (1,1,1); best feasible is two out of three = 1.
    assert best_args[x] * best_args[y] * best_args[z] == 0
    assert best_args[x] + best_args[y] + best_args[z] == 2


def test_to_qubo_linearises_cubic_inequality_constraint():
    m = Model("cubic_leq")
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    m.set_objective(-x - y - z)
    m.add_constraint("triple_cap", LEQ(x * y * z, 0), lagrange_multiplier=20)
    qubo = m.to_qubo(linearization_lagrange_multiplier=50)
    best_args, _ = _qubo_minimum(qubo)
    assert best_args[x] * best_args[y] * best_args[z] == 0


def test_to_qubo_auxiliary_reused_between_objective_and_constraint():
    m = Model("shared_aux")
    x, y, z, w = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z"), BinaryVariable("w")
    m.set_objective(x * y * z)
    m.add_constraint("c", EQ(x * y * w, 0), lagrange_multiplier=10)
    qubo = m.to_qubo()
    aux_labels = [v.label for v in qubo.variables() if v.label.startswith("_linearization_aux")]
    # The x*y pair is shared between the objective monomial and the EQ constraint penalty.
    assert len(aux_labels) == 1


def test_to_qubo_linearize_false_still_rejects_cubic_constraint():
    m = Model("reject")
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    m.set_objective(x + y + z)
    m.add_constraint("c", EQ(x * y * z, 1), lagrange_multiplier=10)
    with pytest.raises(ValueError, match=r"can not contain terms of order 2 or higher"):
        m.to_qubo(linearize=False)


def test_to_qubo_no_auxes_when_terms_are_quadratic():
    m = Model("quadratic_only")
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    m.set_objective(x * y + 2 * z)
    m.add_constraint("c", LEQ(x + y + z, 2), lagrange_multiplier=5)
    qubo = m.to_qubo()
    aux_labels = [v.label for v in qubo.variables() if v.label.startswith("_linearization_aux")]
    assert aux_labels == []


def test_to_qubo_linearises_quartic_monomial():
    m = Model("quartic")
    a, b, c, d = (BinaryVariable(name) for name in ("a", "b", "c", "d"))
    m.set_objective(-(a * b * c * d))  # minimise = maximise abcd
    qubo = m.to_qubo(linearization_lagrange_multiplier=100)
    # A degree-4 monomial needs two substitutions to collapse to quadratic.
    aux_labels = [v.label for v in qubo.variables() if v.label.startswith("_linearization_aux")]
    assert len(aux_labels) == 2
    best_args, _ = _qubo_minimum(qubo)
    assert best_args[a] * best_args[b] * best_args[c] * best_args[d] == 1
