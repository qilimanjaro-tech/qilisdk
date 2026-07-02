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

import numpy as np
import pytest

pytest.importorskip("pyscipopt", reason="ScipSolver tests require the 'scip' optional dependency", exc_type=ImportError)

from pyscipopt import Model as ScipModel

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import EQ, BinaryVariable, Domain, Operation, SpinVariable, Term, Variable
from qilisdk.utils.classical_solvers import ScipSolver
from qilisdk.utils.classical_solvers.scip_solver import _decode_scip_value, _term_to_scip_expr

def test_decode_scip_value_spin_maps_to_sign():
    s = SpinVariable("s")
    assert _decode_scip_value(s, 1.0) == 1
    assert _decode_scip_value(s, 0.0) == -1


def test_decode_scip_value_binary_and_integer_round():
    assert _decode_scip_value(BinaryVariable("b"), 0.9) == 1
    v = Variable("v", Domain.INTEGER, bounds=(0, 7))
    assert _decode_scip_value(v, 4.2) == 4


def test_decode_scip_value_real_is_passthrough():
    v = Variable("v", Domain.REAL, bounds=(0, 10))
    assert np.isclose(_decode_scip_value(v, 3.7), 3.7)


def test_term_to_scip_expr_empty_term_is_zero():
    assert _term_to_scip_expr(Term([], Operation.ADD), {}) == 0.0


def test_term_to_scip_expr_unsupported_operation_raises():
    x = BinaryVariable("x")
    term = Term([x], Operation.SUB)
    scip_model = ScipModel("t")
    with pytest.raises(ValueError, match="not supported"):
        _term_to_scip_expr(term, {x: scip_model.addVar(name="x", vtype="B")})


def test_scip_solver_minimizes_binary():
    x = BinaryVariable("x")
    m = Model("bin")
    m.set_objective(1 * x)
    _, sample = ScipSolver().solve(m)
    assert sample[x] == 0


def test_scip_solver_maximize():
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    _, sample = ScipSolver().solve(m)
    assert sample[x] == 1


def test_scip_solver_quadratic_objective():
    x = Variable("x", Domain.INTEGER, bounds=(0, 7))
    m = Model("int_model")
    m.set_objective((x - 5) * (x - 5))
    _, sample = ScipSolver().solve(m)
    assert sample[x] == 5


def test_scip_solver_real_variable_stays_continuous():
    y = Variable("y", Domain.REAL, bounds=(0, 10))
    m = Model("real_model")
    m.set_objective((y - 3.7) * (y - 3.7))
    _, sample = ScipSolver().solve(m)
    assert np.isclose(sample[y], 3.7, atol=1e-3)


def test_scip_solver_spin_variable():
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(Term([s], Operation.ADD))
    _, sample = ScipSolver().solve(m)
    assert sample[s] == -1


def test_scip_solver_respects_hard_constraint():
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("constrained")
    m.set_objective(x + y)
    m.add_constraint("c1", EQ(x + y, 1))
    results, sample = ScipSolver().solve(m)
    assert sample[x] + sample[y] == 1
    assert results["c1"] == 0


def test_scip_solver_infeasible_raises():
    x = BinaryVariable("x")
    m = Model("infeasible")
    m.set_objective(1 * x)
    m.add_constraint("c1", EQ(x, 1))
    m.add_constraint("c2", EQ(x, 0))
    with pytest.raises(ValueError, match="no feasible solution"):
        ScipSolver().solve(m)


def test_scip_solver_returns_evaluate_of_best():
    x = BinaryVariable("x")
    m = Model("ret_test")
    m.set_objective(1 * x)
    results, sample = ScipSolver().solve(m)
    assert results == m.evaluate(sample)
