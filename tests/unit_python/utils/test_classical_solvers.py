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

from unittest.mock import patch

import numpy as np
import pytest

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import EQ, BinaryVariable, Domain, Operation, SpinVariable, Term, Variable
from qilisdk.utils.classical_solvers import BruteForceSolver, ClassicalSolver, _assert_real

# ---------- _assert_real ----------


def test_assert_real_complex_with_negligible_imag():
    result = _assert_real(3.0 + 1e-20j)
    assert np.isclose(result, 3.0)


def test_assert_real_complex_with_large_imag_raises():
    with pytest.raises(ValueError, match="Complex"):
        _assert_real(1.0 + 2.0j)


def test_assert_real_non_complex_float():
    assert np.isclose(_assert_real(5.0), 5.0)


def test_assert_real_non_complex_int():
    assert np.isclose(_assert_real(7), 7)


# ---------- ClassicalSolver ----------


def test_classical_solver_solve_raises():
    m = Model("m")
    x = BinaryVariable("x")
    m.set_objective(1 * x)
    with pytest.raises(NotImplementedError):
        ClassicalSolver().solve(m)


# ---------- BruteForceSolver ----------


def test_brute_force_binary_variable_domain():
    x = BinaryVariable("x")
    m = Model("bin")
    m.set_objective(1 * x)
    results, sample = BruteForceSolver().solve(m)
    assert sample[x] == 0
    assert results[m.objective.label] == 0


def test_brute_force_maximize():
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    _, sample = BruteForceSolver().solve(m)
    assert sample[x] == 1


def test_brute_force_integer_variable_enumeration():
    v = Variable("v", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    m = Model("int_model")
    m.set_objective(v)
    _, sample = BruteForceSolver().solve(m)
    assert sample[v] == 0


def test_brute_force_unsupported_variable_raises():
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(Term([s], Operation.ADD))
    with pytest.raises(ValueError, match="not supported"):
        BruteForceSolver().solve(m)


def test_brute_force_warns_on_large_model():
    bits = [BinaryVariable(f"b{i}") for i in range(11)]
    obj = bits[0]
    for b in bits[1:]:
        obj = obj + b
    m = Model("large")
    m.set_objective(obj)
    with patch("qilisdk.utils.classical_solvers.logger") as mock_logger:
        BruteForceSolver().solve(m)
    mock_logger.warning.assert_called_once()


def test_brute_force_with_constraint_penalty():
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("constrained")
    m.set_objective(x + y)
    m.add_constraint("c1", EQ(x + y, 1), lagrange_multiplier=10)
    results, sample = BruteForceSolver().solve(m)
    assert sample[x] + sample[y] == 1
    assert results["c1"] == 0


def test_brute_force_best_sample_updated():
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("two_vars")
    m.set_objective(3 * x + 2 * y)
    results, sample = BruteForceSolver().solve(m)
    assert sample[x] == 0
    assert sample[y] == 0
    assert results[m.objective.label] == 0


def test_brute_force_returns_evaluate_of_best():
    x = BinaryVariable("x")
    m = Model("ret_test")
    m.set_objective(1 * x)
    results, sample = BruteForceSolver().solve(m)
    assert results == m.evaluate(sample)
