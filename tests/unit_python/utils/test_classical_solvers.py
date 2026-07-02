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
from qilisdk.utils.classical_solvers import BruteForceSolver, ClassicalSolver, ScipySolver
from qilisdk.utils.classical_solvers.base_solver import _assert_real, _variable_bounds
from qilisdk.utils.classical_solvers.scipy_solver import _decode_value

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


def test_classical_solver_solve_raises():
    m = Model("m")
    x = BinaryVariable("x")
    m.set_objective(1 * x)
    with pytest.raises(NotImplementedError):
        ClassicalSolver().solve(m)


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
    m.set_objective(v * 1)
    _, sample = BruteForceSolver().solve(m)
    assert sample[v] == 0


def test_brute_force_unsupported_variable_raises():
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(Term([s], Operation.ADD))
    with pytest.raises(ValueError, match="not supported"):
        BruteForceSolver().solve(m)


def test_brute_force_warns_on_large_model():
    bits = [BinaryVariable(f"b{i}") for i in range(14)]
    obj = bits[0]
    for b in bits[1:]:
        obj = obj + b
    m = Model("large")
    m.set_objective(obj)
    with patch("qilisdk.utils.classical_solvers.brute_force_solver.logger") as mock_logger:
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


def test_variable_bounds_explicit_bounds():
    v = Variable("v", Domain.REAL, bounds=(-2.5, 4.0))
    assert _variable_bounds(v) == (-2.5, 4.0)


def test_variable_bounds_binary_defaults_to_domain_limits():
    assert _variable_bounds(BinaryVariable("b")) == (0.0, 1.0)


def test_variable_bounds_missing_lower_falls_back_to_domain_min():
    v = Variable("v", Domain.POSITIVE_INTEGER, bounds=(None, 5))
    lower, upper = _variable_bounds(v)
    assert lower == 0.0
    assert upper == 5.0


def test_decode_value_binary_rounds():
    b = BinaryVariable("b")
    assert _decode_value(b, 0.6) == 1
    assert _decode_value(b, 0.4) == 0


def test_decode_value_clamps_outside_bounds():
    b = BinaryVariable("b")
    assert _decode_value(b, 2.0) == 1
    assert _decode_value(b, -3.0) == 0


def test_decode_value_integer_rounds_and_clamps():
    v = Variable("v", Domain.INTEGER, bounds=(0, 7))
    assert _decode_value(v, 3.2) == 3
    assert _decode_value(v, 6.8) == 7
    assert _decode_value(v, 100.0) == 7


def test_decode_value_spin_maps_to_sign():
    s = SpinVariable("s")
    assert _decode_value(s, 0.3) == 1
    assert _decode_value(s, -0.3) == -1
    assert _decode_value(s, 0.0) == 1


def test_decode_value_real_is_passthrough():
    v = Variable("v", Domain.REAL, bounds=(0, 10))
    assert np.isclose(_decode_value(v, 3.7), 3.7)


def test_scipy_solver_minimizes_binary():
    x = BinaryVariable("x")
    m = Model("bin")
    m.set_objective(1 * x)
    _, sample = ScipySolver().solve(m)
    assert sample[x] == 0


def test_scipy_solver_maximize():
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    _, sample = ScipySolver().solve(m, method="differential_evolution", seed=1)
    assert sample[x] == 1


def test_scipy_solver_integer_variable():
    x = Variable("x", Domain.INTEGER, bounds=(0, 7))
    m = Model("int_model")
    m.set_objective((x - 5) * (x - 5))
    _, sample = ScipySolver().solve(m, method="differential_evolution", seed=1)
    assert sample[x] == 5


def test_scipy_solver_real_variable():
    y = Variable("y", Domain.REAL, bounds=(0, 10))
    m = Model("real_model")
    m.set_objective((y - 3.7) * (y - 3.7))
    _, sample = ScipySolver().solve(m, method="differential_evolution", seed=1)
    assert np.isclose(sample[y], 3.7, atol=1e-1)


def test_scipy_solver_unsupported_variable_raises():
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(Term([s], Operation.ADD))
    with pytest.raises(ValueError, match="not supported"):
        ScipySolver().solve(m)


def test_scipy_solver_returns_evaluate_of_best():
    x = BinaryVariable("x")
    m = Model("ret_test")
    m.set_objective(1 * x)
    results, sample = ScipySolver().solve(m)
    assert results == m.evaluate(sample)
