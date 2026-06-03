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

import pytest

from qilisdk.core.model import Model, ObjectiveSense
from qilisdk.core.variables import EQ, BinaryVariable, Domain, Operation, SpinVariable, Term, Variable
from qilisdk.utils.classical_solvers import BruteForceSolver, ClassicalSolver, _assert_real

# ---------- _assert_real ----------


def test_assert_real_complex_with_negligible_imag():
    # Lines 25-27: complex input with imag below atol → returns real part
    result = _assert_real(3.0 + 1e-20j)
    assert result == 3.0


def test_assert_real_complex_with_large_imag_raises():
    # Line 28: complex input with imag above atol → ValueError
    with pytest.raises(ValueError, match="Complex"):
        _assert_real(1.0 + 2.0j)


def test_assert_real_non_complex_float():
    # Line 29: non-complex number returned unchanged
    assert _assert_real(5.0) == 5.0


def test_assert_real_non_complex_int():
    # Line 29: int is not complex → returned directly
    assert _assert_real(7) == 7


# ---------- ClassicalSolver ----------


def test_classical_solver_solve_raises():
    # Line 37: base class raises NotImplementedError
    m = Model("m")
    x = BinaryVariable("x")
    m.set_objective(1 * x)
    with pytest.raises(NotImplementedError):
        ClassicalSolver().solve(m)


# ---------- BruteForceSolver ----------


def test_brute_force_binary_variable_domain():
    # Lines 76-77: BinaryVariable gets domain [0, 1]; minimise picks x=0
    x = BinaryVariable("x")
    m = Model("bin")
    m.set_objective(1 * x)
    results, sample = BruteForceSolver().solve(m)
    assert sample[x] == 0
    assert results[m.objective.label] == 0


def test_brute_force_maximize():
    # Lines 76-77: BinaryVariable with MAXIMIZE sense; best is x=1
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    _, sample = BruteForceSolver().solve(m)
    assert sample[x] == 1


def test_brute_force_integer_variable_enumeration():
    # Lines 78-88: Variable with bounds → bit-pattern enumeration; minimise picks 0
    v = Variable("v", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    m = Model("int_model")
    m.set_objective(v)
    _, sample = BruteForceSolver().solve(m)
    assert sample[v] == 0


def test_brute_force_unsupported_variable_raises():
    # Lines 89-90: SpinVariable is neither BinaryVariable nor Variable → ValueError
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(Term([s], Operation.ADD))
    with pytest.raises(ValueError, match="not supported"):
        BruteForceSolver().solve(m)


def test_brute_force_warns_on_large_model():
    # Lines 95-98: > 1024 combinations triggers logger.warning
    bits = [BinaryVariable(f"b{i}") for i in range(11)]  # 2^11 = 2048 > 1024
    obj = bits[0]
    for b in bits[1:]:
        obj = obj + b
    m = Model("large")
    m.set_objective(obj)
    with patch("qilisdk.utils.classical_solvers.logger") as mock_logger:
        BruteForceSolver().solve(m)
    mock_logger.warning.assert_called_once()


def test_brute_force_with_constraint_penalty():
    # Line 106: penalty is computed from constraint evaluations in the sum
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("constrained")
    m.set_objective(x + y)
    m.add_constraint("c1", EQ(x + y, 1), lagrange_multiplier=10)
    results, sample = BruteForceSolver().solve(m)
    # Valid tour requires x+y=1; constraint is satisfied in the best sample
    assert sample[x] + sample[y] == 1
    assert results["c1"] == 0


def test_brute_force_best_sample_updated():
    # Lines 107-109: best_sample is updated when a lower objective+penalty is found
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("two_vars")
    m.set_objective(3 * x + 2 * y)
    results, sample = BruteForceSolver().solve(m)
    assert sample[x] == 0
    assert sample[y] == 0
    assert results[m.objective.label] == 0


def test_brute_force_returns_evaluate_of_best():
    # Line 110: return value is model.evaluate(best_sample), best_sample
    x = BinaryVariable("x")
    m = Model("ret_test")
    m.set_objective(1 * x)
    results, sample = BruteForceSolver().solve(m)
    assert results == m.evaluate(sample)
