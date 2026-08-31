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

from qilisdk.core.comparison import EQ
from qilisdk.core.model import QUBO, Model, ObjectiveSense
from qilisdk.core.variables import BinaryVariable, Domain, OneHot, SpinVariable, Variable
from qilisdk.utils.classical_solvers import (
    BruteForceSolver,
    ClassicalSolver,
    ClassicalSolverResult,
    ScipySolver,
    SimulatedAnnealingSolver,
)
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


def test_classical_solver_result_exposes_the_solution():
    x, y = BinaryVariable("x"), BinaryVariable("y")
    sample = {x: 1, y: 0}
    result = ClassicalSolverResult({"obj": -3.0, "c1": 0.0, "c2": 5.0}, sample, "obj")

    assert result.objective == -3.0
    assert result.objective_label == "obj"
    assert result.sample == sample
    assert result.results == {"obj": -3.0, "c1": 0.0, "c2": 5.0}
    assert result.constraints == {"c1": 0.0, "c2": 5.0}


def test_classical_solver_result_copies_its_inputs():
    x = BinaryVariable("x")
    results, sample = {"obj": 1.0}, {x: 1}
    result = ClassicalSolverResult(results, sample, "obj")

    # Mutating the inputs, or what the properties hand back, must not change the result
    results["obj"] = 99.0
    sample[x] = 0
    result.results["obj"] = 99.0
    result.sample[x] = 0

    assert result.objective == 1.0
    assert result.sample == {x: 1}


def test_classical_solver_result_without_constraints():
    result = ClassicalSolverResult({"obj": 2.0}, {}, "obj")
    assert result.constraints == {}


def test_classical_solver_result_objective_missing_from_results_raises():
    result = ClassicalSolverResult({"c1": 0.0}, {}, "obj")
    with pytest.raises(KeyError):
        _ = result.objective


def test_classical_solver_result_from_model_evaluates_the_sample():
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("from_model")
    m.set_objective(3 * x + 2 * y)
    m.add_constraint("c1", EQ(x + y, 1), lagrange_multiplier=10)
    sample = {x: 1, y: 0}

    result = ClassicalSolverResult.from_model(m, sample)

    assert result.objective_label == m.objective.label
    assert result.sample == sample
    assert result.results == m.evaluate(sample)
    assert result.objective == 3
    assert result.constraints == {"c1": 0}


def test_classical_solver_result_repr():
    x = BinaryVariable("x")
    result = ClassicalSolverResult({"obj": -1.0}, {x: 1}, "obj")
    assert repr(result) == "ClassicalSolverResult(objective=-1.0, sample={x: 1}, results={'obj': -1.0})"


def test_classical_solver_solve_raises():
    m = Model("m")
    x = BinaryVariable("x")
    m.set_objective(1 * x)
    solver = ClassicalSolver()
    with pytest.raises(NotImplementedError):
        solver.solve(m)


def test_brute_force_binary_variable_domain():
    x = BinaryVariable("x")
    m = Model("bin")
    m.set_objective(1 * x)
    result = BruteForceSolver().solve(m)
    assert result.sample[x] == 0
    assert result.objective == 0


def test_brute_force_maximize():
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    result = BruteForceSolver().solve(m)
    assert result.sample[x] == 1


def test_brute_force_integer_variable_enumeration():
    v = Variable("v", Domain.POSITIVE_INTEGER, bounds=(0, 3))
    m = Model("int_model")
    m.set_objective(v * 1)
    result = BruteForceSolver().solve(m)
    assert result.sample[v] == 0


def test_brute_force_unsupported_variable_raises():
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(s)
    solver = BruteForceSolver()
    with pytest.raises(ValueError, match="not supported"):
        solver.solve(m)


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
    result = BruteForceSolver().solve(m)
    assert result.sample[x] + result.sample[y] == 1
    assert result.results["c1"] == 0


def test_brute_force_best_sample_updated():
    x, y = BinaryVariable("x"), BinaryVariable("y")
    m = Model("two_vars")
    m.set_objective(3 * x + 2 * y)
    result = BruteForceSolver().solve(m)
    assert result.sample[x] == 0
    assert result.sample[y] == 0
    assert result.objective == 0


def test_brute_force_returns_evaluate_of_best():
    x = BinaryVariable("x")
    m = Model("ret_test")
    m.set_objective(1 * x)
    result = BruteForceSolver().solve(m)
    assert result.results == m.evaluate(result.sample)


def test_variable_bounds_explicit_bounds():
    v = Variable("v", Domain.REAL, bounds=(-2.5, 4.0))
    assert np.allclose(_variable_bounds(v), (-2.5, 4.0))


def test_variable_bounds_binary_defaults_to_domain_limits():
    assert np.allclose(_variable_bounds(BinaryVariable("b")), (0.0, 1.0))


def test_variable_bounds_missing_lower_falls_back_to_domain_min():
    v = Variable("v", Domain.POSITIVE_INTEGER, bounds=(None, 5))
    lower, upper = _variable_bounds(v)
    assert np.isclose(lower, 0.0)
    assert np.isclose(upper, 5.0)


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
    result = ScipySolver().solve(m)
    assert result.sample[x] == 0


def test_scipy_solver_maximize():
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    result = ScipySolver(method="differential_evolution", seed=1).solve(m)
    assert result.sample[x] == 1


def test_scipy_solver_integer_variable():
    x = Variable("x", Domain.INTEGER, bounds=(0, 7))
    m = Model("int_model")
    m.set_objective((x - 5) * (x - 5))
    result = ScipySolver(method="differential_evolution", seed=1).solve(m)
    assert result.sample[x] == 5


def test_scipy_solver_real_variable():
    y = Variable("y", Domain.REAL, bounds=(0, 10))
    m = Model("real_model")
    m.set_objective((y - 3.7) * (y - 3.7))
    result = ScipySolver(method="differential_evolution", seed=1).solve(m)
    assert np.isclose(result.sample[y], 3.7, atol=1e-1)


def test_scipy_solver_unsupported_variable_raises():
    s = SpinVariable("s")
    m = Model("spin_model")
    m.set_objective(s)
    solver = ScipySolver()
    with pytest.raises(ValueError, match="not supported"):
        solver.solve(m)


def test_scipy_solver_returns_evaluate_of_best():
    x = BinaryVariable("x")
    m = Model("ret_test")
    m.set_objective(1 * x)
    result = ScipySolver().solve(m)
    assert result.results == m.evaluate(result.sample)


def test_simulated_annealing_minimizes_binary():
    x = BinaryVariable("x")
    qubo = QUBO("bin")
    qubo.set_objective(1 * x)
    result = SimulatedAnnealingSolver().solve(qubo)
    assert result.sample[x] == 0
    assert result.results == qubo.evaluate(result.sample)


def test_simulated_annealing_maximize():
    x = BinaryVariable("x")
    m = Model("max_bin")
    m.set_objective(1 * x, sense=ObjectiveSense.MAXIMIZE)
    result = SimulatedAnnealingSolver().solve(m.to_qubo())
    assert result.sample[x] == 1


def test_simulated_annealing_matches_brute_force_on_random_ising():
    m = Model.random_ising(10, seed=7)
    annealed_result = SimulatedAnnealingSolver(num_reads=50, seed=3).solve(m.to_qubo())
    exhaustive_result = BruteForceSolver().solve(m)
    assert np.isclose(annealed_result.objective, exhaustive_result.objective)


def test_simulated_annealing_respects_constraint_penalties():
    m = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 1], max_weight=3)
    result = SimulatedAnnealingSolver(num_reads=50, seed=1).solve(m.to_qubo())
    assert np.isclose(result.objective, -7)
    # The slack form of the weight constraint is satisfied, so its penalty contributes nothing
    assert np.isclose(result.results["weight"], 0)


def test_simulated_annealing_samples_the_bits_of_an_encoded_variable():
    x = Variable("x", Domain.POSITIVE_INTEGER, bounds=(0, 3), encoding=OneHot)
    m = Model("encoded")
    m.set_objective((x - 2) * (x - 2))
    result = SimulatedAnnealingSolver(num_reads=50, seed=1).solve(m.to_qubo())
    # The QUBO is defined over the bits of the encoding, which decode back to the best value of x
    bits = [result.sample[bit] for bit in x.bin_vars]
    assert x.evaluate({x: bits}) == 2


def test_simulated_annealing_is_deterministic_for_a_given_seed():
    qubo = Model.random_ising(12, seed=2).to_qubo()
    first_result = SimulatedAnnealingSolver(num_reads=8, seed=11).solve(qubo)
    second_result = SimulatedAnnealingSolver(num_reads=8, seed=11).solve(qubo)
    assert first_result.results == second_result.results


def test_simulated_annealing_explicit_beta_range():
    m = Model.random_ising(8, seed=5)
    annealed_result = SimulatedAnnealingSolver(num_reads=20, seed=1, beta_range=(0.01, 10.0)).solve(m.to_qubo())
    exhaustive_result = BruteForceSolver().solve(m)
    assert np.isclose(annealed_result.objective, exhaustive_result.objective)


def test_simulated_annealing_empty_qubo_has_nothing_to_anneal():
    qubo = QUBO("empty")
    result = SimulatedAnnealingSolver().solve(qubo)
    assert result.sample == {}
    assert result.results == qubo.evaluate(result.sample)


def test_simulated_annealing_requires_a_qubo():
    m = Model.random_ising(4, seed=1)
    solver = SimulatedAnnealingSolver()
    with pytest.raises(ValueError, match="requires a QUBO"):
        solver.solve(m)


@pytest.mark.parametrize("beta_range", [(0.0, 1.0), (1.0, -1.0), (10.0, 1.0)])
def test_simulated_annealing_invalid_beta_range_raises(beta_range):
    qubo = Model.random_ising(4, seed=1).to_qubo()
    solver = SimulatedAnnealingSolver(beta_range=beta_range)
    with pytest.raises(ValueError, match="inverse temperature"):
        solver.solve(qubo)


@pytest.mark.parametrize(("num_reads", "num_sweeps"), [(0, 10), (10, 0)])
def test_simulated_annealing_invalid_effort_raises(num_reads, num_sweeps):
    qubo = Model.random_ising(4, seed=1).to_qubo()
    solver = SimulatedAnnealingSolver(num_reads=num_reads, num_sweeps=num_sweeps)
    with pytest.raises(ValueError, match="must be positive"):
        solver.solve(qubo)
