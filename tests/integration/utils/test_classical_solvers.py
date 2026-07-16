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

"""End-to-end tests that the classical solvers recover the true optimum of simple models.

``BruteForceSolver`` exhaustively enumerates every assignment, so its objective value is the
ground-truth optimum. Each other solver is expected to match that optimum on these small,
well-conditioned problems (the global SciPy method is used for ``ScipySolver`` since local
methods stall on the discrete, rounded landscape).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from qilisdk.core import Model
from qilisdk.core.variables import BaseVariable, Domain, Number, Variable
from qilisdk.utils.classical_solvers import BruteForceSolver, ScipSolver, ScipySolver

_HAS_SCIP = importlib.util.find_spec("pyscipopt") is not None


def _cost(model: Model, sample: dict[BaseVariable, Number]) -> float:
    """Objective value plus constraint penalties for a given assignment."""
    results = model.evaluate(sample)
    objective = results[model.objective.label].real
    penalty = sum(results[c.label].real for c in model.constraints)
    return objective + penalty


def _solve(solver_name: str, model: Model) -> dict[BaseVariable, Number]:
    """Solve ``model`` with the named solver and return the sample assignment."""
    if solver_name == "scipy":
        _, sample = ScipySolver(method="differential_evolution", seed=1).solve(model)
        return sample
    _, sample = ScipSolver().solve(model)
    return sample


SOLVERS = [
    pytest.param("scipy", id="scipy"),
    pytest.param(
        "scip",
        id="scip",
        marks=pytest.mark.skipif(not _HAS_SCIP, reason="ScipSolver requires the 'scip' optional dependency"),
    ),
]


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_solves_knapsack(solver_name: str):
    model = Model.knapsack(values=[5, 4, 3], weights=[3, 2, 1], max_weight=3)
    _, brute_sample = BruteForceSolver().solve(model)
    sample = _solve(solver_name, model)
    assert np.isclose(_cost(model, sample), _cost(model, brute_sample))


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_solves_max_cut(solver_name: str):
    model = Model.max_cut(edges=[(0, 1), (1, 2), (2, 0), (2, 3)])
    _, brute_sample = BruteForceSolver().solve(model)
    sample = _solve(solver_name, model)
    assert np.isclose(_cost(model, sample), _cost(model, brute_sample))


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_solves_random_ising(solver_name: str):
    model = Model.random_ising(num_variables=8, seed=1)
    _, brute_sample = BruteForceSolver().solve(model)
    sample = _solve(solver_name, model)
    assert np.isclose(_cost(model, sample), _cost(model, brute_sample))


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_solves_integer_quadratic(solver_name: str):
    x = Variable("x", Domain.INTEGER, bounds=(0, 7))
    model = Model("integer")
    model.set_objective((x - 5) * (x - 5))
    sample = _solve(solver_name, model)
    assert sample[x] == 5


@pytest.mark.parametrize("solver_name", SOLVERS)
def test_solves_real_quadratic(solver_name: str):
    y = Variable("y", Domain.REAL, bounds=(0, 10))
    model = Model("real")
    model.set_objective((y - 3.7) * (y - 3.7))
    sample = _solve(solver_name, model)
    assert np.isclose(sample[y], 3.7, atol=1e-1)
