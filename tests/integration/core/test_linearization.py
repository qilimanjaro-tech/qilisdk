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

"""End-to-end tests for automatic constraint linearization.

These tests exercise the full ``Model`` -> ``QUBO`` -> ``Hamiltonian`` pipeline on problems
whose objective or constraints contain pseudo-Boolean terms of degree greater than two. The
Hamiltonian is diagonalized exactly, the ground state is decoded, and the recovered binary
assignment is compared against the known optimum of the original model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from qilisdk.core import EQ, LEQ, BinaryVariable, Model, ObjectiveSense

if TYPE_CHECKING:
    from qilisdk.core.model import QUBO
    from qilisdk.core.variables import BaseVariable


def _ground_state_assignment(qubo: QUBO) -> dict[BaseVariable, int]:
    """Return the binary variable assignment corresponding to the Hamiltonian ground state.

    The Hamiltonian uses the convention where qubit ``k`` occupies position
    ``nqubits - 1 - k`` inside the computational-basis state index, so we decode the ground
    state index accordingly.
    """
    ham = qubo.to_hamiltonian()
    matrix = ham.to_matrix().toarray()
    _, eigvecs = np.linalg.eigh(matrix)
    gs_index = int(np.argmax(np.abs(eigvecs[:, 0]) ** 2))
    n = ham.nqubits
    return {v: (gs_index >> (n - 1 - i)) & 1 for i, v in enumerate(qubo.qubo_objective.variables())}


def test_cubic_objective_ground_state_matches_minimum():
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    model = Model("min_neg_xyz")
    model.set_objective(-(x * y * z), sense=ObjectiveSense.MINIMIZE)
    qubo = model.to_qubo(linearization_lagrange_multiplier=20)
    assignment = _ground_state_assignment(qubo)
    assert assignment[x] * assignment[y] * assignment[z] == 1


def test_cubic_objective_maximize_ground_state():
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    model = Model("max_xyz")
    model.set_objective(x * y * z, sense=ObjectiveSense.MAXIMIZE)
    qubo = model.to_qubo(linearization_lagrange_multiplier=20)
    assignment = _ground_state_assignment(qubo)
    assert assignment[x] * assignment[y] * assignment[z] == 1


def test_cubic_equality_constraint_ground_state():
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    model = Model("forbid_triple")
    # Unconstrained optimum of -x - y - z is (1, 1, 1); the constraint forbids xyz = 1,
    # so the feasible optimum has exactly two ones.
    model.set_objective(-x - y - z)
    model.add_constraint("no_triple", EQ(x * y * z, 0), lagrange_multiplier=15)
    qubo = model.to_qubo(linearization_lagrange_multiplier=30)
    assignment = _ground_state_assignment(qubo)
    assert assignment[x] * assignment[y] * assignment[z] == 0
    assert assignment[x] + assignment[y] + assignment[z] == 2


def test_cubic_inequality_constraint_ground_state():
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    model = Model("cap_triple")
    model.set_objective(-x - y - z)
    model.add_constraint("triple_cap", LEQ(x * y * z, 0), lagrange_multiplier=15)
    qubo = model.to_qubo(linearization_lagrange_multiplier=30)
    assignment = _ground_state_assignment(qubo)
    assert assignment[x] * assignment[y] * assignment[z] == 0
    assert assignment[x] + assignment[y] + assignment[z] == 2


def test_quartic_objective_ground_state():
    a, b, c, d = (BinaryVariable(name) for name in ("a", "b", "c", "d"))
    model = Model("quartic")
    model.set_objective(a * b * c * d, sense=ObjectiveSense.MAXIMIZE)
    qubo = model.to_qubo(linearization_lagrange_multiplier=40)
    # Two auxiliaries are introduced to collapse a*b*c*d to quadratic form.
    aux_vars = [v for v in qubo.variables() if v.label.startswith("_linearization_aux")]
    assert len(aux_vars) == 2
    assignment = _ground_state_assignment(qubo)
    assert assignment[a] * assignment[b] * assignment[c] * assignment[d] == 1


def test_shared_auxiliary_reduces_hamiltonian_qubit_count():
    x, y, z, w = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z"), BinaryVariable("w")
    model = Model("shared")
    # Both cubic terms reuse the x*y pair, so a single auxiliary suffices.
    model.set_objective(x * y * z + x * y * w, sense=ObjectiveSense.MAXIMIZE)
    qubo = model.to_qubo(linearization_lagrange_multiplier=30)
    aux_vars = [v for v in qubo.variables() if v.label.startswith("_linearization_aux")]
    assert len(aux_vars) == 1
    ham = qubo.to_hamiltonian()
    # 4 original vars + 1 shared aux = 5 qubits.
    assert ham.nqubits == 5
    assignment = _ground_state_assignment(qubo)
    assert assignment[x] == 1
    assert assignment[y] == 1
    # Either z or w (or both) maximises the sum.
    assert assignment[z] == 1 or assignment[w] == 1


def test_linearize_false_rejects_cubic_constraint_end_to_end():
    x, y, z = BinaryVariable("x"), BinaryVariable("y"), BinaryVariable("z")
    model = Model("reject")
    model.set_objective(x + y + z)
    model.add_constraint("forbid", EQ(x * y * z, 1), lagrange_multiplier=10)
    with pytest.raises(ValueError, match=r"can not contain terms of order 2 or higher"):
        model.to_qubo(linearize=False)


def test_linearize_preserves_pure_quadratic_models():
    """A model already in quadratic form should pass through linearization unchanged."""
    x, y = BinaryVariable("x"), BinaryVariable("y")
    model = Model("quadratic")
    model.set_objective(-2 * x - 3 * y + 4 * x * y)
    model.add_constraint("budget", LEQ(x + y, 1), lagrange_multiplier=5)
    qubo_no_lin = model.to_qubo(linearize=False)
    qubo_lin = model.to_qubo(linearize=True)
    # Same variable set (no auxiliaries introduced).
    aux_vars = [v for v in qubo_lin.variables() if v.label.startswith("_linearization_aux")]
    assert aux_vars == []
    # Hamiltonians match up to numerical precision.
    m_no_lin = qubo_no_lin.to_hamiltonian().to_matrix().toarray()
    m_lin = qubo_lin.to_hamiltonian().to_matrix().toarray()
    assert np.allclose(m_no_lin, m_lin)
