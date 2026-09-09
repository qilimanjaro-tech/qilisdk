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

import math

import numpy as np
import pytest

from qilisdk.digital.circuit_transpiler_passes.numeric_helpers import (
    _EPS,
    _is_close_mod_2pi,
    _mat_RX,
    _mat_RY,
    _mat_RZ,
    _mat_U3,
    _round_float,
    _u3_and_phase_from_unitary,
    _unitary_sqrt_2x2,
)
from qilisdk.digital.gates import RX, RY, RZ, U3


@pytest.mark.parametrize("value", [_EPS / 2.0, -_EPS / 2.0])
def test_round_float_snaps_tiny_values_to_zero(value: float) -> None:
    assert np.isclose(_round_float(value), 0.0)


def test_round_float_respects_requested_precision() -> None:
    assert np.isclose(_round_float(1.23456, d=3), 1.235)


def test_is_close_mod_2pi_accepts_wrapped_angles() -> None:
    assert _is_close_mod_2pi(math.pi / 7.0, math.pi / 7.0 + 4.0 * math.pi)


def test_is_close_mod_2pi_rejects_meaningful_difference() -> None:
    assert not _is_close_mod_2pi(0.0, 2.0 * _EPS)


@pytest.mark.parametrize(
    ("helper_matrix", "expected_matrix"),
    [
        (_mat_RZ(math.pi / 5.0), RZ(0, phi=math.pi / 5.0).matrix.dense()),
        (_mat_RY(math.pi / 4.0), RY(0, theta=math.pi / 4.0).matrix.dense()),
        (_mat_RX(math.pi / 3.0), RX(0, theta=math.pi / 3.0).matrix.dense()),
    ],
)
def test_rotation_matrix_helpers_match_gate_matrices(helper_matrix: np.ndarray, expected_matrix: np.ndarray) -> None:
    assert np.allclose(helper_matrix, expected_matrix)


def test_u3_matrix_helper_matches_gate_matrix() -> None:
    theta, phi, lam = (math.pi / 3.0, -math.pi / 4.0, math.pi / 7.0)

    assert np.allclose(_mat_U3(theta, phi, lam), U3(0, theta=theta, phi=phi, gamma=lam).matrix.dense())


@pytest.mark.parametrize(
    ("name", "unitary"),
    [
        ("generic", np.exp(0.3j) * _mat_U3(math.pi / 3.0, -math.pi / 4.0, math.pi / 7.0)),
        ("identity", np.eye(2, dtype=complex)),
        ("diagonal", np.exp(0.7j) * np.diag([1.0, np.exp(1.1j)]).astype(complex)),
        ("anti_diagonal", np.array([[0.0, np.exp(0.4j)], [np.exp(-1.2j), 0.0]], dtype=complex)),
        ("hadamard", np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)),
    ],
)
def test_u3_and_phase_from_unitary_is_exact(name: str, unitary: np.ndarray) -> None:
    theta, phi, gamma, alpha = _u3_and_phase_from_unitary(unitary)

    reconstructed = np.exp(1j * alpha) * U3(0, theta=theta, phi=phi, gamma=gamma).matrix.dense()
    assert np.allclose(unitary, reconstructed), f"Phased U3 reconstruction failed for {name}"


def test_u3_and_phase_from_unitary_rejects_non_2x2() -> None:
    with pytest.raises(ValueError, match="Expected 2x2 unitary"):
        _u3_and_phase_from_unitary(np.ones((3, 2), dtype=complex))


def test_u3_and_phase_from_unitary_rejects_singular() -> None:
    with pytest.raises(ValueError, match="Matrix is singular"):
        _u3_and_phase_from_unitary(np.array([[1, 0], [0, 0]], dtype=complex))


def _haar_unitary(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ginibre = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(ginibre)
    return q * (np.diag(r) / np.abs(np.diag(r)))


SQRT_INPUTS = [
    ("identity", np.eye(2, dtype=complex)),
    ("negative_identity", -np.eye(2, dtype=complex)),  # degenerate spectrum, trace == -2
    ("phased_identity", np.exp(1.234j) * np.eye(2, dtype=complex)),
    ("pauli_x", np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)),
    ("pauli_y", np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)),
    ("pauli_z", np.diag([1.0, -1.0]).astype(complex)),
    ("hadamard", np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)),
    ("rz_pi", _mat_RZ(math.pi)),
    ("ry_two_pi", _mat_RY(2.0 * math.pi)),  # equals -I up to rounding
    ("rx_two_pi", _mat_RX(2.0 * math.pi)),
    ("u3_half_turn", _mat_U3(math.pi, math.pi / 3.0, -math.pi / 5.0)),
    *[(f"haar_{seed}", _haar_unitary(seed)) for seed in range(8)],
]


@pytest.mark.parametrize(("name", "unitary"), SQRT_INPUTS)
def test_unitary_sqrt_squares_back_to_the_input(name: str, unitary: np.ndarray) -> None:
    root = _unitary_sqrt_2x2(unitary)

    assert np.allclose(root @ root, unitary, atol=1e-9), f"Square root does not square back for {name}"


@pytest.mark.parametrize(("name", "unitary"), SQRT_INPUTS)
def test_unitary_sqrt_is_itself_unitary(name: str, unitary: np.ndarray) -> None:
    """A non-unitary "square root" cannot be turned into a gate, even when it squares back correctly."""
    root = _unitary_sqrt_2x2(unitary)

    assert np.allclose(root.conj().T @ root, np.eye(2), atol=1e-9), f"Square root is not unitary for {name}"


def test_unitary_sqrt_rejects_non_2x2() -> None:
    with pytest.raises(ValueError, match="Expected 2x2 unitary"):
        _unitary_sqrt_2x2(np.eye(3, dtype=complex))


def test_unitary_sqrt_rejects_non_unitary() -> None:
    with pytest.raises(ValueError, match="Expected a unitary matrix"):
        _unitary_sqrt_2x2(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex))
