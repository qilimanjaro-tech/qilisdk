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
        (_mat_RZ(math.pi / 5.0), RZ(0, phi=math.pi / 5.0).matrix),
        (_mat_RY(math.pi / 4.0), RY(0, theta=math.pi / 4.0).matrix),
        (_mat_RX(math.pi / 3.0), RX(0, theta=math.pi / 3.0).matrix),
    ],
)
def test_rotation_matrix_helpers_match_gate_matrices(helper_matrix: np.ndarray, expected_matrix: np.ndarray) -> None:
    assert np.allclose(helper_matrix, expected_matrix)


def test_u3_matrix_helper_matches_gate_matrix() -> None:
    theta, phi, lam = (math.pi / 3.0, -math.pi / 4.0, math.pi / 7.0)

    assert np.allclose(_mat_U3(theta, phi, lam), U3(0, theta=theta, phi=phi, gamma=lam).matrix)
