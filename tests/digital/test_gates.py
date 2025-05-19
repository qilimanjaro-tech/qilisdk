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

import numpy as np
import pytest
from hypothesis import example, given, strategies
from numpy.testing import assert_allclose
from scipy.linalg import expm

from qilisdk.digital import CNOT, CZ, RX, RY, RZ, U1, U2, U3, H, M, S, T, X, Y, Z
from qilisdk.digital.exceptions import GateHasNoMatrixError, InvalidParameterNameError, ParametersNotEqualError
from qilisdk.digital.gates import Adjoint, Controlled, Exponential


# ------------------------------------------------------------------------------
# Helper to compare matrices
# ------------------------------------------------------------------------------
def assert_matrix_equal(actual: np.ndarray, expected: np.ndarray, rtol=1e-7, atol=1e-7):
    """
    Utility assertion for comparing two complex matrices for approximate equality.
    """
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"
    assert_allclose(actual, expected, rtol=rtol, atol=atol)


# ------------------------------------------------------------------------------
# Non-Parameterized Single-Qubit Gates
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("gate_class", "expected_name", "expected_matrix"),
    [
        (X, "X", np.array([[0, 1], [1, 0]], dtype=complex)),
        (Y, "Y", np.array([[0, -1j], [1j, 0]], dtype=complex)),
        (Z, "Z", np.array([[1, 0], [0, -1]], dtype=complex)),
        (S, "S", np.array([[1, 0], [0, 1j]], dtype=complex)),
        (T, "T", np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)),
        (H, "H", (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)),
    ],
)
def test_non_parameterized_gate(gate_class: type, expected_name: str, expected_matrix: np.ndarray):
    """Check name, matrix shape, and matrix values for single-qubit non-param gates."""
    qubit = 0
    gate = gate_class(qubit)

    # Check gate name
    assert gate.name == expected_name

    # Check that it is indeed a single-qubit gate
    assert gate.nqubits == 1

    # Check that it has no parameters
    assert gate.is_parameterized is False
    assert gate.nparameters == 0
    assert gate.parameter_names == []
    assert gate.parameters == {}

    # Check target qubits
    assert gate.target_qubits == (qubit,)
    # For these gates, there are no control qubits
    assert gate.control_qubits == ()

    # Check the matrix
    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# Measurement Gate
# ------------------------------------------------------------------------------
def test_measurement_gate():
    """
    Special test for the M (Measurement) gate, which has no matrix representation
    but still shares common structure with Gate.
    """
    qubit = 1
    gate = M(qubit)

    assert gate.name == "M"
    assert gate.nqubits == 1
    assert gate.is_parameterized is False
    assert gate.nparameters == 0
    assert gate.parameter_names == []
    assert gate.parameters == {}
    assert gate.target_qubits == (qubit,)
    assert gate.control_qubits == ()

    # The base Gate sets `_matrix` to an empty array by default (since M has no matrix).
    with pytest.raises(GateHasNoMatrixError):
        gate.matrix


# ------------------------------------------------------------------------------
# Rotation Gates
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_rx_gate(angle: float):
    qubit = 2
    gate = RX(qubit, theta=angle)

    assert gate.name == "RX"
    assert gate.is_parameterized is True
    assert gate.nparameters == 1
    assert gate.parameter_names == ["theta"]
    assert list(gate.parameters.keys()) == ["theta"]
    assert list(gate.parameters.values()) == [angle]

    cos_half = np.cos(angle / 2)
    sin_half = np.sin(angle / 2)
    expected_matrix = np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)
    assert gate.target_qubits == (qubit,)
    assert gate.control_qubits == ()


@pytest.mark.parametrize("angle", [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_ry_gate(angle: float):
    qubit = 3
    gate = RY(qubit, theta=angle)

    assert gate.name == "RY"
    assert gate.is_parameterized is True
    assert gate.nparameters == 1

    cos_half = np.cos(angle / 2)
    sin_half = np.sin(angle / 2)
    expected_matrix = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)


@pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_rz_gate(angle: float):
    qubit = 4
    gate = RZ(qubit, phi=angle)

    assert gate.name == "RZ"
    assert gate.is_parameterized is True
    assert gate.nparameters == 1

    # The file's RZ uses the same form as RY in the code.
    # That might be a placeholder, but let's test correctness.
    cos_half = np.cos(angle / 2)
    sin_half = np.sin(angle / 2)
    expected_matrix = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# U1 Gate Tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("angle", [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_u1_gate(angle: float):
    qubit = 5
    gate = U1(qubit, phi=angle)

    assert gate.name == "U1"
    assert gate.is_parameterized is True
    assert gate.nparameters == 1

    expected_matrix = np.array([[1, 0], [0, np.exp(1j * angle)]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# U2 Gate Tests
# ------------------------------------------------------------------------------
@given(phi=strategies.floats(-1e3, 1e3), gamma=strategies.floats(-1e3, 1e3))
@example(phi=0.0, gamma=0.0)
@example(phi=np.pi / 4, gamma=np.pi / 4)
@example(phi=np.pi / 2, gamma=-np.pi / 2)
@example(phi=np.pi, gamma=2 * np.pi)
def test_u2_gate(phi, gamma):
    """
    Parametrized test for the U2 gate.

    Based on the docstring and the code:
        U2(phi, gamma) = 1/sqrt(2) * [[ 1,    - e^{-i*gamma} ],
                                    [ e^{i*phi},      e^{i*(phi+gamma)}   ]]
    """
    qubit = 7
    gate = U2(qubit, phi=phi, gamma=gamma)

    # Basic checks
    assert gate.name == "U2"
    assert gate.nqubits == 1
    assert gate.is_parameterized is True
    assert gate.nparameters == 2
    assert gate.parameter_names == ["phi", "gamma"]
    assert gate.target_qubits == (qubit,)
    assert gate.control_qubits == ()

    # Check parameter values
    assert gate.parameters["phi"] == phi
    assert gate.parameters["gamma"] == gamma

    # Reconstruct the expected matrix
    factor = 1 / np.sqrt(2)
    a = 1
    b = -np.exp(1j * gamma)
    c = np.exp(1j * phi)
    d = np.exp(1j * (phi + gamma))

    expected_matrix = factor * np.array([[a, b], [c, d]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# U3 Gate Tests
# ------------------------------------------------------------------------------
@given(theta=strategies.floats(-1e3, 1e3), phi=strategies.floats(-1e3, 1e3), gamma=strategies.floats(-1e3, 1e3))
@example(theta=0.0, phi=0.0, gamma=0.0)
@example(theta=np.pi / 4, phi=np.pi / 4, gamma=np.pi / 4)
@example(theta=np.pi / 2, phi=0, gamma=np.pi)
@example(theta=-np.pi, phi=np.e, gamma=-1.11)
def test_u3_gate(theta, phi, gamma):
    """
    Parametrized test for the U3 gate.

    Based on the docstring and the code:
        U2(phi, gamma) = [[cos(theta/2), -exp(i*gamma)*sin(theta/2)],
                                    [exp(i*phi)*sin(theta/2),    exp(i*(phi+gamma))*cos(theta/2)]]
    """
    qubit = 8
    gate = U3(qubit, theta=theta, phi=phi, gamma=gamma)

    # Basic checks
    assert gate.name == "U3"
    assert gate.nqubits == 1
    assert gate.is_parameterized is True
    assert gate.nparameters == 3
    assert gate.parameter_names == ["theta", "phi", "gamma"]
    assert gate.target_qubits == (qubit,)
    assert gate.control_qubits == ()

    # Check parameter values
    assert gate.parameters["theta"] == theta
    assert gate.parameters["phi"] == phi
    assert gate.parameters["gamma"] == gamma

    # Reconstruct the expected matrix.
    a = np.cos(theta / 2)
    b = -np.exp(1j * gamma) * np.sin(theta / 2)
    c = np.exp(1j * phi) * np.sin(theta / 2)
    d = np.exp(1j * (phi + gamma)) * np.cos(theta / 2)

    expected_matrix = np.array([[a, b], [c, d]], dtype=complex)

    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# CNOT Gate Tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("control", "target"),
    [
        (0, 1),
        (1, 0),
        (3, 5),
        (5, 3),
    ],
)
def test_cnot_gate(control: int, target: int):
    """
    Basic tests for the 2-qubit CNOT gate.
    The matrix is:
      [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]]
    irrespective of qubit indices, but we check that the
    control_qubits and target_qubits properties are set properly.
    """
    gate = CNOT(control, target)

    # Check class hierarchy and generics
    assert isinstance(gate, Controlled)
    assert gate.is_modified_from(X)

    # Check name, nqubits, parameters
    assert gate.name == "CNOT"
    assert gate.nqubits == 2
    assert gate.is_parameterized is False
    assert gate.nparameters == 0
    assert gate.parameter_names == []
    assert gate.parameters == {}

    # Check qubits
    assert gate.control_qubits == (control,)
    assert gate.target_qubits == (target,)
    assert gate.qubits == (control, target)

    # Check matrix
    expected_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# CZ Gate Tests
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("control", "target"),
    [
        (0, 1),
        (1, 0),
        (2, 3),
        (4, 7),
    ],
)
def test_cz_gate(control, target):
    """
    Basic tests for the 2-qubit CZ gate.
    The code and docstring currently show a matrix identical to CNOT's,
    but let's just confirm it matches what's in the code:
      [[1, 0, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1],
       [0, 0, 1, 0]]
    """
    gate = CZ(control, target)

    assert gate.name == "CZ"
    assert gate.nqubits == 2
    assert gate.is_parameterized is False
    assert gate.nparameters == 0

    # Check qubits
    assert gate.control_qubits == (control,)
    assert gate.target_qubits == (target,)
    assert gate.qubits == (control, target)

    # Check matrix
    # This is as coded, though it's unusual for a CZ gate:
    expected_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)
    assert_matrix_equal(gate._matrix, expected_matrix)


# ------------------------------------------------------------------------------
# Tests for set_parameters() and set_parameter_values()
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("gate_class", "ctor_kwargs", "valid_dict", "invalid_dict", "valid_list", "invalid_list"),
    [
        # Single-parameter gates
        (RX, {"theta": 0.0}, {"theta": np.pi}, {"phi": np.e}, [1.23], [1.23, 2.34]),
        (RY, {"theta": 0.0}, {"theta": 1.11}, {"gamma": 3.33}, [2.22], []),
        (RZ, {"phi": 0.0}, {"phi": -0.99}, {"theta": 0.77}, [1.11], [1.11, 2.22]),
        (U1, {"phi": 0.0}, {"phi": 4.56}, {"theta": 1.23}, [5.67], []),
        # Two-parameter gate (U2 => phi, gamma)
        (
            U2,
            {"phi": 0.0, "gamma": 0.0},
            {"phi": 1.23, "gamma": 2.34},  # valid dict
            {"theta": 3.45},  # invalid dict -> "theta" not recognized
            [4.56, 5.67],  # valid list, matching 2 parameters
            [1.23],
        ),  # invalid list (only 1 param)
        # Three-parameter gate (U3 => theta, phi, gamma)
        (
            U3,
            {"theta": 0.0, "phi": 0.0, "gamma": 0.0},
            {"theta": 1.11, "phi": 2.22, "gamma": 3.33},  # valid dict
            {"junk": 9.99},  # invalid dict
            [4.44, 5.55, 6.66],  # valid list
            [7.77, 8.88],
        ),  # invalid list (2 instead of 3)
    ],
)
def test_gate_parameter_methods(gate_class, ctor_kwargs, valid_dict, invalid_dict, valid_list, invalid_list):
    """
    Tests set_parameters() and set_parameter_values() for different gate classes.
    Checks both correct usage (happy path) and exceptions for invalid inputs.
    """

    # Create the gate, specifying qubit index = 0 (arbitrary)
    gate = gate_class(0, **ctor_kwargs)

    # 1) set_parameters() with valid_dict
    gate.set_parameters(valid_dict)
    # Verify the gate's 'parameters' match the new values
    for param_name, param_value in valid_dict.items():
        assert gate.parameters[param_name] == param_value, f"Parameter '{param_name}' not updated to {param_value}"

    # 2) set_parameters() with an invalid dict (unknown parameter name)
    with pytest.raises(InvalidParameterNameError):
        gate.set_parameters(invalid_dict)

    # 3) set_parameter_values() with a valid list
    gate.set_parameter_values(valid_list)
    param_names = gate.parameter_names
    for i, val in enumerate(valid_list):
        assert gate.parameters[param_names[i]] == val, f"Parameter '{param_names[i]}' not updated to {val}"

    # 4) set_parameter_values() with an invalid list (length mismatch)
    with pytest.raises(ParametersNotEqualError):
        gate.set_parameter_values(invalid_list)


# ------------------------------------------------------------------------------
# Controlled() Modification Tests - Hardcoded Expected Matrices
# ------------------------------------------------------------------------------
def test_controlled_gate_single_control():
    """
    Test that a single-controlled gate created from a basic X gate behaves as expected.
    """
    qubit = 1
    base_gate = X(qubit)
    control = 0
    controlled_gate = base_gate.controlled(control)

    # Check class hierarchy and generics
    assert isinstance(controlled_gate, Controlled)
    assert controlled_gate.is_modified_from(X)

    # Check qubit assignments.
    assert controlled_gate.control_qubits == (control,)
    assert controlled_gate.target_qubits == (qubit,)
    assert controlled_gate.qubits == (control, qubit)
    # For a single control, the name should be "C" + base_gate.name (e.g. "CX").
    assert controlled_gate.name == "C" + base_gate.name
    # Total qubits should be 1 control + 1 target = 2.
    assert controlled_gate.nqubits == 2

    # Hardcoded expected matrix for a single-controlled X gate (CNOT).
    # It is given by I₄ + kron(P, (X - I₂)), where
    # P = [[0, 0],
    #      [0, 1]] and I₂ is the 2x2 identity.
    # This results in:
    expected_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    assert_matrix_equal(controlled_gate.matrix, expected_matrix)


def test_controlled_gate_multiple_controls():
    """
    Test that a controlled gate with two control qubits created from a basic H gate
    sets the properties and computes the expected matrix correctly.
    """
    qubit = 2
    base_gate = H(qubit)
    controls = (0, 1)
    controlled_gate = base_gate.controlled(*controls)

    # Check class hierarchy and generics
    assert isinstance(controlled_gate, Controlled)
    assert controlled_gate.is_modified_from(H)

    # Check qubit assignments.
    assert controlled_gate.control_qubits == controls
    assert controlled_gate.target_qubits == (qubit,)
    assert controlled_gate.qubits == (*controls, qubit)
    # For 2 controls, the name should be "CC" + base_gate.name (e.g. "CCH").
    assert controlled_gate.name == "CC" + base_gate.name
    # Total qubits should be 2 controls + 1 target = 3.
    assert controlled_gate.nqubits == 3

    # Hardcoded expected matrix for a two-controlled H gate.
    # The overall dimension is 2^(2+1)=8. The controlled gate matrix is:
    # I₈ + kron(P, (H - I₂)) where P = kron([[0,0],[0,1]], [[0,0],[0,1]])
    # For two controls, P is a 4x4 matrix that is all zeros except for a 1 in the bottom-right position.
    # Thus, the lower-right 2x2 block of I₈ is replaced with H.
    expected_matrix = np.eye(8, dtype=complex)
    H_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    # The block corresponding to the control state |11> is at indices 6 and 7.
    expected_matrix[6:8, 6:8] = H_matrix
    assert_matrix_equal(controlled_gate.matrix, expected_matrix)


def test_controlled_gate_parameter_update():
    """
    Test that updating the parameters through the controlled gate correctly propagates
    to the underlying unitary gate and updates the overall matrix.
    """
    qubit = 1
    base_gate = RX(qubit, theta=0.0)
    controlled_gate = base_gate.controlled(0)

    # Check class hierarchy and generics
    assert isinstance(controlled_gate, Controlled)
    assert controlled_gate.is_modified_from(RX)

    new_theta = np.pi / 2
    controlled_gate.set_parameters({"theta": new_theta})
    # Underlying gate parameters should be updated.
    assert controlled_gate.parameters["theta"] == new_theta

    # Compute the updated RX matrix.
    cos_half = np.cos(new_theta / 2)
    sin_half = np.sin(new_theta / 2)
    rx_matrix = np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)

    # For a single control, the expected controlled matrix is:
    # I₄ with the lower-right 2x2 block replaced by the updated RX matrix.
    expected_matrix = np.eye(4, dtype=complex)
    expected_matrix[2:4, 2:4] = rx_matrix
    assert_matrix_equal(controlled_gate.matrix, expected_matrix)


# ------------------------------------------------------------------------------
# Adjoint() Modification Tests
# ------------------------------------------------------------------------------
def test_adjoint_gate():
    """
    Test that the adjoint gate returns the conjugate transpose of the original gate's matrix
    and that its name is suffixed with the adjoint symbol (†).
    """
    qubit = 3
    theta = np.pi / 3
    base_gate = RX(qubit, theta=theta)
    adj_gate = base_gate.adjoint()

    # Check class hierarchy and generics
    assert isinstance(adj_gate, Adjoint)
    assert adj_gate.is_modified_from(RX)

    # The adjoint gate name should be the original name with a '†' appended.
    assert adj_gate.name == base_gate.name + "†"
    # The target qubits should remain unchanged.
    assert adj_gate.target_qubits == base_gate.target_qubits

    # The matrix should be the conjugate transpose of the base matrix.
    expected_matrix = base_gate.matrix.conj().T
    assert_matrix_equal(adj_gate.matrix, expected_matrix)


def test_adjoint_gate_parameter_update():
    """
    Test that updating parameters via the adjoint gate propagates correctly and updates its matrix.
    """
    qubit = 2
    base_gate = RX(qubit, theta=0.0)
    adj_gate = base_gate.adjoint()

    # Check class hierarchy and generics
    assert isinstance(adj_gate, Adjoint)
    assert adj_gate.is_modified_from(RX)

    new_theta = np.pi / 4
    adj_gate.set_parameters({"theta": new_theta})
    # Underlying gate parameters should be updated.
    assert adj_gate.parameters["theta"] == new_theta
    # The adjoint matrix should update to be the conjugate transpose of the new base matrix.
    expected_matrix = base_gate.matrix.conj().T
    assert_matrix_equal(adj_gate.matrix, expected_matrix)


# ------------------------------------------------------------------------------
# Exponential() Modification Tests
# ------------------------------------------------------------------------------
def test_exponential_gate():
    """
    Test that the exponential gate computes the matrix exponential of the original gate's matrix
    and that its name is prefixed with 'e^'.
    """
    qubit = 4
    theta = np.pi / 4
    base_gate = RY(qubit, theta=theta)
    exp_gate = base_gate.exponential()

    # Check class hierarchy and generics
    assert isinstance(exp_gate, Exponential)
    assert exp_gate.is_modified_from(RY)

    # Check that the name is "e^" followed by the base gate's name.
    assert exp_gate.name == f"e^{base_gate.name}"
    # The target qubits should remain unchanged.
    assert exp_gate.target_qubits == base_gate.target_qubits

    # Verify that the matrix is the exponential of the base matrix.
    expected_matrix = expm(base_gate.matrix)
    assert_matrix_equal(exp_gate.matrix, expected_matrix)


def test_exponential_gate_parameter_update():
    """
    Test that updating parameters via the exponential gate propagates correctly and updates its matrix.
    """
    qubit = 3
    base_gate = RY(qubit, theta=0.0)
    exp_gate = base_gate.exponential()

    # Check class hierarchy and generics
    assert isinstance(exp_gate, Exponential)
    assert exp_gate.is_modified_from(RY)

    new_theta = np.pi / 6
    exp_gate.set_parameters({"theta": new_theta})
    # Underlying gate parameters should be updated.
    assert exp_gate.parameters["theta"] == new_theta
    # The matrix should update to the exponential of the new base matrix.
    expected_matrix = expm(base_gate.matrix)
    assert_matrix_equal(exp_gate.matrix, expected_matrix)
