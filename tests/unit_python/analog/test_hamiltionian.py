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

from typing import ClassVar

import numpy as np
import pytest

from qilisdk.analog.exceptions import InvalidHamiltonianOperation
from qilisdk.analog.hamiltonian import (
    Hamiltonian,
    I,
    PauliI,
    PauliOperator,
    PauliX,
    PauliY,
    PauliZ,
    X,
    Y,
    Z,
    _get_pauli,
)
from qilisdk.core import Domain, Parameter, QTensor, Variable
from qilisdk.core.variables import BinaryVariable
from qilisdk.settings import Precision, get_settings

COMPLEX_DTYPE = get_settings().complex_precision.dtype


# Helper function to convert sparse matrix to dense NumPy array.
def dense(ham: Hamiltonian) -> np.ndarray:
    return ham.to_matrix().toarray()


def test_pauli_matrix_dtype_updates_with_settings_change():
    settings = get_settings()
    original_precision = settings.complex_precision
    try:
        settings.complex_precision = Precision.COMPLEX_128
        assert PauliX(0).matrix.dtype == np.dtype(np.complex128)
        settings.complex_precision = Precision.COMPLEX_64
        assert PauliX(0).matrix.dtype == np.dtype(np.complex64)
    finally:
        settings.complex_precision = original_precision


def test_hamiltonian_matrix_dtype_updates_with_settings_change():
    settings = get_settings()
    original_precision = settings.complex_precision
    settings.complex_precision = Precision.COMPLEX_128
    assert Z(0).to_matrix().dtype == np.dtype(np.complex128)
    settings.complex_precision = Precision.COMPLEX_64
    assert Z(0).to_matrix().dtype == np.dtype(np.complex64)
    settings.complex_precision = original_precision


def test_parameters():
    x = BinaryVariable("x")

    with pytest.raises(
        ValueError, match=r"Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
    ):
        Hamiltonian({(PauliX(0),): x, (PauliX(1),): 1})
    with pytest.raises(
        ValueError, match=r"Only Parameters are allowed to be used in hamiltonians. Generic Variables are not supported"
    ):
        Hamiltonian({(PauliX(0),): x + 1, (PauliX(1),): 1})

    y = Parameter("y", 1.5)
    H = Hamiltonian({(PauliX(0),): y + 1, (PauliX(1),): 1})
    assert H.get_parameters() == {"y": 1.5}

    z = Parameter("z", 1.5)
    H = Hamiltonian({(PauliX(0),): z, (PauliX(1),): 1})
    assert H.get_parameters() == {"z": 1.5}


def test_hamiltonian_does_not_expose_public_parameters_attribute():
    hamiltonian = Parameter("theta", 0.4) * Z(0)

    with pytest.raises(AttributeError):
        _ = hamiltonian.parameters

    assert hamiltonian.get_parameters() == {"theta": 0.4}


def test_get_pauli_returns_correct_instance():
    assert isinstance(_get_pauli("X", 0), PauliX)
    assert isinstance(_get_pauli("Y", 1), PauliY)
    assert isinstance(_get_pauli("Z", 2), PauliZ)
    assert isinstance(_get_pauli("I", 3), PauliI)


def test_get_pauli_raises_on_invalid_name():
    with pytest.raises(ValueError, match="Unknown Pauli operator name: W"):
        _get_pauli("W", 0)


# -----------------------------
#  ADDITION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (Z(0) + Z(1), Z(0) + Z(1)),
        (Z(0) + Z(0), 2 * Z(0)),
        (Z(0) + Z(1) + Z(0), 2 * Z(0) + Z(1)),
        (Z(0) + Z(1) + Z(0) + 1, 1 + 2 * Z(0) + Z(1)),
        (X(0) + X(1), X(0) + X(1)),
        (X(0) + X(0), 2 * X(0)),
        (X(0) + X(1) + X(0), 2 * X(0) + X(1)),
        (X(0) + X(1) + X(0) + 1, 1 + 2 * X(0) + X(1)),
        (Y(0) + Y(1), Y(0) + Y(1)),
        (Y(0) + Y(0), 2 * Y(0)),
        (Y(0) + Y(1) + Y(0), 2 * Y(0) + Y(1)),
        (Y(0) + Y(1) + Y(0) + 1, 1 + 2 * Y(0) + Y(1)),
        (1 + Z(0) + Z(1) + Z(0), 1 + 2 * Z(0) + Z(1)),
        (1 + Z(0) + 3 + Z(1) + Z(0) + 2j, (4 + 2j) + 2 * Z(0) + Z(1)),
        ((Z(0)) + (Z(0) - Z(0)), Z(0)),
        ((Z(0) + Z(2) + 0), Z(0) + Z(2)),
    ],
)
def test_addition(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


def test_invalid_addition_operation():
    with pytest.raises(InvalidHamiltonianOperation, match=r"Invalid addition between Hamiltonian and str"):
        _ = (Z(0) + Z(2)) + "Z"


# -----------------------------
#  SUBTRACTION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (Z(0) - Z(1), Z(0) - Z(1)),
        (Z(0) - Z(0), 0),  # i.e. 0
        ((Z(0) + 2) - 1, Z(0) + 1),
        (5 - X(0), 5 + (-1 * X(0))),
        ((Z(0) + Z(1)) - Z(1), Z(0)),
        ((Z(0) + 3) - 3, Z(0)),
        (Z(0) - Z(1) + Z(0), 2 * Z(0) - Z(1)),
        (1 - Z(0) + Z(1) + Z(0), 1 + Z(1)),
        (1 + Z(0) - 3 + Z(1) + Z(0) - 2j, (-2 - 2j) + 2 * Z(0) + Z(1)),
        ((Z(0) + Z(2) - 0), Z(0) + Z(2)),
    ],
)
def test_subtraction(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


def test_invalid_subtraction_operation():
    with pytest.raises(InvalidHamiltonianOperation, match=r"Invalid subtraction between Hamiltonian and str"):
        _ = (Z(0) + Z(2)) - "Z"


# -----------------------------
#  MULTIPLICATION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (Z(0) * Z(0), Hamiltonian({(PauliI(0),): 1})),
        (Z(0) * X(0), 1j * Y(0)),
        (X(0) * Z(0), -1j * Y(0)),
        (Z(0) * Z(1), Z(0) * Z(1)),
        (X(0) * Z(1), X(0) * Z(1)),
        (2 * Z(0), 2 * Z(0)),
        ((2 + Z(0)) * ((2 + Z(0)) * 0), 0),
        ((2 + Z(0)) * 3 * I(0), 6 + 3 * Z(0)),
        (Z(0) * 3, 3 * Z(0)),
        ((Z(0) + Z(1)) * 2, 2 * Z(0) + 2 * Z(1)),
        ((Z(0) + X(1)) * (Z(0) + X(1)), 2 + 2 * (Z(0) * X(1))),
        ((Z(0) + X(1)) * (Z(1) + X(0)), (Z(0) * Z(1)) + 1j * Y(0) + (-1j) * Y(1) + (X(0) * X(1))),
        ((Z(0) + X(0)) * (Z(0) - X(0)), -2j * Y(0)),
        (
            (Z(0) + 1) * (1j * X(0) * X(1) + 1),
            (Z(0) + 1) * (1j * X(0) * X(1) + 1),
        ),
        (
            1 + Z(0) * Z(0) + Z(0) * X(1) + Z(0) * X(1) + X(1) * X(1) + X(1) * Z(1) + Y(1) * Z(1),
            3 + 2 * (Z(0) * X(1)) - 1j * Y(1) + 1j * X(1),
        ),
        ((Z(0) + 1) * (1j * X(0) * X(1) + 1), 1 - Y(0) * X(1) + Z(0) + 1j * X(0) * X(1)),
    ],
)
def test_multiplication(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


def test_invalid_multiplication_operation():
    with pytest.raises(InvalidHamiltonianOperation, match=r"Invalid multiplication between Hamiltonian and str"):
        _ = (Z(0) + Z(2)) * "Z"


@pytest.mark.parametrize(
    ("hamiltonian_rhs", "hamiltonian_lhs", "expected_hamiltonian"),
    [
        (Z(0) + 1, Z(0) + 1, 2 + 2 * Z(0)),
        (Z(0) * Z(1), X(0) * X(1), -1 * Y(0) * Y(1)),
    ],
)
def test_hamiltonian_rmul(hamiltonian_rhs, hamiltonian_lhs, expected_hamiltonian):
    assert hamiltonian_lhs.__rmul__(hamiltonian_rhs) == expected_hamiltonian  # noqa: PLC2801


class MockPauli(PauliOperator):
    _NAME: ClassVar[str] = "M"
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, 1]], dtype=COMPLEX_DTYPE)

    def __init__(self, qubit: int):
        super().__init__(qubit)


def test_multiply_pauli_errors():
    with pytest.raises(ValueError, match=r"Operators must act on the same qubit for multiplication."):
        Hamiltonian._multiply_pauli(PauliZ(0), PauliZ(1))

    with pytest.raises(InvalidHamiltonianOperation, match=r"Multiplying Z\(0\) and M\(0\) not supported."):
        Hamiltonian._multiply_pauli(PauliZ(0), MockPauli(0))


# -----------------------------
#  DIVISION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (Z(0) / 2, 0.5 * Z(0)),
        ((Z(0) + Z(1)) / 2, 0.5 * Z(0) + 0.5 * Z(1)),
        ((Z(0) + 3) / 2, 1.5 + 0.5 * Z(0)),
        (5 / 2, Hamiltonian({(PauliI(0),): 2.5})),
        ((Z(0) + X(1)) / 1j, -1j * Z(0) + -1j * X(1)),
    ],
)
def test_division(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


def test_truediv_raises_not_supported():
    with pytest.raises(InvalidHamiltonianOperation, match="Division by operators is not supported"):
        _ = 2 / Z(0)


def test_equality():
    pauli1 = PauliZ(0)
    assert pauli1 == PauliZ(0)
    assert pauli1 != PauliZ(1)
    assert pauli1 != 1


def test_pauli_hash_consistency():
    pauli1 = PauliZ(2)
    pauli2 = PauliZ(2)
    pauli3 = PauliX(2)

    assert pauli1 == pauli2
    assert hash(pauli1) == hash(pauli2)
    assert pauli1 != pauli3
    assert hash(pauli1) != hash(pauli3)


def test_hamiltonian_hash_order_independent():
    h1 = 2 * Z(0) + 3 * X(1) + 1j * Y(0)
    h2 = 1j * Y(0) + 3 * X(1) + 2 * Z(0)

    assert h1 == h2
    assert hash(h1) == hash(h2)


def test_hamiltonian_hash_changes_when_coefficients_change():
    h1 = Z(0) + X(1)
    h2 = Z(0) + 2 * X(1)

    assert h1 != h2
    assert hash(h1) != hash(h2)


@pytest.mark.parametrize(
    ("pauli", "expected_output"),
    [
        (PauliZ(0), "Z(0)"),
        (PauliZ(9), "Z(9)"),
        (PauliX(0), "X(0)"),
        (PauliX(9), "X(9)"),
        (PauliY(0), "Y(0)"),
        (PauliY(3), "Y(3)"),
        (PauliI(0), "I(0)"),
        (PauliI(5), "I(5)"),
    ],
)
def test_str_and_repr(pauli: PauliOperator, expected_output: str):
    assert str(pauli) == expected_output
    assert repr(pauli) == expected_output


def test_hamiltonian_division_errors():
    H = 1 + 2 * Z(0) + Z(0) * Z(1)
    with pytest.raises(InvalidHamiltonianOperation, match=r"Division by operators is not supported"):
        _ = 2 / H

    with pytest.raises(InvalidHamiltonianOperation, match=r"Division by operators is not supported"):
        _ = H / Z(0)

    with pytest.raises(ZeroDivisionError, match=r"Cannot divide by zero."):
        _ = H / 0


# -----------------------------
#  __STR__
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_str"),
    [
        (Z(0) / 2, "0.5 Z(0)"),
        (Z(0) + Z(1), "Z(0) + Z(1)"),
        (Z(0) + 2, "2 + Z(0)"),
        (Z(0) + 2j, "2j + Z(0)"),
        (Z(0) - 2, "-2 + Z(0)"),
        (Z(0) - 2j, "-2j + Z(0)"),
        (Z(0) - 3 + 2j, "(-3+2j) + Z(0)"),
        (Z(0) - Z(0), "0"),
        (Z(0) - 3 + 2j - 2 * Z(0), "(-3+2j) - Z(0)"),
        (-1 * Z(1) - 2 * Z(0), "- Z(1) - 2 Z(0)"),
        (-1j * Z(1) - 2.5j * Z(0), "-1j Z(1) - 2.5j Z(0)"),
        (Z(0) - 3 + 2.5j - 2 * Z(0), "(-3+2.5j) - Z(0)"),
        (1 + Z(0) - 3 * Z(1) + Z(1) + Z(0) - 2j * Z(1), "1 + 2 Z(0) + (-2-2j) Z(1)"),
    ],
)
def test_str(hamiltonian: Hamiltonian, expected_str: str):
    assert str(hamiltonian) == expected_str
    assert repr(hamiltonian) == expected_str


@pytest.mark.parametrize(
    ("hamiltonian_str", "expected_hamiltonian"),
    [
        ("0", Hamiltonian()),
        ("Z(0)", Z(0)),
        ("X(0)", X(0)),
        ("Y(0)", Y(0)),
        ("- Y(0)", -1 * Y(0)),
        ("Z(0) + 2", 2 + Z(0)),
        ("Z(0) + 2j", 2j + Z(0)),
        ("Z(0) - 2j", -2j + Z(0)),
        ("Z(0) - 2j + 3", (3 - 2j) + Z(0)),
        ("Z(0) - 2j + 3 + 2 Z(0)", (3 - 2j) + 3 * Z(0)),
        ("(2.5+3j) Y(0)", (2.5 + 3j) * Y(0)),
        ("(2.5 + 3j) Y(0)", (2.5 + 3j) * Y(0)),
        ("(2.5+3j)Y(0)", (2.5 + 3j) * Y(0)),
        ("(2.5   +   3j   )    Y(0)   ", (2.5 + 3j) * Y(0)),
        ("  1  Z(0) + X(0)", Z(0) + X(0)),
        ("      ", 1),
        ("   +      Z(0)  ", Z(0)),
    ],
)
def test_parse(hamiltonian_str: str, expected_hamiltonian: Hamiltonian):
    hamiltonian = Hamiltonian.parse(hamiltonian_str)
    assert hamiltonian == expected_hamiltonian


def test_parse_value_error():
    hamiltonian_str = "2 Z(1) + W(0)"
    with pytest.raises(ValueError, match=r"Unrecognized operator format: 'W\(0\)'"):
        Hamiltonian.parse(hamiltonian_str)


@pytest.mark.parametrize(
    ("hamiltonian", "expected"),
    [
        # Identity Hamiltonian on qubit 0.
        (PauliI(0).to_hamiltonian(), np.eye(2, dtype=COMPLEX_DTYPE)),
        # 2 * Z operator: 2*[[1, 0],[0, -1]]
        (2 * PauliZ(0).to_hamiltonian(), 2 * np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)),
        # Sum of 0.5*Z and 1*X: 0.5*[[1, 0],[0, -1]] + [[0, 1],[1, 0]]
        (
            0.5 * Z(0) + X(0),
            0.5 * np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE) + np.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE),
        ),
    ],
)
def test_to_matrix_single_qubit(hamiltonian: Hamiltonian, expected: np.ndarray):
    np.testing.assert_allclose(dense(hamiltonian), expected, atol=1e-8)


def test_to_matrix_zero_hamiltonian():
    """An empty Hamiltonian should produce a zero matrix."""
    H = Hamiltonian()
    expected = np.zeros((1, 1), dtype=COMPLEX_DTYPE)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


# --- Two-Qubit Tests ---


def test_to_matrix_two_qubit_single_term():
    """
    A two-qubit Hamiltonian with a single term (e.g. 2 * (Z(0) ⊗ X(1)))
    should return the correct Kronecker product.
    """
    H = Hamiltonian({(PauliZ(0), PauliX(1)): 2})
    z_matrix = np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    expected = 2 * np.kron(z_matrix, x_matrix)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


def test_to_matrix_two_qubit_multiple_terms():
    """
    For a Hamiltonian defined as 0.5 * (Z(0) ⊗ I) + 1.5 * (I ⊗ X(1)),
    the matrix representation should be the sum of the two Kronecker products.
    """
    H = 0.5 * PauliZ(0).to_hamiltonian() + 1.5 * PauliX(1).to_hamiltonian()
    z_matrix = np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)
    i_matrix = np.eye(2, dtype=COMPLEX_DTYPE)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    expected = 0.5 * np.kron(z_matrix, i_matrix) + 1.5 * np.kron(i_matrix, x_matrix)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


# --- Three-Qubit Test ---


def test_to_matrix_three_qubit():
    """
    Test a Hamiltonian acting on three qubits.
    For example, a term 3*(Z(1) ⊗ X(2)) acting on qubits 1 and 2.
    The full Hamiltonian should be embedded as I ⊗ Z ⊗ X.
    """
    H = Hamiltonian({(PauliZ(1), PauliX(2)): 3})
    I2 = np.eye(2, dtype=COMPLEX_DTYPE)
    z_matrix = np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    expected = 3 * np.kron(I2, np.kron(z_matrix, x_matrix))
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


# --- QTensor Tests ---


def test_to_qtensor_matches_to_matrix():
    H = 0.75 * Z(0) + (1.25 - 0.5j) * (X(1) * Y(2))
    tensor = H.to_qtensor()

    np.testing.assert_allclose(tensor.dense(), dense(H), atol=1e-8)
    assert tensor.nqubits == H.nqubits


def test_to_qtensor_with_padding():
    H = Hamiltonian({(PauliZ(1), PauliX(2)): 3})
    tensor = H.to_qtensor(total_nqubits=4)

    I2 = np.eye(2, dtype=COMPLEX_DTYPE)
    z_matrix = np.array([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    expected = 3 * np.kron(np.kron(np.kron(I2, z_matrix), x_matrix), I2)

    np.testing.assert_allclose(tensor.dense(), expected, atol=1e-8)
    assert tensor.nqubits == 4


def test_apply_operator():
    h = Hamiltonian({(PauliZ(0),): 1})
    with pytest.raises(ValueError, match=r"The list should not contain multiple operators acting on the same qubit."):
        h._apply_operator_on_qubit(terms=[PauliZ(1), PauliZ(1)])
    h_empty = Hamiltonian()
    empty_result = h_empty._apply_operator_on_qubit(terms=[])
    assert empty_result.shape == (1, 1)


def test_to_qtensor_raises_when_total_qubits_smaller():
    H = Hamiltonian({(PauliZ(1),): 1})

    with pytest.raises(ValueError, match=r"total number of qubits can't be less than the number"):
        H.to_qtensor(total_nqubits=1)


# ------ Hamiltonian Simplification Test -------


@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [(Z(0) + I(0) + I(1), Z(0) + 2 * I(0)), (0 * (Z(0) + 2 * Z(3)) + Z(1), Z(1))],
)
def test_simplify_hamiltonian(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


# ---- Equality Tests -----


@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (1 * I(0) + 2 * I(1), 3),
        (0 * (Z(0) + 2 * Z(3)), 0),
        (0.5j * I(0), 0.5j),
        (2 * Z(0) - Z(0), Z(0)),
        (2 * Z(0) - Z(1), 2 * Z(0) - Z(1)),
    ],
)
def test_eq_hamiltonian(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (1 * I(0) + 2 * I(1), "I(0)"),
        (2 * Z(0) - Z(1), Z(1)),
    ],
)
def test_neq_hamiltonian(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian != expected_hamiltonian


# ---- Commuting Partition Tests -----


def test_get_commuting_partitions():
    """
    Test the get_commuting_partitions method of the Hamiltonian class.
    This test creates a Hamiltonian with multiple terms and verifies that the
    returned partitions contain only mutually commuting terms.
    """
    H = Z(0) * Z(1) + X(0) * X(1) + Z(2) + X(2) + X(2) * Y(3)
    partitions = H.get_commuting_partitions()
    for partition in partitions:
        hamiltonians = [Hamiltonian({key: value}) for key, value in partition.items()]
        as_tensors = [part.to_qtensor(4) for part in hamiltonians]
        for i in range(len(as_tensors)):
            for j in range(i + 1, len(as_tensors)):
                commutator = as_tensors[i] @ as_tensors[j] - as_tensors[j] @ as_tensors[i]
                np.testing.assert_allclose(commutator.dense(), np.zeros_like(commutator.dense()), atol=1e-8)


def test_pauli_with_numbers():
    H = PauliX(0)

    # Addition
    assert H + 2 == 2 + H == Hamiltonian({(PauliX(0),): 1, (PauliI(0),): 2})

    # Subtraction
    assert Hamiltonian({(PauliX(0),): 1, (PauliI(0),): -2}) == H - 2
    assert Hamiltonian({(PauliI(0),): 2, (PauliX(0),): -1}) == 2 - H

    # Multiplication
    assert H * 3 == 3 * H == Hamiltonian({(PauliX(0),): 3})
    assert H * 1j == 1j * H == Hamiltonian({(PauliX(0),): 1j})

    # Division
    assert Hamiltonian({(PauliX(0),): 0.5}) == H / 2
    assert Hamiltonian({(PauliX(0),): -1j}) == H / 1j


def test_bad_pauli_division():
    with pytest.raises(InvalidHamiltonianOperation, match="Division by operators is not supported"):
        _ = 3 / PauliZ(0)


def test_to_qtensor_not_enough_qubits():
    H = Z(0) + X(1)

    with pytest.raises(ValueError, match="can't be less than"):
        H.to_qtensor(1)


def test_get_static_hamiltonian():
    x = Parameter("x", 1.5)
    H = PauliZ(0) * PauliZ(1) + 2 * PauliX(0) + x
    h_static = H.get_static_hamiltonian()

    assert h_static == (PauliZ(0) * PauliZ(1) + 2 * PauliX(0) + 1.5)
    assert h_static.get_parameters() == {}


def test_hamiltonian_equal_pauli_operator():
    H = Hamiltonian({(PauliZ(0),): 1})

    assert PauliZ(0) == H
    assert PauliZ(0) == H


def test_dict_with_hamiltonian_key():
    H1 = Hamiltonian({(PauliZ(0),): 1})
    H2 = Hamiltonian({(PauliZ(0),): 1})

    d = {H1: "test"}

    assert d[H2] == "test"


def test_from_qtensor(monkeypatch):
    tensor = QTensor(np.array([[1, 0], [0, -1]], dtype=complex))
    H = Hamiltonian.from_qtensor(tensor)

    assert tensor == H

    non_hermitian_tensor = QTensor(np.array([[1, 1], [0, -1]], dtype=complex))
    with pytest.raises(ValueError, match="not Hermitian"):
        Hamiltonian.from_qtensor(non_hermitian_tensor)

    # rewrite np.trace to give big value so the coversion goes wrong
    def bad_trace(self):
        return 10.0

    monkeypatch.setattr(np, "trace", bad_trace)
    with pytest.raises(ValueError, match="Pauli expansion failed"):
        Hamiltonian.from_qtensor(tensor)


def test_commutator():
    H1 = Z(0) * Z(1) + X(0)
    H2 = X(1) + Y(0) * Y(1)

    assert H1.commutator(H2) == H1 * H2 - H2 * H1
    assert H1.anticommutator(H2) == H1 * H2 + H2 * H1


@pytest.mark.parametrize(
    ("ops1", "ops2", "expected"),
    [
        # Identical single-qubit Paulis commute.
        ((PauliZ(0),), (PauliZ(0),), True),
        # Distinct non-identity Paulis on the same qubit anticommute.
        ((PauliX(0),), (PauliZ(0),), False),
        ((PauliX(0),), (PauliY(0),), False),
        # Operators on disjoint qubits always commute.
        ((PauliZ(0),), (PauliZ(1),), True),
        ((PauliX(0),), (PauliZ(1),), True),
        # Identity commutes with everything.
        ((PauliI(0),), (PauliX(0),), True),
        ((PauliX(0),), (PauliI(0),), True),
        ((PauliI(0),), (PauliI(0),), True),
        # Two disagreements (even parity) => commute, e.g. XX vs ZZ.
        ((PauliX(0), PauliX(1)), (PauliZ(0), PauliZ(1)), True),
        # One disagreement (odd parity) => anticommute, e.g. XX vs ZI.
        ((PauliX(0), PauliX(1)), (PauliZ(0),), False),
        # Agreement on one qubit, disagreement on another => odd => anticommute.
        ((PauliX(0), PauliX(1)), (PauliX(0), PauliZ(1)), False),
        # Three disagreements (odd parity) => anticommute.
        ((PauliX(0), PauliX(1), PauliX(2)), (PauliZ(0), PauliZ(1), PauliZ(2)), False),
        # Empty string (scalar) commutes with anything.
        ((), (PauliX(0),), True),
        # Identity-only strings commute with anything.
        ((PauliI(0), PauliI(1)), (PauliX(0), PauliY(1)), True),
    ],
)
def test_pauli_strings_commute(ops1, ops2, expected):
    assert Hamiltonian._pauli_strings_commute(ops1, ops2) is expected
    # The relation is symmetric.
    assert Hamiltonian._pauli_strings_commute(ops2, ops1) is expected


@pytest.mark.parametrize(
    ("h1", "h2", "expected"),
    [
        # Single Pauli strings.
        (Z(0), Z(1), True),
        (X(0), Z(0), False),
        (X(0) * X(1), Z(0) * Z(1), True),
        (X(0) * X(1), Z(0), False),
        # Identity / scalars commute with everything.
        (X(0), I(0), True),
        (X(0), 3 * I(0), True),
        # Empty (zero) Hamiltonian commutes with everything.
        (Hamiltonian(), Z(0), True),
        # A Hamiltonian commutes with itself.
        (Z(0) * Z(1) + X(0), Z(0) * Z(1) + X(0), True),
        # Sums where individual anticommuting pairs cancel out => commute.
        (X(0) * Y(1), Y(0) * X(1), True),
        # Sums that genuinely do not commute.
        (Z(0) + X(1), X(0) + Z(1), False),
        (2 * X(0) * X(1) + Z(2), Z(0) * Z(1) + X(2), False),
        # Complex coefficients should not affect commutation.
        (1j * X(0), Z(0), False),
        (1j * Z(0), Z(1), True),
    ],
)
def test_commutes_with(h1, h2, expected):
    assert h1.commutes_with(h2) is expected
    # Commutation is symmetric.
    assert h2.commutes_with(h1) is expected
    # Must agree with the explicit commutator computation.
    assert (h1.commutator(h2) == Hamiltonian.ZERO) is expected


def test_commutes_with_matches_commutator_random():
    """Fuzz check: commutes_with must agree with the full commutator on random Hamiltonians."""
    rng = np.random.default_rng(0)
    factories = {"X": X, "Y": Y, "Z": Z}
    coeffs = [1, 2, -1, 1j]
    for _ in range(500):
        nqubits = int(rng.integers(1, 5))

        def random_hamiltonian():
            h = Hamiltonian()
            for _ in range(int(rng.integers(1, 5))):
                term = None
                for qubit in range(nqubits):
                    if rng.random() < 0.6:
                        pauli = factories[rng.choice(["X", "Y", "Z"])](qubit)
                        term = pauli if term is None else term * pauli
                if term is not None:
                    h = h + coeffs[int(rng.integers(0, len(coeffs)))] * term
            return h

        h1, h2 = random_hamiltonian(), random_hamiltonian()
        assert h1.commutes_with(h2) == (h1.commutator(h2) == Hamiltonian.ZERO)


def test_commutes_with_parametrized_coefficients():
    """commutes_with should handle symbolic (Parameter) coefficients."""
    x = Parameter("x", 1.5)
    # Commuting terms remain commuting regardless of the symbolic coefficient.
    assert (x * Z(0)).commutes_with(Z(1)) is True
    # Anticommuting terms do not commute even with a symbolic coefficient.
    assert (x * X(0)).commutes_with(Z(0)) is False


def test_norms():
    H = Z(0) + 2 * X(1) + 3j * Y(2)

    assert np.isclose(H.vector_norm(), np.sqrt(1**2 + 2**2 + 3**2))
    assert np.isclose(H.frobenius_norm(), np.sqrt(14 * 8))  # sqrt(sum |c_i|^2 * 2^n) where n=3 qubits


def test_trace():
    dim = 8

    H = Z(0) + 2 * X(1) + 3j * Y(2) + 1 * I(0)
    assert np.isclose(H.trace(), dim)

    x = Parameter("x", 2.0)
    H = X(1) + Z(2)
    H._elements[PauliI(0),] = x  # to force it to be a parameter not a term
    assert np.isclose(H.trace(), 2.0 * dim)

    x = Parameter("x", 2.0)
    H = (x + 1) * I(0) + X(1) + Z(2)
    assert np.isclose(H.trace(), 3.0 * dim)


def test_hamiltonian_term_arithmetic():
    var = Variable("v", Domain.REAL)
    term = 2 * var

    H = X(0) + Y(1)

    with pytest.raises(ValueError, match="Term provided contains generic variables"):
        _ = H + term

    with pytest.raises(ValueError, match="Term provided contains generic variables"):
        _ = term + H

    with pytest.raises(ValueError, match="Term provided contains generic variables"):
        _ = H - term

    with pytest.raises(ValueError, match="Term provided contains generic variables"):
        _ = term - H

    with pytest.raises(ValueError, match="Term provided contains generic variables"):
        _ = H * term

    with pytest.raises(ValueError, match="Term provided contains generic variables"):
        _ = term * H


def test_hamiltonian_in_place_addition():
    H = X(0) + Y(1)
    var = Variable("v", Domain.REAL)
    term = 2 * var

    param = Parameter("p", 1.0)
    safe_term = param * 2
    H += safe_term
    assert (X(0) + Y(1) + 2.0 * param) == H

    with pytest.raises(ValueError, match="Only Parameters are allowed"):
        H._add_inplace(term)

    H += Z(2)
    assert (X(0) + Y(1) + 2.0 * param + Z(2)) == H

    H += PauliZ(0)
    assert (X(0) + Y(1) + 2.0 * param + Z(2) + Z(0)) == H


def test_hamiltonian_in_place_subtraction():
    H = X(0) + Y(1)
    var = Variable("v", Domain.REAL)
    term = 2 * var

    param = Parameter("p", 1.0)
    safe_term = param * 2
    H -= safe_term
    assert (X(0) + Y(1) - 2.0 * param) == H

    with pytest.raises(ValueError, match="Only Parameters are allowed"):
        H._sub_inplace(term)

    H -= Z(2)
    assert (X(0) + Y(1) - 2.0 * param - Z(2)) == H

    H -= PauliZ(0)
    assert (X(0) + Y(1) - 2.0 * param - Z(2) - Z(0)) == H

    H -= param
    assert (X(0) + Y(1) - 3.0 * param - Z(2) - Z(0)) == H


def test_hamiltonian_in_place_multiplication():
    H: Hamiltonian = X(0) + Y(1)
    var = Variable("v", Domain.REAL)
    term = 2 * var

    param = Parameter("p", 1.0)
    safe_term = param * 2
    H *= safe_term
    assert ((X(0) + Y(1)) * 2.0 * param) == H

    with pytest.raises(ValueError, match="Only Parameters are allowed"):
        H._mul_inplace(term)

    H *= Z(2)
    assert ((X(0) + Y(1)) * 2.0 * param * Z(2)) == H

    H *= PauliZ(0)
    assert ((X(0) + Y(1)) * 2.0 * param * Z(2) * Z(0)) == H

    H *= param
    assert ((X(0) + Y(1)) * 2.0 * param**2 * Z(2) * Z(0)) == H


def test_negation():
    H = Z(0) + 2 * X(1) + 3j * Y(2)

    assert -H == (-1) * H


def test_pauli_operator_rejects_negative_qubit():
    """QSDK-05: a Pauli operator must reject a negative qubit at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        PauliX(-1)


@pytest.mark.parametrize(
    ("built", "expected"),
    [
        # Transverse / longitudinal fields put a single-qubit term on every qubit.
        (Hamiltonian.transverse_field(nqubits=2, x_coefficient=1.3), 1.3 * X(0) + 1.3 * X(1)),
        (Hamiltonian.transverse_field(nqubits=1), X(0)),
        (Hamiltonian.longitudinal_field(nqubits=2, z_coefficient=1.3), 1.3 * Z(0) + 1.3 * Z(1)),
        (Hamiltonian.longitudinal_field(nqubits=1), Z(0)),
        # Ising couples every pair i < j, with an optional longitudinal field.
        (Hamiltonian.ising(nqubits=2, zz_coefficient=2.0), 2.0 * Z(0) * Z(1)),
        (
            Hamiltonian.ising(nqubits=3, zz_coefficient=2.0),
            2.0 * (Z(0) * Z(1) + Z(0) * Z(2) + Z(1) * Z(2)),
        ),
        (
            Hamiltonian.ising(nqubits=2, zz_coefficient=2.0, z_coefficient=0.5),
            2.0 * Z(0) * Z(1) + 0.5 * Z(0) + 0.5 * Z(1),
        ),
        # The default z_coefficient of 0 leaves the field out entirely.
        (
            Hamiltonian.ising(nqubits=2, zz_coefficient=2.0),
            Hamiltonian.ising(nqubits=2, zz_coefficient=2.0, z_coefficient=0.0),
        ),
        # Transverse-field Ising is the sum of the two.
        (
            Hamiltonian.transverse_field_ising(nqubits=2, x_coefficient=1.3, zz_coefficient=-2),
            1.3 * X(0) + 1.3 * X(1) - 2 * Z(0) * Z(1),
        ),
        (
            Hamiltonian.transverse_field_ising(nqubits=2, x_coefficient=1.3, zz_coefficient=-2),
            Hamiltonian.transverse_field(nqubits=2, x_coefficient=1.3)
            + Hamiltonian.ising(nqubits=2, zz_coefficient=-2),
        ),
        (
            Hamiltonian.transverse_field_ising(nqubits=2, x_coefficient=1.0, zz_coefficient=1.0, z_coefficient=0.5),
            X(0) + X(1) + Z(0) * Z(1) + 0.5 * Z(0) + 0.5 * Z(1),
        ),
        # XY: yy_coefficient defaults to xx_coefficient, giving the isotropic model.
        (Hamiltonian.xy(nqubits=2, xx_coefficient=0.5), 0.5 * X(0) * X(1) + 0.5 * Y(0) * Y(1)),
        (
            Hamiltonian.xy(nqubits=2, xx_coefficient=0.5, yy_coefficient=0.25),
            0.5 * X(0) * X(1) + 0.25 * Y(0) * Y(1),
        ),
        # Heisenberg XXX: both other couplings default to xx_coefficient.
        (
            Hamiltonian.heisenberg(nqubits=2, xx_coefficient=0.5),
            0.5 * X(0) * X(1) + 0.5 * Y(0) * Y(1) + 0.5 * Z(0) * Z(1),
        ),
        # Heisenberg XXZ: only the ZZ coupling is anisotropic.
        (
            Hamiltonian.heisenberg(nqubits=2, xx_coefficient=1.0, zz_coefficient=0.3),
            X(0) * X(1) + Y(0) * Y(1) + 0.3 * Z(0) * Z(1),
        ),
        # Heisenberg XYZ with a longitudinal field.
        (
            Hamiltonian.heisenberg(nqubits=2, xx_coefficient=1, yy_coefficient=2, zz_coefficient=3, z_coefficient=0.5),
            X(0) * X(1) + 2 * Y(0) * Y(1) + 3 * Z(0) * Z(1) + 0.5 * Z(0) + 0.5 * Z(1),
        ),
        # Heisenberg reduces to XY plus an Ising ZZ coupling.
        (
            Hamiltonian.heisenberg(nqubits=3, xx_coefficient=1.0),
            Hamiltonian.xy(nqubits=3, xx_coefficient=1.0) + Hamiltonian.ising(nqubits=3, zz_coefficient=1.0),
        ),
    ],
)
def test_named_hamiltonian_constructors(built: Hamiltonian, expected: Hamiltonian):
    assert built == expected


@pytest.mark.parametrize(
    ("built", "expected"),
    [
        # A chain couples adjacent qubits only.
        (Hamiltonian.ising_chain(nqubits=3, zz_coefficient=2.0), 2.0 * Z(0) * Z(1) + 2.0 * Z(1) * Z(2)),
        (
            Hamiltonian.ising_chain(nqubits=4),
            Z(0) * Z(1) + Z(1) * Z(2) + Z(2) * Z(3),
        ),
        # Two qubits give a single bond, matching the all-to-all Ising model.
        (Hamiltonian.ising_chain(nqubits=2), Hamiltonian.ising(nqubits=2)),
        # Closing the ring adds the bond between the two ends.
        (
            Hamiltonian.ising_chain(nqubits=4, periodic=True),
            Z(0) * Z(1) + Z(1) * Z(2) + Z(2) * Z(3) + Z(0) * Z(3),
        ),
        # On three qubits the ring is the complete graph, i.e. the all-to-all model.
        (Hamiltonian.ising_chain(nqubits=3, periodic=True), Hamiltonian.ising(nqubits=3)),
        # Wrapping is skipped on two qubits, where it would double the only bond.
        (Hamiltonian.ising_chain(nqubits=2, periodic=True), Hamiltonian.ising_chain(nqubits=2)),
        # The optional longitudinal field lands on every qubit.
        (
            Hamiltonian.ising_chain(nqubits=3, z_coefficient=0.5),
            Z(0) * Z(1) + Z(1) * Z(2) + 0.5 * Z(0) + 0.5 * Z(1) + 0.5 * Z(2),
        ),
    ],
)
def test_ising_chain(built: Hamiltonian, expected: Hamiltonian):
    assert built == expected


@pytest.mark.parametrize(
    ("built", "expected"),
    [
        # 2x2: qubits 0 1 on the top row, 2 3 on the bottom.
        (
            Hamiltonian.ising_grid(rows=2, columns=2),
            Z(0) * Z(1) + Z(2) * Z(3) + Z(0) * Z(2) + Z(1) * Z(3),
        ),
        # A single row or column degenerates to a chain.
        (Hamiltonian.ising_grid(rows=1, columns=3), Hamiltonian.ising_chain(nqubits=3)),
        (Hamiltonian.ising_grid(rows=3, columns=1), Hamiltonian.ising_chain(nqubits=3)),
        # 2x3, row-major indexing: 0 1 2 / 3 4 5.
        (
            Hamiltonian.ising_grid(rows=2, columns=3, zz_coefficient=2.0),
            2.0
            * (
                Z(0) * Z(1)
                + Z(1) * Z(2)  # top row
                + Z(3) * Z(4)
                + Z(4) * Z(5)  # bottom row
                + Z(0) * Z(3)
                + Z(1) * Z(4)
                + Z(2) * Z(5)  # columns
            ),
        ),
        # Wrapping a 3-wide row adds the bond between its two ends.
        (
            Hamiltonian.ising_grid(rows=1, columns=3, periodic=True),
            Hamiltonian.ising_chain(nqubits=3, periodic=True),
        ),
        # A 2x2 torus would duplicate every bond, so wrapping is skipped in both directions.
        (Hamiltonian.ising_grid(rows=2, columns=2, periodic=True), Hamiltonian.ising_grid(rows=2, columns=2)),
        # The optional longitudinal field lands on every site.
        (
            Hamiltonian.ising_grid(rows=1, columns=2, z_coefficient=0.5),
            Z(0) * Z(1) + 0.5 * Z(0) + 0.5 * Z(1),
        ),
    ],
)
def test_ising_grid(built: Hamiltonian, expected: Hamiltonian):
    assert built == expected


def test_ising_grid_dimensions():
    H = Hamiltonian.ising_grid(rows=3, columns=4)

    assert H.nqubits == 12
    # A rows x columns open lattice has rows*(columns-1) horizontal and (rows-1)*columns vertical bonds.
    assert len(H.elements) == 3 * 3 + 2 * 4


def test_ising_grid_torus_has_two_bonds_per_site():
    H = Hamiltonian.ising_grid(rows=3, columns=3, periodic=True)

    assert H.nqubits == 9
    assert len(H.elements) == 2 * 9


@pytest.mark.parametrize(
    "constructor",
    [
        Hamiltonian.ising,
        Hamiltonian.ising_chain,
        Hamiltonian.transverse_field_ising,
        Hamiltonian.xy,
        Hamiltonian.heisenberg,
    ],
)
def test_two_body_constructors_need_at_least_two_qubits(constructor):
    with pytest.raises(ValueError, match="Hamiltonians need at least 2 qubits, got 1"):
        constructor(nqubits=1)


@pytest.mark.parametrize(
    "constructor",
    [
        Hamiltonian.transverse_field,
        Hamiltonian.longitudinal_field,
        Hamiltonian.ising,
        Hamiltonian.ising_chain,
        Hamiltonian.transverse_field_ising,
        Hamiltonian.xy,
        Hamiltonian.heisenberg,
    ],
)
def test_named_constructors_reject_non_positive_nqubits(constructor):
    with pytest.raises(ValueError, match="nqubits must be greater than zero"):
        constructor(nqubits=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"rows": 0, "columns": 2}, "rows must be greater than zero"),
        ({"rows": -1, "columns": 2}, "rows must be greater than zero"),
        ({"rows": 2, "columns": 0}, "columns must be greater than zero"),
        ({"rows": 1, "columns": 1}, "Ising grid Hamiltonians need at least 2 qubits, got 1"),
    ],
)
def test_ising_grid_rejects_invalid_dimensions(kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        Hamiltonian.ising_grid(**kwargs)


###############################################################################
# Randomized coefficients, requested by passing a (low, high) range
###############################################################################


@pytest.mark.parametrize(
    ("randomized", "fixed"),
    [
        (
            Hamiltonian.transverse_field(nqubits=3, x_coefficient=(-1, 1)),
            Hamiltonian.transverse_field(nqubits=3),
        ),
        (
            Hamiltonian.longitudinal_field(nqubits=3, z_coefficient=(-1, 1)),
            Hamiltonian.longitudinal_field(nqubits=3),
        ),
        (Hamiltonian.ising(nqubits=3, zz_coefficient=(-1, 1)), Hamiltonian.ising(nqubits=3)),
        (
            Hamiltonian.ising(nqubits=3, zz_coefficient=(-1, 1), z_coefficient=(-1, 1)),
            Hamiltonian.ising(nqubits=3, z_coefficient=1.0),
        ),
        (
            Hamiltonian.ising_chain(nqubits=4, zz_coefficient=(-1, 1)),
            Hamiltonian.ising_chain(nqubits=4),
        ),
        (
            Hamiltonian.ising_grid(rows=2, columns=3, zz_coefficient=(-1, 1)),
            Hamiltonian.ising_grid(rows=2, columns=3),
        ),
        (
            Hamiltonian.transverse_field_ising(nqubits=3, x_coefficient=(-1, 1), zz_coefficient=(-1, 1)),
            Hamiltonian.transverse_field_ising(nqubits=3),
        ),
        (Hamiltonian.xy(nqubits=3, xx_coefficient=(-1, 1)), Hamiltonian.xy(nqubits=3)),
        (
            Hamiltonian.heisenberg(nqubits=3, xx_coefficient=(-1, 1)),
            Hamiltonian.heisenberg(nqubits=3),
        ),
    ],
)
def test_a_range_keeps_the_model_structure(randomized: Hamiltonian, fixed: Hamiltonian):
    # Passing a range changes the coefficients only: the operator products are the same as the
    # fixed-coefficient model's.
    assert set(randomized.elements) == set(fixed.elements)
    assert all(-1.0 <= complex(c).real <= 1.0 for c in randomized.elements.values())
    assert all(complex(c).imag == 0 for c in randomized.elements.values())


@pytest.mark.parametrize(
    ("built", "count"),
    [
        (Hamiltonian.transverse_field(nqubits=4, x_coefficient=(2.5, 3.5)), 4),
        (Hamiltonian.longitudinal_field(nqubits=4, z_coefficient=(2.5, 3.5)), 4),
        (Hamiltonian.ising(nqubits=4, zz_coefficient=(2.5, 3.5)), 6),
        (Hamiltonian.ising_chain(nqubits=4, zz_coefficient=(2.5, 3.5)), 3),
        (Hamiltonian.ising_grid(rows=2, columns=2, zz_coefficient=(2.5, 3.5)), 4),
        (
            Hamiltonian.transverse_field_ising(nqubits=4, x_coefficient=(2.5, 3.5), zz_coefficient=(2.5, 3.5)),
            10,
        ),
        (Hamiltonian.xy(nqubits=4, xx_coefficient=(2.5, 3.5)), 12),
        (Hamiltonian.heisenberg(nqubits=4, xx_coefficient=(2.5, 3.5)), 18),
    ],
)
def test_every_term_is_drawn_from_its_range(built: Hamiltonian, count: int):
    assert len(built.elements) == count
    assert all(2.5 <= complex(c).real <= 3.5 for c in built.elements.values())


def test_each_term_gets_an_independent_draw():
    H = Hamiltonian.heisenberg(nqubits=3, xx_coefficient=(-1, 1))

    coefficients = list(H.elements.values())
    assert len(coefficients) == 9
    assert len(set(coefficients)) == 9


def test_fixed_and_random_coefficients_can_be_mixed():
    H = Hamiltonian.transverse_field_ising(nqubits=4, x_coefficient=(-2, 2), zz_coefficient=1.0)

    fields = [c for operators, c in H.elements.items() if len(operators) == 1]
    couplings = [c for operators, c in H.elements.items() if len(operators) == 2]

    assert len(fields) == 4
    assert len(set(fields)) == 4, "the ranged field should be drawn per qubit"
    assert all(-2.0 <= complex(c).real <= 2.0 for c in fields)
    assert set(couplings) == {1.0}, "the fixed coupling should be shared by every pair"


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (Hamiltonian.transverse_field, {"x_coefficient": (-1, 1)}),
        (Hamiltonian.longitudinal_field, {"z_coefficient": (-1, 1)}),
        (Hamiltonian.ising, {"zz_coefficient": (-1, 1)}),
        (Hamiltonian.ising_chain, {"zz_coefficient": (-1, 1)}),
        (Hamiltonian.transverse_field_ising, {"x_coefficient": (-1, 1)}),
        (Hamiltonian.xy, {"xx_coefficient": (-1, 1)}),
        (Hamiltonian.heisenberg, {"xx_coefficient": (-1, 1)}),
    ],
)
def test_ranged_coefficients_are_seeded(constructor, kwargs: dict):
    first = constructor(nqubits=3, seed=7, **kwargs)
    repeat = constructor(nqubits=3, seed=7, **kwargs)
    other = constructor(nqubits=3, seed=8, **kwargs)

    assert first == repeat
    assert first != other


def test_ising_grid_ranged_coefficients_are_seeded():
    first = Hamiltonian.ising_grid(rows=2, columns=3, zz_coefficient=(-1, 1), seed=7)
    repeat = Hamiltonian.ising_grid(rows=2, columns=3, zz_coefficient=(-1, 1), seed=7)
    other = Hamiltonian.ising_grid(rows=2, columns=3, zz_coefficient=(-1, 1), seed=8)

    assert first == repeat
    assert first != other


def test_the_seed_is_ignored_when_no_range_is_given():
    assert Hamiltonian.ising(nqubits=3, zz_coefficient=2.0, seed=1) == Hamiltonian.ising(
        nqubits=3, zz_coefficient=2.0, seed=99
    )


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (Hamiltonian.xy, {"xx_coefficient": (-1, 1)}),
        (Hamiltonian.heisenberg, {"xx_coefficient": (-1, 1)}),
    ],
)
def test_reusing_a_range_still_draws_each_axis_independently(constructor, kwargs: dict):
    H = constructor(nqubits=3, **kwargs)

    by_axis: dict[str, list[complex]] = {"X": [], "Y": [], "Z": []}
    for operators, coefficient in H.elements.items():
        by_axis[operators[0].name].append(coefficient)

    assert by_axis["X"] != by_axis["Y"]
    if by_axis["Z"]:
        assert by_axis["Y"] != by_axis["Z"]


def test_an_explicit_range_matches_the_reused_one():
    default = Hamiltonian.xy(nqubits=3, xx_coefficient=(1.3, 2.3), seed=5)
    explicit = Hamiltonian.xy(nqubits=3, xx_coefficient=(1.3, 2.3), yy_coefficient=(1.3, 2.3), seed=5)

    assert default == explicit


def test_a_degenerate_range_behaves_like_a_fixed_value():
    assert Hamiltonian.ising(nqubits=3, zz_coefficient=(2.0, 2.0)) == Hamiltonian.ising(nqubits=3, zz_coefficient=2.0)


@pytest.mark.parametrize(
    ("constructor", "kwargs", "name"),
    [
        (Hamiltonian.transverse_field, {"x_coefficient": (1.0, -1.0)}, "x_coefficient"),
        (Hamiltonian.longitudinal_field, {"z_coefficient": (1.0, -1.0)}, "z_coefficient"),
        (Hamiltonian.ising, {"zz_coefficient": (1.0, -1.0)}, "zz_coefficient"),
        (Hamiltonian.ising, {"z_coefficient": (1.0, -1.0)}, "z_coefficient"),
        (Hamiltonian.ising_chain, {"zz_coefficient": (1.0, -1.0)}, "zz_coefficient"),
        (Hamiltonian.ising_chain, {"z_coefficient": (1.0, -1.0)}, "z_coefficient"),
        (Hamiltonian.transverse_field_ising, {"x_coefficient": (1.0, -1.0)}, "x_coefficient"),
        (Hamiltonian.transverse_field_ising, {"zz_coefficient": (1.0, -1.0)}, "zz_coefficient"),
        (Hamiltonian.transverse_field_ising, {"z_coefficient": (1.0, -1.0)}, "z_coefficient"),
        (Hamiltonian.xy, {"xx_coefficient": (1.0, -1.0)}, "xx_coefficient"),
        (Hamiltonian.xy, {"yy_coefficient": (1.0, -1.0)}, "yy_coefficient"),
        (Hamiltonian.heisenberg, {"xx_coefficient": (1.0, -1.0)}, "xx_coefficient"),
        (Hamiltonian.heisenberg, {"yy_coefficient": (1.0, -1.0)}, "yy_coefficient"),
        (Hamiltonian.heisenberg, {"zz_coefficient": (1.0, -1.0)}, "zz_coefficient"),
        (Hamiltonian.heisenberg, {"z_coefficient": (1.0, -1.0)}, "z_coefficient"),
    ],
)
def test_misordered_ranges_are_rejected(constructor, kwargs: dict, name: str):
    with pytest.raises(ValueError, match=f"{name} must be a \\(low, high\\) pair"):
        constructor(nqubits=2, **kwargs)


def test_ising_grid_rejects_misordered_ranges():
    with pytest.raises(ValueError, match=r"zz_coefficient must be a \(low, high\) pair"):
        Hamiltonian.ising_grid(rows=2, columns=2, zz_coefficient=(1.0, -1.0))


def test_a_randomized_hamiltonian_is_still_usable():
    H = Hamiltonian.ising_chain(nqubits=3, zz_coefficient=(-1, 1))

    assert H.to_matrix().shape == (8, 8)
    assert H.to_qtensor().is_hermitian()
