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
from qilisdk.core.variables import BinaryVariable


# Helper function to convert sparse matrix to dense NumPy array.
def dense(ham: Hamiltonian) -> np.ndarray:
    return ham.to_matrix().toarray()


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
        (Z(0) + Z(2)) + "Z"


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
        (Z(0) + Z(2)) - "Z"


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
        (Z(0) + Z(2)) * "Z"


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
    _MATRIX: ClassVar[np.ndarray] = np.array([[1, 0], [0, 1]], dtype=complex)

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
        2 / Z(0)


def test_equality():
    pauli1 = PauliZ(0)
    assert pauli1 == PauliZ(0)
    assert pauli1 != PauliZ(1)
    assert pauli1 != 1


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
        2 / H

    with pytest.raises(InvalidHamiltonianOperation, match=r"Division by operators is not supported"):
        H / Z(0)

    with pytest.raises(ZeroDivisionError, match=r"Cannot divide by zero."):
        H / 0


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
        (PauliI(0).to_hamiltonian(), np.eye(2, dtype=complex)),
        # 2 * Z operator: 2*[[1, 0],[0, -1]]
        (2 * PauliZ(0).to_hamiltonian(), 2 * np.array([[1, 0], [0, -1]], dtype=complex)),
        # Sum of 0.5*Z and 1*X: 0.5*[[1, 0],[0, -1]] + [[0, 1],[1, 0]]
        (
            0.5 * Z(0) + X(0),
            0.5 * np.array([[1, 0], [0, -1]], dtype=complex) + np.array([[0, 1], [1, 0]], dtype=complex),
        ),
    ],
)
def test_to_matrix_single_qubit(hamiltonian: Hamiltonian, expected: np.ndarray):
    np.testing.assert_allclose(dense(hamiltonian), expected, atol=1e-8)


def test_to_matrix_zero_hamiltonian():
    """An empty Hamiltonian should produce a zero matrix."""
    H = Hamiltonian()
    expected = np.zeros((1, 1), dtype=complex)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


# --- Two-Qubit Tests ---


def test_to_matrix_two_qubit_single_term():
    """
    A two-qubit Hamiltonian with a single term (e.g. 2 * (Z(0) ⊗ X(1)))
    should return the correct Kronecker product.
    """
    H = Hamiltonian({(PauliZ(0), PauliX(1)): 2})
    Z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = 2 * np.kron(Z_matrix, X_matrix)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


def test_to_matrix_two_qubit_multiple_terms():
    """
    For a Hamiltonian defined as 0.5 * (Z(0) ⊗ I) + 1.5 * (I ⊗ X(1)),
    the matrix representation should be the sum of the two Kronecker products.
    """
    H = 0.5 * PauliZ(0).to_hamiltonian() + 1.5 * PauliX(1).to_hamiltonian()
    Z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    I_matrix = np.eye(2, dtype=complex)
    X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = 0.5 * np.kron(Z_matrix, I_matrix) + 1.5 * np.kron(I_matrix, X_matrix)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


# --- Three-Qubit Test ---


def test_to_matrix_three_qubit():
    """
    Test a Hamiltonian acting on three qubits.
    For example, a term 3*(Z(1) ⊗ X(2)) acting on qubits 1 and 2.
    The full Hamiltonian should be embedded as I ⊗ Z ⊗ X.
    """
    H = Hamiltonian({(PauliZ(1), PauliX(2)): 3})
    I2 = np.eye(2, dtype=complex)
    Z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = 3 * np.kron(I2, np.kron(Z_matrix, X_matrix))
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


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
