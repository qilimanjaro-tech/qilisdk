import numpy as np
import pytest

from qilisdk.analog import Hamiltonian, I, X, Y, Z


# Helper function to convert sparse matrix to dense NumPy array.
def dense(ham: Hamiltonian) -> np.ndarray:
    return ham.to_matrix().toarray()


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
    ],
)
def test_addition(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


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
    ],
)
def test_subtraction(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


# -----------------------------
#  MULTIPLICATION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (Z(0) * Z(0), Hamiltonian({(I(0),): 1})),
        (Z(0) * X(0), 1j * Y(0)),
        (X(0) * Z(0), -1j * Y(0)),
        (Z(0) * Z(1), Z(0) * Z(1)),
        (X(0) * Z(1), X(0) * Z(1)),
        (2 * Z(0), 2 * Z(0)),
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


# -----------------------------
#  DIVISION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_hamiltonian"),
    [
        (Z(0) / 2, 0.5 * Z(0)),
        ((Z(0) + Z(1)) / 2, 0.5 * Z(0) + 0.5 * Z(1)),
        ((Z(0) + 3) / 2, 1.5 + 0.5 * Z(0)),
        (5 / 2, Hamiltonian({(I(0),): 2.5})),
        ((Z(0) + X(1)) / 1j, -1j * Z(0) + -1j * X(1)),
    ],
)
def test_division(hamiltonian: Hamiltonian, expected_hamiltonian: Hamiltonian):
    assert hamiltonian == expected_hamiltonian


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
        (Z(0) - 3 + 2j - 2 * Z(0), "(-3+2j) - Z(0)"),
        (1 + Z(0) - 3 * Z(1) + Z(1) + Z(0) - 2j * Z(1), "1 + 2 Z(0) + (-2-2j) Z(1)"),
    ],
)
def test_str(hamiltonian: Hamiltonian, expected_str: str):
    assert str(hamiltonian) == expected_str


@pytest.mark.parametrize(
    ("hamiltonian_str", "expected_hamiltonian"),
    [
        ("0", Hamiltonian()),
        ("Z(0)", Z(0)),
        ("X(0)", X(0)),
        ("Y(0)", Y(0)),
        ("Z(0) + 2", 2 + Z(0)),
        ("Z(0) + 2j", 2j + Z(0)),
        ("Z(0) - 2j", -2j + Z(0)),
        ("Z(0) - 2j + 3", (3 - 2j) + Z(0)),
        ("Z(0) - 2j + 3 + 2 Z(0)", (3 - 2j) + 3 * Z(0)),
    ],
)
def test_parse(hamiltonian_str: str, expected_hamiltonian: Hamiltonian):
    hamiltonian = Hamiltonian.parse(hamiltonian_str)
    assert hamiltonian == expected_hamiltonian


@pytest.mark.parametrize(
    ("hamiltonian", "expected"),
    [
        # Identity Hamiltonian on qubit 0.
        (I(0).to_hamiltonian(), np.eye(2, dtype=complex)),
        # 2 * Z operator: 2*[[1, 0],[0, -1]]
        (2 * Z(0).to_hamiltonian(), 2 * np.array([[1, 0], [0, -1]], dtype=complex)),
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
    H = Hamiltonian({(Z(0), X(1)): 2})
    Z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = 2 * np.kron(Z_matrix, X_matrix)
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)


def test_to_matrix_two_qubit_multiple_terms():
    """
    For a Hamiltonian defined as 0.5 * (Z(0) ⊗ I) + 1.5 * (I ⊗ X(1)),
    the matrix representation should be the sum of the two Kronecker products.
    """
    H = 0.5 * Z(0).to_hamiltonian() + 1.5 * X(1).to_hamiltonian()
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
    H = Hamiltonian({(Z(1), X(2)): 3})
    I2 = np.eye(2, dtype=complex)
    Z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
    X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    expected = 3 * np.kron(I2, np.kron(Z_matrix, X_matrix))
    np.testing.assert_allclose(dense(H), expected, atol=1e-8)
