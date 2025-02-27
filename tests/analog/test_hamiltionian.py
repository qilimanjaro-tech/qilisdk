import pytest

from qilisdk.analog import Hamiltonian, I, X, Y, Z


# -----------------------------
#  ADDITION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_ham"),
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
def test_addition(hamiltonian: Hamiltonian, expected_ham: Hamiltonian):
    assert hamiltonian == expected_ham


# -----------------------------
#  SUBTRACTION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_ham"),
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
def test_subtraction(hamiltonian: Hamiltonian, expected_ham: Hamiltonian):
    assert hamiltonian == expected_ham


# -----------------------------
#  MULTIPLICATION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_ham"),
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
            # It's perfectly valid just to keep it as an expression, or expand it out.
            # We'll keep it as an expression for brevity.
            (Z(0) + 1) * (1j * X(0) * X(1) + 1),
        ),
        (
            1 + Z(0) * Z(0) + Z(0) * X(1) + Z(0) * X(1) + X(1) * X(1) + X(1) * Z(1) + Y(1) * Z(1),
            3 + 2 * (Z(0) * X(1)) - 1j * Y(1) + 1j * X(1),
        ),
    ],
)
def test_multiplication(hamiltonian: Hamiltonian, expected_ham: Hamiltonian):
    assert hamiltonian == expected_ham


# -----------------------------
#  DIVISION TESTS
# -----------------------------
@pytest.mark.parametrize(
    ("hamiltonian", "expected_ham"),
    [
        (Z(0) / 2, 0.5 * Z(0)),
        ((Z(0) + Z(1)) / 2, 0.5 * Z(0) + 0.5 * Z(1)),
        ((Z(0) + 3) / 2, 1.5 + 0.5 * Z(0)),
        (5 / 2, Hamiltonian({(I(0),): 2.5})),
        ((Z(0) + X(1)) / 1j, -1j * Z(0) + -1j * X(1)),
    ],
)
def test_division(hamiltonian: Hamiltonian, expected_ham: Hamiltonian):
    assert hamiltonian == expected_ham
