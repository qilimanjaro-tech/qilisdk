import pytest

from qilisdk.analog import Hamiltonian, X, Y, Z


@pytest.mark.parametrize(
    ("hamiltonian", "expected_str"),
    [
        (Z(0) + Z(1), "Z(0) + Z(1)"),
        (Z(0) + Z(0), "2 Z(0)"),
        (Z(0) + Z(1) + Z(0), "2 Z(0) + Z(1)"),
        (Z(0) + Z(1) + Z(0) + 1, "1 + 2 Z(0) + Z(1)"),
        (X(0) + X(1), "X(0) + X(1)"),
        (X(0) + X(0), "2 X(0)"),
        (X(0) + X(1) + X(0), "2 X(0) + X(1)"),
        (X(0) + X(1) + X(0) + 1, "1 + 2 X(0) + X(1)"),
        (Y(0) + Y(1), "Y(0) + Y(1)"),
        (Y(0) + Y(0), "2 Y(0)"),
        (Y(0) + Y(1) + Y(0), "2 Y(0) + Y(1)"),
        (Y(0) + Y(1) + Y(0) + 1, "1 + 2 Y(0) + Y(1)"),
        (1 + Z(0) + Z(1) + Z(0), "1 + 2 Z(0) + Z(1)"),
        (1 + Z(0) + 3 + Z(1) + Z(0) + 2j, "(4+2j) + 2 Z(0) + Z(1)"),
    ],
)
def test_addition(hamiltonian: Hamiltonian, expected_str: str):
    assert str(hamiltonian) == expected_str


@pytest.mark.parametrize(
    ("hamiltonian", "expected_str"),
    [
        # Simple subtractions
        (Z(0) - Z(1), "Z(0) - Z(1)"),
        (Z(0) - Z(0), "0"),
        ((Z(0) + 2) - 1, "1 + Z(0)"),
        (5 - X(0), "5 - X(0)"),
        ((Z(0) + Z(1)) - Z(1), "Z(0)"),
        ((Z(0) + 3) - 3, "Z(0)"),
        (Z(0) - Z(1) + Z(0), "2 Z(0) - Z(1)"),
        (1 - Z(0) + Z(1) + Z(0), "1 + Z(1)"),
        (1 + Z(0) - 3 + Z(1) + Z(0) - 2j, "(-2-2j) + 2 Z(0) + Z(1)"),
    ],
)
def test_subtraction(hamiltonian: Hamiltonian, expected_str: str):
    """Test various subtraction scenarios."""
    assert str(hamiltonian) == expected_str


@pytest.mark.parametrize(
    ("hamiltonian", "expected_str"),
    [
        # Pauli-by-Pauli on same qubit
        (Z(0) * Z(0), "1"),
        (Z(0) * X(0), "1j Y(0)"),  # because Z*X = i Y
        (X(0) * Z(0), "-1j Y(0)"),  # because X*Z = -i Y
        # Pauli-by-Pauli on different qubits => direct product (no phase from the table)
        (Z(0) * Z(1), "Z(0) Z(1)"),
        (X(0) * Z(1), "X(0) Z(1)"),
        # Scalar multiplication
        (2 * Z(0), "2 Z(0)"),
        (Z(0) * 3, "3 Z(0)"),
        # Hamiltonian multiplied by scalar
        ((Z(0) + Z(1)) * 2, "2 Z(0) + 2 Z(1)"),
        # Hamiltonian-by-Hamiltonian
        ((Z(0) + X(1)) * (Z(0) + X(1)), "2 + 2 Z(0) X(1)"),
        ((Z(0) + X(1)) * (Z(1) + X(0)), "Z(0) Z(1) + 1j Y(0) - 1j Y(1) + X(0) X(1)"),
        ((Z(0) + X(0)) * (Z(0) - X(0)), "-2j Y(0)"),
        ((Z(0) + 1) * (1j * X(0) * X(1) + 1), "1 - Y(0) X(1) + Z(0) + 1j X(0) X(1)"),
        (
            (1 + Z(0) * Z(0) + Z(0) * X(1) + Z(0) * X(1) + X(1) * X(1) + X(1) * Z(1) + Y(1) * Z(1)),
            "3 + 2 Z(0) X(1) - 1j Y(1) + 1j X(1)",
        ),
    ],
)
def test_multiplication(hamiltonian: Hamiltonian, expected_str: str):
    """Test various multiplication scenarios."""
    assert str(hamiltonian) == expected_str


@pytest.mark.parametrize(
    ("hamiltonian", "expected_str"),
    [
        # Simple division by scalar
        (Z(0) / 2, "0.5 Z(0)"),
        ((Z(0) + Z(1)) / 2, "0.5 Z(0) + 0.5 Z(1)"),
        ((Z(0) + 3) / 2, "1.5 + 0.5 Z(0)"),
        # Division of a pure scalar Hamiltonian
        (5 / 2, "2.5"),  # Just a constant Hamiltonian => "2.5"
        # Combining scalar and operator
        ((Z(0) + X(1)) / 1j, "-1j Z(0) - 1j X(1)"),
        # Explanation: dividing by 1j => multiply by -1j
        # so each term gets an extra -i factor
    ],
)
def test_division(hamiltonian: Hamiltonian, expected_str: str):
    """Test various division scenarios."""
    assert str(hamiltonian) == expected_str
