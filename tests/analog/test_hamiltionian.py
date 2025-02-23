from qilisdk.analog import Hamiltonian, X, Y, Z, I


def test_pauli_addition():
    """
    Test basic pauli operator addition
    """
    H = Z(0) + Z(1)
    assert str(H) == "Z(0) + Z(1)"

    op1 = Z(0)
    op2 = Z(0)
    H = 2 * Z(0)
    assert (op1 + op2) - H == 0