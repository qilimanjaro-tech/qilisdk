from qilisdk.analog import Z


def test_pauli_addition():
    """
    Test basic pauli operator addition
    """
    H = Z(0) + Z(1)
    assert str(H) == "Z(0) + Z(1)"

    H = Z(0) + Z(0)
    assert str(H) == "2 Z(0)"
