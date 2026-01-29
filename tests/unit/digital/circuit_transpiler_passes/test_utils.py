from dataclasses import dataclass

import numpy as np
import pytest

from .utils import (
    _apply_gate_to_state,
    _bits_to_int,
    _expand_gate_to_order,
    _int_to_bits,
    _sequence_matrix,
    _sequences_equivalent,
    _unitaries_equivalent,
    _vectors_equal_up_to_phase,
)


@dataclass(frozen=True)
class Gate:
    matrix: np.ndarray
    qubits: tuple[int, ...]


# ---------------------------
# Bit / integer conversions
# ---------------------------


@pytest.mark.parametrize(
    ("value", "width", "expected"),
    [
        (0, 1, (0,)),
        (1, 1, (1,)),
        (2, 2, (1, 0)),
        (3, 2, (1, 1)),
        (5, 3, (1, 0, 1)),
    ],
)
def test_int_to_bits(value, width, expected):
    assert _int_to_bits(value, width) == expected


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ((0,), 0),
        ((1,), 1),
        ((1, 0), 2),
        ((1, 1), 3),
        ((1, 0, 1), 5),
    ],
)
def test_bits_to_int(bits, expected):
    assert _bits_to_int(bits) == expected


def test_bits_roundtrip():
    for width in range(1, 6):
        for value in range(1 << width):
            assert _bits_to_int(_int_to_bits(value, width)) == value


# ---------------------------
# Gate expansion
# ---------------------------


def test_expand_single_qubit_gate():
    # Pauli-X on qubit 0 in a 2-qubit system
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    gate = Gate(matrix=X, qubits=(0,))
    order = (0, 1)

    expanded = _expand_gate_to_order(gate, order)

    expected = np.kron(X, np.eye(2))
    assert np.allclose(expanded, expected)


def test_expand_two_qubit_gate_swapped_order():
    # CNOT with control=1, target=0
    CNOT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=complex,
    )
    gate = Gate(matrix=CNOT, qubits=(1, 0))
    order = (0, 1)

    expanded = _expand_gate_to_order(gate, order)
    assert expanded.shape == (4, 4)
    assert np.allclose(expanded.conj().T @ expanded, np.eye(4))


# ---------------------------
# State application
# ---------------------------


def test_apply_gate_matches_matrix_application():
    rng = np.random.default_rng(0)

    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    gate = Gate(matrix=H, qubits=(1,))

    nqubits = 2
    dim = 1 << nqubits

    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    state /= np.linalg.norm(state)

    expanded = _expand_gate_to_order(gate, (0, 1))
    out_matrix = expanded @ state
    out_state = _apply_gate_to_state(state, gate, nqubits)

    assert np.allclose(out_matrix, out_state)


# ---------------------------
# Sequence matrix
# ---------------------------


def test_sequence_matrix_composition():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    gates = [
        Gate(matrix=X, qubits=(0,)),
        Gate(matrix=Z, qubits=(0,)),
    ]

    seq = _sequence_matrix(gates, nqubits=1)
    expected = Z @ X

    assert np.allclose(seq, expected)


# ---------------------------
# Unitary equivalence
# ---------------------------


def test_unitaries_equivalent_up_to_phase():
    U = np.eye(2, dtype=complex)
    V = np.exp(1j * 0.123) * np.eye(2, dtype=complex)

    assert _unitaries_equivalent(U, V)
    assert _unitaries_equivalent(V, U)


def test_unitaries_non_equivalent():
    H = (1 / np.sqrt(2)) * np.array(
        [[1, 1], [1, -1]],
        dtype=complex,
    )
    I = np.eye(2, dtype=complex)
    assert not _unitaries_equivalent(H, I)
    assert not _unitaries_equivalent(I, H)


# ---------------------------
# Vector phase equivalence
# ---------------------------


def test_vectors_equal_up_to_global_phase():
    v = np.array([1, 1j], dtype=complex)
    w = np.exp(1j * 0.7) * v

    assert _vectors_equal_up_to_phase(v, w)


def test_vectors_not_equal():
    v = np.array([1, 0], dtype=complex)
    w = np.array([0, 1], dtype=complex)

    assert not _vectors_equal_up_to_phase(v, w)


def test_zero_vectors_equal():
    v = np.array([0, 0], dtype=complex)
    w = np.array([0, 0], dtype=complex)

    assert _vectors_equal_up_to_phase(v, w)


# ---------------------------
# Sequence equivalence
# ---------------------------


def test_sequences_equivalent_small_system():
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    gates_a = [Gate(matrix=X, qubits=(0,)), Gate(matrix=X, qubits=(0,))]
    gates_b = []

    assert _sequences_equivalent(gates_a, gates_b, nqubits=1)


def test_sequences_equivalent_with_basis_states():
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    gates_a = [
        Gate(matrix=H, qubits=(0,)),
        Gate(matrix=X, qubits=(0,)),
        Gate(matrix=H, qubits=(0,)),
    ]
    gates_b = [Gate(matrix=np.array([[1, 0], [0, -1]], dtype=complex), qubits=(0,))]

    basis = [(0,), (1,)]
    assert _sequences_equivalent(gates_a, gates_b, nqubits=1, basis_states=basis)


def test_sequences_equivalent_large_with_basis_states():
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    nqubits = 10
    gates_a = [Gate(matrix=X, qubits=(i,)) for i in range(nqubits)]
    gates_b = [Gate(matrix=X, qubits=(i,)) for i in range(nqubits)]

    assert _sequences_equivalent(gates_a, gates_b, nqubits=nqubits)


def test_sequences_not_equivalent_large_with_basis_states():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    nqubits = 10
    gates_a = [Gate(matrix=X, qubits=(i,)) for i in range(nqubits)]
    gates_b = [Gate(matrix=Y, qubits=(i,)) for i in range(nqubits)]

    assert not _sequences_equivalent(gates_a, gates_b, nqubits=nqubits)
