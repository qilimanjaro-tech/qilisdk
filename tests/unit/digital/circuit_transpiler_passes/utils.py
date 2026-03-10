import numpy as np

from qilisdk.digital.gates import Gate


def _int_to_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in reversed(range(width)))


def _bits_to_int(bits: tuple[int, ...]) -> int:
    acc = 0
    for bit in bits:
        acc = (acc << 1) | bit
    return acc


def _expand_gate_to_order(gate: Gate, order: tuple[int, ...]) -> np.ndarray:
    local_matrix = gate.matrix
    positions = tuple(order.index(q) for q in gate.qubits)
    nqubits = len(order)
    dim = 1 << nqubits
    expanded = np.zeros((dim, dim), dtype=complex)

    for column in range(dim):
        bits = list(_int_to_bits(column, nqubits))
        sub_bits = tuple(bits[pos] for pos in positions)
        local_col = _bits_to_int(sub_bits)
        for local_row in range(1 << len(positions)):
            amplitude = local_matrix[local_row, local_col]
            if np.isclose(amplitude, 0.0):
                continue
            target_bits = bits.copy()
            new_sub_bits = _int_to_bits(local_row, len(positions))
            for pos, bit in zip(positions, new_sub_bits):
                target_bits[pos] = bit
            row = _bits_to_int(tuple(target_bits))
            expanded[row, column] += amplitude
    return expanded


def _apply_gate_to_state(state: np.ndarray, gate: Gate, nqubits: int) -> np.ndarray:
    axes = list(gate.qubits)
    other_axes = [i for i in range(nqubits) if i not in axes]
    perm = axes + other_axes
    inverse = np.argsort(perm)

    reshaped = state.reshape([2] * nqubits).transpose(perm)
    block = reshaped.reshape(2 ** len(axes), -1)
    updated = gate.matrix @ block
    reshaped = updated.reshape([2] * len(axes) + [2] * len(other_axes)).transpose(inverse)
    return reshaped.reshape(-1)


def _sequence_matrix(gates: list[Gate], nqubits: int) -> np.ndarray:
    order = tuple(range(nqubits))
    dim = 1 << nqubits
    aggregate = np.eye(dim, dtype=complex)
    for gate in gates:
        expanded = _expand_gate_to_order(gate, order)
        aggregate = expanded @ aggregate
    return aggregate


def _unitaries_equivalent(first: np.ndarray, second: np.ndarray, atol: float = 1e-9) -> bool:
    diff = first.conj().T @ second
    if not np.allclose(diff, np.diag(np.diag(diff)), atol=atol):
        return False
    diag = np.diag(diff)
    return np.allclose(np.abs(diag), np.ones_like(diag), atol=atol)


def _apply_sequence_to_state(gates: list[Gate], state: np.ndarray, nqubits: int) -> np.ndarray:
    for gate in gates:
        state = _apply_gate_to_state(state, gate, nqubits)
    return state


def _vectors_equal_up_to_phase(vec_a: np.ndarray, vec_b: np.ndarray, atol: float = 1e-9) -> bool:
    idx = np.argmax(np.abs(vec_a) > atol)
    if np.abs(vec_a[idx]) <= atol:
        return np.allclose(vec_a, vec_b, atol=atol)
    phase = vec_b[idx] / vec_a[idx]
    return np.allclose(vec_b, phase * vec_a, atol=atol)


def _sequences_equivalent(
    gates_a: list[Gate], gates_b: list[Gate], nqubits: int, basis_states: list[tuple[int, ...]] | None = None
) -> bool:
    dim = 1 << nqubits
    if dim <= 256 and basis_states is None:
        return _unitaries_equivalent(_sequence_matrix(gates_a, nqubits), _sequence_matrix(gates_b, nqubits))

    if basis_states is None:
        rng = np.random.default_rng(0)
        basis_vectors = []
        for _ in range(3):
            vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            vec /= np.linalg.norm(vec)
            basis_vectors.append(vec)
    else:
        basis_vectors = []
        for bits in basis_states:
            vec = np.zeros(dim, dtype=complex)
            idx = _bits_to_int(bits)
            vec[idx] = 1.0
            basis_vectors.append(vec)

    for vec in basis_vectors:
        out_a = _apply_sequence_to_state(gates_a, vec.copy(), nqubits)
        out_b = _apply_sequence_to_state(gates_b, vec.copy(), nqubits)
        if not _vectors_equal_up_to_phase(out_a, out_b):
            return False
    return True
