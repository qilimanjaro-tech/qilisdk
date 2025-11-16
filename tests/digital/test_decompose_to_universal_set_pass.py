import math
from collections.abc import Callable

import numpy as np
import pytest

from qilisdk.digital import Circuit
from qilisdk.digital.circuit_transpiler_passes.decompose_to_universal_set_pass import (
    _DECOMPOSERS,
    DecomposeToUniversalSetPass,
    UniversalSet,
    decompose_gate_for_universal_set,
)
from qilisdk.digital.circuit_transpiler_passes.numeric_helpers import _zyz_from_unitary
from qilisdk.digital.exceptions import GateHasNoMatrixError
from qilisdk.digital.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U1,
    U2,
    U3,
    Adjoint,
    BasicGate,
    Exponential,
    Gate,
    H,
    I,
    M,
    S,
    T,
    X,
    Y,
    Z,
)

ALLOWED_TYPES = {
    UniversalSet.CLIFFORD_T: (CNOT, H, M, S, T),
    UniversalSet.RZ_RX_CX: (CNOT, M, RX, RZ),
    UniversalSet.U3_CX: (CNOT, M, U3),
}


GateFactory = Callable[[], Gate]


GATE_FACTORIES: list[tuple[str, GateFactory]] = [
    ("M", lambda: M(0)),
    ("I", lambda: I(0)),
    ("RX", lambda: RX(0, theta=math.pi / 2.0)),
    ("RY", lambda: RY(0, theta=math.pi / 2.0)),
    ("RZ", lambda: RZ(0, phi=math.pi / 2.0)),
    ("U1", lambda: U1(0, phi=math.pi / 2.0)),
    ("U2", lambda: U2(0, phi=0.0, gamma=math.pi / 2.0)),
    ("U3", lambda: U3(0, theta=math.pi / 2.0, phi=0.0, gamma=math.pi / 2.0)),
    ("X", lambda: X(0)),
    ("Y", lambda: Y(0)),
    ("Z", lambda: Z(0)),
    ("H", lambda: H(0)),
    ("S", lambda: S(0)),
    ("T", lambda: T(0)),
    ("CNOT", lambda: CNOT(0, 1)),
    ("CZ", lambda: CZ(0, 1)),
    ("SWAP", lambda: SWAP(0, 1)),
    ("Adjoint_I", lambda: I(0).adjoint()),
    ("Adjoint_RX", lambda: RX(0, theta=math.pi / 2.0).adjoint()),
    ("Adjoint_RY", lambda: RY(0, theta=math.pi / 2.0).adjoint()),
    ("Adjoint_RZ", lambda: RZ(0, phi=math.pi / 2.0).adjoint()),
    ("Adjoint_U1", lambda: U1(0, phi=math.pi / 2.0).adjoint()),
    ("Adjoint_U2", lambda: U2(0, phi=0.0, gamma=math.pi / 2.0).adjoint()),
    ("Adjoint_U3", lambda: U3(0, theta=math.pi / 2.0, phi=0.0, gamma=math.pi / 2.0).adjoint()),
    ("Adjoint_X", lambda: X(0).adjoint()),
    ("Adjoint_Y", lambda: Y(0).adjoint()),
    ("Adjoint_Z", lambda: Z(0).adjoint()),
    ("Adjoint_H", lambda: H(0).adjoint()),
    ("Adjoint_S", lambda: S(0).adjoint()),
    ("Adjoint_T", lambda: T(0).adjoint()),
    ("Adjoint_SWAP", lambda: SWAP(0, 1).adjoint()),
]


ADJOINT_FACTORIES: list[tuple[str, GateFactory]] = [
    (name, factory) for name, factory in GATE_FACTORIES if name.startswith("Adjoint_")
]


class _SkewHermitianGenerator(BasicGate):
    def __init__(self, qubit: int, strength: float) -> None:
        self._strength = strength
        super().__init__((qubit,))

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "SkewHermitian"

    def _generate_matrix(self) -> np.ndarray:  # pragma: no cover - trivial
        a = self._strength
        return np.array([[0.0, a], [-a, 0.0]], dtype=complex)


EXPONENTIAL_FACTORIES: list[tuple[str, GateFactory]] = [
    ("Exponential_Skew_0.5", lambda: _SkewHermitianGenerator(0, 0.5).exponential()),
    ("Exponential_Skew_1.0", lambda: _SkewHermitianGenerator(0, 1.0).exponential()),
]


def _int_to_bits(value: int, width: int) -> tuple[int, ...]:
    return tuple((value >> shift) & 1 for shift in reversed(range(width)))


def _bits_to_int(bits: tuple[int, ...]) -> int:
    acc = 0
    for bit in bits:
        acc = (acc << 1) | bit
    return acc


def _qubit_order(gate: Gate) -> tuple[int, ...]:
    order: list[int] = []
    for qubit in gate.qubits:
        if qubit not in order:
            order.append(qubit)
    return tuple(order)


def _gate_signature(gate: Gate) -> tuple:
    params = tuple(sorted(gate.get_parameters().items())) if gate.is_parameterized else ()
    return (type(gate), gate.qubits, params)


def _expand_gate_to_order(gate: Gate, order: tuple[int, ...]) -> np.ndarray | None:
    try:
        local_matrix = gate.matrix
    except GateHasNoMatrixError:
        return None

    positions = tuple(order.index(qubit) for qubit in gate.qubits)
    nqubits = len(order)
    dim = 1 << nqubits
    expanded = np.zeros((dim, dim), dtype=complex)

    for column in range(dim):
        bits = list(_int_to_bits(column, nqubits))
        sub_bits = tuple(bits[pos] for pos in positions)
        local_col = _bits_to_int(sub_bits)
        for local_row in range(1 << len(positions)):
            amplitude = local_matrix[local_row, local_col]
            if amplitude == 0:
                continue
            target_bits = bits.copy()
            new_sub_bits = _int_to_bits(local_row, len(positions))
            for pos, bit in zip(positions, new_sub_bits):
                target_bits[pos] = bit
            row = _bits_to_int(tuple(target_bits))
            expanded[row, column] += amplitude

    return expanded


def _sequence_matrix(gates: list[Gate], order: tuple[int, ...]) -> np.ndarray | None:
    if not gates:
        dim = 1 << len(order)
        return np.eye(dim, dtype=complex)

    dim = 1 << len(order)
    aggregate = np.eye(dim, dtype=complex)
    for gate in gates:
        expanded = _expand_gate_to_order(gate, order)
        if expanded is None:
            return None
        aggregate = aggregate @ expanded
    return aggregate


def _assert_gate_types(gate: Gate, basis: UniversalSet, primitives: list[Gate]) -> None:
    if not primitives:
        assert isinstance(gate, I) or (isinstance(gate, Adjoint) and isinstance(gate.basic_gate, I))
        return

    allowed = ALLOWED_TYPES[basis]
    for primitive in primitives:
        assert isinstance(primitive, allowed)

    if isinstance(gate, M):
        assert primitives == [gate]


def _assert_matrix_equivalence(gate: Gate, primitives: list[Gate]) -> None:
    order = _qubit_order(gate)
    original_matrix = _sequence_matrix([gate], order)
    decomposed_matrix = _sequence_matrix(primitives, order)

    if original_matrix is None or decomposed_matrix is None:
        assert original_matrix is None
        assert decomposed_matrix is None
        return

    assert _unitaries_equivalent(original_matrix, decomposed_matrix)


def _unitaries_equivalent(first: np.ndarray, second: np.ndarray, atol: float = 1e-9) -> bool:
    phase = None
    for value, other in zip(first.flat, second.flat):
        if abs(value) > atol and abs(other) > atol:
            phase = value / other
            break
    if phase is None:
        phase = 1.0
    return np.allclose(first, phase * second, atol=atol)


def _helper_cases() -> list:
    cases = []
    for name, factory in GATE_FACTORIES:
        sample_gate = factory()
        gate_cls = type(sample_gate)
        for basis in UniversalSet:
            if name.startswith("Exponential_") and basis == UniversalSet.CLIFFORD_T:
                continue
            helper = _DECOMPOSERS[gate_cls][basis]
            cases.append(
                pytest.param(factory, basis, helper, id=f"{name}-{basis.name}")
            )
    return cases


def _adjoint_cases() -> list:
    cases = []
    for name, factory in ADJOINT_FACTORIES:
        for basis in UniversalSet:
            cases.append(pytest.param(factory, basis, id=f"{name}-{basis.name}"))
    return cases


def _exponential_cases() -> list:
    cases = []
    for name, factory in EXPONENTIAL_FACTORIES:
        for basis in UniversalSet:
            if basis == UniversalSet.CLIFFORD_T:
                continue
            cases.append(pytest.param(factory, basis, id=f"{name}-{basis.name}"))
    return cases


def _u3_equivalent_from_exponential(gate: Exponential[BasicGate]) -> U3:
    base = gate.basic_gate
    if base.nqubits != 1:
        raise NotImplementedError("Exponential of multi-qubit gates not supported in tests.")
    theta, phi, gamma = _zyz_from_unitary(gate.matrix)
    return U3(base.qubits[0], theta=theta, phi=phi, gamma=gamma)


@pytest.mark.parametrize(("factory", "basis", "helper"), _helper_cases())
def test_universal_set_helpers_preserve_behavior(
    factory: GateFactory, basis: UniversalSet, helper: Callable[[Gate], list[Gate]]
) -> None:
    gate = factory()
    primitives = helper(gate)
    _assert_gate_types(gate, basis, primitives)
    _assert_matrix_equivalence(gate, primitives)


@pytest.mark.parametrize(("factory", "basis", "helper"), _helper_cases())
def test_dispatch_function_matches_helper(
    factory: GateFactory, basis: UniversalSet, helper: Callable[[Gate], list[Gate]]
) -> None:
    gate = factory()
    expected = helper(gate)
    actual = decompose_gate_for_universal_set(gate, basis)
    assert len(actual) == len(expected)
    for produced, reference in zip(actual, expected):
        assert type(produced) is type(reference)
        assert produced.qubits == reference.qubits


def test_pass_rewrites_circuit_to_target_basis() -> None:
    circuit = Circuit(2)
    circuit.add(CZ(0, 1))
    circuit.add(RZ(1, phi=math.pi / 2.0))

    transpiled = DecomposeToUniversalSetPass(UniversalSet.U3_CX).run(circuit)

    assert all(isinstance(g, (CNOT, U3, M)) for g in transpiled.gates)


@pytest.mark.parametrize(("factory", "basis"), _adjoint_cases())
def test_adjoints_match_manual_reverse_sequences(factory: GateFactory, basis: UniversalSet) -> None:
    adjoint_gate = factory()
    base_gate = adjoint_gate.basic_gate
    base_sequence = decompose_gate_for_universal_set(base_gate, basis)

    expected: list[Gate] = []
    for primitive in reversed(base_sequence):
        if hasattr(primitive, "adjoint"):
            adjoint = primitive.adjoint()
            expected.extend(decompose_gate_for_universal_set(adjoint, basis))
        else:
            expected.append(primitive)

    actual = decompose_gate_for_universal_set(adjoint_gate, basis)

    assert len(actual) == len(expected)
    for produced, reference in zip(actual, expected):
        assert _gate_signature(produced) == _gate_signature(reference)


@pytest.mark.parametrize(("factory", "basis"), _adjoint_cases())
def test_adjoints_preserve_unitaries(factory: GateFactory, basis: UniversalSet) -> None:
    gate = factory()
    primitives = decompose_gate_for_universal_set(gate, basis)
    _assert_matrix_equivalence(gate, primitives)


@pytest.mark.parametrize("factory,basis", _exponential_cases())
def test_exponentials_match_u3_decomposition(factory: GateFactory, basis: UniversalSet) -> None:
    gate = factory()
    equivalent_u3 = _u3_equivalent_from_exponential(gate)
    expected = decompose_gate_for_universal_set(equivalent_u3, basis)
    actual = decompose_gate_for_universal_set(gate, basis)

    assert len(actual) == len(expected)
    for produced, reference in zip(actual, expected):
        assert _gate_signature(produced) == _gate_signature(reference)


@pytest.mark.parametrize("factory,basis", _exponential_cases())
def test_exponentials_preserve_unitaries(factory: GateFactory, basis: UniversalSet) -> None:
    gate = factory()
    primitives = decompose_gate_for_universal_set(gate, basis)
    _assert_matrix_equivalence(gate, primitives)
