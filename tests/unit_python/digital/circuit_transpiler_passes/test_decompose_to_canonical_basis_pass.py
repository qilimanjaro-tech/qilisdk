import math
from collections import Counter
from typing import Iterable

import numpy as np
import pytest

import qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass as decompose_module
from qilisdk.digital import Circuit
from qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass import (
    DecomposeToCanonicalBasisPass,
    SingleQubitGateBasis,
    TwoQubitGateBasis,
    _as_basis_1q,
    _cz_in_basis,
    _invert_basis_gate,
    _normalize_single_qubit_gate,
    _single_controlled,
)
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
    Controlled,
    Exponential,
    Gate,
    H,
    I,
    M,
    X,
    Y,
    Z,
)


class DummyBasicGate(BasicGate):
    def __init__(self, target_qubits: Iterable[int], matrix: np.ndarray) -> None:
        self._custom_matrix = np.array(matrix, dtype=complex)
        super().__init__(tuple(target_qubits))

    @property
    def name(self) -> str:
        return "Dummy"

    def _generate_matrix(self) -> np.ndarray:
        return self._custom_matrix


def _count_gate_names(seq: list[Gate]) -> Counter[str]:
    return Counter(g.name for g in seq)


def _nqubits_for_gates(*gates: Gate) -> int:
    highest_qubit = max((max(gate.qubits) for gate in gates if gate.qubits), default=-1)
    return highest_qubit + 1 if highest_qubit >= 0 else 1


def _matrix_of_gates(gates: list[Gate], nqubits: int | None = None) -> np.ndarray:
    circuit = Circuit(_nqubits_for_gates(*gates) if nqubits is None else nqubits)
    for gate in gates:
        circuit.add(gate)
    return circuit.to_matrix()


def _assert_gate_sequence_matches(original: Gate, rewritten: list[Gate]) -> None:
    nqubits = _nqubits_for_gates(original, *rewritten)
    assert np.allclose(
        _matrix_of_gates([original], nqubits=nqubits),
        _matrix_of_gates(rewritten, nqubits=nqubits),
    )


def test_invert_basis_gate_handles_known_types() -> None:
    gates = [
        U3(0, theta=0.2, phi=0.1, gamma=-0.3),
        RX(0, theta=0.4),
        RY(0, theta=-0.5),
        RZ(0, phi=0.7),
        CNOT(0, 1),
        CZ(0, 1),
        H(0),
        X(0),
        Y(0),
        Z(0),
    ]

    for gate in gates:
        inverse = _invert_basis_gate(gate)
        nqubits = _nqubits_for_gates(gate, *inverse)
        identity = np.eye(1 << nqubits, dtype=complex)
        assert np.allclose(_matrix_of_gates([gate, *inverse], nqubits=nqubits), identity)


def test_invert_basis_gate_handles_measurement_and_fallback() -> None:
    assert _invert_basis_gate(M(0))[0].name == "M"

    dummy = DummyBasicGate((0,), np.eye(2))
    assert isinstance(_invert_basis_gate(dummy)[0], Adjoint)


def test_as_basis_1q_handles_standard_and_basic() -> None:
    dummy_gate = DummyBasicGate((0,), np.eye(2, dtype=complex))
    rx_gate = RX(0, theta=0.3)

    assert _as_basis_1q(rx_gate) is rx_gate

    for gate in [H(0), X(0), Y(0), Z(0), U1(0, phi=0.1), U2(0, phi=0.1, gamma=0.2), dummy_gate]:
        basis_gate = _as_basis_1q(gate)
        assert basis_gate.name in {"U3", "RX", "RY", "RZ"}
        assert np.allclose(gate.matrix, basis_gate.matrix)

    with pytest.raises(NotImplementedError):
        _as_basis_1q(Controlled(1, basic_gate=RX(0, theta=0.5)))


def test_cz_in_basis_returns_native_cz_when_requested() -> None:
    sequence = _cz_in_basis(0, 1, TwoQubitGateBasis.CZ)

    assert len(sequence) == 1
    assert isinstance(sequence[0], CZ)
    assert sequence[0].qubits == (0, 1)


def test_normalize_single_qubit_gate_rejects_unknown_basis() -> None:
    with pytest.raises(TypeError, match="Unsupported single-qubit basis"):
        _normalize_single_qubit_gate(H(0), object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("two_qubit_basis", "single_qubit_basis", "target_factory"),
    [
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.U3, lambda: RZ(0, phi=0.2)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.U3, lambda: RX(0, theta=0.2)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.U3, lambda: RY(0, theta=0.2)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.U3, lambda: U3(0, theta=0.3, phi=0.2, gamma=-0.1)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.U3, lambda: DummyBasicGate((5,), np.eye(2))),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.U3, lambda: RZ(0, phi=0.2)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.U3, lambda: RX(0, theta=0.2)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.U3, lambda: RY(0, theta=0.2)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.U3, lambda: U3(0, theta=0.3, phi=0.2, gamma=-0.1)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.U3, lambda: DummyBasicGate((5,), np.eye(2))),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.RxRyRz, lambda: RZ(0, phi=0.2)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.RxRyRz, lambda: RX(0, theta=0.2)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.RxRyRz, lambda: RY(0, theta=0.2)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.RxRyRz, lambda: DummyBasicGate((5,), np.eye(2))),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.RxRyRz, lambda: RZ(0, phi=0.2)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.RxRyRz, lambda: RX(0, theta=0.2)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.RxRyRz, lambda: RY(0, theta=0.2)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.RxRyRz, lambda: DummyBasicGate((5,), np.eye(2))),
    ],
)
def test_single_controlled_paths_preserve_matrix(
    two_qubit_basis: TwoQubitGateBasis,
    single_qubit_basis: SingleQubitGateBasis,
    target_factory,
) -> None:
    target_gate = target_factory()
    sequence = _single_controlled(
        1,
        target_gate,
        basis=two_qubit_basis,
        single_qubit_basis=single_qubit_basis,
    )

    allowed_one_qubit_names = {"U3"} if single_qubit_basis == SingleQubitGateBasis.U3 else {"RX", "RY", "RZ"}
    assert _count_gate_names(sequence)[two_qubit_basis.value] >= 1
    assert all(gate.name in allowed_one_qubit_names | {two_qubit_basis.value} for gate in sequence)
    _assert_gate_sequence_matches(Controlled(1, basic_gate=target_gate), sequence)


def test_rewrite_gate_covers_various_exact_cases() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)

    gates = [
        U3(0, theta=0.1, phi=0.2, gamma=0.3),
        RX(0, theta=0.2),
        RY(0, theta=0.3),
        CZ(0, 1),
        H(0),
        X(0),
        Y(0),
        Z(0),
        U1(0, phi=0.5),
        U2(0, phi=0.6, gamma=0.7),
        CNOT(0, 1),
        SWAP(0, 1),
    ]

    for gate in gates:
        sequence = pass_instance._rewrite_gate(gate)
        assert all(rewritten.name in {"CZ", "U3"} for rewritten in sequence)
        _assert_gate_sequence_matches(gate, sequence)


def test_rewrite_gate_handles_measurement_identity_controlled_and_adjoint() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)

    measurement_sequence = pass_instance._rewrite_gate(M(0))
    assert len(measurement_sequence) == 1
    assert measurement_sequence[0].name == "M"

    assert pass_instance._rewrite_gate(I(0)) == []

    controlled_gate = Controlled(0, basic_gate=RX(1, theta=0.3))
    controlled_sequence = pass_instance._rewrite_gate(controlled_gate)
    assert _count_gate_names(controlled_sequence)["CZ"] >= 1
    _assert_gate_sequence_matches(controlled_gate, controlled_sequence)

    adjoint_gate = Adjoint(RX(0, theta=math.pi / 3))
    adjoint_sequence = pass_instance._rewrite_gate(adjoint_gate)
    assert all(gate.name == "U3" for gate in adjoint_sequence)
    _assert_gate_sequence_matches(adjoint_gate, adjoint_sequence)

    basic_gate = DummyBasicGate((0,), np.eye(2))
    basic_sequence = pass_instance._rewrite_gate(basic_gate)
    assert len(basic_sequence) == 1
    assert basic_sequence[0].name == "U3"
    _assert_gate_sequence_matches(basic_gate, basic_sequence)


def test_rewrite_gate_rejects_unsupported_cases() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)

    multi_controlled = Controlled(0, 1, basic_gate=RX(2, theta=0.3))
    multi_qubit_controlled = Controlled(0, basic_gate=DummyBasicGate((1, 2), np.eye(4)))
    exponential_multi = Exponential(DummyBasicGate((0, 1), np.eye(4)))

    with pytest.raises(NotImplementedError):
        pass_instance._rewrite_gate(multi_controlled)
    with pytest.raises(NotImplementedError):
        pass_instance._rewrite_gate(multi_qubit_controlled)
    with pytest.raises(NotImplementedError):
        pass_instance._rewrite_gate(exponential_multi)


def test_canonical_pass_supports_cnot_basis() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CNOT)

    cnot_rewritten = pass_instance._rewrite_gate(CNOT(0, 1))
    cz_rewritten = pass_instance._rewrite_gate(CZ(0, 1))
    swap_rewritten = pass_instance._rewrite_gate(SWAP(0, 1))

    assert all(gate.name in {"CNOT", "U3"} for gate in cnot_rewritten + cz_rewritten + swap_rewritten)
    _assert_gate_sequence_matches(CNOT(0, 1), cnot_rewritten)
    _assert_gate_sequence_matches(CZ(0, 1), cz_rewritten)
    _assert_gate_sequence_matches(SWAP(0, 1), swap_rewritten)


def test_canonical_pass_supports_rxryrz_single_qubit_basis() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(
        two_qubit_basis=TwoQubitGateBasis.CZ,
        single_qubit_basis=SingleQubitGateBasis.RxRyRz,
    )

    cnot_rewritten = pass_instance._rewrite_gate(CNOT(0, 1))
    swap_rewritten = pass_instance._rewrite_gate(SWAP(0, 1))

    assert all(gate.name in {"CZ", "RX", "RY", "RZ"} for gate in cnot_rewritten + swap_rewritten)
    _assert_gate_sequence_matches(CNOT(0, 1), cnot_rewritten)
    _assert_gate_sequence_matches(SWAP(0, 1), swap_rewritten)


def test_canonical_pass_rejects_invalid_two_qubit_basis() -> None:
    with pytest.raises(TypeError):
        DecomposeToCanonicalBasisPass(two_qubit_basis="CZ")  # type: ignore[arg-type]


def test_canonical_pass_rejects_invalid_single_qubit_basis() -> None:
    with pytest.raises(TypeError):
        DecomposeToCanonicalBasisPass(single_qubit_basis="U3")  # type: ignore[arg-type]


def test_canonical_pass_exposes_configured_basis_properties() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(
        two_qubit_basis=TwoQubitGateBasis.CZ,
        single_qubit_basis=SingleQubitGateBasis.RxRyRz,
    )

    assert pass_instance.single_qubit_basis == SingleQubitGateBasis.RxRyRz
    assert pass_instance.two_qubit_basis == TwoQubitGateBasis.CZ


@pytest.mark.parametrize(
    ("two_qubit_basis", "single_qubit_basis", "expected_gate_names", "first_gate"),
    [
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.U3, {"CZ", "U3"}, CNOT(0, 1)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.U3, {"CNOT", "U3"}, CZ(0, 1)),
        (TwoQubitGateBasis.CZ, SingleQubitGateBasis.RxRyRz, {"CZ", "RX", "RY", "RZ"}, CNOT(0, 1)),
        (TwoQubitGateBasis.CNOT, SingleQubitGateBasis.RxRyRz, {"CNOT", "RX", "RY", "RZ"}, CZ(0, 1)),
    ],
)
def test_canonical_pass_run_produces_basis_circuit_with_same_matrix(
    two_qubit_basis: TwoQubitGateBasis,
    single_qubit_basis: SingleQubitGateBasis,
    expected_gate_names: set[str],
    first_gate: Gate,
) -> None:
    circuit = Circuit(3)
    circuit.add(first_gate)
    circuit.add(SWAP(1, 2))
    circuit.add(Adjoint(RY(2, theta=0.5)))
    circuit.add(RX(0, theta=0.3))

    pass_instance = DecomposeToCanonicalBasisPass(
        two_qubit_basis=two_qubit_basis,
        single_qubit_basis=single_qubit_basis,
    )
    rewritten_circuit = pass_instance.run(circuit)

    assert all(gate.name in expected_gate_names for gate in rewritten_circuit.gates)
    assert len(rewritten_circuit.gates) > 0
    assert np.allclose(circuit.to_matrix(), rewritten_circuit.to_matrix())


def test_rewrite_gate_handles_single_qubit_exponential() -> None:
    anti_hermitian_gate = DummyBasicGate((0,), np.array([[0.0, 0.0], [0.0, 0.25j * math.pi]], dtype=complex))
    pass_instance = DecomposeToCanonicalBasisPass()

    rewritten = pass_instance._rewrite_gate(Exponential(anti_hermitian_gate))

    assert len(rewritten) == 1
    assert rewritten[0].name == "U3"
    _assert_gate_sequence_matches(Exponential(anti_hermitian_gate), rewritten)


@pytest.mark.parametrize(
    ("returned_gate", "expected_name", "expected_values"),
    [
        (U3(9, theta=0.1, phi=0.2, gamma=0.3), "U3", (0.1, 0.2, 0.3)),
        (RX(9, theta=0.4), "RX", (0.4,)),
        (RY(9, theta=0.5), "RY", (0.5,)),
        (RZ(9, phi=0.6), "RZ", (0.6,)),
    ],
)
def test_rewrite_gate_retargets_single_control_basis_gate_to_original_target(
    monkeypatch: pytest.MonkeyPatch,
    returned_gate: Gate,
    expected_name: str,
    expected_values: tuple[float, ...],
) -> None:
    captured: dict[str, Gate] = {}

    def fake_single_controlled(*, control_qubit, target_gate, basis, single_qubit_basis):  # type: ignore[no-untyped-def]
        captured["target_gate"] = target_gate
        return [target_gate]

    monkeypatch.setattr(decompose_module, "_as_basis_1q", lambda gate: returned_gate)
    monkeypatch.setattr(
        decompose_module,
        "_single_controlled",
        lambda control_qubit, target_gate, basis, single_qubit_basis: fake_single_controlled(
            control_qubit=control_qubit,
            target_gate=target_gate,
            basis=basis,
            single_qubit_basis=single_qubit_basis,
        ),
    )

    pass_instance = DecomposeToCanonicalBasisPass()
    result = pass_instance._rewrite_gate(Controlled(0, basic_gate=H(1)))

    assert result == [captured["target_gate"]]
    assert captured["target_gate"].name == expected_name
    assert captured["target_gate"].qubits == (1,)
    assert tuple(captured["target_gate"].get_parameter_values()) == expected_values


def test_rewrite_gate_rejects_unsupported_non_basic_gate() -> None:
    class UnsupportedGate(Gate):
        @property
        def name(self) -> str:
            return "Unsupported"

        @property
        def matrix(self) -> np.ndarray:
            return np.eye(2, dtype=complex)

        @property
        def target_qubits(self) -> tuple[int, ...]:
            return (0,)

    with pytest.raises(NotImplementedError, match="Unsupported"):
        DecomposeToCanonicalBasisPass()._rewrite_gate(UnsupportedGate())
