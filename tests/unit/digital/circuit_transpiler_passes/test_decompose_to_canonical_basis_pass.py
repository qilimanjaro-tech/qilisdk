import math
from collections import Counter
from typing import Iterable

import numpy as np
import pytest

from qilisdk.digital import Circuit
from qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass import (
    DecomposeToCanonicalBasisPass,
    SingleQubitGateBasis,
    TwoQubitGateBasis,
    _as_basis_1q,
    _invert_basis_gate,
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


def _count_gate_names(seq: list) -> Counter:
    return Counter(g.name for g in seq)


def test_invert_basis_gate_handles_known_types():
    gates = [
        U3(0, theta=0.2, phi=0.1, gamma=-0.3),
        RX(0, theta=0.4),
        RY(0, theta=-0.5),
        RZ(0, phi=0.7),
        CNOT(0, 1),
        CZ(0, 1),
        M(0),
        H(0),
        X(0),
        Y(0),
        Z(0),
    ]
    inverted = [_invert_basis_gate(g) for g in gates]
    assert [seq[0].name for seq in inverted[:6]] == ["U3", "RX", "RY", "RZ", "CNOT", "CZ"]
    assert inverted[6][0].name == "M"
    assert [seq[0].name for seq in inverted[7:]] == ["U3", "RX", "RY", "RZ"]

    dummy = DummyBasicGate((0,), np.eye(2))
    assert isinstance(_invert_basis_gate(dummy)[0], Adjoint)


def test_as_basis_1q_handles_standard_and_basic():
    dummy_matrix = np.eye(2, dtype=complex)
    dummy_gate = DummyBasicGate((0,), dummy_matrix)
    rx_gate = RX(0, theta=0.3)
    assert _as_basis_1q(rx_gate) is rx_gate
    assert isinstance(_as_basis_1q(H(0)), U3)
    assert isinstance(_as_basis_1q(X(0)), RX)
    assert isinstance(_as_basis_1q(Y(0)), RY)
    assert isinstance(_as_basis_1q(Z(0)), RZ)
    assert isinstance(_as_basis_1q(U1(0, phi=0.1)), RZ)
    assert isinstance(_as_basis_1q(U2(0, phi=0.1, gamma=0.2)), U3)
    assert isinstance(_as_basis_1q(dummy_gate), U3)
    with pytest.raises(NotImplementedError):
        _as_basis_1q(Controlled(1, basic_gate=RX(0, theta=0.5)))


def test_single_controlled_paths():
    rz_seq = _single_controlled(1, RZ(0, phi=0.2))
    rx_seq = _single_controlled(1, RX(0, theta=0.2))
    ry_seq = _single_controlled(1, RY(0, theta=0.2))
    u3_seq = _single_controlled(1, U3(0, theta=0.3, phi=0.2, gamma=-0.1))
    dummy_gate = DummyBasicGate((5,), np.eye(2))
    recursive_seq = _single_controlled(1, dummy_gate)

    assert _count_gate_names(rz_seq)["CZ"] >= 1
    assert _count_gate_names(rx_seq)["CZ"] >= 1
    assert _count_gate_names(ry_seq)["CZ"] >= 1
    assert _count_gate_names(u3_seq)["CZ"] >= 1
    assert _count_gate_names(recursive_seq)["CZ"] >= 1


def test_single_controlled_paths_cnot_basis():
    rz_seq = _single_controlled(1, RZ(0, phi=0.2), basis=TwoQubitGateBasis.CNOT)
    rx_seq = _single_controlled(1, RX(0, theta=0.2), basis=TwoQubitGateBasis.CNOT)
    ry_seq = _single_controlled(1, RY(0, theta=0.2), basis=TwoQubitGateBasis.CNOT)
    u3_seq = _single_controlled(1, U3(0, theta=0.3, phi=0.2, gamma=-0.1), basis=TwoQubitGateBasis.CNOT)
    dummy_gate = DummyBasicGate((5,), np.eye(2))
    recursive_seq = _single_controlled(1, dummy_gate, basis=TwoQubitGateBasis.CNOT)

    assert _count_gate_names(rz_seq)["CNOT"] >= 1
    assert _count_gate_names(rx_seq)["CNOT"] >= 1
    assert _count_gate_names(ry_seq)["CNOT"] >= 1
    assert _count_gate_names(u3_seq)["CNOT"] >= 1
    assert _count_gate_names(recursive_seq)["CNOT"] >= 1
    assert _count_gate_names(rz_seq)["CZ"] == 0


def test_rewrite_gate_covers_various_cases():
    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)

    gates = [
        M(0),
        U3(0, theta=0.1, phi=0.2, gamma=0.3),
        RX(0, theta=0.2),
        RY(0, theta=0.3),
        RZ(0, phi=0.4),
        CZ(0, 1),
        I(0),
        H(0),
        X(0),
        Y(0),
        Z(0),
        U1(0, phi=0.5),
        U2(0, phi=0.6, gamma=0.7),
        CNOT(0, 1),
        SWAP(0, 1),
    ]

    seqs = [pass_instance._rewrite_gate(g) for g in gates]

    assert len(seqs[0]) == 1
    assert seqs[0][0].name == "M"
    assert len(seqs[5]) == 1
    assert seqs[5][0].name == "CZ"
    assert seqs[6] == []
    assert len(seqs[7]) == 1
    assert seqs[7][0].name == "U3"
    assert len(seqs[8]) == 1
    assert seqs[8][0].name == "U3"
    assert len(seqs[9]) == 1
    assert seqs[9][0].name == "U3"
    assert len(seqs[10]) == 1
    assert seqs[10][0].name == "U3"
    assert len(seqs[11]) == 1
    assert seqs[11][0].name == "U3"
    assert len(seqs[12]) == 1
    assert seqs[12][0].name == "U3"
    assert _count_gate_names(seqs[13])["CZ"] == 1
    assert _count_gate_names(seqs[14])["CZ"] == 3


def test_rewrite_gate_returns_cz_directly():
    real_cz_cls = DecomposeToCanonicalBasisPass._rewrite_gate.__globals__["CZ"]

    class ProxyCZMeta(type):
        count = 0
        real_cls = real_cz_cls

        def __instancecheck__(cls, instance):
            cls.count += 1
            return cls.count >= 1 and isinstance(instance, cls.real_cls)

    class ProxyCZ(metaclass=ProxyCZMeta):
        pass

    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)
    gate = real_cz_cls(0, 1)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(DecomposeToCanonicalBasisPass._rewrite_gate.__globals__, "CZ", ProxyCZ)
    try:
        rewritten = pass_instance._rewrite_gate(gate)
    finally:
        monkeypatch.undo()

    assert len(rewritten) == 1
    assert isinstance(rewritten[0], real_cz_cls)


@pytest.mark.parametrize(
    ("factory", "expected_cls"),
    [
        (lambda q: U3(q, theta=0.1, phi=0.2, gamma=0.3), U3),
        (lambda q: RX(q, theta=0.5), RX),
        (lambda q: RY(q, theta=-0.4), RY),
        (lambda q: RZ(q, phi=0.7), RZ),
    ],
)
def test_controlled_gate_qubit_alignment(monkeypatch, factory, expected_cls):
    pass_instance = DecomposeToCanonicalBasisPass()
    base_gate = DummyBasicGate((3,), np.eye(2))
    controlled = Controlled(0, basic_gate=base_gate)

    original_as_basis = DecomposeToCanonicalBasisPass._rewrite_gate.__globals__["_as_basis_1q"]
    captured = {}

    def fake_as_basis(g):
        if g is base_gate:
            return factory(99)  # wrong qubit on purpose
        return original_as_basis(g)

    monkeypatch.setattr(
        "qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass._as_basis_1q",
        fake_as_basis,
    )

    def fake_single(ctrl, base_1q, _basis, _single_basis):
        captured["base"] = base_1q
        return ["sentinel"]

    monkeypatch.setattr(
        "qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass._single_controlled",
        fake_single,
    )

    seq = pass_instance._rewrite_gate(controlled)
    assert seq == ["sentinel"]
    assert isinstance(captured["base"], expected_cls)
    assert captured["base"].qubits[0] == base_gate.qubits[0]


def test_rewrite_gate_handles_controlled_and_adjoint(monkeypatch):
    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)

    base_gate = DummyBasicGate((2,), np.eye(2))

    def fake_as_basis(gate):
        if gate is base_gate:
            return U3(5, theta=0.1, phi=0.2, gamma=0.3)
        return _as_basis_1q(gate)

    monkeypatch.setattr(
        "qilisdk.digital.circuit_transpiler_passes.decompose_to_canonical_basis_pass._as_basis_1q",
        fake_as_basis,
    )

    controlled = Controlled(0, basic_gate=base_gate)
    adjoint_gate = Adjoint(RX(0, theta=math.pi / 3))
    expo_gate = RX(0, theta=math.pi / 4).exponential()
    basic_gate = DummyBasicGate((0,), np.eye(2))

    ctrl_seq = pass_instance._rewrite_gate(controlled)
    adj_seq = pass_instance._rewrite_gate(adjoint_gate)
    exp_seq = pass_instance._rewrite_gate(expo_gate)
    basic_seq = pass_instance._rewrite_gate(basic_gate)

    assert _count_gate_names(ctrl_seq)["CZ"] >= 1
    assert _count_gate_names(adj_seq)["U3"] >= 1
    assert len(exp_seq) == 1
    assert exp_seq[0].name == "U3"
    assert len(basic_seq) == 1
    assert basic_seq[0].name == "U3"
    assert all(5 not in g.qubits for g in ctrl_seq)

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

    assert len(cnot_rewritten) == 1
    assert cnot_rewritten[0].name == "CNOT"
    assert _count_gate_names(cz_rewritten)["CNOT"] == 1
    assert _count_gate_names(cz_rewritten)["CZ"] == 0
    assert _count_gate_names(swap_rewritten)["CNOT"] == 3
    assert _count_gate_names(swap_rewritten)["CZ"] == 0


def test_canonical_pass_supports_rxryrz_single_qubit_basis() -> None:
    pass_instance = DecomposeToCanonicalBasisPass(
        two_qubit_basis=TwoQubitGateBasis.CZ,
        single_qubit_basis=SingleQubitGateBasis.RxRyRz,
    )

    h_rewritten = pass_instance._rewrite_gate(H(0))
    x_rewritten = pass_instance._rewrite_gate(X(0))
    u2_rewritten = pass_instance._rewrite_gate(U2(0, phi=0.2, gamma=-0.1))

    assert all(g.name in {"RX", "RY", "RZ"} for g in h_rewritten)
    assert all(g.name in {"RX", "RY", "RZ"} for g in x_rewritten)
    assert all(g.name in {"RX", "RY", "RZ"} for g in u2_rewritten)
    assert all(g.name != "U3" for g in h_rewritten + x_rewritten + u2_rewritten)


def test_canonical_pass_rejects_invalid_two_qubit_basis() -> None:
    with pytest.raises(TypeError):
        DecomposeToCanonicalBasisPass(two_qubit_basis="CZ")  # type: ignore[arg-type]


def test_canonical_pass_rejects_invalid_single_qubit_basis() -> None:
    with pytest.raises(TypeError):
        DecomposeToCanonicalBasisPass(single_qubit_basis="U3")  # type: ignore[arg-type]


def test_canonical_pass_run_produces_basis_circuit_for_CZ():
    circuit = Circuit(3)
    circuit.add(CNOT(0, 1))
    circuit.add(SWAP(1, 2))
    circuit.add(Adjoint(RY(2, theta=0.5)))
    circuit.add(RX(0, theta=0.3))

    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CZ)
    out = pass_instance.run(circuit)

    assert all(g.name in {"CZ", "U3"} for g in out.gates)
    assert len(out.gates) > 0
    assert np.allclose(circuit.to_matrix(), out.to_matrix())


def test_canonical_pass_run_produces_basis_circuit_for_cnot() -> None:
    circuit = Circuit(3)
    circuit.add(CZ(0, 1))
    circuit.add(SWAP(1, 2))
    circuit.add(Adjoint(RY(2, theta=0.5)))
    circuit.add(RX(0, theta=0.3))

    pass_instance = DecomposeToCanonicalBasisPass(two_qubit_basis=TwoQubitGateBasis.CNOT)
    out = pass_instance.run(circuit)

    assert all(g.name in {"CNOT", "U3"} for g in out.gates)
    assert len(out.gates) > 0
    assert np.allclose(circuit.to_matrix(), out.to_matrix())


def test_canonical_pass_run_produces_basis_circuit_for_rxryrz() -> None:
    circuit = Circuit(3)
    circuit.add(CNOT(0, 1))
    circuit.add(SWAP(1, 2))
    circuit.add(Adjoint(RY(2, theta=0.5)))
    circuit.add(RX(0, theta=0.3))

    pass_instance = DecomposeToCanonicalBasisPass(single_qubit_basis=SingleQubitGateBasis.RxRyRz)
    out = pass_instance.run(circuit)

    assert all(g.name in {"CNOT", "RX", "RY", "RZ"} for g in out.gates)
    assert len(out.gates) > 0
    assert np.allclose(circuit.to_matrix(), out.to_matrix())


def test_canonical_pass_run_produces_basis_circuit_for_cnot_and_rxryrz() -> None:
    circuit = Circuit(3)
    circuit.add(CZ(0, 1))
    circuit.add(SWAP(1, 2))
    circuit.add(Adjoint(RY(2, theta=0.5)))
    circuit.add(RX(0, theta=0.3))

    pass_instance = DecomposeToCanonicalBasisPass(
        two_qubit_basis=TwoQubitGateBasis.CNOT,
        single_qubit_basis=SingleQubitGateBasis.RxRyRz,
    )
    out = pass_instance.run(circuit)

    assert all(g.name in {"CNOT", "RX", "RY", "RZ"} for g in out.gates)
    assert len(out.gates) > 0
    assert np.allclose(circuit.to_matrix(), out.to_matrix())
