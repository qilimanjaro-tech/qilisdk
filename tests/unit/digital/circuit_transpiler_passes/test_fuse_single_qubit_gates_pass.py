import contextlib

import numpy as np
import pytest

from qilisdk.digital import Circuit
from qilisdk.digital.circuit_transpiler_passes.fuse_single_qubit_gates_pass import FuseSingleQubitGatesPass
from qilisdk.digital.exceptions import GateHasNoMatrixError
from qilisdk.digital.gates import CZ, RX, RY, RZ, SWAP, U3, Exponential, Gate, H, M, S, T, X

from .utils import _sequences_equivalent


def assert_equal_gate(list1: list[Gate], list2: list[Gate]):
    assert len(list1) == len(list2)
    for i, j in zip(list1, list2):
        assert i.qubits == j.qubits
        assert i.name == j.name
        with contextlib.suppress(GateHasNoMatrixError):
            assert i.matrix == pytest.approx(j.matrix, abs=1e-10)


def test_run() -> None:
    # Base test to check an expected case
    test_1 = Circuit(2)
    test_1._gates = [
        RX(0, theta=2),
        RZ(1, phi=1),
        CZ(1, 0),
        RY(0, theta=2),
        U3(1, theta=2, phi=-np.pi / 2, gamma=np.pi / 2),
        M(1),
        RY(0, theta=2),
        RZ(1, phi=1),
    ]
    assert_equal_gate(
        FuseSingleQubitGatesPass().run(test_1)._gates,
        [RZ(1, phi=1), RX(0, theta=2), CZ(1, 0), RX(1, theta=2), M(1), RY(0, theta=4 - np.pi * 2), RZ(1, phi=1)],
    )
    # Checks that the original circuit hasn't been changed
    assert_equal_gate(
        test_1._gates,
        [
            RX(0, theta=2),
            RZ(1, phi=1),
            CZ(1, 0),
            RY(0, theta=2),
            U3(1, theta=2, phi=-np.pi / 2, gamma=np.pi / 2),
            M(1),
            RY(0, theta=2),
            RZ(1, phi=1),
        ],
    )
    # Checks for the order of output gates, exeption handeling and simplification of gates under certain conditions.
    # If this one fails after changeing the code, it might mean that the handeling of edge-cases is diffenent, not necesarely thet the new code is incorrect.
    test_2 = Circuit(3)
    test_2._gates = [
        U3(2, theta=1, phi=np.pi, gamma=-np.pi),
        RZ(1, phi=1),
        SWAP(0, 1),
        RZ(0, phi=1),
        RY(2, theta=1),
        Exponential(RY(0, theta=2)),
        RZ(0, phi=2),
    ]
    assert_equal_gate(
        FuseSingleQubitGatesPass().run(test_2)._gates,
        [RZ(1, phi=1), SWAP(0, 1), RZ(0, phi=1), Exponential(RY(0, theta=2)), RZ(2, phi=0), RZ(0, phi=2)],
    )
    # It makes sure all the inverse equivalences work
    test_3 = Circuit(2)
    test_3._gates = [U3(0, theta=2, phi=np.pi, gamma=np.pi), U3(1, theta=2, phi=np.pi / 2, gamma=-np.pi / 2)]
    assert_equal_gate(FuseSingleQubitGatesPass().run(test_3)._gates, [RY(0, theta=-2), RX(1, theta=-2)])


def test_fuses_any_single_qubit_unitary_sequence() -> None:
    circuit = Circuit(1)
    circuit.add(H(0))
    circuit.add(T(0))
    circuit.add(X(0))

    transpiled = FuseSingleQubitGatesPass().run(circuit)

    assert len(transpiled.gates) == 1
    assert transpiled.gates[0].nqubits == 1
    assert _sequences_equivalent(circuit.gates, transpiled.gates, 1)


def test_fusion_stops_at_multiqubit_gate_boundaries() -> None:
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(T(0))
    circuit.add(CZ(0, 1))
    circuit.add(S(0))
    circuit.add(X(0))

    transpiled = FuseSingleQubitGatesPass().run(circuit)

    assert len(transpiled.gates) == 3
    assert isinstance(transpiled.gates[1], CZ)
    assert transpiled.gates[0].nqubits == 1
    assert transpiled.gates[2].nqubits == 1
    assert _sequences_equivalent(circuit.gates, transpiled.gates, 2)
