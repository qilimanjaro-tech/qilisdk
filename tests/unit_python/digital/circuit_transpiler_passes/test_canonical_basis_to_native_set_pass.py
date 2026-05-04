import contextlib
import math

import pytest

from qilisdk.digital import Circuit, Rmw
from qilisdk.digital.circuit_transpiler_passes import (
    CanonicalBasisToNativeSetPass,
    NativeSingleQubitGateBasis,
    TwoQubitGateBasis,
)
from qilisdk.digital.exceptions import GateHasNoMatrixError
from qilisdk.digital.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    U3,
    Gate,
    M,
    Z,
)


def assert_equal_gate(list1: list[Gate], list2: list[Gate]) -> None:
    assert len(list1) == len(list2)
    for a, b in zip(list1, list2):
        assert a.name == b.name
        assert a.qubits == b.qubits
        with contextlib.suppress(GateHasNoMatrixError):
            assert (a.matrix == b.matrix).all()


class TestCanonicalBasisToNativeSetPass:
    def test_run(self) -> None:
        inp = Circuit(3)
        inp._gates = [
            RZ(0, phi=0.5),
            RX(0, theta=1),
            RZ(1, phi=0.5),
            RZ(0, phi=0.5),
            RY(1, theta=1),
            U3(0, theta=0.5, phi=1, gamma=1.5),
            RZ(2, phi=0.5),
            RZ(1, phi=0.5),
            RZ(1, phi=0.5),
            RZ(0, phi=0.5),
            CZ(1, 2),
            M(0),
        ]
        out_regular = [
            RZ(0, phi=0.5),
            Rmw(0, theta=1, phase=0),
            RZ(1, phi=0.5),
            Rmw(1, theta=1, phase=math.pi / 2),
            RZ(0, phi=0.5),
            Rmw(0, theta=0.5, phase=-math.pi / 2 + 1),
            RZ(1, phi=1),
            RZ(2, phi=0.5),
            CZ(1, 2),
            M(0),
        ]
        assert_equal_gate(CanonicalBasisToNativeSetPass().run(inp).gates, out_regular)

        out_keep_virtual_rz = [
            Rmw(0, theta=1, phase=0),
            Rmw(1, theta=1, phase=math.pi / 2),
            Rmw(0, theta=0.5, phase=-math.pi / 2 + 1),
            CZ(1, 2),
            M(0),
        ]
        assert_equal_gate(CanonicalBasisToNativeSetPass(keep_virtual_rz=False).run(inp).gates, out_keep_virtual_rz)

        out_merge_consecutive_rz = [
            RZ(0, phi=0.5),
            Rmw(0, theta=1, phase=0),
            RZ(1, phi=0.5),
            RZ(0, phi=0.5),
            Rmw(1, theta=1, phase=math.pi / 2),
            Rmw(0, theta=0.5, phase=-math.pi / 2 + 1),
            RZ(0, phi=2.5),
            RZ(2, phi=0.5),
            RZ(1, phi=0.5),
            RZ(1, phi=0.5),
            RZ(0, phi=0.5),
            CZ(1, 2),
            M(0),
        ]
        assert_equal_gate(
            CanonicalBasisToNativeSetPass(merge_consecutive_rz=False).run(inp).gates, out_merge_consecutive_rz
        )

        out_drop_rz_before_measure = [
            RZ(0, phi=0.5),
            Rmw(0, theta=1, phase=0),
            RZ(1, phi=0.5),
            Rmw(1, theta=1, phase=math.pi / 2),
            RZ(0, phi=0.5),
            Rmw(0, theta=0.5, phase=-math.pi / 2 + 1),
            RZ(1, phi=1),
            RZ(2, phi=0.5),
            CZ(1, 2),
            RZ(0, phi=3),
            M(0),
        ]
        assert_equal_gate(
            CanonicalBasisToNativeSetPass(drop_rz_before_measure=False).run(inp).gates, out_drop_rz_before_measure
        )

    def test_edges(self) -> None:
        only_measurement = Circuit(1)
        only_measurement._gates = [M(0)]
        result = CanonicalBasisToNativeSetPass().run(only_measurement)
        # The pass must always return a new circuit, never mutate / alias the input.
        assert result is not only_measurement

        non_canonical_gate = Circuit(1)
        non_canonical_gate._gates = [Z(0)]
        with pytest.raises(NotImplementedError):
            CanonicalBasisToNativeSetPass().run(non_canonical_gate)

        empty = Circuit(1)
        assert CanonicalBasisToNativeSetPass().run(empty)._gates == []

    def test_cnot_two_qubit_basis_forwards_cnot(self) -> None:
        inp = Circuit(2)
        inp._gates = [RX(0, theta=0.3), CNOT(0, 1), M(0)]
        out = CanonicalBasisToNativeSetPass(two_qubit_basis=TwoQubitGateBasis.CNOT).run(inp)
        names = [g.name for g in out.gates]
        assert names == ["Rmw", "CNOT", "M"]

    def test_cnot_basis_rejects_cz_input(self) -> None:
        inp = Circuit(2)
        inp._gates = [CZ(0, 1)]
        with pytest.raises(NotImplementedError):
            CanonicalBasisToNativeSetPass(two_qubit_basis=TwoQubitGateBasis.CNOT).run(inp)

    def test_rejects_invalid_basis_arguments(self) -> None:
        with pytest.raises(TypeError):
            CanonicalBasisToNativeSetPass(single_qubit_basis="Rmw")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            CanonicalBasisToNativeSetPass(two_qubit_basis="CZ")  # type: ignore[arg-type]

    def test_exposes_configured_basis_properties(self) -> None:
        p = CanonicalBasisToNativeSetPass(
            single_qubit_basis=NativeSingleQubitGateBasis.Rmw,
            two_qubit_basis=TwoQubitGateBasis.CNOT,
        )
        assert p.single_qubit_basis == NativeSingleQubitGateBasis.Rmw
        assert p.two_qubit_basis == TwoQubitGateBasis.CNOT
