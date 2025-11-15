import math

import pytest

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.circuit_transpiler_passes import CanonicalBasis, CircuitToCanonicalBasisPass
from qilisdk.digital.gates import CNOT, CZ, H, M, RX, RY, RZ, S, T, U3


def test_zx_basis_produces_rx_rz_and_cnot() -> None:
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CNOT(0, 1))
    circuit.add(RZ(1, phi=math.pi / 3))

    transpiled = CircuitToCanonicalBasisPass(CanonicalBasis.ZX_EULER).run(circuit)

    assert any(isinstance(g, CNOT) for g in transpiled.gates)
    assert not any(isinstance(g, CZ) for g in transpiled.gates)
    assert all(isinstance(g, (RX, RZ, CNOT, M)) for g in transpiled.gates)


def test_xyz_basis_recovers_ry_patterns() -> None:
    circuit = Circuit(1)
    circuit.add(RY(0, theta=0.7))

    transpiled = CircuitToCanonicalBasisPass(CanonicalBasis.XYZ_ROTATIONS).run(circuit)

    assert any(isinstance(g, RY) for g in transpiled.gates)
    assert all(isinstance(g, (RX, RY, RZ, CNOT, M)) for g in transpiled.gates)


def test_u3_basis_only_emits_u3_and_cnot() -> None:
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(CZ(0, 1))

    transpiled = CircuitToCanonicalBasisPass(CanonicalBasis.U3_CX).run(circuit)

    assert any(isinstance(g, CNOT) for g in transpiled.gates)
    assert all(isinstance(g, (U3, CNOT, M)) for g in transpiled.gates)


def test_clifford_t_basis_uses_expected_gates() -> None:
    circuit = Circuit(1)
    circuit.add(RZ(0, phi=math.pi / 2))
    circuit.add(RX(0, theta=math.pi / 4))

    transpiled = CircuitToCanonicalBasisPass(CanonicalBasis.CLIFFORD_T).run(circuit)

    assert all(isinstance(g, (H, S, T, CNOT, M)) for g in transpiled.gates)


def test_clifford_t_basis_rejects_generic_rotations() -> None:
    circuit = Circuit(1)
    circuit.add(RZ(0, phi=0.1))

    with pytest.raises(ValueError):
        CircuitToCanonicalBasisPass(CanonicalBasis.CLIFFORD_T).run(circuit)
