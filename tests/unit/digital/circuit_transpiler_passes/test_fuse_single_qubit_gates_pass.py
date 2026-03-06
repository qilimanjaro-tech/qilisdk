import numpy as np
import pytest

from qilisdk.digital import Circuit
from qilisdk.digital.circuit_transpiler_passes import SingleQubitGateBasis
from qilisdk.digital.circuit_transpiler_passes.fuse_single_qubit_gates_pass import FuseSingleQubitGatesPass
from qilisdk.digital.gates import CZ, RX, RY, RZ, U3, Exponential, Gate, H, M, S, T, X

from .utils import _sequences_equivalent


def _basis_gate_names(single_qubit_basis: SingleQubitGateBasis) -> set[str]:
    return {"U3"} if single_qubit_basis == SingleQubitGateBasis.U3 else {"RX", "RY", "RZ"}


def _describe_gate(gate: Gate) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    return (gate.name, gate.qubits, tuple(gate.get_parameter_values()))


def _describe_circuit(circuit: Circuit) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
    return [_describe_gate(gate) for gate in circuit.gates]


def test_run_respects_u3_basis_and_does_not_mutate_input() -> None:
    circuit = Circuit(2)
    circuit._gates = [
        RX(0, theta=2),
        RZ(1, phi=1),
        CZ(1, 0),
        RY(0, theta=2),
        U3(1, theta=2, phi=-np.pi / 2, gamma=np.pi / 2),
        M(1),
        RY(0, theta=2),
        RZ(1, phi=1),
    ]
    original_snapshot = _describe_circuit(circuit)

    transpiled = FuseSingleQubitGatesPass(single_qubit_basis=SingleQubitGateBasis.U3).run(circuit)

    assert [(gate.name, gate.qubits) for gate in transpiled.gates] == [
        ("U3", (1,)),
        ("U3", (0,)),
        ("CZ", (1, 0)),
        ("U3", (1,)),
        ("M", (1,)),
        ("U3", (0,)),
        ("U3", (1,)),
    ]
    assert all(gate.name in {"U3", "CZ", "M"} for gate in transpiled.gates)
    assert _describe_circuit(circuit) == original_snapshot


def test_run_respects_rxryrz_basis() -> None:
    circuit = Circuit(2)
    circuit._gates = [
        RX(0, theta=2),
        RZ(1, phi=1),
        CZ(1, 0),
        RY(0, theta=2),
        U3(1, theta=2, phi=-np.pi / 2, gamma=np.pi / 2),
        M(1),
        RY(0, theta=2),
        RZ(1, phi=1),
    ]

    transpiled = FuseSingleQubitGatesPass(single_qubit_basis=SingleQubitGateBasis.RxRyRz).run(circuit)

    assert [(gate.name, gate.qubits) for gate in transpiled.gates] == [
        ("RZ", (1,)),
        ("RX", (0,)),
        ("CZ", (1, 0)),
        ("RX", (1,)),
        ("M", (1,)),
        ("RY", (0,)),
        ("RZ", (1,)),
    ]
    assert all(gate.name in {"RX", "RY", "RZ", "CZ", "M"} for gate in transpiled.gates)


@pytest.mark.parametrize(
    ("single_qubit_basis", "expected_names"),
    [
        (SingleQubitGateBasis.U3, ["U3", "U3"]),
        (SingleQubitGateBasis.RxRyRz, ["RY", "RX"]),
    ],
)
def test_fusion_respects_requested_basis_for_axis_aligned_cases(
    single_qubit_basis: SingleQubitGateBasis,
    expected_names: list[str],
) -> None:
    circuit = Circuit(2)
    circuit._gates = [
        U3(0, theta=2, phi=np.pi, gamma=np.pi),
        U3(1, theta=2, phi=np.pi / 2, gamma=-np.pi / 2),
    ]

    transpiled = FuseSingleQubitGatesPass(single_qubit_basis=single_qubit_basis).run(circuit)

    assert [gate.name for gate in transpiled.gates] == expected_names
    assert _sequences_equivalent(circuit.gates, transpiled.gates, circuit.nqubits)


@pytest.mark.parametrize("single_qubit_basis", list(SingleQubitGateBasis))
def test_fuses_any_single_qubit_unitary_sequence_in_requested_basis(
    single_qubit_basis: SingleQubitGateBasis,
) -> None:
    circuit = Circuit(1)
    circuit.add(H(0))
    circuit.add(T(0))
    circuit.add(X(0))

    transpiled = FuseSingleQubitGatesPass(single_qubit_basis=single_qubit_basis).run(circuit)

    assert all(gate.name in _basis_gate_names(single_qubit_basis) for gate in transpiled.gates)
    assert _sequences_equivalent(circuit.gates, transpiled.gates, 1)


@pytest.mark.parametrize("single_qubit_basis", list(SingleQubitGateBasis))
def test_fusion_stops_at_multiqubit_gate_boundaries_in_selected_basis(
    single_qubit_basis: SingleQubitGateBasis,
) -> None:
    circuit = Circuit(2)
    circuit.add(H(0))
    circuit.add(T(0))
    circuit.add(CZ(0, 1))
    circuit.add(S(0))
    circuit.add(X(0))

    transpiled = FuseSingleQubitGatesPass(single_qubit_basis=single_qubit_basis).run(circuit)
    allowed_gate_names = _basis_gate_names(single_qubit_basis) | {"CZ"}

    assert any(isinstance(gate, CZ) for gate in transpiled.gates)
    assert all(gate.name in allowed_gate_names for gate in transpiled.gates)
    assert _sequences_equivalent(circuit.gates, transpiled.gates, 2)


def test_fusion_keeps_non_unitary_boundaries_untouched() -> None:
    circuit = Circuit(3)
    circuit._gates = [
        U3(2, theta=1, phi=np.pi, gamma=-np.pi),
        RZ(1, phi=1),
        Exponential(RY(0, theta=2)),
        RY(2, theta=1),
    ]

    transpiled = FuseSingleQubitGatesPass(single_qubit_basis=SingleQubitGateBasis.U3).run(circuit)

    assert [gate.name for gate in transpiled.gates] == ["e^RY", "U3", "U3"]
    assert isinstance(transpiled.gates[0], Exponential)


def test_fuse_pass_rejects_invalid_single_qubit_basis() -> None:
    with pytest.raises(TypeError):
        FuseSingleQubitGatesPass(single_qubit_basis="U3")  # type: ignore[arg-type]
