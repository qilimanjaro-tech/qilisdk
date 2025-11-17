import math

import pytest

from qilisdk.digital import RX, RY, RZ, U1, U2, U3, Circuit, H, S, T, X, Y, Z
from qilisdk.digital.circuit_transpiler_passes import DecomposeMultiControlledGatesPass
from qilisdk.digital.gates import Controlled, Gate

from .utils import _sequences_equivalent


def _run_pass_with_gate(gate: Gate, nqubits: int) -> Circuit:
    circuit = Circuit(nqubits)
    circuit.add(gate)
    return DecomposeMultiControlledGatesPass().run(circuit)


GATE_FACTORIES = [
    ("X", X),
    ("Y", Y),
    ("Z", Z),
    ("H", H),
    ("S", S),
    ("T", T),
    ("RX", lambda q: RX(q, theta=math.pi / 3.0)),
    ("RY", lambda q: RY(q, theta=math.pi / 4.0)),
    ("RZ", lambda q: RZ(q, phi=math.pi / 5.0)),
    ("U1", lambda q: U1(q, phi=math.pi / 7.0)),
    ("U2", lambda q: U2(q, phi=math.pi / 6.0, gamma=math.pi / 5.0)),
    ("U3", lambda q: U3(q, theta=math.pi / 3.0, phi=math.pi / 4.0, gamma=math.pi / 5.0)),
]

CONTROL_COUNTS = [2, 3, 4]


def _build_controlled_gate(factory, ncontrols: int) -> Controlled:
    controls = tuple(range(ncontrols))
    target = ncontrols
    base_gate = factory(target)
    return Controlled(*controls, basic_gate=base_gate)


def _basis_states_for_controls(ncontrols: int) -> list[tuple[int, ...]]:
    target = ncontrols
    nqubits = ncontrols + 1
    controls = list(range(ncontrols))

    def build(bits_map: dict[int, int]) -> tuple[int, ...]:
        bits = [0] * nqubits
        for idx, bit in bits_map.items():
            bits[idx] = bit
        return tuple(bits)

    states = []
    states.append(build({}))  # all zeros
    partial = dict.fromkeys(controls[1:], 1)  # leave first control zero
    states.append(build(partial))
    all_controls = dict.fromkeys(controls, 1)
    states.extend((build({**all_controls, target: 0}), build({**all_controls, target: 1})))
    return states


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
@pytest.mark.parametrize("ncontrols", CONTROL_COUNTS)
def test_multi_controlled_gates_match_original_unitary(factory_name: str, factory, ncontrols: int) -> None:
    gate = _build_controlled_gate(factory, ncontrols)
    nqubits = ncontrols + 1
    transpiled = _run_pass_with_gate(gate, nqubits)

    states = _basis_states_for_controls(ncontrols)
    assert _sequences_equivalent([gate], transpiled.gates, nqubits, states), f"Vector equality for {factory_name} with {ncontrols} controls"
    assert _sequences_equivalent([gate], transpiled.gates, nqubits, None), f"Unitary equality for {factory_name} with {ncontrols} controls"

    for rewritten in transpiled.gates:
        if isinstance(rewritten, Controlled):
            assert len(rewritten.control_qubits) <= 1


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_single_control_gate_is_not_modified(factory_name: str, factory) -> None:
    gate = _build_controlled_gate(factory, 1)
    transpiled = _run_pass_with_gate(gate, 2)

    assert len(transpiled.gates) == 1
    rewritten = transpiled.gates[0]
    assert isinstance(rewritten, Controlled)
    assert rewritten.control_qubits == gate.control_qubits
    assert rewritten.basic_gate.name == gate.basic_gate.name


def test_other_gates_remain_unchanged() -> None:
    circuit = Circuit(3)
    circuit.add(RZ(0, phi=math.pi / 7.0))
    circuit.add(Controlled(0, 1, basic_gate=X(2)))
    circuit.add(RY(2, theta=math.pi / 9.0))

    transpiled = DecomposeMultiControlledGatesPass().run(circuit)
    assert isinstance(transpiled.gates[0], RZ)
    assert isinstance(transpiled.gates[-1], RY)
