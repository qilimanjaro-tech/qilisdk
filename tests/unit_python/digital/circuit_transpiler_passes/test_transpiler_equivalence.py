# Copyright 2026 Qilimanjaro Quantum Tech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Randomized semantic-equivalence checks for every transpiler pass and pipeline.

Each test builds seeded random circuits over the full gate set and asserts that the transpiled circuit implements
the same unitary up to a single global phase, and that basis-targeting passes only emit gates from their basis.
Routed circuits are compared by embedding a random logical state with the initial layout and reading it back with
the final layout.
"""

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qilisdk.digital import Circuit
from qilisdk.digital.circuit_transpiler import CircuitTranspiler
from qilisdk.digital.circuit_transpiler_passes import (
    CancelIdentityPairsPass,
    CustomLayoutPass,
    DecomposeMultiControlledGatesPass,
    DecomposeToCanonicalBasisPass,
    FuseSingleQubitGatesPass,
    SabreLayoutPass,
    SabreSwapPass,
    SingleQubitGateBasis,
    TwoQubitGateBasis,
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
    Gate,
    H,
    I,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.topology import build_topology_graph

from .utils import _apply_sequence_to_state, _unitaries_equivalent, _vectors_equal_up_to_phase

if TYPE_CHECKING:
    from collections.abc import Callable

SEEDS = range(8)
ROUTED_SEEDS = range(4)
NQUBITS = 4
DEPTH = 24
SINGLE_QUBIT_BASES = list(SingleQubitGateBasis)
TWO_QUBIT_BASES = list(TwoQubitGateBasis)
BASIS_COMBOS = list(itertools.product(SINGLE_QUBIT_BASES, TWO_QUBIT_BASES))
BASIS_GATE_NAMES = {SingleQubitGateBasis.U3: {"U3"}, SingleQubitGateBasis.RxRyRz: {"RX", "RY", "RZ"}}
LINE_TOPOLOGY = [(0, 1), (1, 2), (2, 3), (3, 4)]


def _angle(rng: np.random.Generator) -> float:
    # Mix generic angles with special ones, since special angles hit the exact-match shortcuts in the passes.
    if rng.random() < 0.3:
        return float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2, math.pi / 4]))
    return float(rng.uniform(-2 * math.pi, 2 * math.pi))


def _random_basic_gate(rng: np.random.Generator, qubit: int) -> BasicGate:
    factories: list[Callable[[], BasicGate]] = [
        lambda: I(qubit),
        lambda: X(qubit),
        lambda: Y(qubit),
        lambda: Z(qubit),
        lambda: H(qubit),
        lambda: S(qubit),
        lambda: T(qubit),
        lambda: RX(qubit, theta=_angle(rng)),
        lambda: RY(qubit, theta=_angle(rng)),
        lambda: RZ(qubit, phi=_angle(rng)),
        lambda: U1(qubit, phi=_angle(rng)),
        lambda: U2(qubit, phi=_angle(rng), gamma=_angle(rng)),
        lambda: U3(qubit, theta=_angle(rng), phi=_angle(rng), gamma=_angle(rng)),
    ]
    return factories[rng.integers(len(factories))]()


def _random_circuit(seed: int, nqubits: int = NQUBITS, depth: int = DEPTH, *, max_controls: int = 2) -> Circuit:
    rng = np.random.default_rng(seed)
    circuit = Circuit(nqubits)
    for _ in range(depth):
        kind = rng.random()
        qubits = [int(q) for q in rng.permutation(nqubits)]
        if kind < 0.5 or nqubits == 1:
            gate = _random_basic_gate(rng, qubits[0])
            circuit.add(Adjoint(gate) if rng.random() < 0.15 else gate)
        elif kind < 0.75:
            two_qubit = [CNOT, CZ, SWAP][rng.integers(3)]
            circuit.add(two_qubit(qubits[0], qubits[1]))
        else:
            ncontrols = int(rng.integers(1, min(max_controls, nqubits - 1) + 1))
            target = _random_basic_gate(rng, qubits[0])
            circuit.add(Controlled(*qubits[1 : 1 + ncontrols], basic_gate=target))
    return circuit


def _circuit_unitary(gates: list[Gate], nqubits: int) -> np.ndarray:
    # Column-by-column state application; much faster than utils._sequence_matrix's per-gate dense expansion.
    columns = [_apply_sequence_to_state(gates, column.astype(complex), nqubits) for column in np.eye(1 << nqubits)]
    return np.stack(columns, axis=1)


def _assert_same_unitary(original: Circuit, transpiled: Circuit) -> None:
    nqubits = max(original.nqubits, transpiled.nqubits)
    expected = _circuit_unitary(original.gates, nqubits)
    actual = _circuit_unitary(transpiled.gates, nqubits)
    assert _unitaries_equivalent(expected, actual), "transpiled circuit does not implement the original unitary"


def _assert_routed_equivalent(
    original: Circuit, routed: Circuit, initial_layout: list[int], final_layout: dict[int, int], seed: int
) -> None:
    nlogical = original.nqubits
    nphysical = routed.nqubits
    assert len(initial_layout) >= nlogical
    assert sorted(final_layout) == list(range(nlogical))
    assert len(set(final_layout.values())) == nlogical

    rng = np.random.default_rng(seed)
    for _ in range(3):
        logical_state = rng.normal(size=1 << nlogical) + 1j * rng.normal(size=1 << nlogical)
        logical_state /= np.linalg.norm(logical_state)
        expected = _apply_sequence_to_state(original.gates, logical_state.copy(), nlogical)

        # Logical qubit i starts on physical initial_layout[i]; every other physical qubit starts in |0>.
        embedded = np.zeros([2] * nphysical, dtype=complex)
        index = [0] * nphysical
        for basis_index, amplitude in enumerate(logical_state):
            for logical in range(nlogical):
                index[initial_layout[logical]] = (basis_index >> (nlogical - 1 - logical)) & 1
            embedded[tuple(index)] = amplitude

        output = _apply_sequence_to_state(routed.gates, embedded.reshape(-1), nphysical).reshape([2] * nphysical)
        logical_axes = [final_layout[logical] for logical in range(nlogical)]
        ancilla_axes = [axis for axis in range(nphysical) if axis not in logical_axes]
        extracted = output.transpose(logical_axes + ancilla_axes).reshape(1 << nlogical, -1)

        assert np.allclose(extracted[:, 1:], 0.0, atol=1e-9), "routing leaked amplitude into ancilla qubits"
        assert _vectors_equal_up_to_phase(expected, extracted[:, 0]), "routed circuit does not match the original"


def _assert_only_basis_gates(circuit: Circuit, single: SingleQubitGateBasis, two: TwoQubitGateBasis) -> None:
    allowed = BASIS_GATE_NAMES[single] | {two.value}
    names = {gate.name for gate in circuit.gates}
    assert names <= allowed, f"non-basis gates emitted: {sorted(names - allowed)}"


def _assert_adjacent_on_topology(circuit: Circuit, topology: list[tuple[int, int]]) -> None:
    edges = {frozenset(edge) for edge in topology}
    for gate in circuit.gates:
        if gate.nqubits == 2:
            assert frozenset(gate.qubits) in edges, f"{gate.name} on {gate.qubits} is not a topology edge"


def _canonical_circuit(seed: int) -> Circuit:
    return DecomposeToCanonicalBasisPass().run(DecomposeMultiControlledGatesPass().run(_random_circuit(seed)))


@pytest.mark.parametrize("seed", SEEDS)
def test_decompose_multi_controlled_pass_preserves_unitary(seed: int) -> None:
    circuit = _random_circuit(seed, max_controls=3)

    out = DecomposeMultiControlledGatesPass().run(circuit)

    assert all(not isinstance(gate, Controlled) or len(gate.control_qubits) <= 1 for gate in out.gates)
    _assert_same_unitary(circuit, out)


@pytest.mark.parametrize("seed", SEEDS)
def test_cancel_identity_pairs_pass_preserves_unitary(seed: int) -> None:
    circuit = _random_circuit(seed)
    # Mirror a random block so the pass has real cancellations to make.
    for gate in reversed(_random_circuit(seed + 1000, depth=6, max_controls=1).gates):
        circuit.add(gate)

    _assert_same_unitary(circuit, CancelIdentityPairsPass().run(circuit))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("single", "two"), BASIS_COMBOS)
def test_decompose_to_canonical_basis_pass_preserves_unitary(
    seed: int, single: SingleQubitGateBasis, two: TwoQubitGateBasis
) -> None:
    circuit = DecomposeMultiControlledGatesPass().run(_random_circuit(seed))

    out = DecomposeToCanonicalBasisPass(single_qubit_basis=single, two_qubit_basis=two).run(circuit)

    _assert_only_basis_gates(out, single, two)
    _assert_same_unitary(circuit, out)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("single", SINGLE_QUBIT_BASES)
def test_fuse_single_qubit_gates_pass_preserves_unitary(seed: int, single: SingleQubitGateBasis) -> None:
    circuit = DecomposeMultiControlledGatesPass().run(_random_circuit(seed))

    _assert_same_unitary(circuit, FuseSingleQubitGatesPass(single_qubit_basis=single).run(circuit))


@pytest.mark.parametrize("seed", SEEDS)
def test_sabre_layout_pass_applied_layout_preserves_semantics(seed: int) -> None:
    circuit = _canonical_circuit(seed)
    topology = build_topology_graph(LINE_TOPOLOGY)
    layout_pass = SabreLayoutPass(topology, seed=seed, apply_layout=True)

    out = layout_pass.run(circuit)

    layout = layout_pass.last_layout
    assert layout is not None
    _assert_routed_equivalent(circuit, out, layout, dict(enumerate(layout[: circuit.nqubits])), seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_sabre_swap_pass_preserves_semantics(seed: int) -> None:
    circuit = _canonical_circuit(seed)
    initial_layout = [int(q) for q in np.random.default_rng(seed).permutation(5)[: circuit.nqubits]]
    swap_pass = SabreSwapPass(build_topology_graph(LINE_TOPOLOGY), initial_layout=initial_layout, seed=seed)

    out = swap_pass.run(circuit)

    final_layout = swap_pass.last_final_layout
    assert final_layout is not None
    _assert_adjacent_on_topology(out, LINE_TOPOLOGY)
    _assert_routed_equivalent(circuit, out, initial_layout, dict(enumerate(final_layout[: circuit.nqubits])), seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_custom_layout_pass_preserves_semantics(seed: int) -> None:
    circuit = _canonical_circuit(seed)
    mapping = {logical: int(q) for logical, q in enumerate(np.random.default_rng(seed).permutation(5)[:NQUBITS])}

    out = CustomLayoutPass(build_topology_graph(LINE_TOPOLOGY), mapping).run(circuit)

    _assert_adjacent_on_topology(out, LINE_TOPOLOGY)
    _assert_routed_equivalent(circuit, out, [mapping[q] for q in range(NQUBITS)], mapping, seed)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("single", "two"), BASIS_COMBOS)
def test_default_pipeline_preserves_unitary(seed: int, single: SingleQubitGateBasis, two: TwoQubitGateBasis) -> None:
    circuit = _random_circuit(seed, max_controls=2)

    result = CircuitTranspiler.default(single_qubit_basis=single, two_qubit_basis=two).transpile(circuit)

    _assert_only_basis_gates(result.circuit, single, two)
    _assert_same_unitary(circuit, result.circuit)
    for intermediate in result.intermediate_results:
        _assert_same_unitary(circuit, intermediate.circuit)


@pytest.mark.parametrize("seed", ROUTED_SEEDS)
@pytest.mark.parametrize(("single", "two"), BASIS_COMBOS)
@pytest.mark.parametrize("use_custom_mapping", [False, True])
def test_default_routed_pipeline_preserves_semantics(
    seed: int, single: SingleQubitGateBasis, two: TwoQubitGateBasis, use_custom_mapping: bool
) -> None:
    circuit = _random_circuit(seed, max_controls=2)
    mapping = (
        {logical: int(q) for logical, q in enumerate(np.random.default_rng(seed).permutation(5)[:NQUBITS])}
        if use_custom_mapping
        else None
    )
    transpiler = CircuitTranspiler.default(
        single_qubit_basis=single, two_qubit_basis=two, topology=LINE_TOPOLOGY, qubit_mapping=mapping
    )

    result = transpiler.transpile(circuit)

    _assert_only_basis_gates(result.circuit, single, two)
    _assert_adjacent_on_topology(result.circuit, LINE_TOPOLOGY)
    initial_layout = transpiler._context.initial_layout
    assert result.layout is not None
    _assert_routed_equivalent(circuit, result.circuit, initial_layout, result.layout, seed)


def test_equivalence_helpers_reject_relative_phase_and_misrouting() -> None:
    # Guard the guards: the checks above must fail on the kinds of bugs they exist to catch.
    identity_circuit = Circuit(1)
    identity_circuit.add(I(0))
    z_circuit = Circuit(1)
    z_circuit.add(Z(0))
    with pytest.raises(AssertionError):
        _assert_same_unitary(identity_circuit, z_circuit)

    original = Circuit(2)
    original.add(CNOT(0, 1))
    with pytest.raises(AssertionError):
        _assert_routed_equivalent(original, original, [0, 1], {0: 1, 1: 0}, seed=0)

    reversed_u3 = Circuit(1)
    for gate in (RZ(0, phi=0.7), RY(0, theta=0.3), RZ(0, phi=-1.1)):
        reversed_u3.add(gate)
    u3_circuit = Circuit(1)
    u3_circuit.add(U3(0, theta=0.3, phi=0.7, gamma=-1.1))
    with pytest.raises(AssertionError):
        _assert_same_unitary(u3_circuit, reversed_u3)
