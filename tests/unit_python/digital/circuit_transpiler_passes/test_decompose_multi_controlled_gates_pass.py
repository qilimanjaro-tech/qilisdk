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

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from qilisdk.core import QTensor
from qilisdk.digital import RX, RY, RZ, U1, U2, U3, Circuit, H, I, M, S, T, X, Y, Z
from qilisdk.digital.circuit_transpiler_passes import DecomposeMultiControlledGatesPass
from qilisdk.digital.circuit_transpiler_passes.decompose_multi_controlled_gates_pass import (
    _adjoint_of,
    _phase_on_controls,
    _sqrt_of,
)
from qilisdk.digital.circuit_transpiler_passes.numeric_helpers import _wrap_angle, _zyz_from_unitary
from qilisdk.digital.gates import BasicGate, Controlled, Gate

from .utils import _sequence_matrix, _sequences_equivalent

ATOL = 1e-9


def _run_pass_with_gate(gate: Gate, nqubits: int) -> Circuit:
    circuit = Circuit(nqubits)
    circuit.add(gate)
    return DecomposeMultiControlledGatesPass().run(circuit)


class _MatrixGate(BasicGate):
    """Single-qubit gate defined directly by an arbitrary 2x2 unitary.

    Used to drive the generic (matrix-based) branches of the square-root and adjoint helpers with unitaries that no
    named gate produces, including the degenerate diagonal and anti-diagonal shapes.
    """

    def __init__(self, qubit: int, unitary: np.ndarray) -> None:
        super().__init__(target_qubits=(qubit,))
        self._unitary = np.asarray(unitary, dtype=complex)

    @property
    def name(self) -> str:
        return "MatrixGate"

    def _generate_matrix(self) -> QTensor:
        return QTensor(self._unitary.copy())


def _haar_unitary(seed: int) -> np.ndarray:
    """Draw a Haar-random 2x2 unitary deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    ginibre = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(ginibre)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _haar_gate_factory(seed: int):
    """Build a factory producing a `_MatrixGate` holding the Haar unitary for `seed`."""

    def factory(qubit: int) -> _MatrixGate:
        return _MatrixGate(qubit, _haar_unitary(seed))

    return factory


GATE_FACTORIES = [
    ("I", I),
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

# Angles that stress the wrap-around, zero-phase and half-turn edge cases of the helpers.
EDGE_CASE_FACTORIES = [
    ("RX(0)", lambda q: RX(q, theta=0.0)),
    ("RX(pi)", lambda q: RX(q, theta=math.pi)),
    ("RX(-pi)", lambda q: RX(q, theta=-math.pi)),
    ("RX(2pi)", lambda q: RX(q, theta=2.0 * math.pi)),
    ("RY(pi)", lambda q: RY(q, theta=math.pi)),
    ("RY(-3pi/2)", lambda q: RY(q, theta=-3.0 * math.pi / 2.0)),
    ("RZ(pi)", lambda q: RZ(q, phi=math.pi)),
    ("RZ(-pi)", lambda q: RZ(q, phi=-math.pi)),
    ("RZ(1e-9)", lambda q: RZ(q, phi=1e-9)),
    ("U1(0)", lambda q: U1(q, phi=0.0)),
    ("U1(pi)", lambda q: U1(q, phi=math.pi)),
    ("U1(-pi)", lambda q: U1(q, phi=-math.pi)),
    ("U1(2pi)", lambda q: U1(q, phi=2.0 * math.pi)),
    ("U1(3pi/2)", lambda q: U1(q, phi=3.0 * math.pi / 2.0)),
    ("U2(0,0)", lambda q: U2(q, phi=0.0, gamma=0.0)),
    ("U2(pi,-pi)", lambda q: U2(q, phi=math.pi, gamma=-math.pi)),
    ("U3(0,0,0)", lambda q: U3(q, theta=0.0, phi=0.0, gamma=0.0)),
    ("U3(pi,0,0)", lambda q: U3(q, theta=math.pi, phi=0.0, gamma=0.0)),
    ("U3(pi,pi/3,-pi/5)", lambda q: U3(q, theta=math.pi, phi=math.pi / 3.0, gamma=-math.pi / 5.0)),
    ("U3(-pi/2,pi,pi)", lambda q: U3(q, theta=-math.pi / 2.0, phi=math.pi, gamma=math.pi)),
    ("U3(2pi,0,0)", lambda q: U3(q, theta=2.0 * math.pi, phi=0.0, gamma=0.0)),
]

# Unitaries with no named-gate equivalent, exercising the generic matrix branches.
CUSTOM_UNITARY_FACTORIES = [
    (
        "diagonal",
        lambda q: _MatrixGate(q, np.diag([np.exp(0.37j), np.exp(-1.42j)])),
    ),
    (
        "anti_diagonal",
        lambda q: _MatrixGate(q, np.array([[0.0, np.exp(0.9j)], [np.exp(-2.1j), 0.0]])),
    ),
    (
        "negative_identity",
        lambda q: _MatrixGate(q, -np.eye(2)),
    ),
    (
        "global_phase_only",
        lambda q: _MatrixGate(q, np.exp(1.234j) * np.eye(2)),
    ),
    *[(f"haar_{seed}", _haar_gate_factory(seed)) for seed in range(6)],
]

ALL_FACTORIES = GATE_FACTORIES + EDGE_CASE_FACTORIES + CUSTOM_UNITARY_FACTORIES

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
    assert _sequences_equivalent([gate], transpiled.gates, nqubits, states), (
        f"Vector equality for {factory_name} with {ncontrols} controls"
    )
    assert _sequences_equivalent([gate], transpiled.gates, nqubits, None), (
        f"Unitary equality for {factory_name} with {ncontrols} controls"
    )
    # The decomposition must be exact, not just equal up to phases: a global phase on the square root becomes an
    # observable relative phase once it sits under a control (see issue #339).
    assert np.allclose(_sequence_matrix([gate], nqubits), _sequence_matrix(transpiled.gates, nqubits), atol=1e-9), (
        f"Exact unitary equality for {factory_name} with {ncontrols} controls"
    )

    for rewritten in transpiled.gates:
        if isinstance(rewritten, Controlled):
            assert len(rewritten.control_qubits) <= 1


# ======================= exactness of the decomposed unitary =======================


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
@pytest.mark.parametrize("ncontrols", CONTROL_COUNTS)
def test_decomposition_is_exactly_the_original_unitary(factory_name: str, factory, ncontrols: int) -> None:
    """Every decomposition must reproduce the controlled unitary entry by entry, with no residual phase."""
    gate = _build_controlled_gate(factory, ncontrols)
    circuit = Circuit(ncontrols + 1)
    circuit.add(gate)

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL), (
        f"Exact unitary equality for {factory_name} with {ncontrols} controls"
    )


@pytest.mark.parametrize(("factory_name", "factory"), [("X", X), ("H", H), ("T", T), ("haar_0", _haar_gate_factory(0))])
def test_decomposition_is_exact_for_deep_recursion(factory_name: str, factory) -> None:
    """Five controls exercise four levels of recursion, where dropped phases used to compound."""
    ncontrols = 5
    gate = _build_controlled_gate(factory, ncontrols)
    circuit = Circuit(ncontrols + 1)
    circuit.add(gate)

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL), (
        f"Exact unitary equality for {factory_name} with {ncontrols} controls"
    )


# Control/target placements that are neither contiguous nor ordered.
QUBIT_LAYOUTS = [
    ((1, 0), 2),
    ((2, 0), 1),
    ((3, 1), 0),
    ((0, 2), 3),
    ((3, 0), 2),
    ((2, 0, 3), 1),
    ((3, 1, 0), 2),
    ((1, 3, 2), 0),
    ((3, 2, 1, 0), 4),
    ((4, 0, 3, 1), 2),
]


@pytest.mark.parametrize(("controls", "target"), QUBIT_LAYOUTS)
@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_decomposition_is_exact_for_arbitrary_qubit_layouts(
    controls: tuple[int, ...], target: int, factory_name: str, factory
) -> None:
    """The recursion peels controls off the end of the tuple, so ordering and gaps must not matter."""
    nqubits = max((*controls, target)) + 1
    circuit = Circuit(nqubits)
    circuit.add(Controlled(*controls, basic_gate=factory(target)))

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL), (
        f"Exact unitary equality for {factory_name} on controls {controls} / target {target}"
    )


@pytest.mark.parametrize("ncontrols", CONTROL_COUNTS)
def test_decomposition_acts_as_identity_unless_all_controls_are_set(ncontrols: int) -> None:
    """Outside the all-controls-set subspace the decomposition must be exactly the identity, not a phase."""
    nqubits = ncontrols + 1
    circuit = Circuit(nqubits)
    circuit.add(Controlled(*range(ncontrols), basic_gate=H(ncontrols)))

    matrix = DecomposeMultiControlledGatesPass().run(circuit).to_matrix()

    # Qubit 0 is the most significant bit, so the triggered subspace is the top 2 rows/columns.
    triggered = [index for index in range(1 << nqubits) if index >> 1 == (1 << ncontrols) - 1]
    untriggered = [index for index in range(1 << nqubits) if index not in triggered]
    inactive_block = matrix[np.ix_(untriggered, untriggered)]

    assert np.allclose(inactive_block, np.eye(len(untriggered)), atol=ATOL)
    assert np.allclose(matrix[np.ix_(untriggered, triggered)], 0.0, atol=ATOL)
    assert np.allclose(matrix[np.ix_(triggered, untriggered)], 0.0, atol=ATOL)
    assert np.allclose(matrix[np.ix_(triggered, triggered)], H(0).matrix.dense(), atol=ATOL)


def test_decomposition_of_a_full_circuit_with_several_multi_controlled_gates() -> None:
    """Phases must stay consistent when several decompositions are composed in one circuit."""
    circuit = Circuit(4)
    circuit.add(H(0))
    circuit.add(Controlled(0, 1, basic_gate=X(2)))
    circuit.add(RZ(3, phi=math.pi / 3.0))
    circuit.add(Controlled(0, 1, 2, basic_gate=U3(3, theta=0.7, phi=-1.1, gamma=2.3)))
    circuit.add(Controlled(3, 2, basic_gate=Z(0)))
    circuit.add(Controlled(1, basic_gate=Y(0)))
    circuit.add(Controlled(2, 3, 0, basic_gate=T(1)))

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL)


@pytest.mark.parametrize("ncontrols", CONTROL_COUNTS)
def test_nested_controlled_gate_is_decomposed_exactly(ncontrols: int) -> None:
    """`Controlled` flattens nested controls, so a nested build must decompose like a flat one."""
    target = ncontrols
    nested: Controlled = Controlled(0, basic_gate=X(target))
    for control in range(1, ncontrols):
        nested = Controlled(control, basic_gate=nested)

    circuit = Circuit(ncontrols + 1)
    circuit.add(nested)

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL)


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_pass_is_idempotent(factory_name: str, factory) -> None:
    """A second run has nothing left to decompose and must not perturb the unitary."""
    gate = _build_controlled_gate(factory, 3)
    once = _run_pass_with_gate(gate, 4)
    twice = DecomposeMultiControlledGatesPass().run(once)

    assert len(twice.gates) == len(once.gates)
    assert np.allclose(once.to_matrix(), twice.to_matrix(), atol=ATOL), f"Idempotence failed for {factory_name}"


# ======================= structural guarantees of the output =======================


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
@pytest.mark.parametrize("ncontrols", CONTROL_COUNTS)
def test_output_gates_have_at_most_one_control(factory_name: str, factory, ncontrols: int) -> None:
    transpiled = _run_pass_with_gate(_build_controlled_gate(factory, ncontrols), ncontrols + 1)

    assert transpiled.gates, f"No gates produced for {factory_name}"
    for rewritten in transpiled.gates:
        if isinstance(rewritten, Controlled):
            assert len(rewritten.control_qubits) == 1, f"Leftover multi-control for {factory_name}"
            assert not isinstance(rewritten.basic_gate, Controlled)


@pytest.mark.parametrize(("controls", "target"), QUBIT_LAYOUTS)
def test_output_gates_only_touch_the_original_qubits(controls: tuple[int, ...], target: int) -> None:
    nqubits = max((*controls, target)) + 1
    circuit = Circuit(nqubits)
    circuit.add(Controlled(*controls, basic_gate=H(target)))

    transpiled = DecomposeMultiControlledGatesPass().run(circuit)

    allowed = {*controls, target}
    for rewritten in transpiled.gates:
        assert set(rewritten.qubits) <= allowed


def test_output_preserves_circuit_width_and_surrounding_gates() -> None:
    circuit = Circuit(5)
    circuit.add(RZ(4, phi=0.3))
    circuit.add(Controlled(0, 1, 2, basic_gate=X(3)))
    circuit.add(H(4))

    transpiled = DecomposeMultiControlledGatesPass().run(circuit)

    assert transpiled.nqubits == circuit.nqubits
    assert isinstance(transpiled.gates[0], RZ)
    assert isinstance(transpiled.gates[-1], H)
    assert np.allclose(circuit.to_matrix(), transpiled.to_matrix(), atol=ATOL)


# ======================= _sqrt_of / _adjoint_of =======================


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
def test_sqrt_of_gate_squares_back_exactly(factory_name: str, factory) -> None:
    gate = factory(0)
    sqrt_gate, phase = _sqrt_of(gate)
    square_root = np.exp(1j * phase) * sqrt_gate.matrix.dense()

    assert np.allclose(gate.matrix.dense(), square_root @ square_root, atol=ATOL), f"Sqrt failed for {factory_name}"


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
def test_sqrt_of_gate_is_unitary_and_keeps_the_target_qubit(factory_name: str, factory) -> None:
    gate = factory(2)
    sqrt_gate, _ = _sqrt_of(gate)

    assert sqrt_gate.qubits == (2,), f"Target qubit changed for {factory_name}"
    assert np.allclose(sqrt_gate.matrix.dense().conj().T @ sqrt_gate.matrix.dense(), np.eye(2), atol=ATOL)


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
def test_adjoint_of_gate_inverts_exactly(factory_name: str, factory) -> None:
    gate = factory(0)
    adjoint_gate, phase = _adjoint_of(gate)
    inverse = np.exp(1j * phase) * adjoint_gate.matrix.dense()

    assert np.allclose(inverse @ gate.matrix.dense(), np.eye(2), atol=ATOL), f"Adjoint failed for {factory_name}"
    assert np.allclose(gate.matrix.dense() @ inverse, np.eye(2), atol=ATOL), f"Adjoint failed for {factory_name}"


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
def test_adjoint_of_sqrt_composes_to_the_gate_inverse(factory_name: str, factory) -> None:
    """This is exactly the composition the recursion relies on: sqrt(U)† · sqrt(U)† = U†."""
    gate = factory(0)
    sqrt_gate, sqrt_phase = _sqrt_of(gate)
    adjoint_gate, adjoint_phase = _adjoint_of(sqrt_gate)
    inverse_square_root = np.exp(1j * (adjoint_phase - sqrt_phase)) * adjoint_gate.matrix.dense()

    assert np.allclose(inverse_square_root @ inverse_square_root, gate.matrix.dense().conj().T, atol=ATOL), (
        f"sqrt adjoint composition failed for {factory_name}"
    )


@pytest.mark.parametrize(("factory_name", "factory"), ALL_FACTORIES)
def test_double_adjoint_returns_the_original_gate(factory_name: str, factory) -> None:
    gate = factory(0)
    adjoint_gate, phase = _adjoint_of(gate)
    twice_adjoint_gate, twice_phase = _adjoint_of(adjoint_gate)

    assert np.allclose(
        np.exp(1j * (twice_phase - phase)) * twice_adjoint_gate.matrix.dense(), gate.matrix.dense(), atol=ATOL
    ), f"Double adjoint failed for {factory_name}"


@pytest.mark.parametrize(
    ("gate", "expected_type", "expected_parameters"),
    [
        (Z(0), S, {}),
        (S(0), T, {}),
        (T(0), U1, {"phi": math.pi / 8.0}),
        (U1(0, phi=math.pi / 3.0), U1, {"phi": math.pi / 6.0}),
        (I(0), I, {}),
        (RX(0, theta=1.2), RX, {"theta": 0.6}),
        (RY(0, theta=1.2), RY, {"theta": 0.6}),
        (RZ(0, phi=1.2), RZ, {"phi": 0.6}),
    ],
)
def test_sqrt_of_uses_the_exact_gate_and_reports_no_residual_phase(
    gate: BasicGate, expected_type: type[BasicGate], expected_parameters: dict[str, float]
) -> None:
    """Diagonal and rotation gates have exact square roots, so no phase correction should be emitted."""
    sqrt_gate, phase = _sqrt_of(gate)

    assert isinstance(sqrt_gate, expected_type)
    assert phase == 0.0
    for name, value in expected_parameters.items():
        assert np.isclose(getattr(sqrt_gate, name), value)


@pytest.mark.parametrize(
    ("gate", "expected_type", "expected_parameters"),
    [
        (U1(0, phi=math.pi / 3.0), U1, {"phi": -math.pi / 3.0}),
        (S(0), U1, {"phi": -math.pi / 2.0}),
        (T(0), U1, {"phi": -math.pi / 4.0}),
        (RX(0, theta=1.2), RX, {"theta": -1.2}),
        (RY(0, theta=1.2), RY, {"theta": -1.2}),
        (RZ(0, phi=1.2), RZ, {"phi": -1.2}),
        (X(0), X, {}),
        (Y(0), Y, {}),
        (Z(0), Z, {}),
        (H(0), H, {}),
        (I(0), I, {}),
        (U3(0, theta=0.4, phi=0.5, gamma=0.6), U3, {"theta": -0.4, "phi": -0.6, "gamma": -0.5}),
    ],
)
def test_adjoint_of_uses_the_exact_gate_and_reports_no_residual_phase(
    gate: BasicGate, expected_type: type[BasicGate], expected_parameters: dict[str, float]
) -> None:
    adjoint_gate, phase = _adjoint_of(gate)

    assert isinstance(adjoint_gate, expected_type)
    assert phase == 0.0
    for name, value in expected_parameters.items():
        assert np.isclose(getattr(adjoint_gate, name), value)


@pytest.mark.parametrize(("factory_name", "factory"), [("X", X), ("Y", Y)])
def test_sqrt_of_pauli_x_and_y_reports_the_quarter_turn_phase(factory_name: str, factory) -> None:
    """sqrt(X) and sqrt(Y) are only RX(pi/2) / RY(pi/2) up to e^{i·pi/4}; that phase must be reported."""
    sqrt_gate, phase = _sqrt_of(factory(0))

    assert np.isclose(phase, math.pi / 4.0), f"Wrong residual phase for {factory_name}"
    assert not np.allclose(sqrt_gate.matrix.dense() @ sqrt_gate.matrix.dense(), factory(0).matrix.dense(), atol=ATOL)


def test_sqrt_of_u2_and_h_report_a_non_zero_phase() -> None:
    """Gates routed through the generic U3 branch generally carry a residual phase; it must not be dropped."""
    for gate in (H(0), U2(0, phi=math.pi / 6.0, gamma=math.pi / 5.0)):
        sqrt_gate, phase = _sqrt_of(gate)
        square_root = np.exp(1j * phase) * sqrt_gate.matrix.dense()
        assert isinstance(sqrt_gate, U3)
        assert not np.isclose(phase, 0.0)
        assert np.allclose(square_root @ square_root, gate.matrix.dense(), atol=ATOL)


# ======================= _phase_on_controls =======================


@pytest.mark.parametrize("seed", range(40))
def test_random_haar_controlled_gates_decompose_exactly(seed: int) -> None:
    """Randomised sweep over Haar unitaries, control counts and qubit layouts, seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    ncontrols = int(rng.integers(2, 5))
    nqubits = ncontrols + 1 + int(rng.integers(0, 2))
    qubits = [int(q) for q in rng.permutation(nqubits)]
    controls, target = tuple(qubits[:ncontrols]), qubits[ncontrols]

    circuit = Circuit(nqubits)
    circuit.add(Controlled(*controls, basic_gate=_MatrixGate(target, _haar_unitary(1000 + seed))))

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL), (
        f"Exact unitary equality failed for seed {seed} (controls {controls}, target {target})"
    )


@pytest.mark.parametrize("seed", range(40))
def test_random_named_gate_circuits_decompose_exactly(seed: int) -> None:
    """Randomised sweep over multi-gate circuits mixing controlled and plain gates at random angles."""
    rng = np.random.default_rng(10_000 + seed)

    def random_gate(qubit: int) -> BasicGate:
        angles = [float(rng.uniform(-4.0 * math.pi, 4.0 * math.pi)) for _ in range(3)]
        candidates = [
            RX(qubit, theta=angles[0]),
            RY(qubit, theta=angles[0]),
            RZ(qubit, phi=angles[0]),
            U1(qubit, phi=angles[0]),
            U2(qubit, phi=angles[0], gamma=angles[1]),
            U3(qubit, theta=angles[0], phi=angles[1], gamma=angles[2]),
            [I, X, Y, Z, H, S, T][int(rng.integers(0, 7))](qubit),
        ]
        return candidates[int(rng.integers(0, len(candidates)))]

    nqubits = int(rng.integers(3, 6))
    circuit = Circuit(nqubits)
    for _ in range(int(rng.integers(1, 5))):
        qubits = [int(q) for q in rng.permutation(nqubits)]
        ncontrols = int(rng.integers(1, nqubits))
        circuit.add(Controlled(*qubits[:ncontrols], basic_gate=random_gate(qubits[ncontrols])))
        circuit.add(random_gate(int(rng.integers(0, nqubits))))

    decomposed = DecomposeMultiControlledGatesPass().run(circuit)

    assert np.allclose(circuit.to_matrix(), decomposed.to_matrix(), atol=ATOL), f"Failed for seed {seed}"


def test_empty_circuit_is_returned_unchanged() -> None:
    transpiled = DecomposeMultiControlledGatesPass().run(Circuit(3))

    assert transpiled.nqubits == 3
    assert transpiled.gates == []


def test_measurement_gates_pass_through() -> None:
    circuit = Circuit(3)
    circuit.add(Controlled(0, 1, basic_gate=X(2)))
    circuit.add(M(0, 1, 2))

    transpiled = DecomposeMultiControlledGatesPass().run(circuit)

    assert isinstance(transpiled.gates[-1], M)
    assert sum(isinstance(gate, M) for gate in transpiled.gates) == 1


# ======================= _phase_on_controls =======================


@pytest.mark.parametrize("phase", [0.0, 2.0 * math.pi, -2.0 * math.pi, 4.0 * math.pi, 1e-15])
def test_phase_on_controls_emits_nothing_for_a_trivial_phase(phase: float) -> None:
    assert _phase_on_controls(phase, (0, 1, 2)) == []


def test_phase_on_controls_with_a_single_control_is_a_bare_u1() -> None:
    gates = _phase_on_controls(0.75, (3,))

    assert len(gates) == 1
    assert isinstance(gates[0], U1)
    assert gates[0].qubits == (3,)
    assert np.isclose(gates[0].phi, 0.75)


@pytest.mark.parametrize("ncontrols", [1, 2, 3, 4])
@pytest.mark.parametrize("phase", [0.75, math.pi, -math.pi / 3.0])
def test_phase_on_controls_applies_the_phase_only_when_every_control_is_set(ncontrols: int, phase: float) -> None:
    controls = tuple(range(ncontrols))
    circuit = Circuit(ncontrols)
    for gate in _phase_on_controls(phase, controls):
        circuit.add(gate)

    expected = np.eye(1 << ncontrols, dtype=complex)
    expected[-1, -1] = np.exp(1j * phase)

    assert np.allclose(circuit.to_matrix(), expected, atol=ATOL)
    for gate in circuit.gates:
        if isinstance(gate, Controlled):
            assert len(gate.control_qubits) == 1


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_single_control_gate_is_not_modified(factory_name: str, factory) -> None:
    gate = _build_controlled_gate(factory, 1)
    transpiled = _run_pass_with_gate(gate, 2)

    assert len(transpiled.gates) == 1
    rewritten = transpiled.gates[0]
    assert isinstance(rewritten, Controlled)
    assert rewritten.control_qubits == gate.control_qubits
    assert rewritten.basic_gate.name == gate.basic_gate.name


def test_toffoli_decomposition_has_no_spurious_relative_phase() -> None:
    """Regression test for issue #339: the |11x> block used to pick up an extra -i."""
    circuit = Circuit(3)
    circuit.add(Controlled(0, 1, basic_gate=X(2)))

    reference = circuit.to_matrix()
    decomposed = DecomposeMultiControlledGatesPass().run(circuit).to_matrix()

    assert np.allclose(reference, decomposed)


def test_other_gates_remain_unchanged() -> None:
    circuit = Circuit(3)
    circuit.add(RZ(0, phi=math.pi / 7.0))
    circuit.add(Controlled(0, 1, basic_gate=X(2)))
    circuit.add(RY(2, theta=math.pi / 9.0))

    transpiled = DecomposeMultiControlledGatesPass().run(circuit)
    assert isinstance(transpiled.gates[0], RZ)
    assert isinstance(transpiled.gates[-1], RY)


def test_wrap_angle():
    assert _wrap_angle(0) == 0
    assert _wrap_angle(math.pi) == math.pi
    assert _wrap_angle(-math.pi) == math.pi
    assert _wrap_angle(3 * math.pi) == math.pi
    assert _wrap_angle(-3 * math.pi) == math.pi


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_zyz_unitary(factory_name: str, factory) -> None:
    gate = factory(0)
    unitary = gate.matrix.dense()
    theta, phi, gamma = _zyz_from_unitary(unitary)
    reconstructed = U3(0, theta=theta, phi=phi, gamma=gamma).matrix.dense()
    assert np.allclose(unitary, reconstructed), f"ZYZ reconstruction failed for {factory_name}"


def test_zyz_unitary_errors():
    bad_unitary = np.ones((3, 2), dtype=complex)
    with pytest.raises(ValueError, match="Expected 2x2 unitary"):
        _zyz_from_unitary(bad_unitary)

    singular = np.array([[1, 0], [0, 0]], dtype=complex)
    with pytest.raises(ValueError, match="Matrix is singular"):
        _zyz_from_unitary(singular)


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_adjoint_of_gate(factory_name: str, factory) -> None:
    gate = factory(0)
    adjoint_gate, phase = _adjoint_of(gate)
    # The reported phase must make the factorisation exact, not merely equal up to a global phase.
    assert np.allclose(gate.matrix.dense().conj().T, np.exp(1j * phase) * adjoint_gate.matrix.dense()), (
        f"Adjoint computation failed for {factory_name}"
    )


@pytest.mark.parametrize(("factory_name", "factory"), GATE_FACTORIES)
def test_sqrt_of_gate(factory_name: str, factory) -> None:
    gate = factory(0)
    sqrt_gate, phase = _sqrt_of(gate)
    square_root = np.exp(1j * phase) * sqrt_gate.matrix.dense()
    # V · V must reproduce the gate exactly; a leftover global phase becomes a relative phase under a control.
    assert np.allclose(gate.matrix.dense(), square_root @ square_root), f"Sqrt computation failed for {factory_name}"


def test_sqrt_of_gate_errors():
    custom_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # X gate
    custom_gate = MagicMock(spec=BasicGate)
    custom_gate.matrix = QTensor(custom_matrix)
    custom_gate.qubits = (
        0,
        1,
    )
    custom_gate.nqubits = 2
    with pytest.raises(NotImplementedError, match="only supports 1-qubit gates"):
        _sqrt_of(custom_gate)


def test_adjoint_of_generic_gate():
    custom_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # X gate
    custom_gate = MagicMock(spec=BasicGate)
    custom_gate.matrix = QTensor(custom_matrix)
    custom_gate.qubits = (0,)
    custom_gate.nqubits = 1
    adjoint_gate, phase = _adjoint_of(custom_gate)
    assert np.allclose(custom_matrix.conj().T, np.exp(1j * phase) * adjoint_gate.matrix.dense()), (
        "Adjoint computation failed for generic gate"
    )


def test_adjoint_of_generic_multi_qubit_gate():
    custom_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # X gate
    custom_gate = MagicMock(spec=BasicGate)
    custom_gate.matrix = QTensor(custom_matrix)
    custom_gate.qubits = (
        0,
        1,
    )
    custom_gate.nqubits = 2
    with pytest.raises(NotImplementedError, match="only supports 1-qubit gates"):
        _adjoint_of(custom_gate)


def test_decompose_pass_of_multi_controlled_generic_gate():
    multi_qubit_gate = MagicMock(spec=BasicGate)
    multi_qubit_gate.nqubits = 2
    multi_qubit_gate._parameters = {}
    controlled_gate = Controlled(2, basic_gate=multi_qubit_gate)
    circuit = Circuit(3)
    circuit.add(controlled_gate)
    with pytest.raises(NotImplementedError, match="Controlled version of multi-qubit gates is not supported"):
        DecomposeMultiControlledGatesPass().run(circuit)
