# Copyright 2025 Qilimanjaro Quantum Tech
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
"""Property tests for SABRE layout + routing: semantics, bookkeeping, determinism and quality."""

import itertools
import math
import random
from typing import Any

import numpy as np
import pytest

from qilisdk.digital import CNOT, CZ, RX, RY, RZ, SWAP, U3, Circuit, H
from qilisdk.digital.circuit_transpiler import CircuitTranspiler
from qilisdk.digital.circuit_transpiler_passes import SabreLayoutPass, SabreSwapPass
from qilisdk.digital.topology import build_topology_graph

from .utils import _apply_sequence_to_state, _vectors_equal_up_to_phase

TOPOLOGIES: dict[str, list[tuple[int, int]]] = {
    "line": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "ring": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    "grid": [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)],
    "star": [(0, 1), (0, 2), (0, 3), (0, 4)],
    "sparse_labels": [(0, 2), (2, 4), (4, 7), (7, 9), (2, 9)],
}


def _random_circuit(
    nqubits: int, ngates: int, rng: random.Random, two_qubit_gates: tuple[type, ...] = (CNOT, CZ, SWAP)
) -> Circuit:
    """Mix 1Q rotations with 2Q gates, including repeated and reversed pairs."""
    circuit = Circuit(nqubits)
    previous_pair: tuple[int, int] | None = None
    for _ in range(ngates):
        if rng.random() < 0.5:
            if previous_pair is not None and rng.random() < 0.2:
                qubit_a, qubit_b = previous_pair if rng.random() < 0.5 else previous_pair[::-1]
            else:
                qubit_a, qubit_b = rng.sample(range(nqubits), 2)
            previous_pair = (qubit_a, qubit_b)
            circuit.add(rng.choice(two_qubit_gates)(qubit_a, qubit_b))
            continue
        qubit = rng.randrange(nqubits)
        angle = rng.uniform(-math.pi, math.pi)
        circuit.add(
            rng.choice(
                [
                    lambda: H(qubit),
                    lambda: RX(qubit, theta=angle),
                    lambda: RY(qubit, theta=angle),
                    lambda: RZ(qubit, phi=angle),
                    lambda: U3(qubit, theta=angle, phi=angle / 2, gamma=-angle / 3),
                ]
            )()
        )
    return circuit


def _random_state(nqubits: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    state = generator.normal(size=2**nqubits) + 1j * generator.normal(size=2**nqubits)
    return state / np.linalg.norm(state)


def _embed(logical_state: np.ndarray, placement: dict[int, int], num_physical: int) -> np.ndarray:
    """Place logical qubit q on physical qubit ``placement[q]``; every other physical qubit is |0>."""
    num_logical = len(placement)
    tensor = np.zeros([2] * num_physical, dtype=complex)
    tensor[(Ellipsis,) + (0,) * (num_physical - num_logical)] = logical_state.reshape([2] * num_logical)
    # Axes 0..n-1 now hold the logical qubits and the rest hold |0>; move each one to its physical wire.
    source_axis = [-1] * num_physical
    for logical_qubit, physical_qubit in placement.items():
        source_axis[physical_qubit] = logical_qubit
    spare_axes = iter(range(num_logical, num_physical))
    source_axis = [axis if axis >= 0 else next(spare_axes) for axis in source_axis]
    return tensor.transpose(source_axis).reshape(-1)


def _assert_same_action(
    circuit: Circuit, routed: Circuit, initial: dict[int, int], final: dict[int, int], seed: int
) -> None:
    """Routed circuit maps embed(psi, initial) to embed(U psi, final) for a random entangled psi."""
    logical_state = _random_state(circuit.nqubits, seed)
    expected = _apply_sequence_to_state(circuit.gates, logical_state.copy(), circuit.nqubits)
    actual = _apply_sequence_to_state(routed.gates, _embed(logical_state, initial, routed.nqubits), routed.nqubits)
    assert _vectors_equal_up_to_phase(_embed(expected, final, routed.nqubits), actual)


def _assert_gates_on_edges(routed: Circuit, edges: list[tuple[int, int]]) -> None:
    edge_set = {frozenset(edge) for edge in edges}
    for gate in routed.gates:
        if gate.nqubits == 2:
            assert frozenset(gate.qubits) in edge_set, gate


def test_embed_places_logical_qubits_on_their_physical_wires() -> None:
    # |01> on logical (0, 1) placed at physical (2, 0) of 3 wires is |100> (wire 0 is the leftmost bit).
    logical_state = np.zeros(4, dtype=complex)
    logical_state[0b01] = 1.0

    embedded = _embed(logical_state, {0: 2, 1: 0}, 3)

    assert embedded[0b100] == 1.0
    assert np.count_nonzero(embedded) == 1


@pytest.mark.parametrize("apply_layout", [False, True])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("topology_name", sorted(TOPOLOGIES))
def test_layout_and_routing_preserve_circuit_action(topology_name: str, seed: int, apply_layout: bool) -> None:
    edges = TOPOLOGIES[topology_name]
    graph = build_topology_graph(edges)
    rng = random.Random(seed)
    circuit = _random_circuit(rng.randint(2, 5), 25, rng)
    layout_pass = SabreLayoutPass(graph, seed=seed, apply_layout=apply_layout)
    transpiler = CircuitTranspiler([layout_pass, SabreSwapPass(graph, seed=seed)])

    result = transpiler.transpile(circuit)

    initial = dict(enumerate(layout_pass.last_layout))
    # With the layout applied first, the router sees physical wires, so its final map is keyed by them.
    final = {q: result.layout[initial[q]] for q in initial} if apply_layout else result.layout
    _assert_gates_on_edges(result.circuit, edges)
    _assert_same_action(circuit, result.circuit, initial, final, seed)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("topology_name", sorted(TOPOLOGIES))
def test_default_pipeline_preserves_circuit_action(topology_name: str, seed: int) -> None:
    edges = TOPOLOGIES[topology_name]
    rng = random.Random(100 + seed)
    circuit = _random_circuit(rng.randint(2, 5), 20, rng)
    transpiler = CircuitTranspiler.default(topology=edges)

    result = transpiler.transpile(circuit)

    layout_pass = next(p for p in transpiler._pipeline if isinstance(p, SabreLayoutPass))
    _assert_gates_on_edges(result.circuit, edges)
    _assert_same_action(circuit, result.circuit, dict(enumerate(layout_pass.last_layout)), result.layout, seed)


def _replay_routing(circuit: Circuit, routed: Circuit, initial_layout: list[int]) -> int:
    """Check routed == original gates with SWAPs interleaved, and return the number of inserted SWAPs."""
    occupant: dict[int, int | None] = {physical: logical for logical, physical in enumerate(initial_layout)}
    original_gates = iter(circuit.gates)
    expected = next(original_gates, None)
    inserted_swaps = 0
    for gate in routed.gates:
        logical_qubits = tuple(occupant.get(qubit) for qubit in gate.qubits)
        if expected is not None and type(gate) is type(expected) and logical_qubits == expected.qubits:
            expected = next(original_gates, None)
            continue
        # The circuits replayed here contain no SWAPs, so anything else must be an inserted one.
        assert isinstance(gate, SWAP), gate
        physical_a, physical_b = gate.qubits
        occupant[physical_a], occupant[physical_b] = occupant.get(physical_b), occupant.get(physical_a)
        inserted_swaps += 1
    assert expected is None
    return inserted_swaps


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("topology_name", sorted(TOPOLOGIES))
def test_swap_routing_keeps_gate_order_from_any_initial_layout(topology_name: str, seed: int) -> None:
    edges = TOPOLOGIES[topology_name]
    physical_nodes = sorted({node for edge in edges for node in edge})
    rng = random.Random(seed)
    nqubits = rng.randint(2, len(physical_nodes))
    circuit = _random_circuit(nqubits, 30, rng, two_qubit_gates=(CNOT, CZ))
    initial_layout = rng.sample(physical_nodes, nqubits)
    swap_pass = SabreSwapPass(edges, initial_layout=initial_layout, seed=seed)

    routed = swap_pass.run(circuit)

    assert _replay_routing(circuit, routed, initial_layout) == swap_pass.last_swap_count
    _assert_gates_on_edges(routed, edges)


def test_layout_trials_chain_reverse_and_forward_simulations(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = Circuit(3)
    for qubit_a, qubit_b in [(0, 1), (1, 2), (0, 2), (2, 1)]:
        circuit.add(CZ(qubit_a, qubit_b))
    pairs = [(0, 1), (1, 2), (0, 2), (2, 1)]
    layouts = [list(permutation) for permutation in itertools.permutations([0, 1, 2])]
    forward_scores = iter([5.0, 3.0, 4.0, 3.0, 2.0, 2.0])
    calls: list[dict[str, Any]] = []

    def fake_simulate(self: SabreLayoutPass, layout: list[int], _distances, sim_pairs, per_qubit, *_args) -> Any:
        is_reverse = len(calls) % 2 == 0
        output = layouts[len(calls) % len(layouts)]
        calls.append({"input": list(layout), "pairs": list(sim_pairs), "per_qubit": per_qubit, "output": output})
        return list(output), 0.0 if is_reverse else next(forward_scores)

    monkeypatch.setattr(SabreLayoutPass, "_sabre_simulate_layout", fake_simulate)
    layout_pass = SabreLayoutPass([(0, 1), (1, 2)], num_trials=2, num_iterations=3, seed=0, apply_layout=False)

    layout_pass.run(circuit)

    assert len(calls) == 2 * 2 * 3
    for index, call in enumerate(calls):
        expected_pairs = pairs[::-1] if index % 2 == 0 else pairs
        assert call["pairs"] == expected_pairs
        # Per-qubit index lists must list, in order, the interactions touching each qubit.
        for qubit in range(3):
            assert call["per_qubit"][qubit] == [i for i, pair in enumerate(expected_pairs) if qubit in pair]
        # Each forward pass starts from the reverse output; later reverse passes continue from the forward output.
        if index % 2 == 1 or index % 6 != 0:
            assert call["input"] == calls[index - 1]["output"]
    # The fifth forward pass scores best (2.0); the later tie keeps the earlier candidate.
    assert layout_pass.last_layout == calls[8]["output"]
    assert math.isclose(layout_pass.last_score, 2.0)


@pytest.mark.parametrize("apply_layout", [False, True])
def test_layout_and_routing_are_deterministic_for_a_seed(apply_layout: bool) -> None:
    edges = TOPOLOGIES["grid"]
    circuit = _random_circuit(6, 40, random.Random(3))

    def transpile() -> tuple[list[tuple[str, tuple[int, ...]]], list[int]]:
        layout_pass = SabreLayoutPass(edges, seed=11, apply_layout=apply_layout)
        result = CircuitTranspiler([layout_pass, SabreSwapPass(edges, seed=11)]).transpile(circuit)
        return [(type(gate).__name__, gate.qubits) for gate in result.circuit.gates], layout_pass.last_layout

    first_run = transpile()
    second_run = transpile()

    assert first_run == second_run


def test_layout_simulation_does_not_mutate_its_input_layout() -> None:
    layout_pass = SabreLayoutPass(TOPOLOGIES["line"], num_trials=1, seed=0)
    physical_nodes = [0, 1, 2, 3, 4]
    physical_to_dense_index = {node: node for node in physical_nodes}
    distance_matrix = layout_pass._all_pairs_shortest_path_unweighted(
        layout_pass.topology, physical_nodes, physical_to_dense_index
    )
    layout = [0, 4, 2]

    final_layout, score = layout_pass._sabre_simulate_layout(
        layout,
        distance_matrix,
        [(0, 1), (1, 2), (0, 2)],
        [[0, 2], [0, 1], [1, 2]],
        random.Random(0),
        physical_nodes,
        physical_to_dense_index,
    )

    assert layout == [0, 4, 2]
    assert final_layout is not layout
    assert score > 0


@pytest.mark.parametrize(
    ("front_positions", "max_size", "expected"),
    [
        ([0, 0, 0], 2, {1, 2}),  # budget is shared across qubits in qubit order
        ([0, 0, 0], 10, {1, 2, 3}),  # stops when every qubit is exhausted
        ([2, 0, 0], 1, {2}),  # a qubit with no gates left is skipped
        ([0, 0, 0], 0, set()),  # zero budget
        ([0, 0, 0], -1, set()),  # negative budget
    ],
)
def test_layout_extended_set_scans_past_each_front_pointer(
    front_positions: list[int], max_size: int, expected: set[int]
) -> None:
    gate_indices_per_qubit = [[0, 1], [0, 2, 3], [4]]

    assert SabreLayoutPass._extended_set(gate_indices_per_qubit, front_positions, max_size) == expected


@pytest.mark.parametrize("seed", range(6))
def test_layout_finds_swap_free_placement_for_scrambled_ring(seed: int) -> None:
    ring = [(i, (i + 1) % 8) for i in range(8)]
    order = list(range(8))
    random.Random(seed).shuffle(order)
    circuit = Circuit(8)
    for position in range(8):
        circuit.add(CNOT(order[position], order[(position + 1) % 8]))
    layout_pass = SabreLayoutPass(ring, seed=seed, apply_layout=False)
    swap_pass = SabreSwapPass(ring, seed=seed)

    CircuitTranspiler([layout_pass, swap_pass]).transpile(circuit)

    assert math.isclose(layout_pass.last_score, 0.0)
    assert swap_pass.last_swap_count == 0


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for qubit in range(rows * cols):
        if (qubit % cols) + 1 < cols:
            edges.append((qubit, qubit + 1))
        if qubit + cols < rows * cols:
            edges.append((qubit, qubit + cols))
    return edges


def test_layout_usually_recovers_hidden_grid_placement() -> None:
    # The interaction graph is the 4x4 grid itself under a random relabelling, so a SWAP-free layout exists.
    edges = _grid_edges(4, 4)
    found = 0
    for seed in range(10):
        relabel = list(range(16))
        random.Random(seed).shuffle(relabel)
        circuit = Circuit(16)
        for _ in range(3):
            for qubit_a, qubit_b in edges:
                circuit.add(CZ(relabel[qubit_a], relabel[qubit_b]))
        layout_pass = SabreLayoutPass(edges, seed=seed, apply_layout=False)
        layout_pass.run(circuit)
        found += layout_pass.last_score == 0
    assert found >= 8


def test_layout_beats_identity_placement_on_random_circuits() -> None:
    edges = _grid_edges(4, 4)
    sabre_swaps = identity_swaps = 0
    for seed in range(10):
        rng = random.Random(100 + seed)
        circuit = Circuit(16)
        for _ in range(40):
            circuit.add(CZ(*rng.sample(range(16), 2)))
        layout_pass = SabreLayoutPass(edges, seed=seed, apply_layout=False)
        layout_pass.run(circuit)
        sabre_route = SabreSwapPass(edges, initial_layout=layout_pass.last_layout, seed=seed)
        identity_route = SabreSwapPass(edges, initial_layout=list(range(16)), seed=seed)
        sabre_route.run(circuit)
        identity_route.run(circuit)
        sabre_swaps += sabre_route.last_swap_count
        identity_swaps += identity_route.last_swap_count
    # Currently about two thirds; a regression to forward-only layouts lands near parity.
    assert sabre_swaps < 0.8 * identity_swaps


def test_swap_routing_ignores_unused_qubit_placeholder_outside_the_graph() -> None:
    # Unused logical qubit 3 is hinted onto label 99, which is not a coupling-graph node.
    circuit = Circuit(4)
    circuit.add(CZ(0, 1))
    circuit.add(RX(2, theta=0.3))
    swap_pass = SabreSwapPass([(0, 1), (1, 2)], initial_layout=[0, 2, 1, 99], seed=0)

    routed = swap_pass.run(circuit)

    assert _replay_routing(circuit, routed, [0, 2, 1, 99]) == swap_pass.last_swap_count == 1
    assert swap_pass.last_final_layout[3] == 99


def _random_cz_circuit(nqubits: int, ngates: int, seed: int) -> Circuit:
    rng = random.Random(seed)
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        circuit.add(CZ(*rng.sample(range(nqubits), 2)))
    return circuit


def test_layout_simulation_routes_about_as_well_as_the_router() -> None:
    # From the same start, the layout simulation may also reorder independent gates, so it should not need
    # noticeably more SWAPs than SabreSwapPass. Stale front bookkeeping inflates this by about a quarter.
    edges = _grid_edges(4, 4)
    nodes = list(range(16))
    identity = {node: node for node in nodes}
    simulated_swaps = routed_swaps = 0.0
    for seed in range(10):
        circuit = _random_cz_circuit(16, 40, 100 + seed)
        layout_pass = SabreLayoutPass(edges, seed=seed)
        distance_matrix = layout_pass._all_pairs_shortest_path_unweighted(layout_pass.topology, nodes, identity)
        _, pairs, per_qubit = layout_pass._two_qubit_structure(circuit)
        _, swaps = layout_pass._sabre_simulate_layout(
            list(nodes), distance_matrix, pairs, per_qubit, random.Random(seed), nodes, identity
        )
        swap_pass = SabreSwapPass(edges, initial_layout=list(nodes), seed=seed)
        swap_pass.run(circuit)
        simulated_swaps += swaps
        routed_swaps += swap_pass.last_swap_count
    assert simulated_swaps <= 1.1 * routed_swaps


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("topology_name", ["grid4x4", "line"])
def test_swap_routing_picks_the_cheapest_swap_by_full_recomputation(topology_name: str, seed: int) -> None:
    # Oracle: re-score every candidate SWAP with the full SABRE cost (all F and E gates, eager decay) and check
    # the pass picked a minimum, preferring SWAPs that shorten the gate being routed.
    edges = _grid_edges(4, 4) if topology_name == "grid4x4" else TOPOLOGIES["line"]
    nodes = sorted({node for edge in edges for node in edge})
    circuit = _random_cz_circuit(len(nodes), 30, seed)
    swap_pass = SabreSwapPass(edges, initial_layout=list(nodes), seed=seed)
    routed = swap_pass.run(circuit)

    graph = swap_pass.topology
    identity = {node: node for node in nodes}
    distance = swap_pass._all_pairs_shortest_path_unweighted(graph, nodes, identity)
    _, pairs, per_qubit = swap_pass._two_qubit_structure(circuit)
    position = list(nodes)
    front_positions = [0] * len(nodes)
    decay = [0.0] * len(nodes)
    next_gate = 0

    def full_cost(layout: list[int]) -> float:
        front = {gates[front_positions[q]] for q, gates in enumerate(per_qubit) if front_positions[q] < len(gates)}
        extended = SabreSwapPass._extended_set(per_qubit, front_positions, swap_pass.lookahead_size)
        front_cost = sum(
            distance[layout[a]][layout[b]] * (1 + decay[layout[a]] + decay[layout[b]])
            for a, b in (pairs[g] for g in front)
        )
        return front_cost + swap_pass.beta * sum(
            distance[layout[a]][layout[b]] for a, b in (pairs[g] for g in extended)
        )

    def swapped(layout: list[int], edge: tuple[int, int]) -> list[int]:
        return [edge[1] if p == edge[0] else edge[0] if p == edge[1] else p for p in layout]

    checked = 0
    for gate in routed.gates:
        if not isinstance(gate, SWAP):
            logical_a, logical_b = pairs[next_gate]
            front_positions[logical_a] += 1
            front_positions[logical_b] += 1
            next_gate += 1
            continue
        logical_a, logical_b = pairs[next_gate]
        decay = [value * swap_pass.decay_lambda for value in decay]
        front_qubits = {q for q, gates in enumerate(per_qubit) if front_positions[q] < len(gates)}
        candidates = {
            (min(position[q], int(n)), max(position[q], int(n)))
            for q in front_qubits
            for n in graph.neighbors(position[q])
        }
        costs = {edge: full_cost(swapped(position, edge)) for edge in candidates}
        current = distance[position[logical_a]][position[logical_b]]
        improving = {
            edge: cost
            for edge, cost in costs.items()
            if distance[swapped(position, edge)[logical_a]][swapped(position, edge)[logical_b]] < current
        }
        pool = improving or costs
        chosen = (min(gate.qubits), max(gate.qubits))
        assert chosen in pool
        assert pool[chosen] <= min(pool.values()) + 1e-9
        position = swapped(position, chosen)
        decay[chosen[0]] += swap_pass.decay_delta
        decay[chosen[1]] += swap_pass.decay_delta
        checked += 1
    assert checked == swap_pass.last_swap_count > 0
