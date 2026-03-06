import math
import random
from itertools import starmap

import pytest
from rustworkx import PyGraph

from qilisdk.digital import (
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
    Circuit,
    Controlled,
    Exponential,
    Gate,
    H,
    M,
    S,
    T,
    X,
    Y,
    Z,
)
from qilisdk.digital.circuit_transpiler_passes.sabre_layout_pass import SabreLayoutPass
from qilisdk.digital.circuit_transpiler_passes.transpilation_context import TranspilationContext


def make_graph(edges, nodes=None):
    graph = PyGraph()
    if nodes is None:
        if edges:
            max_node = max(starmap(max, edges))
            nodes = range(max_node + 1)
        else:
            nodes = []
    graph.add_nodes_from(nodes)
    for u, v in edges:
        graph.add_edge(u, v, None)
    return graph


def test_sabre_layout_requires_pygraph():
    with pytest.raises(TypeError, match=r"requires a rustworkx\.PyGraph"):
        SabreLayoutPass("not-a-graph")  # type: ignore[arg-type]


def test_sabre_layout_rejects_empty_topology():
    topology = PyGraph()
    layout_pass = SabreLayoutPass(topology)
    circuit = Circuit(1)
    with pytest.raises(ValueError, match="Coupling graph has no nodes"):
        layout_pass.run(circuit)


def test_sabre_layout_raises_when_circuit_needs_more_qubits():
    topology = make_graph([(0, 1)], nodes=range(2))
    layout_pass = SabreLayoutPass(topology)
    circuit = Circuit(3)
    with pytest.raises(ValueError, match="circuit needs 3 qubits"):
        layout_pass.run(circuit)


def test_sabre_layout_identity_when_no_two_qubit_gates():
    topology = make_graph([(0, 1), (1, 2)])
    layout_pass = SabreLayoutPass(topology, seed=123)
    context = TranspilationContext()
    layout_pass.attach_context(context)

    circuit = Circuit(2)
    circuit.add(RX(0, theta=0.1))
    circuit.add(RY(1, theta=0.2))

    out = layout_pass.run(circuit)

    assert out.nqubits == 3
    assert [gate.qubits for gate in out.gates] == [(0,), (1,)]
    assert layout_pass.last_layout == [0, 1]
    assert math.isclose(layout_pass.last_score, 0.0)
    assert context.initial_layout == [0, 1]
    assert context.circuits == {}

    # Original circuit untouched
    assert [gate.qubits for gate in circuit.gates] == [(0,), (1,)]


def test_sabre_layout_retargets_generic_gate_set_and_updates_diagnostics():
    topology = make_graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    layout_pass = SabreLayoutPass(topology, num_trials=1, seed=7, lookahead_size=2)
    context = TranspilationContext()
    layout_pass.attach_context(context)

    circuit = Circuit(4)
    circuit.add(H(0))
    circuit.add(S(1))
    circuit.add(T(2))
    circuit.add(X(3))
    circuit.add(Y(0))
    circuit.add(Z(1))
    circuit.add(RX(2, theta=0.1))
    circuit.add(RY(3, theta=0.2))
    circuit.add(RZ(0, phi=0.3))
    circuit.add(U1(1, phi=0.4))
    circuit.add(U2(2, phi=0.5, gamma=0.6))
    circuit.add(U3(3, theta=0.7, phi=0.8, gamma=0.9))
    circuit.add(CNOT(0, 1))
    circuit.add(CZ(2, 3))
    circuit.add(SWAP(1, 2))
    circuit.add(Controlled(0, 2, basic_gate=RY(3, theta=1.1)))
    circuit.add(Adjoint(RX(0, theta=1.2)))
    circuit.add(Exponential(RZ(1, phi=1.3)))
    circuit.add(M(3))

    out = layout_pass.run(circuit)

    assert out.nqubits == 5
    assert len(out.gates) == len(circuit.gates)
    assert all(type(o) is type(i) for o, i in zip(out.gates, circuit.gates))
    assert layout_pass.last_layout is not None
    for in_gate, out_gate in zip(circuit.gates, out.gates):
        mapped_qubits = tuple(layout_pass.last_layout[q] for q in in_gate.qubits)
        assert out_gate.qubits == mapped_qubits
    assert layout_pass.last_layout is not None
    assert len(layout_pass.last_layout) == circuit.nqubits
    assert sorted(layout_pass.last_layout) == sorted(set(layout_pass.last_layout))
    assert layout_pass.last_score is not None
    assert context.initial_layout == layout_pass.last_layout
    assert any(key.startswith("SabreLayoutPass") for key in context.circuits)


def test_sabre_layout_handles_sparse_physical_indices():
    topology = PyGraph()
    topology.add_nodes_from(range(5))
    for node in (3, 1):
        topology.remove_node(node)
    topology.add_edge(0, 2, None)
    topology.add_edge(2, 4, None)
    assert sorted(topology.node_indices()) == [0, 2, 4]

    layout_pass = SabreLayoutPass(topology, num_trials=1, seed=5)

    circuit = Circuit(2)
    circuit.add(CZ(0, 1))

    out = layout_pass.run(circuit)

    assert out.nqubits >= 5
    assert layout_pass.last_layout is not None
    assert set(layout_pass.last_layout).issubset({0, 2, 4})
    assert len(set(layout_pass.last_layout)) == circuit.nqubits
    assert all(q in {0, 2, 4} for gate in out.gates for q in gate.qubits)


def _line_graph(num_nodes: int) -> PyGraph:
    graph = PyGraph()
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1, None)
    return graph


def _circuit_with_cz() -> Circuit:
    circuit = Circuit(3)
    circuit.add(CZ(0, 2))
    return circuit


def test_layout_breaks_when_no_candidates(monkeypatch):
    graph = _line_graph(3)
    layout_pass = SabreLayoutPass(graph, lookahead_size=1, seed=0)
    circuit = _circuit_with_cz()

    # Force all logical qubits onto the same physical node so touched_phys contains a single entry.
    monkeypatch.setattr(layout_pass, "_random_initial_layout", lambda rng, n_logical, phys_nodes: [0, 0, 0])

    layout_pass.run(circuit)
    assert layout_pass.last_layout is not None


def test_layout_handles_empty_extended_set(monkeypatch):
    graph = _line_graph(3)
    layout_pass = SabreLayoutPass(graph, lookahead_size=0, seed=0)
    circuit = _circuit_with_cz()

    layout_pass.run(circuit)

    assert layout_pass.last_layout is not None


def test_layout_tie_break(monkeypatch):
    graph = _line_graph(3)
    layout_pass = SabreLayoutPass(graph, lookahead_size=1, seed=0)
    circuit = _circuit_with_cz()

    # Force candidate swaps to have exactly equal cost
    monkeypatch.setattr(layout_pass, "_cost_front", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(random.Random, "random", lambda self: 0.4)

    layout_pass.run(circuit)

    assert layout_pass.last_layout is not None


def test_sabre_layout_simulation_skips_stale_scheduled_front_entries() -> None:
    graph = _line_graph(3)
    layout_pass = SabreLayoutPass(graph, num_trials=1, seed=0)
    physical_nodes = [0, 1, 2]
    physical_to_dense_index = {node: node for node in physical_nodes}
    distance_matrix = layout_pass._all_pairs_shortest_path_unweighted(graph, physical_nodes, physical_to_dense_index)

    final_layout, score = layout_pass._sabre_simulate_layout(
        [0, 1, 2],
        distance_matrix,
        [(0, 1), (1, 2)],
        [[0], [0, 1], [0, 1]],
        random.Random(0),
        physical_nodes,
        physical_to_dense_index,
    )

    assert final_layout == [0, 1, 2]
    assert score > 0.0


def test_sabre_layout_simulation_breaks_on_degenerate_no_candidate_case() -> None:
    graph = make_graph([], nodes=[0, 1])
    layout_pass = SabreLayoutPass(graph, num_trials=1, seed=0)
    physical_nodes = [0, 1]
    physical_to_dense_index = {node: node for node in physical_nodes}
    distance_matrix = layout_pass._all_pairs_shortest_path_unweighted(graph, physical_nodes, physical_to_dense_index)

    final_layout, score = layout_pass._sabre_simulate_layout(
        [0, 0],
        distance_matrix,
        [(0, 1)],
        [[0], [0]],
        random.Random(0),
        physical_nodes,
        physical_to_dense_index,
    )

    assert final_layout == [0, 0]
    assert math.isclose(score, 0.0)


def test_sabre_layout_simulation_uses_sampled_candidate_when_neighbors_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SampledCandidateUsed(RuntimeError):
        pass

    graph = make_graph([], nodes=[0, 1])
    layout_pass = SabreLayoutPass(graph, num_trials=1, seed=0)
    monkeypatch.setattr(
        layout_pass,
        "_cost_front",
        lambda *args, **kwargs: (_ for _ in ()).throw(SampledCandidateUsed()),
    )

    with pytest.raises(SampledCandidateUsed):
        layout_pass._sabre_simulate_layout(
            [0, 1],
            [[0, 2], [2, 0]],
            [(0, 1)],
            [[0], [0]],
            random.Random(0),
            [0, 1],
            {0: 0, 1: 1},
        )


def test_sabre_layout_cost_front_penalizes_unreachable_interactions() -> None:
    cost = SabreLayoutPass._cost_front(
        {0},
        [0, 1],
        [(0, 1)],
        [[0, SabreLayoutPass._UNREACHABLE_DISTANCE], [SabreLayoutPass._UNREACHABLE_DISTANCE, 0]],
        None,
        {0: 0, 1: 1},
    )

    assert math.isclose(cost, 1e6)


def test_sabre_layout_random_initial_layout_requires_enough_physical_nodes() -> None:
    with pytest.raises(ValueError, match="need ≥ 2"):
        SabreLayoutPass._random_initial_layout(random.Random(0), 2, [0])


def test_sabre_layout_remap_gate_qubits_rejects_unsupported_type() -> None:
    class UnsupportedGate(Gate):
        @property
        def name(self) -> str:
            return "Unsupported"

        @property
        def matrix(self):
            return None

        @property
        def target_qubits(self) -> tuple[int, ...]:
            return (0,)

    with pytest.raises(NotImplementedError, match="UnsupportedGate"):
        SabreLayoutPass._remap_gate_qubits_inplace(UnsupportedGate(), {0: 1})
