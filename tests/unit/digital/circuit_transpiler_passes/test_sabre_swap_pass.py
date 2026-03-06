import builtins
import math

import pytest
from rustworkx import PyGraph

from qilisdk.digital import CZ, RX, RY, RZ, Adjoint, Circuit, Exponential, M
from qilisdk.digital.circuit_transpiler_passes.sabre_swap_pass import SabreSwapPass
from qilisdk.digital.circuit_transpiler_passes.transpilation_context import TranspilationContext


def make_line_topology(n: int) -> PyGraph:
    graph = PyGraph()
    graph.add_nodes_from(range(n))
    for i in range(n - 1):
        graph.add_edge(i, i + 1, None)
    return graph


def test_sabre_swap_requires_pygraph():
    with pytest.raises(TypeError, match=r"requires a rustworkx.PyGraph"):
        SabreSwapPass("not-a-graph")  # type: ignore[arg-type]


def test_sabre_swap_rejects_empty_topology_graph():
    topology = PyGraph()
    swap_pass = SabreSwapPass(topology)
    circuit = Circuit(1)
    circuit.add(RX(0, theta=0.3))

    with pytest.raises(ValueError, match="Topology graph has no nodes"):
        swap_pass.run(circuit)


def test_sabre_swap_routes_adjacent_gate_without_swaps_and_updates_context():
    topology = make_line_topology(2)
    swap_pass = SabreSwapPass(topology, seed=4)
    context = TranspilationContext()
    swap_pass.attach_context(context)

    circuit = Circuit(2)
    circuit.add(RY(0, theta=0.2))
    circuit.add(CZ(0, 1))
    circuit.add(M(1))

    out = swap_pass.run(circuit)

    assert [type(g).__name__ for g in out.gates] == ["RY", "CZ", "M"]
    assert [gate.qubits for gate in out.gates] == [(0,), (0, 1), (1,)]
    assert swap_pass.last_swap_count == 0
    assert swap_pass.last_final_layout == [0, 1]
    assert context.final_layout == {0: 0, 1: 1}
    assert context.metrics["swap_count"] == 0
    assert any(name.startswith("SabreSwapPass") for name in context.circuits)


def test_sabre_swap_inserts_swaps_for_non_adjacent_gate():
    topology = make_line_topology(3)
    swap_pass = SabreSwapPass(topology, seed=0)

    circuit = Circuit(3)
    circuit.add(RZ(0, phi=0.1))
    circuit.add(CZ(0, 2))

    out = swap_pass.run(circuit)

    names = [type(g).__name__ for g in out.gates]
    qubits = [gate.qubits for gate in out.gates]
    assert names == ["RZ", "SWAP", "CZ"]
    assert qubits[0] == (0,)
    assert qubits[1] in {(0, 1), (1, 0)}
    assert qubits[2] in {(1, 2), (2, 1)}
    assert swap_pass.last_swap_count == 1
    assert set(swap_pass.last_final_layout or []) == {0, 1, 2}


def test_sabre_swap_respects_custom_initial_layout():
    topology = PyGraph()
    topology.add_nodes_from(range(3))
    topology.add_edge(0, 1, None)
    topology.add_edge(1, 2, None)
    topology.add_edge(0, 2, None)

    swap_pass = SabreSwapPass(topology, initial_layout=[2, 0], seed=2)

    circuit = Circuit(2)
    circuit.add(RX(0, theta=0.5))
    circuit.add(CZ(0, 1))

    out = swap_pass.run(circuit)

    assert next(gate.qubits for gate in out.gates) == (2,)
    assert swap_pass.last_final_layout[:2] == [2, 0]


def test_sabre_swap_rejects_multi_qubit_gate():
    topology = make_line_topology(3)
    swap_pass = SabreSwapPass(topology)

    class FakeGate:
        def __init__(self, qubits):
            self.qubits = qubits
            self.nqubits = len(qubits)

    circuit = Circuit(3)
    circuit._gates = [FakeGate((0, 1, 2))]

    with pytest.raises(NotImplementedError, match="FakeGate"):
        swap_pass.run(circuit)


def test_sabre_swap_raises_when_swap_budget_exceeded():
    topology = make_line_topology(3)
    swap_pass = SabreSwapPass(topology, seed=1, max_swaps_factor=0.0)

    circuit = Circuit(3)
    circuit.add(CZ(0, 2))

    with pytest.raises(RuntimeError, match="Exceeded swap budget"):
        swap_pass.run(circuit)


def test_sabre_swap_handles_sparse_physical_indices_with_default_layout():
    topology = PyGraph()
    topology.add_nodes_from(range(5))
    for node in (3, 1):
        topology.remove_node(node)
    topology.add_edge(0, 2, None)
    topology.add_edge(2, 4, None)
    assert sorted(topology.node_indices()) == [0, 2, 4]

    swap_pass = SabreSwapPass(topology, seed=3)

    circuit = Circuit(2)
    circuit.add(CZ(0, 1))

    out = swap_pass.run(circuit)

    assert out.nqubits >= 5
    assert all(q in {0, 2, 4} for gate in out.gates for q in gate.qubits)
    assert swap_pass.last_final_layout is not None
    assert set(swap_pass.last_final_layout).issubset({0, 2, 4})


def test_sabre_swap_handles_sparse_physical_indices_with_custom_layout():
    topology = PyGraph()
    topology.add_nodes_from(range(5))
    for node in (3, 1):
        topology.remove_node(node)
    topology.add_edge(0, 2, None)
    topology.add_edge(2, 4, None)
    assert sorted(topology.node_indices()) == [0, 2, 4]

    swap_pass = SabreSwapPass(topology, initial_layout=[0, 4], seed=5)

    circuit = Circuit(2)
    circuit.add(CZ(0, 1))

    out = swap_pass.run(circuit)

    assert out.nqubits >= 5
    assert any(type(g).__name__ == "SWAP" for g in out.gates)
    assert all(q in {0, 2, 4} for gate in out.gates for q in gate.qubits)
    assert swap_pass.last_final_layout is not None
    assert set(swap_pass.last_final_layout).issubset({0, 2, 4})
    assert swap_pass.last_swap_count
    assert swap_pass.last_swap_count > 0


def test_sabre_swap_accepts_padding_qubits_after_layout():
    topology = PyGraph()
    topology.add_nodes_from(range(5))
    topology.remove_node(3)
    topology.add_edge(0, 1, None)
    topology.add_edge(1, 2, None)
    topology.add_edge(2, 4, None)
    assert sorted(topology.node_indices()) == [0, 1, 2, 4]

    swap_pass = SabreSwapPass(topology, seed=7)

    circuit = Circuit(5)
    circuit.add(CZ(0, 4))

    out = swap_pass.run(circuit)

    assert out.nqubits >= 5
    assert swap_pass.last_final_layout is not None
    assert len(swap_pass.last_final_layout) == circuit.nqubits
    assert {q for gate in out.gates for q in gate.qubits}.issubset({0, 1, 2, 4})
    assert 3 not in swap_pass.last_final_layout


def _empty_graph(num_nodes: int) -> PyGraph:
    graph = PyGraph()
    graph.add_nodes_from(range(num_nodes))
    return graph


def _chain_graph(num_nodes: int) -> PyGraph:
    graph = _empty_graph(num_nodes)
    for i in range(num_nodes - 1):
        graph.add_edge(i, i + 1, None)
    return graph


def _two_qubit_circuit() -> Circuit:
    circuit = Circuit(2)
    circuit.add(CZ(0, 1))
    return circuit


def test_run_raises_after_all_attempts(monkeypatch):
    swap_pass = SabreSwapPass(_chain_graph(2), max_attempts=5)
    circuit = Circuit(1)  # trivial circuit; loop won't run

    original_max = builtins.max

    def fake_max(*args, **kwargs):
        if len(args) == 2 and args[0] == 1 and isinstance(args[1], int):
            return 0
        return original_max(*args, **kwargs)

    monkeypatch.setattr("builtins.max", fake_max)

    with pytest.raises(RuntimeError, match="SABRE routing failed after 0 attempts"):
        swap_pass.run(circuit)


def test_run_no_candidate_swaps_available(monkeypatch):
    graph = _empty_graph(2)  # no edges
    swap_pass = SabreSwapPass(graph, max_attempts=1)
    circuit = _two_qubit_circuit()

    def fake_init_layout(self, n_logical, phys_nodes, active_qubits, layout_hint):
        return [0, 0]  # force both logical qubits onto the same physical node

    monkeypatch.setattr(SabreSwapPass, "_init_layout", fake_init_layout)

    with pytest.raises(RuntimeError, match="No candidate swaps available; coupling graph likely degenerate"):
        swap_pass.run(circuit)


def test_run_no_swap_candidate_selected(monkeypatch):
    swap_pass = SabreSwapPass(_chain_graph(3), max_attempts=1, initial_layout=[0, 2])
    circuit = _two_qubit_circuit()

    # Keep layout stable so distances never improve.
    monkeypatch.setattr(SabreSwapPass, "_swap_mapping", lambda *args, **kwargs: None)
    monkeypatch.setattr(SabreSwapPass, "_cost_set", lambda *args, **kwargs: math.inf)

    with pytest.raises(RuntimeError, match="SABRE heuristic could not select a swap candidate"):
        swap_pass.run(circuit)


def test_extended_set_zero_budget():
    swap_pass = SabreSwapPass(_chain_graph(2))
    per_qubit = [[0], [0]]
    pos = [0, 0]
    assert swap_pass._extended_set(per_qubit, pos, 0) == set()


def test_sabre_swap_uses_context_layout_hint_when_no_explicit_layout() -> None:
    topology = make_line_topology(2)
    swap_pass = SabreSwapPass(topology, seed=5)
    context = TranspilationContext()
    context.initial_layout = [1, 0]
    swap_pass.attach_context(context)

    circuit = Circuit(2)
    circuit.add(RX(0, theta=0.1))
    circuit.add(CZ(0, 1))

    out = swap_pass.run(circuit)

    assert [gate.qubits for gate in out.gates] == [(1,), (1, 0)]
    assert swap_pass.last_final_layout == [1, 0]
    assert context.final_layout == {0: 1, 1: 0}


def test_sabre_swap_rejects_multi_qubit_gate_after_entering_main_routing_loop() -> None:
    class FakeGate:
        def __init__(self, qubits):
            self.qubits = qubits
            self.nqubits = len(qubits)

    topology = make_line_topology(3)
    swap_pass = SabreSwapPass(topology)
    circuit = Circuit(3)
    circuit._gates = [FakeGate((0, 1, 2)), CZ(0, 2)]

    with pytest.raises(NotImplementedError, match="FakeGate"):
        swap_pass.run(circuit)


def test_sabre_swap_fallback_samples_candidate_edge_when_no_neighbors_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    class SampledCandidateUsed(RuntimeError):
        pass

    graph = _empty_graph(2)
    swap_pass = SabreSwapPass(graph, max_attempts=1)
    circuit = _two_qubit_circuit()

    monkeypatch.setattr(
        swap_pass,
        "_cost_set",
        lambda *args, **kwargs: (_ for _ in ()).throw(SampledCandidateUsed()),
    )

    with pytest.raises(SampledCandidateUsed):
        swap_pass.run(circuit)


def test_front_set_skips_already_scheduled_entries() -> None:
    front_gate_indices = SabreSwapPass._front_set([[0, 1], [0], [1]], [0, 0, 0], [True, False])

    assert front_gate_indices == {1}


def test_extended_set_stops_when_budget_is_exhausted() -> None:
    extended_gate_indices = SabreSwapPass._extended_set([[0, 1, 2], [0, 1]], [0, 0], 1)

    assert extended_gate_indices == {1}


def test_cost_set_penalizes_unreachable_pairs_without_decay() -> None:
    cost = SabreSwapPass._cost_set(
        {0},
        [0, 1],
        [(0, 1)],
        [[0, SabreSwapPass._UNREACHABLE_DISTANCE], [SabreSwapPass._UNREACHABLE_DISTANCE, 0]],
        None,
        {0: 0, 1: 1},
    )

    assert cost == 1e6


def test_init_layout_supports_active_only_hints() -> None:
    layout = SabreSwapPass._init_layout(4, [0, 1, 2, 3], {1, 3}, [2, 0])

    assert layout == [1, 2, 3, 0]


def test_init_layout_supports_prefix_hints_and_placeholder_fill() -> None:
    layout = SabreSwapPass._init_layout(3, [10, 11], {0, 1}, [11])

    assert layout == [11, 10, 10]


def test_init_layout_rejects_unknown_active_physical_nodes() -> None:
    with pytest.raises(ValueError, match="not present in the coupling graph"):
        SabreSwapPass._init_layout(2, [0, 1], {0, 1}, [0, 99])


def test_init_layout_rejects_duplicate_active_targets() -> None:
    with pytest.raises(ValueError, match="unique physical qubits"):
        SabreSwapPass._init_layout(2, [0, 1], {0, 1}, [0, 0])


def test_init_layout_returns_empty_layout_for_empty_identity_case() -> None:
    assert SabreSwapPass._init_layout(0, [], set(), None) == []


def test_init_layout_rejects_too_few_physical_nodes_without_hint() -> None:
    with pytest.raises(ValueError, match="Coupling graph has 2 qubits but circuit needs 3"):
        SabreSwapPass._init_layout(3, [0, 1], {2}, None)


def test_remap_gate_qubits_inplace_handles_wrapped_gates() -> None:
    adjoint_gate = Adjoint(RY(0, theta=0.3))
    exponential_gate = Exponential(RZ(0, phi=0.2))

    SabreSwapPass._remap_gate_qubits_inplace(adjoint_gate, {0: 2})
    SabreSwapPass._remap_gate_qubits_inplace(exponential_gate, {0: 1})

    assert adjoint_gate.qubits == (2,)
    assert exponential_gate.qubits == (1,)
