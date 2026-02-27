import pytest
from rustworkx import PyGraph

from qilisdk.digital import CZ, RX, Circuit, M
from qilisdk.digital.circuit_transpiler_passes.custom_layout_pass import CustomLayoutPass
from qilisdk.digital.circuit_transpiler_passes.transpilation_context import TranspilationContext


def _make_topology(n_nodes: int) -> PyGraph:
    graph = PyGraph()
    graph.add_nodes_from(range(n_nodes))
    return graph


def test_custom_layout_routing_inserts_swaps_when_needed():
    topology = _make_topology(3)
    topology.add_edge(0, 1, None)
    topology.add_edge(1, 2, None)

    circuit = Circuit(2)
    circuit.add(RX(0, theta=0.5))
    circuit.add(CZ(0, 1))
    circuit.add(M(1))

    user_mapping = {0: 2, 1: 0}
    layout_pass = CustomLayoutPass(topology, user_mapping)

    context = TranspilationContext()
    layout_pass.attach_context(context)

    output = layout_pass.run(circuit)

    assert output.nqubits == 3
    assert [type(g).__name__ for g in output.gates] == ["RX", "SWAP", "CZ", "SWAP", "M"]
    assert [gate.qubits for gate in output.gates] == [(2,), (1, 0), (2, 1), (1, 0), (0,)]
    assert [gate.qubits for gate in circuit.gates] == [(0,), (0, 1), (1,)]
    assert layout_pass.last_layout == [2, 0]
    assert context.initial_layout == [2, 0]
    assert "CustomLayoutPass" in context.circuits
    assert context.circuits["CustomLayoutPass"] is output


def test_custom_layout_respects_adjacent_mapping_without_swaps():
    topology = _make_topology(3)
    topology.add_edge(0, 1, None)
    topology.add_edge(1, 2, None)

    circuit = Circuit(2)
    circuit.add(RX(0, theta=0.5))
    circuit.add(CZ(0, 1))
    circuit.add(M(1))

    user_mapping = {0: 0, 1: 1}
    layout_pass = CustomLayoutPass(topology, user_mapping)

    output = layout_pass.run(circuit)

    assert [type(g).__name__ for g in output.gates] == ["RX", "CZ", "M"]
    assert [gate.qubits for gate in output.gates] == [(0,), (0, 1), (1,)]
    assert layout_pass.last_layout == [0, 1]


def test_custom_layout_restores_mapping_after_two_qubit_gate():
    topology = _make_topology(3)
    topology.add_edge(0, 1, None)
    topology.add_edge(1, 2, None)

    circuit = Circuit(2)
    circuit.add(CZ(0, 1))
    circuit.add(RX(0, theta=0.7))

    user_mapping = {0: 2, 1: 0}
    layout_pass = CustomLayoutPass(topology, user_mapping)

    output = layout_pass.run(circuit)

    assert [type(g).__name__ for g in output.gates] == ["SWAP", "CZ", "SWAP", "RX"]
    assert [gate.qubits for gate in output.gates] == [(1, 0), (2, 1), (1, 0), (2,)]
    assert layout_pass.last_layout == [2, 0]


def test_custom_layout_mapping_requires_all_logical_qubits():
    topology = _make_topology(2)
    circuit = Circuit(2)

    layout_pass = CustomLayoutPass(topology, {0: 0})

    with pytest.raises(ValueError, match=r"missing logical qubits \[1\]"):
        layout_pass.run(circuit)


def test_custom_layout_mapping_rejects_extra_logical_keys():
    topology = _make_topology(2)
    circuit = Circuit(2)

    layout_pass = CustomLayoutPass(topology, {0: 0, 1: 1, 2: 2})

    with pytest.raises(ValueError, match=r"extraneous logical keys \[2\]"):
        layout_pass.run(circuit)


def test_custom_layout_mapping_must_be_injective():
    topology = _make_topology(2)
    circuit = Circuit(2)

    layout_pass = CustomLayoutPass(topology, {0: 0, 1: 0})

    with pytest.raises(ValueError, match=r"duplicated physical qubits: \[0\]"):
        layout_pass.run(circuit)


def test_custom_layout_rejects_out_of_range_physical_indices():
    topology = _make_topology(3)
    circuit = Circuit(2)

    layout_pass = CustomLayoutPass(topology, {0: 0, 1: 3})

    with pytest.raises(ValueError, match=r"Mapping refers to physical qubits not present in the coupling graph: \[3\]"):
        layout_pass.run(circuit)


def test_custom_layout_raises_when_no_route_possible():
    topology = _make_topology(3)
    topology.add_edge(0, 1, None)
    # Physical qubit 2 is isolated: no edges

    circuit = Circuit(2)
    circuit.add(CZ(0, 1))

    layout_pass = CustomLayoutPass(topology, {0: 0, 1: 2})

    with pytest.raises(ValueError, match=r"no path between physical qubits 0 and 2"):
        layout_pass.run(circuit)


def test_custom_layout_requires_pygraph_topology():
    with pytest.raises(TypeError, match=r"requires a rustworkx.PyGraph"):
        CustomLayoutPass("not a graph", {0: 0})


def test_custom_layout_rejects_empty_topology():
    topology = PyGraph()
    layout_pass = CustomLayoutPass(topology, {0: 0})
    circuit = Circuit(1)

    with pytest.raises(ValueError, match="Coupling graph has no nodes"):
        layout_pass.run(circuit)


def test_custom_layout_raises_for_multi_qubit_gate():
    topology = _make_topology(3)
    topology.add_edge(0, 1, None)
    topology.add_edge(1, 2, None)

    class FakeGate:
        def __init__(self, qubits):
            self.qubits = qubits
            self.nqubits = len(qubits)

    circuit = Circuit(3)
    circuit._gates = [FakeGate((0, 1, 2))]

    layout_pass = CustomLayoutPass(topology, {0: 0, 1: 1, 2: 2})

    with pytest.raises(NotImplementedError, match="FakeGate"):
        layout_pass.run(circuit)
