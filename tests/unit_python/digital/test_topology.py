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
import pytest
from rustworkx import PyGraph, is_connected

from qilisdk.digital import CNOT, Circuit, CircuitTranspiler, H, Topology
from qilisdk.digital.circuit_transpiler_passes import CustomLayoutPass, SabreLayoutPass, SabreSwapPass
from qilisdk.digital.topology import build_topology_graph


def test_line() -> None:
    topology = Topology.line(4)
    assert topology.nqubits == 4
    assert topology.qubits == [0, 1, 2, 3]
    assert topology.edges == [(0, 1), (1, 2), (2, 3)]


def test_ring() -> None:
    topology = Topology.ring(4)
    assert topology.nqubits == 4
    assert topology.edges == [(0, 1), (0, 3), (1, 2), (2, 3)]
    assert topology.neighbors(0) == [1, 3]


def test_grid() -> None:
    topology = Topology.grid(rows=2, cols=3)
    assert topology.nqubits == 6
    assert topology.edges == [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]
    assert Topology.grid(3, 3).neighbors(4) == [1, 3, 5, 7]


@pytest.mark.parametrize(("rows", "cols"), [(4, 5), (1, 6), (6, 1)])
def test_grid_edge_count(rows: int, cols: int) -> None:
    topology = Topology.grid(rows, cols)
    assert topology.nqubits == rows * cols
    assert len(topology.edges) == rows * (cols - 1) + cols * (rows - 1)


def test_all_to_all() -> None:
    topology = Topology.all_to_all(6)
    assert topology.nqubits == 6
    assert len(topology.edges) == 15
    assert topology.neighbors(2) == [0, 1, 3, 4, 5]


@pytest.mark.parametrize(("distance", "nqubits"), [(3, 19), (5, 57), (7, 115)])
def test_heavy_hex(distance: int, nqubits: int) -> None:
    topology = Topology.heavy_hex(distance=distance)
    assert topology.nqubits == nqubits
    assert topology.qubits == list(range(nqubits))
    assert is_connected(topology.graph)
    degrees = [len(topology.neighbors(qubit)) for qubit in topology.qubits]
    assert max(degrees) == 3
    assert min(degrees) == 1


def test_from_coupling_map_edge_list_keeps_sparse_labels() -> None:
    topology = Topology.from_coupling_map([(0, 2), (2, 5)])
    assert topology.nqubits == 3
    assert topology.qubits == [0, 2, 5]
    assert topology.edges == [(0, 2), (2, 5)]


def test_from_coupling_map_wraps_graph_and_topology() -> None:
    graph = build_topology_graph([(0, 1), (1, 2)])
    topology = Topology.from_coupling_map(graph)
    assert topology.graph is graph
    assert Topology(topology).graph is graph


def test_edges_are_deduplicated_and_normalised() -> None:
    assert Topology([(1, 0), (0, 1), (2, 1)]).edges == [(0, 1), (1, 2)]


def test_build_topology_graph_unwraps_topology() -> None:
    topology = Topology.line(3)
    assert build_topology_graph(topology) is topology.graph
    assert isinstance(topology.graph, PyGraph)


def test_repr() -> None:
    assert repr(Topology.line(3)) == "Topology(nqubits=3, edges=[(0, 1), (1, 2)])"


def test_neighbors_rejects_unknown_qubit() -> None:
    topology = Topology.from_coupling_map([(0, 2)])
    with pytest.raises(ValueError, match="Qubit 1 is not part of the topology"):
        topology.neighbors(1)


@pytest.mark.parametrize(
    ("constructor", "args", "message"),
    [
        (Topology.line, (1,), "nqubits must be at least 2"),
        (Topology.ring, (2,), "nqubits must be at least 3"),
        (Topology.grid, (0, 3), "rows must be at least 1"),
        (Topology.grid, (3, 0), "cols must be at least 1"),
        (Topology.grid, (1, 1), r"rows \* cols must be at least 2"),
        (Topology.all_to_all, (1,), "nqubits must be at least 2"),
        (Topology.heavy_hex, (1,), "distance must be at least 3"),
        (Topology.heavy_hex, (4,), "distance must be odd"),
    ],
)
def test_constructors_reject_invalid_sizes(constructor, args, message) -> None:
    with pytest.raises(ValueError, match=message):
        constructor(*args)


def test_rejects_malformed_coupling_map() -> None:
    malformed = [(0, 1, 2)]
    with pytest.raises(ValueError, match="topology has incorrect format"):
        Topology(malformed)  # ty: ignore[invalid-argument-type]


def test_passes_accept_topology() -> None:
    topology = Topology.line(3)
    assert SabreLayoutPass(topology).topology is topology.graph
    assert SabreSwapPass(topology).topology is topology.graph
    assert CustomLayoutPass(topology, {0: 0}).topology is topology.graph


def test_default_transpiler_routes_onto_grid() -> None:
    topology = Topology.grid(2, 3)
    circuit = Circuit(5)
    circuit.add(H(0))
    for target in range(1, 5):
        circuit.add(CNOT(0, target))

    result = CircuitTranspiler.default(topology=topology).transpile(circuit)

    for gate in result.circuit.gates:
        if len(gate.qubits) == 2:
            assert topology.graph.has_edge(*gate.qubits)
