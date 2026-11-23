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
from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, TypeGuard

from loguru import logger
from rustworkx import PyGraph, generators
from typing_extensions import override

if TYPE_CHECKING:
    from .types import TopologyLike


_TOPOLOGY_TUPLE_ARITY = 2


def _is_topology_graph(topology: TopologyLike) -> TypeGuard[PyGraph[int, None]]:
    return isinstance(topology, PyGraph) and all(isinstance(node, int) for node in topology.nodes())


def _is_topology_list(topology: TopologyLike) -> TypeGuard[list[tuple[int, int]]]:
    return isinstance(topology, list) and all(
        isinstance(element, tuple)
        and len(element) == _TOPOLOGY_TUPLE_ARITY
        and isinstance(element[0], int)
        and isinstance(element[1], int)
        for element in topology
    )


def build_topology_graph(topology: TopologyLike) -> PyGraph[int, None]:
    """Return a topology as a ``rustworkx.PyGraph``.

    Args:
        topology (TopologyLike): Coupling map provided as a :class:`TopologyPreset`, a list of
            connected physical-qubit pairs or an existing ``PyGraph``.

    Raises:
        ValueError: If the edge list is empty or ``topology`` has an unsupported format.

    Returns:
        PyGraph[int, None]: Coupling map in ``PyGraph`` form.
    """
    if isinstance(topology, TopologyPreset):
        return topology.graph
    if _is_topology_graph(topology):
        logger.debug("[Topology] Using provided PyGraph with {} nodes", len(topology.node_indices()))
        return topology
    if _is_topology_list(topology):
        logger.debug("[Topology] Building topology graph from edge list with {} edges", len(topology))
        graph = PyGraph[int, None]()

        # Collect the physical qubit labels that actually appear in the coupling map.
        active_nodes = {int(qubit) for pair in topology for qubit in pair}
        if not active_nodes:
            raise ValueError("Topology edge list cannot be empty.")

        max_label = max(active_nodes)

        # Add a dense block of nodes so that node indices match physical labels.
        graph.add_nodes_from(range(max_label + 1))
        for physical_qubit_a, physical_qubit_b in topology:
            graph.add_edge(int(physical_qubit_a), int(physical_qubit_b), None)

        # Remove any indices that are not populated in the topology. This keeps
        # rustworkx node indices aligned with the real physical labels.
        missing_nodes = {node for node in range(max_label + 1) if node not in active_nodes}
        for missing_node in sorted(missing_nodes, reverse=True):
            graph.remove_node(missing_node)

        logger.debug(
            "[Topology] Built topology graph with {} nodes and {} edges",
            graph.num_nodes(),
            graph.num_edges(),
        )
        return graph
    raise ValueError("topology has incorrect format.")


def _require_at_least(name: str, value: int, minimum: int) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")


class TopologyPreset:
    """Hardware coupling map: physical qubits and the pairs that support two-qubit gates.

    Build one with the named constructors for common device layouts, or wrap an
    existing coupling map with :meth:`from_coupling_map`. Physical qubits are
    labelled by integers, and every constructor labels them ``0 .. nqubits - 1``.

    Anywhere a topology is accepted (``CircuitTranspiler.default``, ``SabreLayoutPass``,
    ``SabreSwapPass``, ``CustomLayoutPass``) a :class:`TopologyPreset`, a list of qubit pairs
    or a ``rustworkx.PyGraph`` can be passed interchangeably.

    Example:
        .. code-block:: python

            from qilisdk.digital import TopologyPreset

            TopologyPreset.line(8)
            TopologyPreset.ring(8)
            TopologyPreset.grid(rows=4, cols=5)
            TopologyPreset.all_to_all(6)
            TopologyPreset.heavy_hex(distance=3)
            TopologyPreset.from_coupling_map([(0, 1), (1, 2)])
    """

    def __init__(self, coupling_map: TopologyLike) -> None:
        """Wrap a coupling map.

        Args:
            coupling_map (TopologyLike): Another :class:`TopologyPreset`, a list of connected
                physical-qubit pairs, or a ``rustworkx.PyGraph`` whose node payloads are ints.

        Raises:
            ValueError: If the edge list is empty or the coupling map has an unsupported format.
        """
        self._graph: PyGraph[int, None] = build_topology_graph(coupling_map)

    @classmethod
    def from_coupling_map(cls, coupling_map: TopologyLike) -> TopologyPreset:
        """Build a topology from an explicit coupling map.

        Args:
            coupling_map (TopologyLike): A list of connected physical-qubit pairs, a
                ``rustworkx.PyGraph`` or another :class:`TopologyPreset`.

        Returns:
            TopologyPreset: The wrapped coupling map.
        """
        return cls(coupling_map)

    @classmethod
    def line(cls, nqubits: int) -> TopologyPreset:
        """Build a linear chain ``0 - 1 - ... - (nqubits - 1)``.

        Args:
            nqubits (int): Number of qubits, at least 2.

        Returns:
            TopologyPreset: The line topology.

        Raises:
            ValueError: If ``nqubits`` is smaller than 2.
        """
        _require_at_least("nqubits", nqubits, 2)
        return cls([(qubit, qubit + 1) for qubit in range(nqubits - 1)])

    @classmethod
    def ring(cls, nqubits: int) -> TopologyPreset:
        """Build a closed chain, i.e. a line whose last qubit is coupled back to qubit 0.

        Args:
            nqubits (int): Number of qubits, at least 3.

        Returns:
            TopologyPreset: The ring topology.

        Raises:
            ValueError: If ``nqubits`` is smaller than 3.
        """
        _require_at_least("nqubits", nqubits, 3)
        return cls([(qubit, (qubit + 1) % nqubits) for qubit in range(nqubits)])

    @classmethod
    def grid(cls, rows: int, cols: int) -> TopologyPreset:
        """Build a square lattice with nearest-neighbour couplings.

        Qubits are labelled row-major, so qubit ``r * cols + c`` sits at row ``r`` and column ``c``.

        Args:
            rows (int): Number of rows, at least 1.
            cols (int): Number of columns, at least 1.

        Returns:
            TopologyPreset: The grid topology.

        Raises:
            ValueError: If either dimension is smaller than 1 or the grid has fewer than 2 qubits.
        """
        _require_at_least("rows", rows, 1)
        _require_at_least("cols", cols, 1)
        _require_at_least("rows * cols", rows * cols, 2)
        edges = [(r * cols + c, r * cols + c + 1) for r in range(rows) for c in range(cols - 1)]
        edges += [(r * cols + c, (r + 1) * cols + c) for r in range(rows - 1) for c in range(cols)]
        return cls(edges)

    @classmethod
    def all_to_all(cls, nqubits: int) -> TopologyPreset:
        """Build a fully connected topology where every pair of qubits is coupled.

        Args:
            nqubits (int): Number of qubits, at least 2.

        Returns:
            TopologyPreset: The all-to-all topology.

        Raises:
            ValueError: If ``nqubits`` is smaller than 2.
        """
        _require_at_least("nqubits", nqubits, 2)
        return cls(list(combinations(range(nqubits), 2)))

    @classmethod
    def heavy_hex(cls, distance: int) -> TopologyPreset:
        """Build an IBM-style heavy-hex lattice for a code of the given distance.

        The lattice has ``(5 * distance**2 - 2 * distance - 1) / 2`` qubits, e.g. 19 for
        ``distance=3`` and 57 for ``distance=5``. The qubit labelling follows
        ``rustworkx.generators.heavy_hex_graph``.

        Args:
            distance (int): Code distance, an odd integer of at least 3.

        Returns:
            TopologyPreset: The heavy-hex topology.

        Raises:
            ValueError: If ``distance`` is smaller than 3 or even.
        """
        _require_at_least("distance", distance, 3)
        if distance % 2 == 0:
            raise ValueError(f"distance must be odd, got {distance}.")
        return cls(list(generators.heavy_hex_graph(distance).edge_list()))

    @property
    def graph(self) -> PyGraph[int, None]:
        """The coupling map as a ``rustworkx.PyGraph`` whose node indices are the physical qubit labels."""
        return self._graph

    @property
    def nqubits(self) -> int:
        """Number of physical qubits in the topology."""
        return self._graph.num_nodes()

    @property
    def qubits(self) -> list[int]:
        """Physical qubit labels in ascending order."""
        return sorted(self._graph.node_indices())

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Coupled qubit pairs as sorted ``(low, high)`` tuples in ascending order, without duplicates."""
        return sorted({(min(a, b), max(a, b)) for a, b in self._graph.edge_list()})

    def neighbors(self, qubit: int) -> list[int]:
        """Return the qubits coupled to ``qubit``.

        Args:
            qubit (int): Physical qubit label.

        Returns:
            list[int]: Neighbouring qubit labels in ascending order.

        Raises:
            ValueError: If ``qubit`` is not part of the topology.
        """
        if not self._graph.has_node(qubit):
            raise ValueError(f"Qubit {qubit} is not part of the topology.")
        return sorted(self._graph.neighbors(qubit))

    @override
    def __repr__(self) -> str:
        return f"TopologyPreset(nqubits={self.nqubits}, edges={self.edges})"
