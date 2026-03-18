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

from typing import TYPE_CHECKING, TypeGuard

from rustworkx import PyGraph

if TYPE_CHECKING:
    from .types import Topology


_TOPOLOGY_TUPLE_ARITY = 2


def _is_topology_graph(topology: list[tuple[int, int]] | PyGraph[int, None]) -> TypeGuard[PyGraph[int, None]]:
    return isinstance(topology, PyGraph) and all(isinstance(node, int) for node in topology.nodes())


def _is_topology_list(topology: list[tuple[int, int]] | PyGraph[int, None]) -> TypeGuard[list[tuple[int, int]]]:
    return isinstance(topology, list) and all(
        isinstance(element, tuple)
        and len(element) == _TOPOLOGY_TUPLE_ARITY
        and isinstance(element[0], int)
        and isinstance(element[1], int)
        for element in topology
    )


def build_topology_graph(topology: Topology) -> PyGraph[int, None]:
    """Return a topology as a ``rustworkx.PyGraph``.

    Args:
        topology (Topology): Coupling map provided either as a list of
            connected physical-qubit pairs or as an existing ``PyGraph``.

    Raises:
        ValueError:

    Returns:
        PyGraph[int, None]: Coupling map in ``PyGraph`` form.
    """
    if _is_topology_graph(topology):
        return topology
    if _is_topology_list(topology):
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

        return graph
    raise ValueError("topology has incorrect format.")
