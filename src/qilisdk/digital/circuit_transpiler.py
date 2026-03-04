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
from typing import Self, TypeGuard

from rustworkx import PyGraph

from qilisdk.digital import Circuit

from .circuit_transpiler_passes import (
    CancelIdentityPairsPass,
    CircuitToCanonicalBasisPass,
    CircuitTranspilerPass,
    CustomLayoutPass,
    DecomposeMultiControlledGatesPass,
    FuseSingleQubitGatesPass,
    SabreLayoutPass,
    SabreSwapPass,
    TranspilationContext,
)


def _is_topology_graph(topology: list[tuple[int, int]] | PyGraph[int, None]) -> TypeGuard[PyGraph[int, None]]:
    return isinstance(topology, PyGraph)


class CircuitTranspiler:
    """Apply an ordered pipeline of circuit transpilation passes.

    The transpiler acts as a thin orchestrator: each pass receives the circuit from the previous
    pass and must return a brand-new circuit, allowing both structural rewrites and device-specific
    lowering steps to be chained deterministically. Today the pipeline defaults to a single
    `DecomposeMultiControlledGatesPass`, but the API is designed so additional passes—e.g. layout,
    routing, or hardware-aware optimizers—can be composed in future iterations without changing
    backend code.

    Args:
        pipeline (list[CircuitTranspilerPass]): Sequential list of passes to execute while transpiling.
    """

    def __init__(self, pipeline: list[CircuitTranspilerPass]) -> None:
        self._pipeline: list[CircuitTranspilerPass] = pipeline
        self._context = TranspilationContext()

        for p in self._pipeline:
            p.attach_context(self._context)

    @classmethod
    def default(cls, topology: list[tuple[int, int]] | PyGraph[int, None], qubit_mapping: dict[int, int] | None = None) -> Self:
        topology = topology if _is_topology_graph(topology) else CircuitTranspiler._build_topology_graph(topology)
        layout_routing_passes: list[CircuitTranspilerPass] = (
            [SabreLayoutPass(topology), SabreSwapPass(topology)]
            if qubit_mapping is None
            else [CustomLayoutPass(topology, qubit_mapping)]
        )

        return CircuitTranspiler([
            DecomposeMultiControlledGatesPass(),
            CancelIdentityPairsPass(),
            CircuitToCanonicalBasisPass(),
            FuseSingleQubitGatesPass(),
            *layout_routing_passes,
            CircuitToCanonicalBasisPass(),
            FuseSingleQubitGatesPass(),
        ])

    def transpile(self, circuit: Circuit) -> Circuit:
        """Run the configured pass pipeline over the provided circuit.

        Args:
            circuit (Circuit): Circuit to be rewritten by the transpiler passes.
        Returns:
            Circuit: The circuit returned by the last pass in the pipeline.
        """
        for transpiler_pass in self._pipeline:
            circuit = transpiler_pass.run(circuit)
        return circuit

    @staticmethod
    def _build_topology_graph(topology: tuple[int, int]) -> PyGraph[int, None]:
        graph = PyGraph[int, None]()

        # Collect the physical qubit labels that actually appear in the coupling map.
        active_nodes = {int(qubit) for pair in topology for qubit in pair}
        max_label = max(active_nodes)

        # Add a dense block of nodes so that node indices match physical labels.
        graph.add_nodes_from(range(max_label + 1))
        for a, b in topology:
            graph.add_edge(int(a), int(b), None)

        # Remove any indices that are not populated in the topology. This keeps
        # rustworkx node indices aligned with the real physical labels.
        for missing in sorted({node for node in range(max_label + 1) if node not in active_nodes}, reverse=True):
            graph.remove_node(missing)
        return graph
