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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, TypeAlias, TypeGuard

from rustworkx import PyGraph

from .circuit_transpiler_passes import (
    CancelIdentityPairsPass,
    CircuitTranspilerPass,
    CustomLayoutPass,
    DecomposeMultiControlledGatesPass,
    DecomposeToCanonicalBasisPass,
    FuseSingleQubitGatesPass,
    SabreLayoutPass,
    SabreSwapPass,
    SingleQubitGateBasis,
    TranspilationContext,
    TwoQubitGateBasis,
)

if TYPE_CHECKING:
    from qilisdk.digital import Circuit


def _is_topology_graph(topology: list[tuple[int, int]] | PyGraph[int, None]) -> TypeGuard[PyGraph[int, None]]:
    return isinstance(topology, PyGraph) and all(isinstance(node, int) for node in topology.nodes())


LayoutMap: TypeAlias = dict[int, int]


class TranspilerPassResult:
    """Per-pass transpilation artifact containing pass identity and output circuit."""

    def __init__(self, name: str, circuit: Circuit) -> None:
        self._name = name
        self._circuit = circuit

    @property
    def name(self) -> str:
        """Pass class name."""
        return self._name

    @property
    def circuit(self) -> Circuit:
        """Circuit produced right after this pass."""
        return self._circuit


class CircuitTranspilerResult:
    """Result of a full transpiler run with pass-by-pass diagnostics."""

    def __init__(
        self,
        circuit: Circuit,
        intermediate_results: list[TranspilerPassResult] | None = None,
        layout: LayoutMap | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._circuit: Circuit = circuit
        self._intermediate_results: list[TranspilerPassResult] = (
            list(intermediate_results) if intermediate_results is not None else []
        )
        self._layout: LayoutMap | None = dict(layout) if layout is not None else None
        self._metrics: dict[str, Any] = dict(metrics) if metrics is not None else {}

    @property
    def circuit(self) -> Circuit:
        """Alias for the final transpiled circuit."""
        return self._circuit

    @property
    def intermediate_results(self) -> list[TranspilerPassResult]:
        """Ordered per-pass outputs."""
        return list(self._intermediate_results)

    @property
    def layout(self) -> LayoutMap | None:
        """Final user-facing logical-to-physical mapping captured after the pipeline finishes."""
        return dict(self._layout) if self._layout is not None else None

    @property
    def metrics(self) -> dict[str, Any]:
        """Metrics collected by transpiler passes."""
        return dict(self._metrics)


class CircuitTranspiler:
    """Apply an ordered pipeline of circuit transpilation passes.

    The transpiler acts as a thin orchestrator: each pass receives the circuit from the previous
    pass and must return a brand-new circuit, allowing both structural rewrites and device-specific
    lowering steps to be chained deterministically. Without topology information, the default
    pipeline applies decomposition and cheap local simplification passes. With topology
    information, a richer hardware-aware pipeline can be built through :meth:`default`.

    Args:
        pipeline (list[CircuitTranspilerPass]): Sequential list of passes to execute while transpiling.
    """

    def __init__(self, pipeline: list[CircuitTranspilerPass]) -> None:
        self._pipeline: list[CircuitTranspilerPass] = list(pipeline)
        self._context: TranspilationContext = TranspilationContext()
        self._attach_context_to_pipeline()

    @classmethod
    def default(
        cls,
        single_qubit_basis: SingleQubitGateBasis = SingleQubitGateBasis.U3,
        two_qubit_basis: TwoQubitGateBasis = TwoQubitGateBasis.CNOT,
        topology: list[tuple[int, int]] | PyGraph[int, None] | None = None,
        qubit_mapping: dict[int, int] | None = None,
    ) -> Self:
        if topology is None:
            return cls(
                [
                    DecomposeMultiControlledGatesPass(),
                    CancelIdentityPairsPass(),
                    DecomposeToCanonicalBasisPass(
                        single_qubit_basis=single_qubit_basis, two_qubit_basis=two_qubit_basis
                    ),
                    FuseSingleQubitGatesPass(single_qubit_basis=single_qubit_basis),
                ]
            )

        topology = CircuitTranspiler._build_topology_graph(topology)  # ty:ignore[invalid-argument-type]
        layout_routing_passes: list[CircuitTranspilerPass] = (
            [SabreLayoutPass(topology), SabreSwapPass(topology)]
            if qubit_mapping is None
            else [CustomLayoutPass(topology, qubit_mapping)]
        )

        return cls(
            [
                DecomposeMultiControlledGatesPass(),
                CancelIdentityPairsPass(),
                DecomposeToCanonicalBasisPass(single_qubit_basis=single_qubit_basis, two_qubit_basis=two_qubit_basis),
                FuseSingleQubitGatesPass(single_qubit_basis=single_qubit_basis),
                *layout_routing_passes,
                DecomposeToCanonicalBasisPass(single_qubit_basis=single_qubit_basis, two_qubit_basis=two_qubit_basis),
                FuseSingleQubitGatesPass(single_qubit_basis=single_qubit_basis),
            ]
        )

    def transpile(self, circuit: Circuit) -> CircuitTranspilerResult:
        """Run the pipeline and return pass-by-pass transpilation diagnostics.

        Args:
            circuit (Circuit): Input circuit to transpile.
        Returns:
            CircuitTranspilerRunResult: Final circuit and intermediate pass outputs.
        """
        self._reset_context()

        pass_results: list[TranspilerPassResult] = []
        transpiled_circuit = circuit

        for transpiler_pass in self._pipeline:
            transpiled_circuit = transpiler_pass.run(transpiled_circuit)
            pass_results.append(
                TranspilerPassResult(
                    name=transpiler_pass.__class__.__name__,
                    circuit=transpiled_circuit,
                )
            )

        return CircuitTranspilerResult(
            circuit=transpiled_circuit,
            intermediate_results=pass_results,
            layout=self._context.final_layout,
            metrics=self._context.metrics,
        )

    def _attach_context_to_pipeline(self) -> None:
        for transpiler_pass in self._pipeline:
            transpiler_pass.attach_context(self._context)

    def _reset_context(self) -> None:
        self._context = TranspilationContext()
        self._attach_context_to_pipeline()

    @staticmethod
    def _build_topology_graph(topology: list[tuple[int, int]]) -> PyGraph[int, None]:
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
