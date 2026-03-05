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
    DecomposeToCanonicalBasisPass,
    CircuitTranspilerPass,
    CustomLayoutPass,
    DecomposeMultiControlledGatesPass,
    FuseSingleQubitGatesPass,
    SabreLayoutPass,
    SabreSwapPass,
    TranspilationContext,
)

if TYPE_CHECKING:
    from qilisdk.digital import Circuit


def _is_topology_graph(topology: list[tuple[int, int]] | PyGraph[int, None]) -> TypeGuard[PyGraph[int, None]]:
    return isinstance(topology, PyGraph)


LayoutMap: TypeAlias = dict[int, int]


class TranspilationLayoutResult:
    """Immutable-style view over layout diagnostics produced by transpiler passes."""

    def __init__(self, initial_layout: LayoutMap | None = None, final_layout: LayoutMap | None = None) -> None:
        self._initial_layout: LayoutMap | None = dict(initial_layout) if initial_layout else None
        self._final_layout: LayoutMap | None = dict(final_layout) if final_layout else None

    @property
    def initial_layout(self) -> LayoutMap | None:
        """Logical-to-physical mapping at the end of the layout stage, if available."""
        return dict(self._initial_layout) if self._initial_layout is not None else None

    @property
    def final_layout(self) -> LayoutMap | None:
        """Logical-to-physical mapping at the end of routing, if available."""
        return dict(self._final_layout) if self._final_layout is not None else None


class TranspilerPassResult:
    """Per-pass transpilation artifact containing pass identity and output circuit."""

    def __init__(
        self,
        pass_name: str,
        transpiled_circuit: Circuit,
        layout: TranspilationLayoutResult | None = None,
    ) -> None:
        self._pass_name = pass_name
        self._transpiled_circuit = transpiled_circuit
        self._layout = layout if layout is not None else TranspilationLayoutResult()

    @property
    def pass_name(self) -> str:
        """Concrete pass class name."""
        return self._pass_name

    @property
    def transpiled_circuit(self) -> Circuit:
        """Circuit produced right after this pass."""
        return self._transpiled_circuit

    @property
    def layout(self) -> TranspilationLayoutResult:
        """Layout snapshot captured after this pass."""
        return self._layout


class CircuitTranspilerRunResult:
    """Result of a full transpiler run with pass-by-pass diagnostics."""

    def __init__(
        self,
        final_circuit: Circuit,
        pass_results: list[TranspilerPassResult] | None = None,
        layout: TranspilationLayoutResult | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._final_circuit: Circuit = final_circuit
        self._pass_results: list[TranspilerPassResult] = list(pass_results) if pass_results is not None else []
        self._layout: TranspilationLayoutResult = layout if layout is not None else TranspilationLayoutResult()
        self._metrics: dict[str, Any] = dict(metrics) if metrics is not None else {}

    @property
    def circuit(self) -> Circuit:
        """Alias for the final transpiled circuit."""
        return self._final_circuit

    @property
    def final_circuit(self) -> Circuit:
        """Circuit returned by the last transpiler pass."""
        return self._final_circuit

    @property
    def pass_results(self) -> list[TranspilerPassResult]:
        """Ordered per-pass outputs."""
        return list(self._pass_results)

    @property
    def layout(self) -> TranspilationLayoutResult:
        """Final layout snapshot captured after the pipeline finishes."""
        return self._layout

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
        topology: list[tuple[int, int]] | PyGraph[int, None] | None = None,
        qubit_mapping: dict[int, int] | None = None,
    ) -> Self:
        if topology is None:
            return cls(
                [
                    DecomposeMultiControlledGatesPass(),
                    CancelIdentityPairsPass(),
                    DecomposeToCanonicalBasisPass(),
                    FuseSingleQubitGatesPass(),
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
                DecomposeToCanonicalBasisPass(),
                FuseSingleQubitGatesPass(),
                *layout_routing_passes,
                DecomposeToCanonicalBasisPass(),
                FuseSingleQubitGatesPass(),
            ]
        )

    def run(self, circuit: Circuit) -> CircuitTranspilerRunResult:
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
                    pass_name=transpiler_pass.__class__.__name__,
                    transpiled_circuit=transpiled_circuit,
                    layout=self._build_layout_result(),
                )
            )

        return CircuitTranspilerRunResult(
            final_circuit=transpiled_circuit,
            pass_results=pass_results,
            layout=self._build_layout_result(),
            metrics=self._context.metrics,
        )

    def transpile(self, circuit: Circuit) -> Circuit:
        """Run the configured pass pipeline over the provided circuit.

        Args:
            circuit (Circuit): Circuit to be rewritten by the transpiler passes.
        Returns:
            Circuit: The circuit returned by the last pass in the pipeline.
        """
        return self.run(circuit).final_circuit

    def _attach_context_to_pipeline(self) -> None:
        for transpiler_pass in self._pipeline:
            transpiler_pass.attach_context(self._context)

    def _reset_context(self) -> None:
        self._context = TranspilationContext()
        self._attach_context_to_pipeline()

    def _build_layout_result(self) -> TranspilationLayoutResult:
        initial_layout = CircuitTranspiler._initial_layout_to_mapping(self._context.initial_layout)
        final_layout = CircuitTranspiler._normalized_final_layout(self._context.final_layout)
        return TranspilationLayoutResult(initial_layout=initial_layout, final_layout=final_layout)

    @staticmethod
    def _initial_layout_to_mapping(initial_layout: list[int]) -> LayoutMap | None:
        if not initial_layout:
            return None
        return dict(enumerate(initial_layout))

    @staticmethod
    def _normalized_final_layout(final_layout: dict[int, int]) -> LayoutMap | None:
        if not final_layout:
            return None
        return {int(logical_qubit): int(physical_qubit) for logical_qubit, physical_qubit in final_layout.items()}

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
