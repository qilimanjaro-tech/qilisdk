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

from typing import TYPE_CHECKING, Any, Self

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
from .topology import build_topology_graph

if TYPE_CHECKING:
    from qilisdk.digital import Circuit

    from .types import LayoutMap, Topology


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
    """Orchestrate an ordered sequence of circuit transpilation passes.

    Instantiate this class directly when you need full control over the pass
    pipeline. For common workflows, use :meth:`default` to build a standard
    pipeline for basis conversion only or for topology-aware transpilation.

    Each pass receives the circuit produced by the previous pass and must return
    a new circuit, which makes the overall transpilation flow deterministic and
    easy to inspect through :meth:`transpile`.

    Args:
        pipeline (list[CircuitTranspilerPass]): Ordered passes to run during transpilation.
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
        topology: Topology | None = None,
        qubit_mapping: LayoutMap | None = None,
    ) -> Self:
        """Build the default transpiler pipeline.

        Use this constructor for the standard transpilation flow. When
        ``topology`` is omitted, the returned transpiler only performs generic
        decomposition and simplification passes. When ``topology`` is provided,
        the pipeline becomes hardware aware: it either computes a placement with
        SABRE and routes the circuit automatically, or it enforces a
        user-provided logical-to-physical layout through ``qubit_mapping``.

        Args:
            single_qubit_basis (SingleQubitGateBasis): Target single-qubit basis
                used by canonical decomposition and final single-qubit fusion.
            two_qubit_basis (TwoQubitGateBasis): Target two-qubit entangling gate
                used by canonical decomposition.
            topology (Topology | None): Optional hardware coupling map. It can
                be provided either as a list of connected physical-qubit pairs
                or as a ``rustworkx`` ``PyGraph`` whose node indices are
                physical qubits.
            qubit_mapping (LayoutMap | None): Optional logical-to-physical
                qubit assignment. This argument is only used when ``topology`` is
                provided. If omitted, the pipeline uses ``SabreLayoutPass`` and
                ``SabreSwapPass``. If provided, the pipeline uses
                ``CustomLayoutPass`` to preserve the requested mapping.

        Returns:
            CircuitTranspiler: A ``CircuitTranspiler`` instance configured with the default pass
            sequence for the requested transpilation mode.
        """
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

        topology = build_topology_graph(topology)
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
