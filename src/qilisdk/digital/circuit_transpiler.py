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
from qilisdk.digital import Circuit

from .circuit_transpiler_passes import CircuitTranspilerPass, DecomposeMultiControlledGatesPass


class CircuitTranspiler:
    """Apply an ordered pipeline of circuit transpilation passes.

    The transpiler acts as a thin orchestrator: each pass receives the circuit from the previous
    pass and must return a brand-new circuit, allowing both structural rewrites and device-specific
    lowering steps to be chained deterministically. Today the pipeline defaults to a single
    `DecomposeMultiControlledGatesPass`, but the API is designed so additional passes—e.g. layout,
    routing, or hardware-aware optimizers—can be composed in future iterations without changing
    backend code.

    Args:
        pipeline (list[CircuitTranspilerPass] | None): Sequential list of passes to execute while transpiling.
    """

    def __init__(self, pipeline: list[CircuitTranspilerPass] | None = None) -> None:
        self._pipeline = pipeline or [DecomposeMultiControlledGatesPass()]

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
