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
    """Apply a configurable pipeline of passes to digital circuits.

    Args:
        pipeline (list[CircuitTranspilerPass] | None): Sequential list of passes to execute while transpiling.

    Returns:
        CircuitTranspiler: New transpiler instance bound to the provided pipeline.
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
