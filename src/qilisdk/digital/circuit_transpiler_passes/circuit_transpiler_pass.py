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

from abc import ABC, abstractmethod

from qilisdk.digital import Circuit

from .transpilation_context import TranspilationContext, TranspilationPassOutput


class CircuitTranspilerPass(ABC):
    """Base class for non-mutating circuit transpiler passes.

    Returns:
        CircuitTranspilerPass: Instances expose the `run` API required by the transpiler.
    """
    context: TranspilationContext | None = None  # injected by CircuitTranspiler

    @abstractmethod
    def run(self, circuit: Circuit) -> Circuit:
        """Create a new circuit built from `circuit` without mutating the input.

        Args:
            circuit (Circuit): Circuit to be transpiled.
        Returns:
            Circuit: Newly transpiled circuit.
        """
        ...

    def attach_context(self, ctx: TranspilationContext) -> None:
        self.context = ctx

    def add_output_to_context(self, circuit: Circuit) -> None:
        name = self.__class__.__name__
        if self.context is not None:
            self.context.outputs.append(TranspilationPassOutput(name, circuit))
