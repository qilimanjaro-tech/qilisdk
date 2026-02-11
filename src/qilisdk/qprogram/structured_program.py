# Copyright 2023 Qilimanjaro Quantum Tech
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
# ruff: noqa: SLF001, ANN001
from __future__ import annotations

from collections import deque
from typing import cast

from qilisdk.yaml import yaml

from .blocks import Average, Block, ForLoop
from .exceptions import VariableAllocated
from .variables import FloatVariable, IntVariable, QProgramDomain, QProgramVariable


@yaml.register_class
class VariableInfo:
    def __init__(self) -> None:
        self.is_allocated: bool = False
        self.allocated_by: Block | None = None

    def allocate(self, block: Block) -> None:
        self.is_allocated = True
        self.allocated_by = block

    def free(self) -> None:
        self.is_allocated = False
        self.allocated_by = None


class StructuredProgram:
    """Represents a structured quantum program with various control flow blocks."""

    def __init__(self) -> None:
        """Initializes a StructuredProgram instance, setting up the body, block stack, variables, and buses."""
        self._body: Block = Block()
        self._block_stack: deque[Block] = deque([self._body])
        self._variables: dict[QProgramVariable, VariableInfo] = {}
        self._buses: set[str] = set()

    def _append_to_block_stack(self, block: Block) -> None:
        """Appends a block to the internal block stack.

        Args:
            block (Block): The block to append.
        """
        self._block_stack.append(block)

    def _pop_from_block_stack(self) -> Block:
        """Removes and returns the last block from the block stack.

        Returns:
            Block: The popped block.
        """
        return self._block_stack.pop()

    @property
    def _active_block(self) -> Block:
        """Returns the currently active block on top of the block stack.

        Returns:
            Block: The active block.
        """
        return self._block_stack[-1]

    @property
    def body(self) -> Block:
        """Get the body of the QProgram

        Returns:
            Block: The block of the body
        """
        return self._body

    @property
    def buses(self) -> set[str]:
        """Get the buses.

        Returns:
            set[str]: A set of the names of the buses
        """
        return self._buses

    @property
    def variables(self) -> list[QProgramVariable]:
        """Get the variables.

        Returns:
            list[Variable]: A list of variables
        """
        return list(self._variables)

    def insert_block(self, block: Block) -> None:
        for element in block.elements:
            self._active_block.append(element)

    def block(self) -> _BlockContext:
        """Define a generic block for scoping operations.

        Blocks need to open a scope.

        Returns:
            Block: The block.

        Examples:

            >>> with qp.block() as block:
            >>> # operations that shall be executed in the block
        """
        return StructuredProgram._BlockContext(program=self)

    def for_loop(self, variable: QProgramVariable, start: float, stop: float, step: float = 1) -> _ForLoopContext:
        """Define a for_loop block to iterate values over a variable.

        Blocks need to open a scope.

        Args:
            variable (Variable): The variable to be affected from the loop.
            start (int | float): The start value.
            stop (int | float): The stop value.
            step (int | float, optional): The step value. Defaults to 1.

        Returns:
            Loop: The loop block.

        Examples:

            >>> variable = qp.variable(int)
            >>> with qp.for_loop(variable=variable, start=0, stop=100, step=5)):
            >>> # operations that shall be executed in the for_loop block
        """

        return StructuredProgram._ForLoopContext(program=self, variable=variable, start=start, stop=stop, step=step)

    def average(self, shots: int) -> _AverageContext:
        """Define a measurement loop block with averaging in real time.

        Blocks need to open a scope.

        Args:
            shots (int): The number of measurement shots.

        Returns:
            Average: The average block.

        Examples:

            >>> with qp.average(shots=1000):
            >>> # operations that shall be executed in the average block
        """
        return StructuredProgram._AverageContext(program=self, shots=shots)

    def variable(self, label: str, domain: QProgramDomain, type: type[int | float] | None = None) -> QProgramVariable:
        """Declare a variable.

        Args:
            type (int | float): The type of the variable.

        Raises:
            NotImplementedError: If an unsupported type is provided.
            ValueError: If a scalar variable is declared without a type or a non-scalar variable is declared with a type.

        Returns:
            QProgramVariable: The newly created variable.
        """

        def _int_variable(label: str, domain: QProgramDomain = QProgramDomain.Scalar) -> IntVariable:
            variable = IntVariable(label, domain)
            self._variables[variable] = VariableInfo()
            return variable

        def _float_variable(label: str, domain: QProgramDomain = QProgramDomain.Scalar) -> FloatVariable:
            variable = FloatVariable(label, domain)
            self._variables[variable] = VariableInfo()
            return variable

        if domain is QProgramDomain.Scalar and type is None:
            raise ValueError("You must specify a type in a scalar variable.")
        if domain is not QProgramDomain.Scalar and type is not None:
            raise ValueError("When declaring a variable of a specific domain, its type is inferred by its domain.")

        if domain is QProgramDomain.Scalar:
            if type is int:
                return _int_variable(label, domain)
            if type is float:
                return _float_variable(label, domain)

        if domain == QProgramDomain.Time:
            return _int_variable(label, domain)
        if domain in {QProgramDomain.Frequency, QProgramDomain.Phase, QProgramDomain.Voltage, QProgramDomain.Flux}:
            return _float_variable(label, domain)
        raise NotImplementedError

    class _BlockContext:
        def __init__(self, program: StructuredProgram) -> None:
            self.program = program
            self.block: Block = Block()

        def __enter__(self) -> Block:
            self.program._append_to_block_stack(block=self.block)
            return self.block

        def __exit__(self, exc_type, exc_value, exc_tb) -> None:
            block = self.program._pop_from_block_stack()
            self.program._active_block.append(block)

    class _ForLoopContext(_BlockContext):
        def __init__(
            self,
            program: "StructuredProgram",
            variable: QProgramVariable,
            start: float,
            stop: float,
            step: float,
        ) -> None:
            self.program = program
            self.block: ForLoop = ForLoop(variable=variable, start=start, stop=stop, step=step)

        def __enter__(self) -> ForLoop:
            if self.program._variables[self.block.variable].is_allocated:
                raise VariableAllocated(self.block.variable)
            self.program._variables[self.block.variable].allocate(self.block)
            return cast("ForLoop", super().__enter__())

        def __exit__(self, exc_type, exc_value, exc_tb) -> None:
            self.program._variables[self.block.variable].free()
            super().__exit__(exc_type, exc_value, exc_tb)

    class _AverageContext(_BlockContext):
        def __init__(self, program: StructuredProgram, shots: int) -> None:
            self.program = program
            self.block: Average = Average(shots=shots)
