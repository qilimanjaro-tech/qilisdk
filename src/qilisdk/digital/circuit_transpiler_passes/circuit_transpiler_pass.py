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

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from loguru import logger

from qilisdk.digital import Adjoint, Circuit, Controlled, Exponential, M
from qilisdk.digital.gates import BasicGate, Modified

if TYPE_CHECKING:
    from qilisdk.digital import Gate
    from qilisdk.digital.types import LayoutMap

    from .transpilation_context import TranspilationContext


class CircuitTranspilerPass(ABC):
    """Base class for non-mutating circuit transpiler passes.

    Returns:
        CircuitTranspilerPass: Instances expose the `run` API required by the transpiler.
    """

    context: TranspilationContext | None = None

    def attach_context(self, context: TranspilationContext) -> None:
        logger.debug("[CircuitTranspilerPass] Attaching context to {}", self.__class__.__name__)
        self.context = context

    def append_circuit_to_context(self, circuit: Circuit) -> None:
        name = self.__class__.__name__
        if self.context is not None:
            key, i = name, 1
            while key in self.context.circuits:
                i += 1
                key = f"{name}_{i}"
            self.context.circuits[key] = circuit
            logger.trace("[CircuitTranspilerPass] Stored circuit in context under key {}", key)

    @staticmethod
    def _retarget_gate(gate: Gate, mapped_qubits: tuple[int, ...]) -> Gate:
        """
        Recreate ``gate`` on ``mapped_qubits`` by deep-copying and remapping indices.

        The method deep-copies the gate object to preserve concrete gate type and all parameters, then updates control and target qubit tuples recursively through wrapped gates.

        Args:
            gate (Gate): Original gate to retarget.
            mapped_qubits (tuple[int, ...]): Physical qubits where the gate should act.

        Returns:
            Gate: Retargeted gate equivalent to ``gate`` but acting on ``mapped_qubits``.
        """
        # A gate's matrix does not depend on which qubits it acts on, so share any cached matrix rather than copying it.
        gates = (gate, gate.basic_gate) if isinstance(gate, Modified) else (gate,)
        memo = {id(matrix): matrix for g in gates if (matrix := g.__dict__.get("matrix")) is not None}
        retargeted = deepcopy(gate, memo)
        qubit_map: LayoutMap = dict(zip(gate.qubits, mapped_qubits, strict=True))
        CircuitTranspilerPass._remap_gate_qubits_inplace(retargeted, qubit_map)
        return retargeted

    @staticmethod
    def _remap_gate_qubits_inplace(gate: Gate, qubit_map: LayoutMap) -> None:
        """Recursively remap control/target qubits of a copied gate.

        Args:
            gate (Gate): Gate object to modify in place.
            qubit_map (LayoutMap): Logical-to-physical mapping for qubits touched by ``gate``.

        Raises:
            NotImplementedError: If the gate type does not provide known internal storage for control or target qubits.
        """
        if isinstance(gate, (BasicGate, M)):
            gate._target_qubits = tuple(qubit_map[logical_qubit] for logical_qubit in gate.target_qubits)  # ruff: ignore[private-member-access]
            return

        if isinstance(gate, Controlled):
            gate._control_qubits = tuple(qubit_map[control_qubit] for control_qubit in gate.control_qubits)  # ruff: ignore[private-member-access]
            CircuitTranspilerPass._remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
            return

        if isinstance(gate, (Adjoint, Exponential)):
            CircuitTranspilerPass._remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
            return

        raise NotImplementedError(
            f"Retargeting not implemented for gate type {type(gate).__name__} with arity {gate.nqubits}"
        )

    @abstractmethod
    def run(self, circuit: Circuit) -> Circuit:
        """Create a new circuit built from `circuit` without mutating the input.

        Args:
            circuit (Circuit): Circuit to be transpiled.
        Returns:
            Circuit: Newly transpiled circuit.
        """
        ...
