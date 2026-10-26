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

from copy import deepcopy
from typing import TYPE_CHECKING

from qilisdk.digital import Adjoint, Controlled, Exponential, Gate, M
from qilisdk.digital.gates import BasicGate

if TYPE_CHECKING:
    from qilisdk.digital.types import LayoutMap


def _retarget_gate(gate: Gate, mapped_qubits: tuple[int, ...]) -> Gate:
    """
    Recreate ``gate`` on ``mapped_qubits`` by deep-copying and remapping indices.

    The gate is deep-copied to preserve its concrete type and parameters, then its control and target qubit tuples are updated recursively through wrapped gates.

    Args:
        gate (Gate): Original gate to retarget.
        mapped_qubits (tuple[int, ...]): Physical qubits where the gate should act, in the order of ``gate.qubits``.

    Returns:
        Gate: Retargeted gate equivalent to ``gate`` but acting on ``mapped_qubits``.
    """
    retargeted = deepcopy(gate)
    qubit_map: LayoutMap = dict(zip(gate.qubits, mapped_qubits, strict=True))
    _remap_gate_qubits_inplace(retargeted, qubit_map)
    return retargeted


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
        _remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
        return

    if isinstance(gate, (Adjoint, Exponential)):
        _remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
        return

    raise NotImplementedError(
        f"Retargeting not implemented for gate type {type(gate).__name__} with arity {gate.nqubits}"
    )
