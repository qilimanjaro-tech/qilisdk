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

from typing import Iterator, TypeAlias

from numpy import pi

from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.core.variables import Domain, Parameter, Term
from qilisdk.digital.gates import CNOT, RX, RZ, BasicGate, H

CommutingParts: TypeAlias = list[dict[tuple[PauliOperator, ...], complex | Term | Parameter]]
TimeParameter: TypeAlias = complex | Term | Parameter
GateIterator: TypeAlias = Iterator[BasicGate | CNOT]


def _pauli_evolution(
    term: tuple[PauliOperator, ...],
    coeff: complex | Term | Parameter,
    time: complex | Term | Parameter,
) -> Iterator[BasicGate | CNOT]:
    """
    An iterator of parameterized gates performing the evolution of a given Pauli string.

    Args:
        term (tuple[PauliOperator, ...]): The Pauli string to evolve under.
        coeff (complex | Term | Parameter): The coefficient of the Pauli string.
        time (complex | Term | Parameter): The evolution time parameter (gamma or alpha).

    Yields:
        Iterator[BasicGate]: Gates implementing the evolution under the Pauli string.
    """

    qubit_indices = [pauli.qubit for pauli in term if pauli.name != "I"]
    if len(qubit_indices) == 0:
        return

    # Move everything to Z basis
    for pauli in term:
        q = pauli.qubit
        name = pauli.name
        if name == "X":
            yield H(q)
        elif name == "Y":
            gate_val = pi / 2
            yield RX(q, theta=Parameter("fixed_" + str(gate_val), gate_val, Domain.REAL, (gate_val, gate_val)))

    # Apply CNOT ladder
    for i in range(len(qubit_indices) - 1):
        yield CNOT(qubit_indices[i], qubit_indices[i + 1])

    # Apply RZ rotation on last qubit
    last_qubit = qubit_indices[-1]
    if isinstance(coeff, complex):
        coeff = coeff.real
    if isinstance(time, complex):
        time = time.real
    yield RZ(last_qubit, phi=(2 * coeff * time))

    # Undo CNOT ladder
    for i in reversed(range(len(qubit_indices) - 1)):
        yield CNOT(qubit_indices[i], qubit_indices[i + 1])

    # Move back from Z basis
    for pauli in term:
        q = pauli.qubit
        name = pauli.name
        if name == "X":
            yield H(q)
        elif name == "Y":
            gate_val = -pi / 2
            yield RX(q, theta=Parameter("fixed_" + str(gate_val), gate_val, Domain.REAL, (gate_val, gate_val)))


def trotter_evolution(
    hamiltonian: Hamiltonian,
    time: TimeParameter,
    trotter_steps: int,
) -> GateIterator:
    """
    An iterator of parameterized gates performing Trotterized evolution of a commuting Hamiltonian part.

    Args:
        hamiltonian (Hamiltonian): Hamiltonian object to be trotterized.
        time (complex | Term | Parameter): The evolution time parameter.
        trotter_steps (int): Number of Trotter steps.

    Yields:
        Iterator[BasicGate]: Gates implementing the Trotterized evolution.
    """
    commuting_parts = hamiltonian.get_commuting_partitions()
    yield from _commuting_trotter_evolution(commuting_parts=commuting_parts, time=time, trotter_steps=trotter_steps)


def _commuting_trotter_evolution(
    commuting_parts: CommutingParts,
    time: TimeParameter,
    trotter_steps: int,
) -> GateIterator:
    """
    An iterator of parameterized gates performing Trotterized evolution of a commuting Hamiltonian part.

    Args:
        commuting_parts (CommutingParts): List of commuting Hamiltonian parts.
        time (TimeParameter): The evolution time parameter.
        trotter_steps (int): Number of Trotter steps.

    Yields:
        Iterator[BasicGate]: Gates implementing the Trotterized evolution.
    """
    for _ in range(trotter_steps):
        for part in commuting_parts:
            for term, coeff in part.items():
                for gate in _pauli_evolution(term, coeff / trotter_steps, time):
                    yield gate
