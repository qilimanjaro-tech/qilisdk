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
from abc import ABC
from typing import Iterator, Literal, Type

from qilisdk.digital.circuit import Circuit
from qilisdk.analog.hamiltonian import Hamiltonian, PauliOperator
from qilisdk.digital.gates import CNOT, CZ, U1, U2, U3, RX, BasicGate, H, RZ, RZConstant
from qilisdk.yaml import yaml
from qilisdk.core.variables import Parameter
from numpy import pi

Connectivity = Literal["circular", "linear", "full"] | list[tuple[int, int]]
Structure = Literal["grouped", "interposed"]
MixerType = Literal["X"]



class Ansatz(Circuit, ABC):
    """Abstract template for parameterised digital circuits."""

    def __init__(self, nqubits: int) -> None:
        """
        Args:
            nqubits (int): Number of logical qubits in the circuit.
        """
        super().__init__(nqubits=nqubits)


@yaml.register_class
class HardwareEfficientAnsatz(Ansatz):
    """
    Hardware-efficient ansatz with `(layers + 1)` single-qubit blocks and ``layers``
    entangling blocks.

    Example:
        .. code-block:: python

            from qilisdk.digital.ansatz import HardwareEfficientAnsatz
            from qilisdk.digital.gates import U3, CNOT

            ansatz = HardwareEfficientAnsatz(
                nqubits=4,
                layers=3,
                connectivity="linear",
                structure="grouped",
                one_qubit_gate=U3,
                two_qubit_gate=CNOT,
            )
            ansatz.draw()

    Notes:
        ``structure="grouped"`` applies full single-qubit layers followed by entanglers,
        while ``structure="interposed"`` alternates single-qubit updates with entanglers
        per qubit. No measurements are added automatically.
    """

    def __init__(
        self,
        nqubits: int,
        layers: int = 1,
        connectivity: Connectivity = "linear",
        structure: Structure = "grouped",
        one_qubit_gate: Type[U1 | U2 | U3] = U1,
        two_qubit_gate: Type[CZ | CNOT] = CZ,
    ) -> None:
        """
        Args:
            nqubits (int): Number of qubits in the circuit.
            layers (int, optional): Number of entangling layers. Defaults to 1.
            connectivity (Connectivity, optional): Topology used for two-qubit gates.
                Accepts ``"linear"``, ``"circular"``, ``"full"``, or an explicit list of edges.
                Defaults to ``"linear"``.
            structure (Structure, optional): Layout of single- and two-qubit gates within each layer.
                ``"grouped"`` applies all single-qubit gates before the entangler block; ``"interposed"``
                interleaves them per qubit. Defaults to ``"grouped"``.
            one_qubit_gate (Type[U1 | U2 | U3], optional): Parameterised single-qubit gate class. Defaults to :class:`U1`.
            two_qubit_gate (Type[CZ | CNOT], optional): Entangling gate class. Defaults to :class:`CZ`.

        Raises:
            ValueError: If ``layers`` is negative or the connectivity definition is invalid.
        """
        super().__init__(nqubits)

        if layers < 0:
            raise ValueError("layers must be >= 0")

        self._layers = int(layers)
        self._connectivity = tuple(self._normalize_connectivity(connectivity))
        self._structure: Structure = "grouped" if structure.lower() == "grouped" else "interposed"
        self._one_qubit_gate: type[U1 | U2 | U3] = one_qubit_gate
        self._two_qubit_gate: type[CZ | CNOT] = two_qubit_gate

        self._build_circuit()

    @property
    def layers(self) -> int:
        """Number of entangling layers."""
        return self._layers

    @property
    def connectivity(self) -> tuple[tuple[int, int], ...]:
        """Entangling edges as an immutable tuple of (control, target) pairs."""
        return self._connectivity

    @property
    def structure(self) -> Structure:
        """Declared structure ('grouped' or 'interposed')."""
        return self._structure

    @property
    def one_qubit_gate(self) -> type[U1 | U2 | U3]:
        """Single-qubit gate class used for parameterized layers (U1, U2, or U3)."""
        return self._one_qubit_gate

    @property
    def two_qubit_gate(self) -> type[CZ | CNOT]:
        """Two-qubit entangling gate class (CZ or CNOT)."""
        return self._two_qubit_gate

    def _normalize_connectivity(self, connectivity: Connectivity) -> list[tuple[int, int]]:
        """
        Returns:
            list[tuple[int, int]]: a validated list of entangling edges derived from ``connectivity``.

        Raises:
            ValueError: If ``connectivity`` is invalid.
        """
        if isinstance(connectivity, list):
            edges = connectivity
        else:
            kind = connectivity.lower()
            if kind == "full":
                edges = [(i, j) for i in range(self.nqubits) for j in range(i + 1, self.nqubits)]
            elif kind == "circular":
                edges = (
                    []
                    if self.nqubits < 2  # noqa: PLR2004
                    else [(i, i + 1) for i in range(self.nqubits - 1)] + [(self.nqubits - 1, 0)]
                )
            elif kind == "linear":
                edges = [(i, i + 1) for i in range(self.nqubits - 1)]
            else:
                raise ValueError(f"Unrecognized connectivity: {connectivity!r}")

        # basic validation
        for a, b in edges:
            if not (0 <= a < self.nqubits and 0 <= b < self.nqubits):
                raise ValueError(f"Edge {(a, b)} out of range for {self.nqubits} qubits.")
            if a == b:
                raise ValueError(f"Self-edge {(a, b)} is not allowed.")
        return edges

    def _parameter_blocks(self) -> Iterator[dict[str, float]]:
        """Yield dictionaries initialised for each parameterised single-qubit gate in build order."""
        names = tuple(self.one_qubit_gate.PARAMETER_NAMES)
        blocks = (self.layers + 1) * self.nqubits

        zero = dict.fromkeys(names, 0.0)
        for _ in range(blocks):
            # fresh dict each time
            yield dict(zero)

    def _apply_single_qubit(self, qubit: int, parameter_iterator: Iterator[dict[str, float]]) -> None:
        """Apply one parameterised single-qubit gate to ``qubit`` using the next parameter block."""
        params = next(parameter_iterator)
        self.add(self.one_qubit_gate(qubit, **params))

    def _apply_single_qubit_block(self, parameter_iterator: Iterator[dict[str, float]]) -> None:
        """Apply a parameterised gate to every qubit using successive parameter blocks."""
        for qubit in range(self.nqubits):
            params = next(parameter_iterator)
            self.add(self.one_qubit_gate(qubit, **params))

    def _apply_entanglers(self) -> None:
        """Append the entangling block across all connectivity edges."""
        for i, j in self.connectivity:
            self.add(self.two_qubit_gate(i, j))

    def _build_circuit(self) -> None:
        """Populate the circuit according to the current structure and connectivity settings."""
        # Parameter iterator covering all single-qubit blocks, in order
        parameter_iterator = iter(self._parameter_blocks())

        # For each remaining layer: U -> E
        if self.structure == "grouped":
            for _ in range(self.layers):
                self._apply_single_qubit_block(parameter_iterator)
                self._apply_entanglers()
        else:
            for _ in range(self.layers):
                for q in range(self.nqubits):
                    self._apply_single_qubit(q, parameter_iterator)
                    self._apply_entanglers()

@yaml.register_class
class QAOA(Ansatz):
    """
    Quantum Approximate Optimization Algorithm (QAOA) ansatz.

    This ansatz alternates between applying a problem Hamiltonian and a mixer Hamiltonian,
    parameterized by angles alpha and beta, respectively.

    By default, the mixer Hamiltonian is chosen to be a transverse field (X gates on all qubits).

    Example:
        .. code-block:: python

            from qilisdk.digital.ansatz import QAOA

            ansatz = QAOA(
                nqubits=4,
                hamiltonian=your_problem_hamiltonian,
                mixer_type='X',
                layers=3,
            )
            ansatz.draw()

    """

    def __init__(
        self,
        nqubits: int,
        hamiltonian: Hamiltonian,
        layers: int = 1,
        mixer_type: MixerType = "X",
    ) -> None:
        """
        Args:
            nqubits (int): Number of qubits in the circuit.
            hamiltonian (Hamiltonian): The problem Hamiltonian encoding the cost function.
            layers (int, optional): Number of QAOA layers. Defaults to 1.
            mixer_type (str, optional): The type of mixer Hamiltonian to use. Defaults to 'X'.

        Raises:
            ValueError: If ``nqubits`` is non-positive.
            ValueError: If ``layers`` is negative or the connectivity definition is invalid.
            ValueError: If ``hamiltonian`` has no qubits.
            ValueError: If ``nqubits`` does not match the number of qubits in ``hamiltonian``.
        """
        super().__init__(nqubits)

        if layers < 0:
            raise ValueError("layers must be >= 0")

        if hamiltonian.nqubits <= 0:
            raise ValueError("hamiltonian must have at least one qubit")
        
        if hamiltonian.nqubits != nqubits:
            raise ValueError("nqubits must match the number of qubits in hamiltonian")

        self._layers = int(layers)
        self._hamiltonian = hamiltonian
        self._mixer_type = mixer_type

        self._build_circuit()

    @property
    def layers(self) -> int:
        """Number of entangling layers."""
        return self._layers

    @property
    def hamiltonian(self) -> Hamiltonian:
        """The problem Hamiltonian encoding the cost function."""
        return self._hamiltonian

    @property
    def mixer_type(self) -> str:
        """The type of mixer Hamiltonian used ('X' or 'XY')."""
        return self._mixer_type

    def _build_circuit(self) -> None:
        """Populate the circuit according to the Hamiltonian and mixer settings."""

        print("Problem Hamiltonian:")
        print(self.hamiltonian)

        # Split the problem hamiltonian into commuting parts
        commuting_parts = self.hamiltonian.get_commuting_partitions()

        print()
        print("Commuting parts:")
        for i, part in enumerate(commuting_parts):
            print(f" Part {i}:", part)

        def _pauli_evolution(term: tuple[PauliOperator, ...], coeff: float, time: Parameter) -> Iterator[BasicGate]:
            """
                An iterator of parameterized gates performing the evolution of a given Pauli string
            """

            qubit_indices = sorted([pauli.qubit for pauli in term if pauli.name != "I"])
            if len(qubit_indices) == 0:
                return

            # Move everything to Z basis
            for pauli in term:
                q = pauli.qubit
                name = pauli.name
                if name == "X":
                    yield H(q)
                elif name == "Y":
                    yield RX(q, **{"theta": pi / 2})

            # Apply CNOT ladder
            for i in range(len(qubit_indices) - 1):
                yield CNOT(qubit_indices[i], qubit_indices[i + 1])

            # Apply RZ rotation on last qubit
            last_qubit = qubit_indices[-1]
            yield RZConstant(last_qubit, phi=(-2 * coeff).real)
            yield RZ(last_qubit, **{"phi": time})

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
                    yield RX(q, **{"theta": -pi / 2})

        # Applied to the equal superposition initial state |+>
        for qubit in range(self.nqubits):
            self.add(H(qubit))

        # Build the layers
        for _ in range(self.layers):

            # Apply problem Hamiltonian
            alpha_param = Parameter("alpha_" + str(_), 0.0)
            for part in commuting_parts:
                for term, coeff in part.items():
                    for gate in _pauli_evolution(term, coeff, alpha_param):
                        self.add(gate)
                    
            # Apply mixer Hamiltonian
            beta_param = Parameter("beta_" + str(_), 0.0)
            if self.mixer_type == "X":
                for qubit in range(self.nqubits):
                    self.add(RX(qubit, **{"theta": beta_param}))
        
        # Print all gates
        for gate in self.gates:
            print(gate)

        print(self.get_parameters())