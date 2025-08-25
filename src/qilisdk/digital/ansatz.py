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
from typing import Iterator, Literal, Sequence, Type

from qilisdk.digital.circuit import Circuit
from qilisdk.digital.gates import CNOT, CZ, U1, U2, U3
from qilisdk.yaml import yaml

Connectivity = Literal["circular", "linear", "full"] | list[tuple[int, int]]
Structure = Literal["grouped", "interposed"]


class Ansatz(Circuit, ABC):
    def __init__(self, nqubits: int) -> None:
        super().__init__(nqubits=nqubits)


@yaml.register_class
class HardwareEfficientAnsatz(Ansatz):
    def __init__(
        self,
        nqubits: int,
        layers: int = 1,
        connectivity: Connectivity = "linear",
        structure: Structure = "grouped",
        one_qubit_gate: Type[U1 | U2 | U3] = U1,
        two_qubit_gate: Type[CZ | CNOT] = CZ,
        initial_parameters: Sequence[float] | None = None,
    ) -> None:
        """Constructs a hardware-efficient ansatz circuit for variational quantum algorithms.

        The ansatz is composed of multiple layers of parameterized single-qubit gates
        and two-qubit entangling gates, arranged according to the specified connectivity
        and structural strategy.

        Args:
            nqubits (int): The number of qubits in the circuit.
            layers (int, optional): Number of repeating layers of gates.. Defaults to 1.
            connectivity (Literal["circular", "linear", "full"] | list[tuple[int, int]], optional): Topology of qubit connectivity. Options include:
                - 'circular': Qubits form a closed loop.
                - 'linear' : Qubits are connected in a line.
                - 'full'   : All-to-all connectivity between qubits
                Defaults to "linear".
            structure (Literal["grouped", "interposed"], optional): Structure of each layer. Options include:
                - 'grouped'   : Applies all single-qubit gates followed by all two-qubit gates.
                - 'interposed': Interleaves single- and two-qubit gates.
                Defaults to "grouped".
            one_qubit_gate (Type[U1] | Type[U2] | Type[U3], optional): the single-qubit gate class name to be used
                as parameterized gates (e.g., `U1`, `U2`, or `U3`). Defaults to U1.
            two_qubit_gate (Type[CZ] | Type[CNOT], optional):  the two-qubit gate class name used for
                entanglement (e.g., `CNOT` or `CZ`). Defaults to CZ.

        Raises:
            ValueError: If an unsupported connectivity or structure type is specified.

        Example:

        .. code-block:: python

            from qilisdk.digital.ansatz import HardwareEfficientAnsatz
            from qilisdk.digital.gates import U3, CNOT

            ansatz = HardwareEfficientAnsatz(
                num_qubits=4,
                layers=3,
                connectivity="linear",
                structure="grouped",
                on_qubit_gates=U3,
                two_qubit_gates=CNOT,
            )
            ansatz.draw()
        """
        super().__init__(nqubits)

        if layers < 0:
            raise ValueError("layers must be >= 0")

        self._layers = int(layers)
        self._connectivity = self._normalize_connectivity(connectivity)
        self._structure: Structure = "grouped" if structure.lower() == "grouped" else "interposed"
        self._one_qubit_gate: type[U1 | U2 | U3] = one_qubit_gate
        self._two_qubit_gate: type[CZ | CNOT] = two_qubit_gate

        self._build_circuit(initial_parameters)

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
        if isinstance(connectivity, list):
            edges = connectivity
        else:
            kind = connectivity.lower()
            if kind == "full":
                edges = [(i, j) for i in range(self.nqubits) for j in range(i + 1, self.nqubits)]
            elif kind == "circular":
                edges = [] if self.nqubits < 2 else [(i, i + 1) for i in range(self.nqubits - 1)] + [(self.nqubits - 1, 0)]  # noqa: PLR2004
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

    def _param_blocks(self, initial_parameters: Sequence[float] | None) -> Iterator[dict[str, float]]:
        names = tuple(self.one_qubit_gate.PARAMETER_NAMES)
        per_gate = len(names)
        blocks = (self.layers + 1) * self.nqubits
        required = per_gate * blocks

        if initial_parameters is None:
            zero = dict.fromkeys(names, 0.0)
            for _ in range(blocks):
                # fresh dict each time
                yield dict(zero)
            return

        values = list(initial_parameters)
        if len(values) != required:
            raise ValueError(
                f"Expected exactly {required} parameters "
                f"({blocks} applications x {per_gate} per gate), got {len(values)}."
            )

        it = iter(values)
        for _ in range(blocks):
            vals = [next(it) for _ in range(per_gate)]
            yield dict(zip(names, vals))

    def _apply_single_qubit_block(self, param_block_iter: Iterator[dict[str, float]]) -> None:
        for q in range(self.nqubits):
            params = next(param_block_iter)
            self.add(self.one_qubit_gate(q, **params))

    def _apply_entanglers(self) -> None:
        for i, j in self.connectivity:
            self.add(self.two_qubit_gate(i, j))

    def _build_circuit(self, initial_parameters: Sequence[float] | None) -> None:
        # Parameter iterator covering all single-qubit blocks, in order
        param_iter = iter(self._param_blocks(initial_parameters))

        # U(0)
        self._apply_single_qubit_block(param_iter)

        # For each remaining layer: E -> U
        for _ in range(self.layers):
            self._apply_entanglers()
            self._apply_single_qubit_block(param_iter)
