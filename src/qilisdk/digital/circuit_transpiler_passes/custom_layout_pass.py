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

from collections import deque
from typing import TYPE_CHECKING, Mapping

from rustworkx import PyGraph

from qilisdk.digital import CNOT, CZ, RX, RY, RZ, SWAP, U3, Circuit, Gate, M

from .circuit_transpiler_pass import CircuitTranspilerPass

if TYPE_CHECKING:
    from qilisdk.digital.circuit_transpiler import LayoutMap


class CustomLayoutPass(CircuitTranspilerPass):
    """
    Apply a user-specified initial layout (logical→physical mapping) and retarget
    all gates accordingly. The returned circuit is resized to the device size
    (i.e., the number of qubits of the coupling graph).

    Parameters
    ----------
    topology : PyGraph
        Undirected coupling graph whose node indices represent *physical* qubits.
        Nodes should be 0..(n_physical-1). Edges indicate allowed 2Q interactions.
    mapping : LayoutMap
        Logical→physical mapping provided by the user. For example, for a 2-qubit
        circuit: {0: 5, 1: 2} means logical q0→phys 5, logical q1→phys 2.

    Behavior
    --------
    - Validates that `mapping` covers *every* logical qubit in the input circuit,
      is injective (no repeated physical indices), and only references physical
      qubits that exist in `topology`.
    - Stores the user-requested layout in `context.initial_layout` (same field
      set by `SabreLayoutPass`).
    - Inserts SWAPs (with corresponding un-swaps) along shortest paths of the
      coupling graph when a user-requested 2Q interaction would otherwise
      violate connectivity, keeping logical qubits on their requested physical
      nodes for subsequent operations.
    - Returns a *new* `Circuit` whose `nqubits` equals the chip size
      (len of topology's qubits) and whose gates are retargeted to the mapped
      physical qubits.
    - Exposes `last_layout` (list[int]) for diagnostics, mirroring SabreLayout.

    Notes
    -----
    * SWAPs are emitted eagerly using shortest paths whenever a 2Q gate is not
      locally executable on the chosen mapping, and are undone immediately so the
      mapping remains stable for subsequent operations.
    * ``last_layout`` matches the user-provided mapping; the same list is stored
      in ``context.initial_layout`` for diagnostics.
    """

    _TWO_QUBIT_ARITY = 2

    def __init__(self, topology: PyGraph[int, None], layout: LayoutMap) -> None:
        """Initialize the pass with a fixed logical-to-physical layout.

        Args:
            topology (PyGraph[int, None]): Undirected coupling graph whose node indices are physical qubits.
            layout (LayoutMap): User-specified layout ``logical -> physical``.

        Raises:
            TypeError: If ``topology`` is not a ``rustworkx.PyGraph`` instance.
        """
        if not isinstance(topology, PyGraph):
            raise TypeError("CustomLayoutPass requires a rustworkx.PyGraph (undirected).")
        self.topology = topology
        # Store a copy with explicit int coercion
        self._user_layout: LayoutMap = {int(k): int(v) for k, v in layout.items()}

    def run(self, circuit: Circuit) -> Circuit:
        """Retarget a circuit using the user layout, inserting temporary SWAPs when needed.

        Args:
            circuit (Circuit): Logical circuit to retarget.

        Returns:
            Circuit: New circuit in physical qubit indices sized to the topology.

        Raises:
            ValueError: If the topology is empty, mapping coverage is invalid, mapping is not injective, mapping targets unknown physical qubits, or a two-qubit interaction is unroutable.
            NotImplementedError: If a gate with arity greater than two is encountered.
            RuntimeError: If routing swaps fail to make a two-qubit gate adjacent.
        """
        num_logical_qubits = circuit.nqubits

        physical_nodes = [int(node_index) for node_index in self.topology.node_indices()]
        if not physical_nodes:
            raise ValueError("Coupling graph has no nodes.")
        # As in SabreLayoutPass: assume nodes are 0..(n_physical-1)
        num_physical_qubits = max(physical_nodes) + 1
        valid_physical_nodes = set(physical_nodes)

        # ---- validations on provided mapping ----
        mapping_keys = set(self._user_layout.keys())
        expected_logical_qubits = set(range(num_logical_qubits))
        if mapping_keys != expected_logical_qubits:
            missing_logical_qubits = sorted(expected_logical_qubits - mapping_keys)
            extraneous_logical_qubits = sorted(mapping_keys - expected_logical_qubits)
            error_parts = []
            if missing_logical_qubits:
                error_parts.append(f"missing logical qubits {missing_logical_qubits}")
            if extraneous_logical_qubits:
                error_parts.append(f"extraneous logical keys {extraneous_logical_qubits}")
            raise ValueError(
                "User mapping must map *every* logical qubit in the circuit exactly once; " + "; ".join(error_parts)
            )

        mapped_physical_qubits = list(self._user_layout.values())
        if len(set(mapped_physical_qubits)) != len(mapped_physical_qubits):
            # find duplicates for a clearer error
            seen_physical_qubits: set[int] = set()
            duplicate_physical_qubits: list[int] = []
            for physical_qubit in mapped_physical_qubits:
                if physical_qubit in seen_physical_qubits:
                    duplicate_physical_qubits.append(physical_qubit)
                else:
                    seen_physical_qubits.add(physical_qubit)
            duplicate_physical_qubits.sort()
            raise ValueError(f"User mapping is not injective; duplicated physical qubits: {duplicate_physical_qubits}")

        invalid_physical_qubits = sorted(set(mapped_physical_qubits) - valid_physical_nodes)
        if invalid_physical_qubits:
            raise ValueError(
                f"Mapping refers to physical qubits not present in the coupling graph: {invalid_physical_qubits}"
            )

        # Build the layout list[int] where layout[logical] = physical and keep a copy for diagnostics
        logical_to_physical_layout = [self._user_layout[logical_qubit] for logical_qubit in range(num_logical_qubits)]
        initial_layout = logical_to_physical_layout.copy()

        # Track which logical qubit currently occupies each physical node
        inverse_layout: list[int | None] = [None] * num_physical_qubits
        for logical_qubit, physical_qubit in enumerate(logical_to_physical_layout):
            inverse_layout[physical_qubit] = logical_qubit

        # Retarget gates, inserting SWAPs along shortest paths whenever a 2Q interaction
        # would otherwise violate the coupling constraints. SWAPs are added in pairs so
        # that the logical-to-physical mapping is restored after each routed 2Q gate.
        output_circuit = Circuit(num_physical_qubits)

        for gate in circuit.gates:
            gate_qubits = gate.qubits

            if len(gate_qubits) <= 1 or isinstance(gate, M):
                mapped_qubits = tuple(logical_to_physical_layout[logical_qubit] for logical_qubit in gate_qubits)
                output_circuit.add(self._retarget_gate(gate, mapped_qubits))
                continue

            if len(gate_qubits) != CustomLayoutPass._TWO_QUBIT_ARITY:
                raise NotImplementedError(
                    f"CustomLayoutPass currently supports routing for 1Q/2Q gates only; "
                    f"received {type(gate).__name__} acting on {len(gate_qubits)} qubits."
                )

            logical_qubit_a, logical_qubit_b = gate_qubits

            if self.topology.has_edge(
                logical_to_physical_layout[logical_qubit_a],
                logical_to_physical_layout[logical_qubit_b],
            ):
                mapped_qubits = (
                    logical_to_physical_layout[logical_qubit_a],
                    logical_to_physical_layout[logical_qubit_b],
                )
                output_circuit.add(self._retarget_gate(gate, mapped_qubits))
                continue

            routing_path = self._shortest_path(
                logical_to_physical_layout[logical_qubit_a],
                logical_to_physical_layout[logical_qubit_b],
            )
            if routing_path is None or len(routing_path) < CustomLayoutPass._TWO_QUBIT_ARITY:
                raise ValueError(
                    "User mapping cannot be routed on the provided topology; no path between "
                    f"physical qubits {logical_to_physical_layout[logical_qubit_a]} and "
                    f"{logical_to_physical_layout[logical_qubit_b]}."
                )

            applied_swap_edges: list[tuple[int, int]] = []
            # Move the second qubit along the path until it neighbors the first one.
            for path_position in range(len(routing_path) - 1, 1, -1):
                physical_node_a, physical_node_b = routing_path[path_position - 1], routing_path[path_position]
                output_circuit.add(SWAP(physical_node_a, physical_node_b))
                self._apply_swap_to_layout(
                    logical_to_physical_layout,
                    inverse_layout,
                    physical_node_a,
                    physical_node_b,
                )
                applied_swap_edges.append((physical_node_a, physical_node_b))

            mapped_qubits = (
                logical_to_physical_layout[logical_qubit_a],
                logical_to_physical_layout[logical_qubit_b],
            )
            if not self.topology.has_edge(*mapped_qubits):
                raise RuntimeError(
                    "Failed to route gate after inserting swaps; resulting qubits are still non-adjacent."
                )
            output_circuit.add(self._retarget_gate(gate, mapped_qubits))

            # Restore the original mapping so later 1Q gates remain on the requested qubits.
            for physical_node_a, physical_node_b in reversed(applied_swap_edges):
                output_circuit.add(SWAP(physical_node_a, physical_node_b))
                self._apply_swap_to_layout(
                    logical_to_physical_layout,
                    inverse_layout,
                    physical_node_a,
                    physical_node_b,
                )

        if self.context is not None:
            self.context.initial_layout = initial_layout
            self.context.final_layout = self._user_layout

        self.append_circuit_to_context(output_circuit)

        return output_circuit

    # --------- retargeting helpers (mirrors SabreLayoutPass) ---------

    @staticmethod
    def _retarget_gate(gate: Gate, mapped_qubits: tuple[int, ...]) -> Gate:
        """Recreate a supported gate on new qubit indices.

        Args:
            gate (Gate): Gate to retarget.
            mapped_qubits (tuple[int, ...]): Physical qubits where the gate should act.

        Returns:
            Gate: Retargeted gate equivalent to ``gate`` on ``mapped_qubits``.

        Raises:
            NotImplementedError: If the gate type is not supported by this pass.
        """
        # 1-qubit basics
        if isinstance(gate, RX):
            return RX(mapped_qubits[0], theta=gate.theta)
        if isinstance(gate, RY):
            return RY(mapped_qubits[0], theta=gate.theta)
        if isinstance(gate, RZ):
            return RZ(mapped_qubits[0], phi=gate.phi)
        if isinstance(gate, U3):
            return U3(mapped_qubits[0], theta=gate.theta, phi=gate.phi, gamma=gate.gamma)

        # 2-qubit basics
        if isinstance(gate, CNOT):
            return CNOT(mapped_qubits[0], mapped_qubits[1])
        if isinstance(gate, CZ):
            return CZ(mapped_qubits[0], mapped_qubits[1])
        if isinstance(gate, SWAP):
            return SWAP(mapped_qubits[0], mapped_qubits[1])

        # Measurement (possibly multi-qubit)
        if isinstance(gate, M):
            return M(*mapped_qubits)

        raise NotImplementedError(
            f"Retargeting not implemented for gate type {type(gate).__name__} with arity {gate.nqubits}"
        )

    @staticmethod
    def _apply_swap_to_layout(
        logical_to_physical_layout: list[int],
        inverse_layout: list[int | None],
        physical_node_a: int,
        physical_node_b: int,
    ) -> None:
        """Update forward and inverse mappings after applying a physical SWAP.

        Args:
            logical_to_physical_layout (list[int]): Forward mapping ``logical -> physical``.
            inverse_layout (list[int | None]): Inverse mapping ``physical -> logical``.
            physical_node_a (int): First swapped physical node.
            physical_node_b (int): Second swapped physical node.
        """
        logical_qubit_on_a = inverse_layout[physical_node_a]
        logical_qubit_on_b = inverse_layout[physical_node_b]

        inverse_layout[physical_node_a], inverse_layout[physical_node_b] = logical_qubit_on_b, logical_qubit_on_a

        if logical_qubit_on_a is not None:
            logical_to_physical_layout[logical_qubit_on_a] = physical_node_b
        if logical_qubit_on_b is not None:
            logical_to_physical_layout[logical_qubit_on_b] = physical_node_a

    def _shortest_path(self, start: int, goal: int) -> list[int] | None:
        """Return one shortest path between two physical qubits in the coupling graph.

        Args:
            start (int): Source physical qubit.
            goal (int): Target physical qubit.

        Returns:
            list[int] | None: Path from ``start`` to ``goal`` if reachable; otherwise ``None``.
        """

        if start == goal:
            return [start]

        visited_nodes = {start}
        parent_by_node: dict[int, int] = {}
        queue: deque[int] = deque([start])

        while queue:
            current_node = queue.popleft()
            for neighbor in self.topology.neighbors(current_node):
                neighbor_node = int(neighbor)
                if neighbor_node in visited_nodes:
                    continue
                parent_by_node[neighbor_node] = current_node
                if neighbor_node == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(parent_by_node[path[-1]])
                    path.reverse()
                    return path
                visited_nodes.add(neighbor_node)
                queue.append(neighbor_node)

        return None
