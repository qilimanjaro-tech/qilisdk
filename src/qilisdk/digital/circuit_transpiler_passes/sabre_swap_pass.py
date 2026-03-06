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

import math
import random
from collections import deque
from copy import deepcopy
from typing import Iterable, TypeGuard

from rustworkx import PyGraph

from qilisdk.digital import (
    SWAP,
    Adjoint,
    Circuit,
    Controlled,
    Exponential,
    Gate,
    M,
)
from qilisdk.digital.gates import BasicGate

from .circuit_transpiler_pass import CircuitTranspilerPass


def _is_controlled_gate(gate: Gate) -> TypeGuard[Controlled[BasicGate]]:
    """Return whether a gate is a controlled gate with a basic-gate payload.

    Args:
        gate (Gate): Candidate gate.

    Returns:
        TypeGuard[Controlled[BasicGate]]: ``True`` when ``gate`` is a ``Controlled`` gate whose payload type is ``BasicGate``.
    """
    return isinstance(gate, Controlled)


def _is_adjoint_or_exponential(
    gate: Gate,
) -> TypeGuard[Adjoint[BasicGate] | Exponential[BasicGate]]:
    """Return whether a gate is an adjoint or exponential wrapper.

    Args:
        gate (Gate): Candidate gate.

    Returns:
        TypeGuard[Adjoint[BasicGate] | Exponential[BasicGate]]: ``True`` when ``gate`` is either ``Adjoint`` or ``Exponential``.
    """
    return isinstance(gate, (Adjoint, Exponential))


class SabreSwapPass(CircuitTranspilerPass):
    """
    SABRE routing (SWAP insertion) for 1Q/2Q circuits on an undirected coupling graph.

    Inputs
    ------
    coupling : rustworkx.PyGraph
        Undirected device graph. Node indices are physical qubits (labels need not be contiguous).
    initial_layout : list[int] | None
        Logical -> physical mapping to start from (e.g., from SabreLayoutPass.last_layout).
        If None, uses the lowest-indexed physical qubits provided by the coupling graph.

    Heuristic (SABRE-style)
    -----------------------
    When a 2Q gate's mapped endpoints are non-adjacent, choose one SWAP on an edge
    touching the mapped endpoints of the current front-set gates that minimizes:

        cost = sum_{g in F} distance_matrix(p_u, p_v) * (1 + decay[p_u] + decay[p_v])
             + beta * sum_{g in E} distance_matrix(p_u, p_v)

    where F is the per-qubit first unscheduled 2Q gate and E is a small look-ahead set.
    A light decay penalty discourages thrashing.

    Behavior
    --------
    - Returns a **new** Circuit with 1Q gates mapped and SWAPs inserted before each
      non-adjacent 2Q gate so that every emitted 2Q gate acts on an edge of `coupling`.
    - Preserves the original **gate order** (no reordering/commutation across the list).

    Notes
    -----
    - Supports 1Q and 2Q gates (CNOT, CZ, SWAP, generic Controlled with one control).
      Multi-qubit (>2) non-SWAP gates should be decomposed before routing.
    """

    _TWO_QUBIT_ARITY = 2
    _UNREACHABLE_DISTANCE = 1_000_000_000
    _TIE_EPSILON = 1e-12
    _TIE_BREAK_PROBABILITY = 0.5

    def __init__(
        self,
        topology: PyGraph,
        *,
        initial_layout: list[int] | None = None,
        seed: int | None = None,
        lookahead_size: int = 10,
        beta: float = 0.8,
        decay_delta: float = 0.001,
        decay_lambda: float = 0.99,
        max_swaps_factor: float = 64.0,
        max_attempts: int = 10,
    ) -> None:
        """Configure SABRE swap routing behavior.

        Args:
            topology (PyGraph): Undirected coupling graph for valid physical two-qubit connections.
            initial_layout (list[int] | None): Optional initial logical-to-physical mapping; uses identity-style mapping when omitted.
            seed (int | None): Base seed for stochastic swap scoring and retry attempts.
            lookahead_size (int): Number of future two-qubit gates used in the extended SABRE cost.
            beta (float): Weight balancing extended-set cost against immediate front-set cost.
            decay_delta (float): Penalty increment added to swapped physical qubits.
            decay_lambda (float): Per-iteration multiplier that relaxes accumulated decay penalties.
            max_swaps_factor (float): Per-gate swap budget factor derived from current endpoint distance.
            max_attempts (int): Maximum number of independent routing attempts with varied seeds.

        Raises:
            TypeError: If ``topology`` is not a ``rustworkx.PyGraph`` instance.
        """
        if not isinstance(topology, PyGraph):
            raise TypeError("SabreSwapPass requires a rustworkx.PyGraph (undirected).")
        self.topology = topology
        self.initial_layout = initial_layout
        self.seed = seed
        self.lookahead_size = int(lookahead_size)
        self.beta = float(beta)
        self.decay_delta = float(decay_delta)
        self.decay_lambda = float(decay_lambda)
        self.max_swaps_factor = float(max_swaps_factor)
        self.max_attempts = int(max_attempts)

        # Diagnostics
        self.last_swap_count: int | None = None
        self.last_final_layout: list[int] | None = None

    # ----------------------- public API -----------------------

    def run(self, circuit: Circuit) -> Circuit:
        """Route a circuit onto the coupling graph using SABRE swap insertion.

        The pass retries with different seeds when an attempt exceeds swap budget and stores diagnostics in context.

        Args:
            circuit (Circuit): Logical circuit to route.

        Returns:
            Circuit: Routed circuit with gates mapped to physical qubits and inserted SWAPs.

        Raises:
            RuntimeError: If all attempts fail due to exhausted swap budgets.
        """
        max_attempt_count = max(1, self.max_attempts)
        last_exception: RuntimeError | None = None
        base_seed = self.seed
        # Obtain layout hint without mutating instance attributes so repeated runs
        # do not accidentally persist stale mappings.
        layout_hint: list[int] | None = None
        if self.initial_layout is not None:
            layout_hint = list(self.initial_layout)
        elif self.context is not None and self.context.initial_layout:
            layout_hint = list(self.context.initial_layout)

        for attempt in range(max_attempt_count):
            attempt_seed = None if base_seed is None else base_seed + attempt
            try:
                routed_circuit, swap_count, final_layout = self._run_once(
                    circuit,
                    attempt_seed,
                    layout_hint,
                )
            except RuntimeError as exc:
                if "Exceeded swap budget" not in str(exc):
                    raise
                last_exception = exc
                continue

            self.last_swap_count = swap_count
            self.last_final_layout = final_layout

            if self.context is not None:
                if layout_hint:
                    self.context.final_layout = {
                        logical_qubit: self.last_final_layout[logical_qubit] for logical_qubit in sorted(layout_hint)
                    }
                else:
                    self.context.final_layout = {
                        logical_qubit: self.last_final_layout[logical_qubit]
                        for logical_qubit in range(len(final_layout))
                    }
                self.context.metrics["swap_count"] = self.last_swap_count

            self.append_circuit_to_context(routed_circuit)
            return routed_circuit

        if last_exception is not None:
            raise last_exception
        raise RuntimeError(f"SABRE routing failed after {max_attempt_count} attempts.")

    def _run_once(
        self,
        circuit: Circuit,
        attempt_seed: int | None,
        layout_hint: list[int] | None,
    ) -> tuple[Circuit, int, list[int]]:
        """Execute one SABRE routing attempt with a fixed random seed.

        Args:
            circuit (Circuit): Logical circuit to route.
            attempt_seed (int | None): Seed used for stochastic tie-breaking.
            layout_hint (list[int] | None): Optional initial logical-to-physical mapping hint.

        Returns:
            tuple[Circuit, int, list[int]]: Routed circuit, number of inserted SWAPs, and final logical-to-physical layout.

        Raises:
            ValueError: If the topology has no nodes or layout initialization is invalid for active qubits.
            NotImplementedError: If a gate with arity greater than two needs routing.
            RuntimeError: If swap budget is exceeded or no valid swap can be selected.
        """
        random_generator = random.Random(attempt_seed)

        num_logical_qubits = circuit.nqubits
        physical_nodes = sorted(int(x) for x in self.topology.node_indices())
        if not physical_nodes:
            raise ValueError("Topology graph has no nodes.")
        num_physical_nodes = len(physical_nodes)
        physical_to_dense_index = {node: dense_index for dense_index, node in enumerate(physical_nodes)}
        max_physical_label_plus_one = max(physical_nodes) + 1

        active_qubits = {int(qubit) for gate in circuit.gates for qubit in gate.qubits}
        layout = self._init_layout(num_logical_qubits, physical_nodes, active_qubits, layout_hint)

        if self._is_layout_routable(circuit, layout):
            output_num_qubits = max(
                max_physical_label_plus_one,
                max(layout) + 1 if layout else max_physical_label_plus_one,
                circuit.nqubits,
            )
            routed_circuit = Circuit(output_num_qubits)
            for gate in circuit.gates:
                mapped_qubits = tuple(layout[logical_qubit] for logical_qubit in gate.qubits)
                routed_circuit.add(self._retarget_gate(gate, mapped_qubits))
            return routed_circuit, 0, layout[:]

        inverse_layout = self._invert_layout(layout, physical_to_dense_index)
        distance_matrix = self._all_pairs_shortest_path_unweighted(
            self.topology,
            physical_nodes,
            physical_to_dense_index,
        )

        # Preprocess 2Q structure for SABRE scoring
        two_qubit_operation_indices, two_qubit_pairs, gate_indices_per_qubit = self._two_qubit_structure(circuit)
        num_two_qubit_operations = len(two_qubit_pairs)
        scheduled = [False] * num_two_qubit_operations
        front_positions = [0] * num_logical_qubits

        def advance_front_for(logical_qubit: int) -> None:
            qubit_gate_indices = gate_indices_per_qubit[logical_qubit]
            cursor = front_positions[logical_qubit]
            while cursor < len(qubit_gate_indices) and scheduled[qubit_gate_indices[cursor]]:
                cursor += 1
            front_positions[logical_qubit] = cursor

        for logical_qubit in range(num_logical_qubits):
            advance_front_for(logical_qubit)

        # Output circuit will use physical indices; ensure capacity for max physical index we may touch.
        output_num_qubits = max(
            max_physical_label_plus_one,
            max(layout) + 1 if layout else max_physical_label_plus_one,
            circuit.nqubits,
        )
        routed_circuit = Circuit(output_num_qubits)

        # Decay penalties on physical qubits
        decay = [0.0] * num_physical_nodes
        swap_count = 0

        # Map from op index -> local 2Q index
        operation_to_two_qubit_index = {
            operation_idx: two_qubit_index for two_qubit_index, operation_idx in enumerate(two_qubit_operation_indices)
        }

        # --- main sweep over original gates, preserving order ---
        for operation_index, gate in enumerate(circuit.gates):
            gate_qubits = gate.qubits

            # 1Q (or measurement / any non-2Q) -> just map and emit
            if len(gate_qubits) <= 1 or isinstance(gate, M):
                mapped_qubits = tuple(layout[logical_qubit] for logical_qubit in gate_qubits)
                routed_circuit.add(self._retarget_gate(gate, mapped_qubits))
                continue

            # We currently support only 2Q routing (SWAP, CNOT, CZ, single-control Controlled)
            if len(gate_qubits) != SabreSwapPass._TWO_QUBIT_ARITY:
                raise NotImplementedError(
                    f"Routing of {type(gate).__name__} with arity {len(gate_qubits)} not supported. "
                    "Please decompose to 1Q/2Q gates before routing."
                )

            logical_qubit_a, logical_qubit_b = gate_qubits
            two_qubit_index = operation_to_two_qubit_index[operation_index]  # local 2Q index for SABRE bookkeeping

            # While mapped endpoints are not adjacent, insert a SABRE-chosen SWAP
            endpoint_distance = distance_matrix[physical_to_dense_index[layout[logical_qubit_a]]][
                physical_to_dense_index[layout[logical_qubit_b]]
            ]
            max_swaps_this_gate = int(self.max_swaps_factor * max(1, endpoint_distance))
            swap_steps = 0
            while (
                distance_matrix[physical_to_dense_index[layout[logical_qubit_a]]][
                    physical_to_dense_index[layout[logical_qubit_b]]
                ]
                != 1
            ):
                swap_steps += 1
                if swap_steps > max_swaps_this_gate:
                    raise RuntimeError(
                        f"Exceeded swap budget while routing gate {type(gate).__name__}{gate_qubits}: "
                        "graph may be disconnected or heuristic stuck."
                    )

                # Front set: first unscheduled 2Q gate per logical qubit
                front_gate_indices = self._front_set(gate_indices_per_qubit, front_positions, scheduled)
                # Ensure current gate is in the front set (it usually is, but be robust)
                front_gate_indices.add(two_qubit_index)
                # Extended look-ahead set
                extended_gate_indices = self._extended_set(gate_indices_per_qubit, front_positions, self.lookahead_size)

                # Build candidate physical edges from neighbors of endpoints of all front-set gates
                candidate_swap_edges: set[tuple[int, int]] = set()
                touched_physical_nodes: set[int] = set()
                for front_gate_index in front_gate_indices:
                    gate_qubit_a, gate_qubit_b = two_qubit_pairs[front_gate_index]
                    physical_qubit_a, physical_qubit_b = layout[gate_qubit_a], layout[gate_qubit_b]
                    touched_physical_nodes.add(physical_qubit_a)
                    touched_physical_nodes.add(physical_qubit_b)
                    for neighbor in self.topology.neighbors(physical_qubit_a):
                        edge_start, edge_end = int(physical_qubit_a), int(neighbor)
                        if edge_start != edge_end:
                            candidate_swap_edges.add((min(edge_start, edge_end), max(edge_start, edge_end)))
                    for neighbor in self.topology.neighbors(physical_qubit_b):
                        edge_start, edge_end = int(physical_qubit_b), int(neighbor)
                        if edge_start != edge_end:
                            candidate_swap_edges.add((min(edge_start, edge_end), max(edge_start, edge_end)))

                if not candidate_swap_edges:
                    # Fallback: try swaps among touched physical nodes (should be rare)
                    candidate_physical_nodes = (
                        sorted(touched_physical_nodes) if touched_physical_nodes else physical_nodes[:]
                    )
                    if len(candidate_physical_nodes) >= SabreSwapPass._TWO_QUBIT_ARITY:
                        sampled_a, sampled_b = random_generator.sample(
                            candidate_physical_nodes,
                            SabreSwapPass._TWO_QUBIT_ARITY,
                        )
                        candidate_swap_edges.add((min(sampled_a, sampled_b), max(sampled_a, sampled_b)))
                    else:
                        raise RuntimeError("No candidate swaps available; coupling graph likely degenerate.")

                # Decay relaxation
                for physical_dense_index in range(num_physical_nodes):
                    decay[physical_dense_index] *= self.decay_lambda

                # Evaluate SABRE cost for each candidate swap (virtually)
                current_distance = distance_matrix[physical_to_dense_index[layout[logical_qubit_a]]][
                    physical_to_dense_index[layout[logical_qubit_b]]
                ]
                improving_edge: tuple[int, int] | None = None
                improving_cost: float = math.inf
                best_edge: tuple[int, int] | None = None
                best_cost: float = math.inf

                for physical_node_a, physical_node_b in candidate_swap_edges:
                    logical_at_a = inverse_layout[physical_to_dense_index[physical_node_a]]
                    logical_at_b = inverse_layout[physical_to_dense_index[physical_node_b]]
                    # Virtually swap mapping
                    self._swap_mapping(
                        layout,
                        inverse_layout,
                        physical_to_dense_index,
                        physical_node_a,
                        physical_node_b,
                        logical_at_a,
                        logical_at_b,
                    )
                    # Distance after the hypothetical swap
                    new_distance = distance_matrix[physical_to_dense_index[layout[logical_qubit_a]]][
                        physical_to_dense_index[layout[logical_qubit_b]]
                    ]
                    front_cost = self._cost_set(
                        front_gate_indices,
                        layout,
                        two_qubit_pairs,
                        distance_matrix,
                        decay,
                        physical_to_dense_index,
                    )
                    extended_cost = self._cost_set(
                        extended_gate_indices,
                        layout,
                        two_qubit_pairs,
                        distance_matrix,
                        None,
                        physical_to_dense_index,
                    )
                    total_cost = front_cost + self.beta * extended_cost
                    # Revert
                    self._swap_mapping(
                        layout,
                        inverse_layout,
                        physical_to_dense_index,
                        physical_node_a,
                        physical_node_b,
                        logical_at_b,
                        logical_at_a,
                    )

                    if new_distance < current_distance:
                        if total_cost < improving_cost - SabreSwapPass._TIE_EPSILON:
                            improving_cost = total_cost
                            improving_edge = (physical_node_a, physical_node_b)
                        elif (
                            abs(total_cost - improving_cost) <= SabreSwapPass._TIE_EPSILON
                            and random_generator.random() < SabreSwapPass._TIE_BREAK_PROBABILITY
                        ):
                            improving_edge = (physical_node_a, physical_node_b)

                    if total_cost < best_cost - SabreSwapPass._TIE_EPSILON:
                        best_cost = total_cost
                        best_edge = (physical_node_a, physical_node_b)
                    elif (
                        abs(total_cost - best_cost) <= SabreSwapPass._TIE_EPSILON
                        and random_generator.random() < SabreSwapPass._TIE_BREAK_PROBABILITY
                    ):
                        best_edge = (physical_node_a, physical_node_b)

                # Apply chosen SWAP physically and in the mapping
                chosen_edge = improving_edge if improving_edge is not None else best_edge
                if chosen_edge is None:
                    raise RuntimeError("SABRE heuristic could not select a swap candidate.")
                chosen_physical_a, chosen_physical_b = chosen_edge
                routed_circuit.add(SWAP(chosen_physical_a, chosen_physical_b))
                swap_count += 1
                logical_at_a = inverse_layout[physical_to_dense_index[chosen_physical_a]]
                logical_at_b = inverse_layout[physical_to_dense_index[chosen_physical_b]]
                self._swap_mapping(
                    layout,
                    inverse_layout,
                    physical_to_dense_index,
                    chosen_physical_a,
                    chosen_physical_b,
                    logical_at_a,
                    logical_at_b,
                )
                decay[physical_to_dense_index[chosen_physical_a]] += self.decay_delta
                decay[physical_to_dense_index[chosen_physical_b]] += self.decay_delta

            # Now adjacent: emit the mapped 2Q gate
            mapped_qubits = (layout[logical_qubit_a], layout[logical_qubit_b])
            routed_circuit.add(self._retarget_gate(gate, mapped_qubits))
            scheduled[two_qubit_index] = True
            # Advance front pointers for the logical qubits touched
            advance_front_for(logical_qubit_a)
            advance_front_for(logical_qubit_b)

        return routed_circuit, swap_count, layout[:]

    # ----------------------- SABRE helpers -----------------------

    def _is_layout_routable(self, circuit: Circuit, layout: list[int]) -> bool:
        """Check whether all two-qubit gates are adjacent under a layout.

        Args:
            circuit (Circuit): Circuit in logical qubit indices.
            layout (list[int]): Logical-to-physical mapping.

        Returns:
            bool: ``True`` if every two-qubit gate acts on a coupling-graph edge; otherwise ``False``.
        """
        for gate in circuit.gates:
            gate_qubits = gate.qubits
            if len(gate_qubits) != SabreSwapPass._TWO_QUBIT_ARITY:
                continue
            if not self.topology.has_edge(layout[gate_qubits[0]], layout[gate_qubits[1]]):
                return False
        return True

    @staticmethod
    def _front_set(
        gate_indices_per_qubit: list[list[int]], front_positions: list[int], scheduled: list[bool]
    ) -> set[int]:
        """Return the current SABRE front set of unscheduled two-qubit gates.

        Args:
            gate_indices_per_qubit (list[list[int]]): Per-logical-qubit interaction indices touching that qubit.
            front_positions (list[int]): Per-qubit cursor into ``gate_indices_per_qubit``.
            scheduled (list[bool]): Flags for interactions already scheduled.

        Returns:
            set[int]: One earliest unscheduled interaction per logical qubit.
        """
        front_gate_indices: set[int] = set()
        for logical_qubit in range(len(gate_indices_per_qubit)):
            qubit_gate_indices = gate_indices_per_qubit[logical_qubit]
            cursor = front_positions[logical_qubit]
            while cursor < len(qubit_gate_indices) and scheduled[qubit_gate_indices[cursor]]:
                cursor += 1
            if cursor < len(qubit_gate_indices):
                front_gate_indices.add(qubit_gate_indices[cursor])
        return front_gate_indices

    @staticmethod
    def _extended_set(gate_indices_per_qubit: list[list[int]], front_positions: list[int], max_size: int) -> set[int]:
        """Build SABRE's look-ahead interaction set after the front layer.

        Args:
            gate_indices_per_qubit (list[list[int]]): Per-logical-qubit interaction indices touching that qubit.
            front_positions (list[int]): Per-qubit cursor into ``gate_indices_per_qubit``.
            max_size (int): Maximum number of interactions to collect.

        Returns:
            set[int]: Bounded set of future interaction indices.
        """
        extended_gate_indices: set[int] = set()
        if max_size <= 0:
            return extended_gate_indices
        budget = max_size
        for logical_qubit in range(len(gate_indices_per_qubit)):
            qubit_gate_indices = gate_indices_per_qubit[logical_qubit]
            cursor = front_positions[logical_qubit] + 1
            while cursor < len(qubit_gate_indices) and budget > 0:
                extended_gate_indices.add(qubit_gate_indices[cursor])
                cursor += 1
                budget -= 1
                if budget == 0:
                    break
            if budget == 0:
                break
        return extended_gate_indices

    @staticmethod
    def _cost_set(
        gate_indices: Iterable[int],
        layout: list[int],
        two_qubit_pairs: list[tuple[int, int]],
        distance_matrix: list[list[int]],
        decay: list[float] | None,
        physical_to_dense_index: dict[int, int],
    ) -> float:
        """Score a set of interactions under a candidate layout.

        Args:
            gate_indices (Iterable[int]): Interaction indices to evaluate.
            layout (list[int]): Logical-to-physical mapping.
            two_qubit_pairs (list[tuple[int, int]]): Logical qubit pairs for each interaction index.
            distance_matrix (list[list[int]]): Physical all-pairs shortest-path distance matrix.
            decay (list[float] | None): Optional physical decay penalties added to front-set costs.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            float: Sum of pairwise distances with optional decay weighting.
        """
        total_cost = 0.0
        for gate_index in gate_indices:
            logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_index]
            physical_qubit_a, physical_qubit_b = layout[logical_qubit_a], layout[logical_qubit_b]
            physical_index_a = physical_to_dense_index[physical_qubit_a]
            physical_index_b = physical_to_dense_index[physical_qubit_b]
            path_distance = distance_matrix[physical_index_a][physical_index_b]
            if path_distance >= SabreSwapPass._UNREACHABLE_DISTANCE:
                # Disconnected; make it very expensive
                path_distance = 1e6
            if decay is not None:
                total_cost += path_distance * (1.0 + decay[physical_index_a] + decay[physical_index_b])
            else:
                total_cost += path_distance
        return total_cost

    # ----------------------- structure & mapping -----------------------

    @staticmethod
    def _two_qubit_structure(circuit: Circuit) -> tuple[list[int], list[tuple[int, int]], list[list[int]]]:
        """Extract two-qubit interaction indices and adjacency structure.

        Args:
            circuit (Circuit): Input logical circuit.

        Returns:
            tuple[list[int], list[tuple[int, int]], list[list[int]]]: Two-qubit operation indices, corresponding logical-qubit pairs, and per-logical-qubit interaction index lists.
        """
        num_logical_qubits = circuit.nqubits
        two_qubit_operation_indices: list[int] = []
        two_qubit_pairs: list[tuple[int, int]] = []
        for operation_index, gate in enumerate(circuit.gates):
            gate_qubits = gate.qubits
            if len(gate_qubits) == SabreSwapPass._TWO_QUBIT_ARITY:
                two_qubit_operation_indices.append(operation_index)
                two_qubit_pairs.append((gate_qubits[0], gate_qubits[1]))
        gate_indices_per_qubit: list[list[int]] = [[] for _ in range(num_logical_qubits)]
        for two_qubit_index, (logical_qubit_a, logical_qubit_b) in enumerate(two_qubit_pairs):
            gate_indices_per_qubit[logical_qubit_a].append(two_qubit_index)
            gate_indices_per_qubit[logical_qubit_b].append(two_qubit_index)
        return two_qubit_operation_indices, two_qubit_pairs, gate_indices_per_qubit

    @staticmethod
    def _invert_layout(
        layout: list[int],
        physical_to_dense_index: dict[int, int],
    ) -> list[int | None]:
        """Build inverse mapping from dense physical indices to logical qubits.

        Args:
            layout (list[int]): Logical-to-physical mapping.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            list[int | None]: Inverse map where each dense physical index stores the occupying logical qubit or ``None``.
        """
        inverse_layout = [None] * len(physical_to_dense_index)
        for logical_qubit, physical_qubit in enumerate(layout):
            physical_dense_index = physical_to_dense_index.get(physical_qubit)
            if physical_dense_index is not None and inverse_layout[physical_dense_index] is None:
                inverse_layout[physical_dense_index] = logical_qubit
        return inverse_layout

    @staticmethod
    def _swap_mapping(
        layout: list[int],
        inverse_layout: list[int | None],
        physical_to_dense_index: dict[int, int],
        physical_node_a: int,
        physical_node_b: int,
        logical_at_a: int | None,
        logical_at_b: int | None,
    ) -> None:
        """Apply a physical swap to both forward and inverse layout mappings.

        Args:
            layout (list[int]): Logical-to-physical mapping to mutate.
            inverse_layout (list[int | None]): Dense physical-to-logical mapping to mutate.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.
            physical_node_a (int): First physical node in the swap.
            physical_node_b (int): Second physical node in the swap.
            logical_at_a (int | None): Logical qubit currently assigned to ``physical_node_a``.
            logical_at_b (int | None): Logical qubit currently assigned to ``physical_node_b``.
        """
        # update inverse
        dense_index_a = physical_to_dense_index[physical_node_a]
        dense_index_b = physical_to_dense_index[physical_node_b]
        inverse_layout[dense_index_a], inverse_layout[dense_index_b] = logical_at_b, logical_at_a
        # update forward
        if logical_at_a is not None:
            layout[logical_at_a] = physical_node_b
        if logical_at_b is not None:
            layout[logical_at_b] = physical_node_a

    @staticmethod
    def _all_pairs_shortest_path_unweighted(
        graph: PyGraph,
        physical_nodes: list[int],
        physical_to_dense_index: dict[int, int],
    ) -> list[list[int]]:
        """Compute unweighted all-pairs physical distances with BFS.

        Args:
            graph (PyGraph): Undirected coupling graph.
            physical_nodes (list[int]): Physical node labels.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            list[list[int]]: Dense distance matrix with shortest path lengths and sentinel values for unreachable pairs.
        """
        unreachable_distance = SabreSwapPass._UNREACHABLE_DISTANCE
        distance_matrix = [[unreachable_distance] * len(physical_nodes) for _ in physical_nodes]
        for source_node in physical_nodes:
            source_dense_index = physical_to_dense_index[source_node]
            distance_matrix[source_dense_index][source_dense_index] = 0
            queue = deque([source_node])
            seen = {source_node}
            while queue:
                current_node = queue.popleft()
                current_dense_index = physical_to_dense_index[current_node]
                current_distance = distance_matrix[source_dense_index][current_dense_index]
                for neighbor in graph.neighbors(current_node):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        neighbor_dense_index = physical_to_dense_index[neighbor]
                        distance_matrix[source_dense_index][neighbor_dense_index] = current_distance + 1
                        queue.append(neighbor)
        return distance_matrix

    @staticmethod
    def _init_layout(
        num_logical_qubits: int,
        physical_nodes: list[int],
        active_qubits: set[int],
        layout_hint: list[int] | None,
    ) -> list[int]:
        """Create an initial logical-to-physical layout for routing.

        The method accepts full, active-only, or prefix layout hints and fills unassigned logical qubits with remaining physical nodes.

        Args:
            num_logical_qubits (int): Number of logical qubits in the circuit.
            physical_nodes (list[int]): Physical node labels available in the coupling graph.
            active_qubits (set[int]): Logical qubits that appear in at least one circuit operation.
            layout_hint (list[int] | None): Optional user-provided initial mapping.

        Returns:
            list[int]: Logical-to-physical mapping used to start SABRE routing.

        Raises:
            ValueError: If active logical qubits map outside the coupling graph, if active mappings are not unique, or if there are fewer physical nodes than logical qubits when no valid identity-style mapping exists.
        """
        if layout_hint is not None:
            layout_hint_copy = list(layout_hint)
            active_logical_qubits = sorted(active_qubits)
            unassigned = -1
            layout: list[int]
            if len(layout_hint_copy) == num_logical_qubits:
                layout = layout_hint_copy[:num_logical_qubits]
            elif active_logical_qubits and len(layout_hint_copy) == len(active_logical_qubits):
                layout = [unassigned] * num_logical_qubits
                for hint_index, logical_qubit in enumerate(active_logical_qubits):
                    layout[logical_qubit] = layout_hint_copy[hint_index]
            else:
                # Fallback: treat as prefix mapping and pad/trim
                layout = layout_hint_copy[:num_logical_qubits]
                if len(layout) < num_logical_qubits:
                    layout.extend([unassigned] * (num_logical_qubits - len(layout)))
            used_physical_nodes = {physical_qubit for physical_qubit in layout if physical_qubit != unassigned}
            remaining_physical_nodes = [node for node in physical_nodes if node not in used_physical_nodes]
            placeholder = physical_nodes[0] if physical_nodes else 0
            for logical_qubit in range(num_logical_qubits):
                if layout[logical_qubit] == unassigned:
                    layout[logical_qubit] = remaining_physical_nodes.pop(0) if remaining_physical_nodes else placeholder
            available_physical_nodes = set(physical_nodes)
            active_targets = [layout[logical_qubit] for logical_qubit in active_qubits]
            missing = sorted(node for node in active_targets if node not in available_physical_nodes)
            if missing:
                raise ValueError(
                    "initial_layout refers to physical qubits not present in the coupling graph for active logical "
                    f"qubits: {missing}"
                )
            if len(set(active_targets)) != len(active_targets):
                raise ValueError("initial_layout must map active logical qubits to unique physical qubits.")
            return layout
        # identity by default
        physical_node_set = set(physical_nodes)
        uses_physical_labels = active_qubits <= physical_node_set
        if uses_physical_labels:
            if not physical_nodes:
                return []
            layout = [-1] * num_logical_qubits
            for logical_qubit in active_qubits:
                layout[logical_qubit] = logical_qubit
            placeholder = physical_nodes[0]
            for logical_qubit in range(num_logical_qubits):
                if layout[logical_qubit] == -1:
                    layout[logical_qubit] = logical_qubit if logical_qubit in physical_node_set else placeholder
            return layout
        if len(physical_nodes) < num_logical_qubits:
            raise ValueError(f"Coupling graph has {len(physical_nodes)} qubits but circuit needs {num_logical_qubits}.")
        return physical_nodes[:num_logical_qubits]

    # ----------------------- gate (re)construction -----------------------

    def _retarget_gate(self, gate: Gate, mapped_qubits: tuple[int, ...]) -> Gate:
        """
        Recreate ``gate`` on ``mapped_qubits`` by deep-copying and remapping indices.

        The method deep-copies the gate object to preserve concrete type and parameters, then recursively updates control and target qubit tuples through wrapped gates.

        Args:
            gate (Gate): Original gate to retarget.
            mapped_qubits (tuple[int, ...]): Physical qubits where the gate should act.

        Returns:
            Gate: Retargeted gate equivalent to ``gate`` but acting on ``mapped_qubits``.
        """
        retargeted = deepcopy(gate)
        qubit_map = dict(zip(gate.qubits, mapped_qubits, strict=True))
        self._remap_gate_qubits_inplace(retargeted, qubit_map)
        return retargeted

    @staticmethod
    def _remap_gate_qubits_inplace(gate: Gate, qubit_map: dict[int, int]) -> None:
        """Recursively remap control/target qubits of a copied gate.

        Args:
            gate (Gate): Gate object to modify in place.
            qubit_map (dict[int, int]): Logical-to-physical mapping for qubits touched by ``gate``.

        Raises:
            NotImplementedError: If the gate type does not provide known internal storage for control or target qubits.
        """
        if isinstance(gate, (BasicGate, M)):
            gate._target_qubits = tuple(qubit_map[logical_qubit] for logical_qubit in gate.target_qubits)  # noqa: SLF001
            return

        if _is_controlled_gate(gate):
            gate._control_qubits = tuple(qubit_map[control_qubit] for control_qubit in gate.control_qubits)  # noqa: SLF001
            SabreSwapPass._remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
            return

        if _is_adjoint_or_exponential(gate):
            SabreSwapPass._remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
            return

        raise NotImplementedError(
            f"Retargeting not implemented for gate type {type(gate).__name__} with arity {gate.nqubits}"
        )
