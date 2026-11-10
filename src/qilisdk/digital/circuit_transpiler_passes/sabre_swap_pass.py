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
from typing import TYPE_CHECKING

from loguru import logger

from qilisdk.digital import SWAP, Circuit, M
from qilisdk.digital.topology import build_topology_graph

from .circuit_transpiler_pass import CircuitTranspilerPass

# Cost of an interaction between disconnected physical qubits, used instead of the sentinel distance.
_UNREACHABLE_PENALTY = 1e6
# Lazily stored decay values are folded back into the scale once it drops below this.
_DECAY_RESCALE_THRESHOLD = 1e-100

if TYPE_CHECKING:
    from rustworkx import PyGraph

    from qilisdk.digital.types import LayoutMap, Topology


class SabreSwapPass(CircuitTranspilerPass):
    """
    SABRE routing (SWAP insertion) for 1Q/2Q circuits on an undirected coupling graph.

    Inputs
    ------
    topology : Topology
        Coupling map provided either as an edge list or as an undirected
        ``PyGraph`` whose node indices are physical qubits.
    initial_layout : list[int] | None
        Logical -> physical mapping to start from (e.g., from SabreLayoutPass.last_layout).
        If None, falls back to ``context.initial_layout`` when a layout pass recorded one
        without applying it, and otherwise uses the lowest-indexed physical qubits
        provided by the coupling graph.

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
      non-adjacent 2Q gate so that every emitted 2Q gate acts on an edge of `topology`.
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
        topology: Topology,
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
            topology (Topology): Coupling map as an edge list or ``PyGraph``.
            initial_layout (list[int] | None): Optional initial logical-to-physical mapping; uses identity-style mapping when omitted.
            seed (int | None): Base seed for stochastic swap scoring and retry attempts.
            lookahead_size (int): Number of future two-qubit gates used in the extended SABRE cost.
            beta (float): Weight balancing extended-set cost against immediate front-set cost.
            decay_delta (float): Penalty increment added to swapped physical qubits.
            decay_lambda (float): Per-iteration multiplier that relaxes accumulated decay penalties.
            max_swaps_factor (float): Per-gate swap budget factor derived from current endpoint distance.
            max_attempts (int): Maximum number of independent routing attempts with varied seeds.
        """
        self.topology: PyGraph[int, None] = build_topology_graph(topology)
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
        logger.debug(
            "[SabreSwapPass] Running on circuit with {} gates, up to {} attempts",
            len(circuit.gates),
            max_attempt_count,
        )
        last_exception: RuntimeError | None = None
        base_seed = self.seed
        # Obtain layout hint without mutating instance attributes so repeated runs
        # do not accidentally persist stale mappings.
        layout_hint: list[int] | None = None
        if self.initial_layout is not None:
            layout_hint = list(self.initial_layout)
        elif self.context is not None and self.context.initial_layout and not self.context.layout_applied:
            # A layout pass that already retargeted the circuit leaves it in physical
            # coordinates, so re-applying its mapping here would compose it with itself.
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
                logger.warning("[SabreSwapPass] Attempt {} exceeded swap budget, retrying", attempt)
                last_exception = exc
                continue

            logger.debug(
                "[SabreSwapPass] Attempt {} routed successfully with {} swaps inserted",
                attempt,
                swap_count,
            )
            self.last_swap_count = swap_count
            self.last_final_layout = final_layout

            if self.context is not None:
                # Keys are always logical qubits. Indexing by the layout hint would key
                # the map on physical labels and can run past the end of final_layout.
                final_layout_map: LayoutMap = {
                    logical_qubit: final_layout[logical_qubit] for logical_qubit in range(len(final_layout))
                }
                self.context.final_layout = final_layout_map
                self.context.layout_applied = True
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
            logger.debug("[SabreSwapPass] Initial layout is already routable, no swaps needed")
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
        front_positions = [0] * num_logical_qubits

        # Dense positions for scoring; unused logical qubits may hold a label outside the graph, hence -1.
        position = [physical_to_dense_index.get(physical_qubit, -1) for physical_qubit in layout]
        incident_edges = [
            [
                (min(dense_index, neighbor_index), max(dense_index, neighbor_index))
                for neighbor_index in (
                    physical_to_dense_index[int(neighbor)] for neighbor in self.topology.neighbors(node)
                )
                if neighbor_index != dense_index
            ]
            for dense_index, node in enumerate(physical_nodes)
        ]
        # Disconnected pairs are scored with a large finite penalty instead of the sentinel distance.
        if max(map(max, distance_matrix), default=0) >= SabreSwapPass._UNREACHABLE_DISTANCE:
            distance_costs = [
                [
                    _UNREACHABLE_PENALTY if distance >= SabreSwapPass._UNREACHABLE_DISTANCE else distance
                    for distance in row
                ]
                for row in distance_matrix
            ]
        else:
            distance_costs = distance_matrix

        # F holds each logical qubit's first unscheduled 2Q gate; front_partners[q] lists q's partners in F.
        front_gates: set[int] = set()
        front_partners: list[list[int]] = [[] for _ in range(num_logical_qubits)]

        def add_front_head(logical_qubit: int) -> None:
            qubit_gate_indices = gate_indices_per_qubit[logical_qubit]
            if front_positions[logical_qubit] < len(qubit_gate_indices):
                gate_idx = qubit_gate_indices[front_positions[logical_qubit]]
                if gate_idx not in front_gates:
                    front_gates.add(gate_idx)
                    gate_qubit_a, gate_qubit_b = two_qubit_pairs[gate_idx]
                    front_partners[gate_qubit_a].append(gate_qubit_b)
                    front_partners[gate_qubit_b].append(gate_qubit_a)

        for logical_qubit in range(num_logical_qubits):
            add_front_head(logical_qubit)

        # Output circuit will use physical indices; ensure capacity for max physical index we may touch.
        output_num_qubits = max(
            max_physical_label_plus_one,
            max(layout) + 1 if layout else max_physical_label_plus_one,
            circuit.nqubits,
        )
        routed_circuit = Circuit(output_num_qubits)

        # Decay penalty of dense qubit d is decay[d] * decay_scale, so relaxing every qubit is a single multiply.
        decay = [0.0] * num_physical_nodes
        decayed_indices: list[int] = []
        decay_scale = 1.0
        swap_count = 0
        tie_epsilon = SabreSwapPass._TIE_EPSILON
        tie_break_probability = SabreSwapPass._TIE_BREAK_PROBABILITY
        beta = self.beta
        no_partners: tuple[int, ...] = ()
        extended_partners: list[tuple[int, ...]] = []
        extended_stale = True

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
            current_distance = distance_matrix[position[logical_qubit_a]][position[logical_qubit_b]]
            max_swaps_this_gate = int(self.max_swaps_factor * max(1, current_distance))
            swap_steps = 0
            while current_distance != 1:
                swap_steps += 1
                if swap_steps > max_swaps_this_gate:
                    raise RuntimeError(
                        f"Exceeded swap budget while routing gate {type(gate).__name__}{gate_qubits}: "
                        "graph may be disconnected or heuristic stuck."
                    )

                if extended_stale:
                    extended_partners = [no_partners] * num_logical_qubits
                    for gate_idx in self._extended_set(gate_indices_per_qubit, front_positions, self.lookahead_size):
                        gate_qubit_a, gate_qubit_b = two_qubit_pairs[gate_idx]
                        extended_partners[gate_qubit_a] += (gate_qubit_b,)
                        extended_partners[gate_qubit_b] += (gate_qubit_a,)
                    extended_stale = False

                # Candidate SWAPs are the coupling edges touching a front-gate endpoint.
                candidate_swap_edges: set[tuple[int, int]] = set()
                for front_qubit, partners in enumerate(front_partners):
                    if partners:
                        candidate_swap_edges.update(incident_edges[position[front_qubit]])

                if not candidate_swap_edges:
                    # Fallback: try swaps among touched physical nodes (should be rare)
                    touched_dense_indices = sorted(
                        {position[front_qubit] for front_qubit, partners in enumerate(front_partners) if partners}
                    )
                    if len(touched_dense_indices) < SabreSwapPass._TWO_QUBIT_ARITY:
                        raise RuntimeError("No candidate swaps available; coupling graph likely degenerate.")
                    sampled_a, sampled_b = random_generator.sample(
                        touched_dense_indices,
                        SabreSwapPass._TWO_QUBIT_ARITY,
                    )
                    candidate_swap_edges.add((min(sampled_a, sampled_b), max(sampled_a, sampled_b)))

                decay_scale *= self.decay_lambda
                if decay_scale < _DECAY_RESCALE_THRESHOLD:
                    for dense_index in decayed_indices:
                        decay[dense_index] *= decay_scale
                    decay_scale = 1.0

                # Score candidates by the cost change of their qubits' F/E gates; a gate between the two keeps its cost.
                position_a, position_b = position[logical_qubit_a], position[logical_qubit_b]
                improving_edge: tuple[int, int] | None = None
                improving_delta = math.inf
                best_edge = next(iter(candidate_swap_edges))
                best_delta = math.inf
                for dense_a, dense_b in candidate_swap_edges:
                    logical_on_a = inverse_layout[dense_a]
                    logical_on_b = inverse_layout[dense_b]
                    costs_from_a = distance_costs[dense_a]
                    costs_from_b = distance_costs[dense_b]
                    decay_a = decay[dense_a]
                    decay_b = decay[dense_b]
                    front_delta = 0.0
                    extended_delta = 0.0
                    if logical_on_a is not None:
                        for partner in front_partners[logical_on_a]:
                            if partner != logical_on_b:
                                partner_dense = position[partner]
                                partner_decay = decay[partner_dense]
                                front_delta += costs_from_b[partner_dense] * (
                                    1.0 + (decay_b + partner_decay) * decay_scale
                                ) - costs_from_a[partner_dense] * (1.0 + (decay_a + partner_decay) * decay_scale)
                        for partner in extended_partners[logical_on_a]:
                            if partner != logical_on_b:
                                partner_dense = position[partner]
                                extended_delta += costs_from_b[partner_dense] - costs_from_a[partner_dense]
                    if logical_on_b is not None:
                        for partner in front_partners[logical_on_b]:
                            if partner != logical_on_a:
                                partner_dense = position[partner]
                                partner_decay = decay[partner_dense]
                                front_delta += costs_from_a[partner_dense] * (
                                    1.0 + (decay_a + partner_decay) * decay_scale
                                ) - costs_from_b[partner_dense] * (1.0 + (decay_b + partner_decay) * decay_scale)
                        for partner in extended_partners[logical_on_b]:
                            if partner != logical_on_a:
                                partner_dense = position[partner]
                                extended_delta += costs_from_a[partner_dense] - costs_from_b[partner_dense]
                    candidate_delta = front_delta + beta * extended_delta

                    # Distance of the gate being routed after the hypothetical swap
                    new_position_a = position_a
                    if position_a == dense_a:
                        new_position_a = dense_b
                    elif position_a == dense_b:
                        new_position_a = dense_a
                    new_position_b = position_b
                    if position_b == dense_a:
                        new_position_b = dense_b
                    elif position_b == dense_b:
                        new_position_b = dense_a
                    if distance_matrix[new_position_a][new_position_b] < current_distance:
                        if candidate_delta < improving_delta - tie_epsilon:
                            improving_delta = candidate_delta
                            improving_edge = (dense_a, dense_b)
                        elif (
                            candidate_delta <= improving_delta + tie_epsilon
                            and random_generator.random() < tie_break_probability
                        ):
                            improving_edge = (dense_a, dense_b)

                    if candidate_delta < best_delta - tie_epsilon:
                        best_delta = candidate_delta
                        best_edge = (dense_a, dense_b)
                    elif (
                        candidate_delta <= best_delta + tie_epsilon
                        and random_generator.random() < tie_break_probability
                    ):
                        best_edge = (dense_a, dense_b)

                # Apply chosen SWAP physically and in the mapping
                chosen_a, chosen_b = improving_edge if improving_edge is not None else best_edge
                chosen_physical_a, chosen_physical_b = physical_nodes[chosen_a], physical_nodes[chosen_b]
                routed_circuit.add(SWAP(chosen_physical_a, chosen_physical_b))
                swap_count += 1
                logger.trace(
                    "[SabreSwapPass] Inserted SWAP on physical qubits {} and {}",
                    chosen_physical_a,
                    chosen_physical_b,
                )
                logical_at_a = inverse_layout[chosen_a]
                logical_at_b = inverse_layout[chosen_b]
                self._swap_mapping(
                    layout,
                    inverse_layout,
                    physical_to_dense_index,
                    chosen_physical_a,
                    chosen_physical_b,
                    logical_at_a,
                    logical_at_b,
                )
                for moved_logical in (logical_at_a, logical_at_b):
                    if moved_logical is not None:
                        position[moved_logical] = physical_to_dense_index[layout[moved_logical]]
                for dense_index in (chosen_a, chosen_b):
                    if not decay[dense_index]:
                        decayed_indices.append(dense_index)
                    decay[dense_index] += self.decay_delta / decay_scale
                current_distance = distance_matrix[position[logical_qubit_a]][position[logical_qubit_b]]

            # Now adjacent: emit the mapped 2Q gate
            mapped_qubits = (layout[logical_qubit_a], layout[logical_qubit_b])
            routed_circuit.add(self._retarget_gate(gate, mapped_qubits))
            # The routed gate is the head of both its qubits; replace it in F with their next gates.
            front_gates.discard(two_qubit_index)
            front_partners[logical_qubit_a].remove(logical_qubit_b)
            front_partners[logical_qubit_b].remove(logical_qubit_a)
            front_positions[logical_qubit_a] += 1
            front_positions[logical_qubit_b] += 1
            add_front_head(logical_qubit_a)
            add_front_head(logical_qubit_b)
            extended_stale = True

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
            if len(gate_qubits) != SabreSwapPass._TWO_QUBIT_ARITY or isinstance(gate, M):
                continue
            if not self.topology.has_edge(layout[gate_qubits[0]], layout[gate_qubits[1]]):
                return False
        return True

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
            if len(gate_qubits) == SabreSwapPass._TWO_QUBIT_ARITY and not isinstance(gate, M):
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
