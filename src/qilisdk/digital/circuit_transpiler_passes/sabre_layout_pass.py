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

from __future__ import annotations

import math
import random
from collections import deque
from copy import deepcopy
from typing import Iterable, TypeGuard

from rustworkx import PyGraph

from qilisdk.digital import (
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


class SabreLayoutPass(CircuitTranspilerPass):
    """
    A SABRE-style initial layout pass (no SWAP insertion).
    It computes a good logical→physical qubit mapping for a given coupling graph
    and returns a *new* Circuit with all gates retargeted to the chosen physical qubits.

    Key features
    ------------
    - Uses rustworkx PyGraph as the coupling graph (undirected, unweighted).
    - Implements SABRE's heuristic:
        * Front layer (first unscheduled 2Q gates on each qubit).
        * Extended look-ahead set of upcoming 2Q gates.
        * Cost = sum distances(front) + beta * sum distances(lookahead),
          with a light decay penalty on recently swapped physical qubits.
    - Runs several randomized trials and keeps the best layout.

    Parameters
    ----------
    coupling : PyGraph
        Undirected coupling graph whose node indices represent *physical* qubits.
        Node labels need not form a contiguous range. Edges indicate allowed 2Q interactions.
    num_trials : int
        Number of random initializations to try (keeps the best).
    seed : int | None
        RNG seed for reproducibility (affects initial layout and tie breaks).
    lookahead_size : int
        Max size of the extended set (SABRE "E").
    beta : float
        Weight for the extended set in the cost function.
    decay_delta : float
        Increment added to the decay penalty for the two physical qubits whenever
        a swap *would* be applied during simulation.
    decay_lambda : float
        Decay multiplier applied each iteration to gradually forget old penalties.

    Results
    -------
    - Returns a *new* Circuit with all gates retargeted to physical qubits.
    - Exposes `last_layout` (list[int]) mapping logical → physical.
      Also `last_score` as a diagnostic (lower is better).

    Notes
    -----
    * This pass performs **layout only**. Routing/SWAP insertion should be done by a separate pass.
    * If the coupling graph has more physical qubits than the input circuit, the returned
      circuit's `nqubits` will be enlarged as needed so that physical indices are in range.
    """

    _TWO_QUBIT_ARITY = 2
    _UNREACHABLE_DISTANCE = 1_000_000_000
    _TIE_EPSILON = 1e-12
    _TIE_BREAK_PROBABILITY = 0.5

    def __init__(
        self,
        topology: PyGraph,
        *,
        num_trials: int = 8,
        seed: int | None = None,
        lookahead_size: int = 10,
        beta: float = 0.5,
        decay_delta: float = 0.001,
        decay_lambda: float = 0.99,
    ) -> None:
        """Initialize a SABRE layout pass.

        Args:
            topology (PyGraph): Undirected coupling graph where node indices are physical qubits.
            num_trials (int): Number of randomized layout trials.
            seed (int | None): RNG seed for reproducible trials.
            lookahead_size (int): Maximum number of gates in the look-ahead set.
            beta (float): Weight assigned to look-ahead cost.
            decay_delta (float): Increment applied to decay on selected swap endpoints.
            decay_lambda (float): Per-iteration decay factor for penalties.

        Raises:
            TypeError: If ``topology`` is not a ``rustworkx.PyGraph`` instance.
        """
        self.topology = topology
        self.num_trials = max(1, int(num_trials))
        self.seed = seed
        self.lookahead_size = int(lookahead_size)
        self.beta = float(beta)
        self.decay_delta = float(decay_delta)
        self.decay_lambda = float(decay_lambda)

        self.last_layout: list[int] | None = None
        self.last_score: float | None = None

        # Validate coupling graph is undirected PyGraph (rustworkx enforces this by type)
        if not isinstance(topology, PyGraph):
            raise TypeError("SabreLayoutPass requires a rustworkx.PyGraph (undirected).")

    # --------- public API ---------

    def run(self, circuit: Circuit) -> Circuit:
        """Compute a layout and return a retargeted copy of the circuit.

        Args:
            circuit (Circuit): Logical circuit to place onto physical qubits.

        Returns:
            Circuit: New circuit retargeted according to the selected layout.

        Raises:
            ValueError: If the topology is empty or has fewer physical qubits than required by ``circuit``.
        """
        random_generator = random.Random(self.seed)

        num_logical_qubits = circuit.nqubits
        physical_nodes = sorted(int(x) for x in self.topology.node_indices())
        if not physical_nodes:
            raise ValueError("Coupling graph has no nodes.")
        num_physical_nodes = len(physical_nodes)
        if num_logical_qubits > num_physical_nodes:
            raise ValueError(
                f"Coupling graph has {num_physical_nodes} nodes but circuit needs {num_logical_qubits} qubits."
            )
        physical_to_dense_index = {node: dense_index for dense_index, node in enumerate(physical_nodes)}
        max_physical_label_plus_one = max(physical_nodes) + 1

        # Precompute all-pairs shortest-path distances on the coupling graph.
        # We use an internal BFS on the rustworkx graph to avoid relying on version-specific APIs.
        distance_matrix = self._all_pairs_shortest_path_unweighted(
            self.topology, physical_nodes, physical_to_dense_index
        )

        # Build the list of 2Q gates and per-qubit indices for the SABRE simulation.
        two_qubit_gate_indices, two_qubit_pairs, gate_indices_per_qubit = self._two_qubit_structure(circuit)

        # edge case: circuits with no 2Q gates -> trivial identity layout
        if not two_qubit_gate_indices:
            layout = physical_nodes[:num_logical_qubits]
            self.last_layout = layout
            self.last_score = 0.0
            if self.context is not None:
                self.context.initial_layout = self.last_layout
            return self._retarget_circuit(circuit, layout, max_physical_label_plus_one)

        # Multi-trial SABRE simulation; keep the best layout according to final cost.
        best_layout: list[int] | None = None
        best_score: float = math.inf

        for _ in range(self.num_trials):
            initial_layout = self._random_initial_layout(random_generator, num_logical_qubits, physical_nodes)
            layout, score = self._sabre_simulate_layout(
                initial_layout,
                distance_matrix,
                two_qubit_pairs,
                gate_indices_per_qubit,
                random_generator,
                physical_nodes,
                physical_to_dense_index,
            )
            if score < best_score:
                best_layout = layout
                best_score = score

        self.last_layout = best_layout
        self.last_score = best_score

        if self.context is not None:
            self.context.initial_layout = self.last_layout or []

        new_circuit = self._retarget_circuit(circuit, best_layout, max_physical_label_plus_one)  # type: ignore[arg-type]

        self.append_circuit_to_context(new_circuit)

        # Return a new circuit with all gates mapped to the chosen physical qubits.
        return new_circuit

    # --------- SABRE core ---------

    def _sabre_simulate_layout(
        self,
        layout: list[int],  # logical -> physical
        distance_matrix: list[list[int]],
        two_qubit_pairs: list[tuple[int, int]],
        gate_indices_per_qubit: list[list[int]],
        random_generator: random.Random,
        physical_nodes: list[int],
        physical_to_dense_index: dict[int, int],
    ) -> tuple[list[int], float]:
        """Run one SABRE simulation trial and score the resulting layout.

        This is a layout-only variant: SWAPs are simulated on the mapping but
        never inserted into the output circuit.

        Args:
            layout (list[int]): Initial logical-to-physical mapping.
            distance_matrix (list[list[int]]): All-pairs shortest-path distance matrix for physical qubits.
            two_qubit_pairs (list[tuple[int, int]]): Ordered 2-qubit interactions as logical-qubit pairs.
            gate_indices_per_qubit (list[list[int]]): For each logical qubit, indices of ``two_qubit_pairs`` touching it.
            random_generator (random.Random): Random generator used for tie-breaking.
            physical_nodes (list[int]): Physical node labels.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            tuple[list[int], float]: Final mapping and heuristic score for this trial.
        """
        num_logical_qubits = len(layout)
        num_physical_qubits = len(physical_nodes)
        # inverse layout: physical -> logical (dense indices)
        inverse_layout = self._invert_layout(layout, physical_to_dense_index)

        scheduled = [False] * len(two_qubit_pairs)
        front_positions = [
            0
        ] * num_logical_qubits  # per-qubit pointer to the next 2Q gate index in gate_indices_per_qubit[logical_qubit]
        decay = [0.0] * num_physical_qubits  # physical-qubit penalty stored with dense indexing

        # Sum of executed front distances (diagnostic score)
        score_accum = 0.0

        def advance_front_for(logical_qubit: int) -> None:
            """Move ``front_positions[logical_qubit]`` to the first unscheduled interaction.

            Args:
                logical_qubit (int): Logical qubit index whose front pointer is advanced.
            """
            qubit_gate_indices = gate_indices_per_qubit[logical_qubit]
            cursor = front_positions[logical_qubit]
            while cursor < len(qubit_gate_indices) and scheduled[qubit_gate_indices[cursor]]:
                cursor += 1
            front_positions[logical_qubit] = cursor

        def front_set() -> set[int]:
            """Build the current front layer of unscheduled interactions.

            Returns:
                set[int]: Indices of first unscheduled 2-qubit interactions across all logical qubits.
            """
            front_gate_indices: set[int] = set()
            for logical_qubit in range(num_logical_qubits):
                if front_positions[logical_qubit] < len(gate_indices_per_qubit[logical_qubit]):
                    front_gate_indices.add(gate_indices_per_qubit[logical_qubit][front_positions[logical_qubit]])
            return front_gate_indices

        # Initialize pointers
        for logical_qubit in range(num_logical_qubits):
            advance_front_for(logical_qubit)

        # Main loop: continue until all 2Q gates are scheduled
        remaining = len(two_qubit_pairs)
        while remaining:
            # Try to greedily "apply" any executable front gates (whose mapped endpoints are adjacent).
            progressed = True
            while progressed:
                progressed = False
                front_gate_indices = list(front_set())
                for gate_idx in front_gate_indices:
                    if scheduled[gate_idx]:
                        continue
                    logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_idx]
                    physical_qubit_a, physical_qubit_b = layout[logical_qubit_a], layout[logical_qubit_b]
                    if (
                        distance_matrix[physical_to_dense_index[physical_qubit_a]][
                            physical_to_dense_index[physical_qubit_b]
                        ]
                        == 1
                    ):
                        # "Execute" this 2Q gate in the simulation.
                        scheduled[gate_idx] = True
                        remaining -= 1
                        progressed = True
                        # Diagnostic score: adjacency cost is 1 by definition
                        score_accum += 1.0
                        # Advance fronts for both logical qubits involved
                        advance_front_for(logical_qubit_a)
                        advance_front_for(logical_qubit_b)

            if not remaining:
                break  # done

            # No front gate was executable: pick a simulated SWAP using SABRE heuristic.
            front_gate_indices: list[int] = list(front_set())
            # Build candidate physical edges from neighbors of endpoints of non-executable front gates.
            candidate_edges: set[tuple[int, int]] = set()
            touched_physical_nodes: set[int] = set()
            for gate_idx in front_gate_indices:
                logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_idx]
                physical_qubit_a, physical_qubit_b = layout[logical_qubit_a], layout[logical_qubit_b]
                touched_physical_nodes.add(physical_qubit_a)
                touched_physical_nodes.add(physical_qubit_b)
                for neighbor in self.topology.neighbors(physical_qubit_a):
                    edge_start, edge_end = int(physical_qubit_a), int(neighbor)
                    if edge_start != edge_end:
                        candidate_edges.add((min(edge_start, edge_end), max(edge_start, edge_end)))
                for neighbor in self.topology.neighbors(physical_qubit_b):
                    edge_start, edge_end = int(physical_qubit_b), int(neighbor)
                    if edge_start != edge_end:
                        candidate_edges.add((min(edge_start, edge_end), max(edge_start, edge_end)))

            if not candidate_edges:
                # Graph might be disconnected or trivial; fall back to a random valid swap over touched qubits.
                # This won't crash; it just gives the algorithm a way to keep moving.
                candidate_physical_nodes = (
                    sorted(touched_physical_nodes) if touched_physical_nodes else physical_nodes.copy()
                )
                if len(candidate_physical_nodes) >= SabreLayoutPass._TWO_QUBIT_ARITY:
                    sampled_a, sampled_b = random_generator.sample(
                        candidate_physical_nodes, SabreLayoutPass._TWO_QUBIT_ARITY
                    )
                    candidate_edges.add((min(sampled_a, sampled_b), max(sampled_a, sampled_b)))
                else:
                    # Degenerate case: nothing to do; break to avoid infinite loop.
                    break

            # Decay relaxation
            for physical_dense_index in range(num_physical_qubits):
                decay[physical_dense_index] *= self.decay_lambda

            # Evaluate heuristic for each candidate swap
            best_edge: tuple[int, int] | None = None
            best_cost: float = math.inf

            # Precompute lookahead set E (extended)
            extended_gate_indices = self._extended_set(
                two_qubit_pairs, gate_indices_per_qubit, front_positions, self.lookahead_size
            )

            for physical_a, physical_b in candidate_edges:
                # Virtually swap logical assignments at physical nodes a and b
                logical_on_a = inverse_layout[physical_to_dense_index[physical_a]]
                logical_on_b = inverse_layout[physical_to_dense_index[physical_b]]
                # Logical qubit may be "unassigned" if device > logical; ensure we handle that.
                # If either side is None, this swap doesn't affect distances; we still allow it.
                # Simulate new layout distances
                # (temporarily mutate layout/inverse_layout, compute cost, then revert).
                self._swap_mapping(
                    layout,
                    inverse_layout,
                    physical_to_dense_index,
                    physical_a,
                    physical_b,
                    logical_on_a,
                    logical_on_b,
                )

                front_cost = self._cost_front(
                    front_gate_indices, layout, two_qubit_pairs, distance_matrix, decay, physical_to_dense_index
                )
                extended_cost = self._cost_front(
                    extended_gate_indices,
                    layout,
                    two_qubit_pairs,
                    distance_matrix,
                    None,
                    physical_to_dense_index,
                )  # no decay on E
                candidate_cost = front_cost + self.beta * extended_cost

                # Revert the swap
                self._swap_mapping(
                    layout,
                    inverse_layout,
                    physical_to_dense_index,
                    physical_a,
                    physical_b,
                    logical_on_b,
                    logical_on_a,
                )

                if candidate_cost < best_cost - SabreLayoutPass._TIE_EPSILON:
                    best_cost = candidate_cost
                    best_edge = (physical_a, physical_b)
                elif abs(candidate_cost - best_cost) <= SabreLayoutPass._TIE_EPSILON:
                    # Tie-break randomly for diversification
                    if random_generator.random() < SabreLayoutPass._TIE_BREAK_PROBABILITY:
                        best_edge = (physical_a, physical_b)

            # Apply the chosen swap *to the mapping only* (layout-only SABRE).
            # assert best_edge is not None
            chosen_physical_a, chosen_physical_b = best_edge
            logical_on_a = inverse_layout[physical_to_dense_index[chosen_physical_a]]
            logical_on_b = inverse_layout[physical_to_dense_index[chosen_physical_b]]
            self._swap_mapping(
                layout,
                inverse_layout,
                physical_to_dense_index,
                chosen_physical_a,
                chosen_physical_b,
                logical_on_a,
                logical_on_b,
            )
            # Increase decay on touched physical qubits
            decay[physical_to_dense_index[chosen_physical_a]] += self.decay_delta
            decay[physical_to_dense_index[chosen_physical_b]] += self.decay_delta

        # Final diagnostic cost: re-evaluate sum of distances for the entire 2Q list under final layout
        final_distance_cost = self._cost_front(
            set(range(len(two_qubit_pairs))), layout, two_qubit_pairs, distance_matrix, None, physical_to_dense_index
        )
        # Combine both measures mildly
        blended_score = 0.5 * final_distance_cost + 0.5 * score_accum
        return layout, float(blended_score)

    # --------- helpers: front / extended set / costs ---------

    @staticmethod
    def _extended_set(
        two_qubit_pairs: list[tuple[int, int]],
        gate_indices_per_qubit: list[list[int]],
        front_positions: list[int],
        max_size: int,
    ) -> set[int]:
        """Collect upcoming 2-qubit gates beyond the current front layer.

        Args:
            two_qubit_pairs (list[tuple[int, int]]): Ordered 2-qubit interaction list.
            gate_indices_per_qubit (list[list[int]]): Per-logical-qubit interaction indices.
            front_positions (list[int]): Current front pointers into ``gate_indices_per_qubit``.
            max_size (int): Maximum number of indices to include.

        Returns:
            set[int]: Set of selected interaction indices for look-ahead.
        """
        extended_gate_indices: set[int] = set()
        if max_size <= 0:
            return extended_gate_indices

        # Start from the qubits that currently participate in front gates
        # (i.e., those qubits whose front_positions[logical_qubit] points to some gate).
        frontier_qubits = [
            logical_qubit
            for logical_qubit in range(len(front_positions))
            if front_positions[logical_qubit] < len(gate_indices_per_qubit[logical_qubit])
        ]
        # Scan forward a few steps on each such qubit
        budget = max_size
        for logical_qubit in frontier_qubits:
            cursor = front_positions[logical_qubit] + 1
            while cursor < len(gate_indices_per_qubit[logical_qubit]) and budget > 0:
                gate_idx = gate_indices_per_qubit[logical_qubit][cursor]
                extended_gate_indices.add(gate_idx)
                budget -= 1
                cursor += 1
                if budget == 0:
                    break
            if budget == 0:
                break
        return extended_gate_indices

    @staticmethod
    def _cost_front(
        gate_indices: Iterable[int],
        layout: list[int],
        two_qubit_pairs: list[tuple[int, int]],
        distance_matrix: list[list[int]],
        decay: list[float] | None,
        physical_to_dense_index: dict[int, int],
    ) -> float:
        """Evaluate distance-based heuristic cost for a set of 2-qubit gates.

        Args:
            gate_indices (Iterable[int]): Indices of interactions to evaluate.
            layout (list[int]): Logical-to-physical mapping.
            two_qubit_pairs (list[tuple[int, int]]): Logical interaction pairs.
            distance_matrix (list[list[int]]): Physical shortest-path distance matrix.
            decay (list[float] | None): Optional per-physical-node decay penalties.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            float: Accumulated heuristic cost.
        """
        total_cost = 0.0
        for gate_idx in gate_indices:
            logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_idx]
            physical_qubit_a, physical_qubit_b = layout[logical_qubit_a], layout[logical_qubit_b]
            physical_index_a = physical_to_dense_index[physical_qubit_a]
            physical_index_b = physical_to_dense_index[physical_qubit_b]
            path_distance = distance_matrix[physical_index_a][physical_index_b]
            if path_distance >= SabreLayoutPass._UNREACHABLE_DISTANCE:
                # Disconnected: incur a large penalty to discourage this layout
                path_distance = 1e6
            if decay is not None:
                total_cost += path_distance * (1.0 + decay[physical_index_a] + decay[physical_index_b])
            else:
                total_cost += path_distance
        return total_cost

    @staticmethod
    def _random_initial_layout(
        random_generator: random.Random,
        num_logical_qubits: int,
        physical_nodes: list[int],
    ) -> list[int]:
        """Sample a random injective logical-to-physical layout.

        Args:
            random_generator (random.Random): Random generator used for shuffling.
            num_logical_qubits (int): Number of logical qubits.
            physical_nodes (list[int]): Available physical node labels.

        Returns:
            list[int]: Mapping ``layout[q_logical] = q_physical``.

        Raises:
            ValueError: If there are fewer physical nodes than logical qubits.
        """
        if len(physical_nodes) < num_logical_qubits:
            raise ValueError(f"Coupling graph has only {len(physical_nodes)} nodes; need ≥ {num_logical_qubits}.")
        nodes = physical_nodes.copy()  # copy so we can shuffle deterministically
        random_generator.shuffle(nodes)
        return nodes[:num_logical_qubits]

    # --------- helpers: mapping & structure ---------

    @staticmethod
    def _invert_layout(layout: list[int], physical_to_dense_index: dict[int, int]) -> list[int | None]:
        """Build inverse mapping from dense physical index to logical qubit.

        Args:
            layout (list[int]): Logical-to-physical mapping.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            list[int | None]: Inverse mapping where missing assignments are represented by ``None``.
        """
        inverse_layout = [None] * len(physical_to_dense_index)
        for logical_qubit, physical_qubit in enumerate(layout):
            physical_dense_index = physical_to_dense_index.get(physical_qubit)
            if physical_dense_index is not None:
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
        """Swap logical assignments between two physical nodes.

        Args:
            layout (list[int]): Logical-to-physical mapping to mutate.
            inverse_layout (list[int | None]): Inverse dense-physical mapping to mutate.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.
            physical_node_a (int): First physical node label.
            physical_node_b (int): Second physical node label.
            logical_at_a (int | None): Logical qubit currently at ``physical_node_a``, if any.
            logical_at_b (int | None): Logical qubit currently at ``physical_node_b``, if any.
        """
        # Update inverse first
        dense_index_a = physical_to_dense_index[physical_node_a]
        dense_index_b = physical_to_dense_index[physical_node_b]
        inverse_layout[dense_index_a], inverse_layout[dense_index_b] = logical_at_b, logical_at_a
        # Then forward mapping (only if logicals exist)
        if logical_at_a is not None:
            layout[logical_at_a] = physical_node_b
        if logical_at_b is not None:
            layout[logical_at_b] = physical_node_a

    @staticmethod
    def _two_qubit_structure(circuit: Circuit) -> tuple[list[int], list[tuple[int, int]], list[list[int]]]:
        """Extract 2-qubit interaction structure from a circuit.

        Args:
            circuit (Circuit): Input circuit in logical qubits.

        Returns:
            tuple[list[int], list[tuple[int, int]], list[list[int]]]: Tuple with indices of 2-qubit gates in ``circuit.gates``, corresponding logical-qubit pairs, and per-logical-qubit interaction index lists.
        """
        num_logical_qubits = circuit.nqubits
        two_qubit_gate_indices: list[int] = []
        two_qubit_pairs: list[tuple[int, int]] = []
        for gate_idx, gate in enumerate(circuit.gates):
            gate_qubits = gate.qubits
            if len(gate_qubits) == SabreLayoutPass._TWO_QUBIT_ARITY:
                two_qubit_gate_indices.append(gate_idx)
                two_qubit_pairs.append((gate_qubits[0], gate_qubits[1]))
        gate_indices_per_qubit: list[list[int]] = [[] for _ in range(num_logical_qubits)]
        for interaction_idx, (logical_qubit_a, logical_qubit_b) in enumerate(two_qubit_pairs):
            gate_indices_per_qubit[logical_qubit_a].append(interaction_idx)
            gate_indices_per_qubit[logical_qubit_b].append(interaction_idx)
        return two_qubit_gate_indices, two_qubit_pairs, gate_indices_per_qubit

    @staticmethod
    def _all_pairs_shortest_path_unweighted(
        graph: PyGraph,
        physical_nodes: list[int],
        physical_to_dense_index: dict[int, int],
    ) -> list[list[int]]:
        """Compute all-pairs unweighted shortest-path distances by BFS.

        Args:
            graph (PyGraph): Undirected coupling graph.
            physical_nodes (list[int]): Physical node labels.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            list[list[int]]: Dense distance matrix indexed by ``physical_to_dense_index[label]``.
        """

        inf_distance = SabreLayoutPass._UNREACHABLE_DISTANCE
        distance_matrix = [[inf_distance] * len(physical_nodes) for _ in physical_nodes]
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
                    neighbor_label = int(neighbor)
                    if neighbor_label not in seen:
                        seen.add(neighbor_label)
                        neighbor_dense_index = physical_to_dense_index[neighbor_label]
                        distance_matrix[source_dense_index][neighbor_dense_index] = current_distance + 1
                        queue.append(neighbor_label)
        return distance_matrix

    # --------- retargeting to the chosen layout ---------

    def _retarget_circuit(self, circuit: Circuit, layout: list[int], max_physical_label_plus_one: int) -> Circuit:
        """Retarget all gates in ``circuit`` using a chosen layout.

        Args:
            circuit (Circuit): Input logical circuit.
            layout (list[int]): Logical-to-physical mapping.
            max_physical_label_plus_one (int): Upper bound needed to keep output qubit range valid for sparse physical labels.

        Returns:
            Circuit: New circuit with all gates mapped to physical qubits.
        """
        # The output circuit must have enough qubits to accommodate the maximum physical index.
        output_num_qubits = max(
            circuit.nqubits,
            (max(layout) + 1) if layout else circuit.nqubits,
            max_physical_label_plus_one,
        )
        retargeted_circuit = Circuit(output_num_qubits)
        for gate in circuit.gates:
            mapped_qubits = tuple(layout[logical_qubit] for logical_qubit in gate.qubits)
            retargeted_circuit.add(self._retarget_gate(gate, mapped_qubits))
        return retargeted_circuit

    def _retarget_gate(self, gate: Gate, mapped_qubits: tuple[int, ...]) -> Gate:
        """
        Recreate ``gate`` on ``mapped_qubits`` by deep-copying and remapping indices.

        The method deep-copies the gate object to preserve concrete gate type and all parameters, then updates control and target qubit tuples recursively through wrapped gates.

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
            SabreLayoutPass._remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
            return

        if _is_adjoint_or_exponential(gate):
            SabreLayoutPass._remap_gate_qubits_inplace(gate.basic_gate, qubit_map)
            return

        raise NotImplementedError(
            f"Retargeting not implemented for gate type {type(gate).__name__} with arity {gate.nqubits}"
        )
