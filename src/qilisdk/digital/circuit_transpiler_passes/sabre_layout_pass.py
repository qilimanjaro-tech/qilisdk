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
from typing import TYPE_CHECKING

from loguru import logger

from qilisdk.digital import Circuit, M
from qilisdk.digital.topology import build_topology_graph

from .circuit_transpiler_pass import CircuitTranspilerPass

# Cost of an interaction between disconnected physical qubits, used instead of the sentinel distance.
_UNREACHABLE_PENALTY = 1e6
# Lazily stored decay values are folded back into the scale once it drops below this.
_DECAY_RESCALE_THRESHOLD = 1e-100

if TYPE_CHECKING:
    from rustworkx import PyGraph

    from qilisdk.digital.types import Topology


class SabreLayoutPass(CircuitTranspilerPass):
    """
    A SABRE-style initial layout pass (no SWAP insertion).
    It computes a good logical→physical qubit mapping for a given coupling graph
    and, unless ``apply_layout`` is ``False``, returns a *new* Circuit with all gates
    retargeted to the chosen physical qubits.

    Key features
    ------------
    - Accepts either a topology edge list or a rustworkx ``PyGraph`` and
      normalizes it to an undirected coupling graph internally.
    - Implements SABRE's heuristic:
        * Front layer (unscheduled 2Q gates that are the next gate on both of their qubits).
        * Extended look-ahead set of upcoming 2Q gates.
        * Cost = sum distances(front) + beta * sum distances(lookahead),
          with a light decay penalty on recently swapped physical qubits.
    - Each trial starts from a random layout and refines it over several
      rounds: routing the reversed circuit ends on a mapping tuned for the
      start of the circuit, and routing the circuit forward from it scores that
      mapping and seeds the next round. The candidate needing the fewest
      simulated SWAPs across all trials and rounds wins.

    Parameters
    ----------
    topology : Topology
        Coupling map provided either as an edge list or as an undirected
        ``PyGraph`` whose node indices represent *physical* qubits.
    num_trials : int
        Number of random initializations to try (keeps the best).
    num_iterations : int
        Backward-forward refinement rounds per trial. Each round routes the
        reversed circuit and then the circuit itself, and scores the layout
        it produces by the SWAPs the forward simulation needed.
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
    release_valve_factor : float
        Number of simulated swaps per physical qubit allowed without executing a
        gate before the closest front gate is forced through along a shortest path.
    apply_layout : bool
        Whether to return the circuit retargeted onto the chosen physical qubits.
        When ``False`` the pass only records the layout and returns the input
        circuit untouched, leaving it to a following router such as
        ``SabreSwapPass`` to apply the mapping. Use ``False`` when this pass is
        paired with a router that reads ``context.initial_layout``, otherwise the
        layout is applied twice.

    Results
    -------
    - Returns a *new* Circuit with all gates retargeted to physical qubits, unless
      ``apply_layout`` is ``False``.
    - Exposes `last_layout` (list[int]) mapping logical → physical.
      Also `last_score`, the number of SWAPs the forward simulation needed (lower is better).

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
        topology: Topology,
        *,
        num_trials: int = 2,
        num_iterations: int = 4,
        seed: int | None = None,
        lookahead_size: int = 10,
        beta: float = 0.5,
        decay_delta: float = 0.001,
        decay_lambda: float = 0.99,
        release_valve_factor: float = 10.0,
        apply_layout: bool = True,
    ) -> None:
        """Initialize a SABRE layout pass.

        Args:
            topology (Topology): Coupling map as an edge list or ``PyGraph``.
            num_trials (int): Number of randomized layout trials.
            num_iterations (int): Backward-forward refinement rounds per trial.
            seed (int | None): RNG seed for reproducible trials.
            lookahead_size (int): Maximum number of gates in the look-ahead set.
            beta (float): Weight assigned to look-ahead cost.
            decay_delta (float): Increment applied to decay on selected swap endpoints.
            decay_lambda (float): Per-iteration decay factor for penalties.
            release_valve_factor (float): Swaps allowed per physical qubit without executing a gate before
                the closest front gate is forced through.
            apply_layout (bool): Whether to retarget the circuit onto the chosen physical
                qubits. Set to ``False`` when a following router applies
                ``context.initial_layout`` itself.
        """
        self.topology: PyGraph[int, None] = build_topology_graph(topology)
        self.num_trials = max(1, int(num_trials))
        self.num_iterations = max(1, int(num_iterations))
        self.seed = seed
        self.lookahead_size = int(lookahead_size)
        self.beta = float(beta)
        self.decay_delta = float(decay_delta)
        self.decay_lambda = float(decay_lambda)
        self.release_valve_factor = float(release_valve_factor)
        self.apply_layout = bool(apply_layout)

        self.last_layout: list[int] | None = None
        self.last_score: float | None = None

    # --------- public API ---------

    def run(self, circuit: Circuit) -> Circuit:
        """Compute a layout and return a retargeted copy of the circuit.

        Args:
            circuit (Circuit): Logical circuit to place onto physical qubits.

        Returns:
            Circuit: New circuit retargeted according to the selected layout, or ``circuit``
            unchanged when ``apply_layout`` is ``False``.

        Raises:
            ValueError: If the topology is empty or has fewer physical qubits than required by ``circuit``.
        """
        random_generator = random.Random(self.seed)

        num_logical_qubits = circuit.nqubits
        logger.debug(
            "[SabreLayoutPass] Running on {} logical qubits over {} trials",
            num_logical_qubits,
            self.num_trials,
        )
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
            logger.debug("[SabreLayoutPass] No two-qubit gates, using trivial identity layout")
            layout = physical_nodes[:num_logical_qubits]
            self.last_layout = layout
            self.last_score = 0.0
            if self.context is not None:
                self.context.initial_layout = self.last_layout
                self.context.layout_applied = self.apply_layout
            if not self.apply_layout:
                return circuit
            return self._retarget_circuit(circuit, layout, max_physical_label_plus_one)

        # Each trial refines its layout by routing the circuit backwards and forwards.
        last_pair_index = len(two_qubit_pairs) - 1
        reversed_pairs = two_qubit_pairs[::-1]
        reversed_gate_indices_per_qubit = [
            [last_pair_index - gate_idx for gate_idx in reversed(qubit_gate_indices)]
            for qubit_gate_indices in gate_indices_per_qubit
        ]

        # Multi-trial SABRE: keep the layout whose forward simulation needs the fewest SWAPs.
        best_layout: list[int] | None = None
        best_score: float = math.inf

        for trial in range(self.num_trials):
            layout = self._random_initial_layout(random_generator, num_logical_qubits, physical_nodes)
            for iteration in range(self.num_iterations):
                # Reverse routing tunes a layout for the circuit start; forward routing scores it and seeds the next round.
                candidate_layout, _ = self._sabre_simulate_layout(
                    layout,
                    distance_matrix,
                    reversed_pairs,
                    reversed_gate_indices_per_qubit,
                    random_generator,
                    physical_nodes,
                    physical_to_dense_index,
                )
                layout, score = self._sabre_simulate_layout(
                    candidate_layout,
                    distance_matrix,
                    two_qubit_pairs,
                    gate_indices_per_qubit,
                    random_generator,
                    physical_nodes,
                    physical_to_dense_index,
                )
                logger.trace("[SabreLayoutPass] Trial {} iteration {} scored {}", trial, iteration, score)
                if score < best_score:
                    best_layout = candidate_layout
                    best_score = score

        self.last_layout = best_layout
        self.last_score = best_score
        logger.debug("[SabreLayoutPass] Selected layout {} with score {}", best_layout, best_score)

        if self.context is not None:
            self.context.initial_layout = self.last_layout or []
            self.context.layout_applied = self.apply_layout

        # When the layout is not applied here, a following router applies
        # context.initial_layout instead. Applying it in both places composes the
        # mapping with itself and scrambles the placement.
        new_circuit = (
            self._retarget_circuit(circuit, best_layout, max_physical_label_plus_one)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]
            if self.apply_layout
            else circuit
        )

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
        """Run one SABRE routing simulation and count the SWAPs it needs.

        This is a layout-only variant: SWAPs are simulated on the mapping but
        never inserted into the output circuit. A SWAP only moves two logical
        qubits, so each candidate is scored by the cost change of the front and
        look-ahead gates on those two qubits rather than by re-scoring every gate.

        Args:
            layout (list[int]): Initial logical-to-physical mapping. It is not modified.
            distance_matrix (list[list[int]]): All-pairs shortest-path distance matrix for physical qubits.
            two_qubit_pairs (list[tuple[int, int]]): Ordered 2-qubit interactions as logical-qubit pairs.
            gate_indices_per_qubit (list[list[int]]): For each logical qubit, indices of ``two_qubit_pairs`` touching it.
            random_generator (random.Random): Random generator used for tie-breaking.
            physical_nodes (list[int]): Physical node labels.
            physical_to_dense_index (dict[int, int]): Physical label to dense index mapping.

        Returns:
            tuple[list[int], float]: Final logical-to-physical mapping and the number of simulated SWAPs.
        """
        num_logical_qubits = len(layout)
        num_physical_qubits = len(physical_nodes)
        unreachable_distance = SabreLayoutPass._UNREACHABLE_DISTANCE
        tie_epsilon = SabreLayoutPass._TIE_EPSILON
        tie_break_probability = SabreLayoutPass._TIE_BREAK_PROBABILITY
        beta = self.beta

        # Everything below works on dense physical indices. rustworkx lists neighbors in hash order, so sort them.
        neighbors = [
            sorted(physical_to_dense_index[int(neighbor)] for neighbor in self.topology.neighbors(node))
            for node in physical_nodes
        ]
        swap_neighbors = [
            [neighbor for neighbor in node_neighbors if neighbor != dense_index]
            for dense_index, node_neighbors in enumerate(neighbors)
        ]
        # Disconnected pairs are scored with a large finite penalty instead of the sentinel distance.
        if max(map(max, distance_matrix), default=0) >= unreachable_distance:
            distance_costs = [
                [_UNREACHABLE_PENALTY if distance >= unreachable_distance else distance for distance in row]
                for row in distance_matrix
            ]
        else:
            distance_costs = distance_matrix

        position = [physical_to_dense_index[physical_qubit] for physical_qubit in layout]
        occupant: list[int | None] = [None] * num_physical_qubits
        for logical_qubit, dense_index in enumerate(position):
            occupant[dense_index] = logical_qubit

        # cursor[q] points at the next unexecuted interaction of logical qubit q in gate_indices_per_qubit[q].
        cursor = [0] * num_logical_qubits
        front: set[int] = set()
        # front_partner[q] is the other qubit of q's front gate, or -1 when q has none.
        front_partner = [-1] * num_logical_qubits
        ready_to_check: list[int] = []

        def head(logical_qubit: int) -> int:
            qubit_gate_indices = gate_indices_per_qubit[logical_qubit]
            index = cursor[logical_qubit]
            return qubit_gate_indices[index] if index < len(qubit_gate_indices) else -1

        def enter_front(logical_qubit: int) -> None:
            """Add the next gate of ``logical_qubit`` to the front once it is also next on its partner."""
            gate_idx = head(logical_qubit)
            if gate_idx < 0:
                return
            logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_idx]
            partner = logical_qubit_b if logical_qubit == logical_qubit_a else logical_qubit_a
            if head(partner) == gate_idx:
                front.add(gate_idx)
                front_partner[logical_qubit_a] = logical_qubit_b
                front_partner[logical_qubit_b] = logical_qubit_a
                ready_to_check.append(gate_idx)

        def apply_swap(dense_a: int, dense_b: int) -> None:
            logical_on_a, logical_on_b = occupant[dense_a], occupant[dense_b]
            occupant[dense_a], occupant[dense_b] = logical_on_b, logical_on_a
            for logical_qubit, new_dense_index in ((logical_on_a, dense_b), (logical_on_b, dense_a)):
                if logical_qubit is not None:
                    position[logical_qubit] = new_dense_index
                    ready_to_check.append(head(logical_qubit))

        for logical_qubit in range(num_logical_qubits):
            enter_front(logical_qubit)

        # The decay of dense qubit d is decay[d] * decay_scale, so relaxing every qubit is a single multiply.
        decay = [0.0] * num_physical_qubits
        decayed_indices: list[int] = []
        decay_scale = 1.0

        release_valve_threshold = int(self.release_valve_factor * num_physical_qubits)
        swaps_since_progress = 0
        swap_count = 0
        no_partners: tuple[int, ...] = ()
        extended_partners: list[tuple[int, ...]] = []
        extended_stale = True

        remaining = len(two_qubit_pairs)
        while remaining:
            # Execute every front gate whose endpoints are adjacent; only gates whose qubits moved need checking.
            while ready_to_check:
                gate_idx = ready_to_check.pop()
                if gate_idx not in front:
                    continue
                logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_idx]
                if distance_matrix[position[logical_qubit_a]][position[logical_qubit_b]] != 1:
                    continue
                front.discard(gate_idx)
                front_partner[logical_qubit_a] = front_partner[logical_qubit_b] = -1
                remaining -= 1
                cursor[logical_qubit_a] += 1
                cursor[logical_qubit_b] += 1
                enter_front(logical_qubit_a)
                enter_front(logical_qubit_b)
                swaps_since_progress = 0
                for dense_index in decayed_indices:
                    decay[dense_index] = 0.0
                decayed_indices.clear()
                decay_scale = 1.0
                extended_stale = True

            if not remaining:
                break

            if swaps_since_progress >= release_valve_threshold:
                # Greedy swaps are cycling: walk the closest front gate's qubits together along a shortest path.
                current_distance, closest_gate_idx = min(
                    (
                        distance_matrix[position[two_qubit_pairs[gate_idx][0]]][position[two_qubit_pairs[gate_idx][1]]],
                        gate_idx,
                    )
                    for gate_idx in front
                )
                if current_distance >= unreachable_distance:
                    # Disconnected topology: no swap sequence helps, so let SabreSwapPass report it.
                    break
                logical_qubit_a, logical_qubit_b = two_qubit_pairs[closest_gate_idx]
                current_dense = position[logical_qubit_a]
                target_dense = position[logical_qubit_b]
                while current_distance > 1:
                    next_dense = next(
                        neighbor
                        for neighbor in neighbors[current_dense]
                        if distance_matrix[neighbor][target_dense] == current_distance - 1
                    )
                    apply_swap(current_dense, next_dense)
                    swap_count += 1
                    current_dense = next_dense
                    current_distance -= 1
                ready_to_check.append(closest_gate_idx)
                swaps_since_progress = 0
                continue

            if extended_stale:
                extended_partners = [no_partners] * num_logical_qubits
                for gate_idx in self._extended_set(gate_indices_per_qubit, cursor, self.lookahead_size):
                    logical_qubit_a, logical_qubit_b = two_qubit_pairs[gate_idx]
                    extended_partners[logical_qubit_a] += (logical_qubit_b,)
                    extended_partners[logical_qubit_b] += (logical_qubit_a,)
                extended_stale = False

            decay_scale *= self.decay_lambda
            if decay_scale < _DECAY_RESCALE_THRESHOLD:
                for dense_index in decayed_indices:
                    decay[dense_index] *= decay_scale
                decay_scale = 1.0

            # Candidates are the coupling edges at front-gate endpoints, scored by the cost change of their qubits' gates.
            best_edge: tuple[int, int] | None = None
            best_delta = math.inf
            for gate_idx in front:
                for logical_on_a in two_qubit_pairs[gate_idx]:
                    dense_a = position[logical_on_a]
                    costs_from_a = distance_costs[dense_a]
                    decay_a = decay[dense_a]
                    partner_dense_a = position[front_partner[logical_on_a]]
                    partner_decay_a = decay[partner_dense_a]
                    front_cost_a = costs_from_a[partner_dense_a] * (1.0 + (decay_a + partner_decay_a) * decay_scale)
                    extended_dense_a = [position[partner] for partner in extended_partners[logical_on_a]]
                    for dense_b in swap_neighbors[dense_a]:
                        logical_on_b = occupant[dense_b]
                        partner_b = -1 if logical_on_b is None else front_partner[logical_on_b]
                        # An edge joining two front endpoints is reached from both ends; score it from the lower one.
                        if partner_b >= 0 and dense_b < dense_a:
                            continue
                        costs_from_b = distance_costs[dense_b]
                        decay_b = decay[dense_b]
                        # A gate between the two swapped qubits keeps its cost.
                        front_delta = (
                            0.0
                            if partner_dense_a == dense_b
                            else costs_from_b[partner_dense_a] * (1.0 + (decay_b + partner_decay_a) * decay_scale)
                            - front_cost_a
                        )
                        extended_delta = 0.0
                        for partner_dense in extended_dense_a:
                            if partner_dense != dense_b:
                                extended_delta += costs_from_b[partner_dense] - costs_from_a[partner_dense]
                        if logical_on_b is not None:
                            if partner_b >= 0 and partner_b != logical_on_a:
                                partner_dense = position[partner_b]
                                partner_decay = decay[partner_dense]
                                front_delta += costs_from_a[partner_dense] * (
                                    1.0 + (decay_a + partner_decay) * decay_scale
                                ) - costs_from_b[partner_dense] * (1.0 + (decay_b + partner_decay) * decay_scale)
                            for partner in extended_partners[logical_on_b]:
                                partner_dense = position[partner]
                                if partner_dense != dense_a:
                                    extended_delta += costs_from_a[partner_dense] - costs_from_b[partner_dense]
                        candidate_delta = front_delta + beta * extended_delta

                        if candidate_delta < best_delta - tie_epsilon:
                            best_delta = candidate_delta
                            best_edge = (dense_a, dense_b)
                        elif (
                            candidate_delta <= best_delta + tie_epsilon
                            # Tie-break randomly for diversification
                            and random_generator.random() < tie_break_probability
                        ):
                            best_edge = (dense_a, dense_b)

            if best_edge is None:
                # Graph might be disconnected or trivial; fall back to a random swap over touched qubits.
                touched_dense_indices = sorted(
                    {position[logical_qubit] for gate_idx in front for logical_qubit in two_qubit_pairs[gate_idx]}
                )
                if len(touched_dense_indices) < SabreLayoutPass._TWO_QUBIT_ARITY:
                    # Degenerate case: nothing to do; break to avoid an infinite loop.
                    break
                sampled_a, sampled_b = random_generator.sample(touched_dense_indices, SabreLayoutPass._TWO_QUBIT_ARITY)
                best_edge = (sampled_a, sampled_b)

            chosen_a, chosen_b = best_edge
            apply_swap(chosen_a, chosen_b)
            for dense_index in best_edge:
                if not decay[dense_index]:
                    decayed_indices.append(dense_index)
                decay[dense_index] += self.decay_delta / decay_scale
            swap_count += 1
            swaps_since_progress += 1

        return [physical_nodes[dense_index] for dense_index in position], float(swap_count)

    # --------- helpers: extended set ---------

    @staticmethod
    def _extended_set(
        gate_indices_per_qubit: list[list[int]],
        front_positions: list[int],
        max_size: int,
    ) -> set[int]:
        """Collect upcoming 2-qubit gates beyond the current front layer.

        Args:
            gate_indices_per_qubit (list[list[int]]): Per-logical-qubit interaction indices.
            front_positions (list[int]): Current front pointers into ``gate_indices_per_qubit``.
            max_size (int): Maximum number of indices to include.

        Returns:
            set[int]: Set of selected interaction indices for look-ahead.
        """
        extended_gate_indices: set[int] = set()
        budget = max_size
        # Scan forward a few steps past the front pointer of each qubit, stopping once the budget is spent.
        for logical_qubit, qubit_gate_indices in enumerate(gate_indices_per_qubit):
            if budget <= 0:
                break
            start = front_positions[logical_qubit] + 1
            upcoming = qubit_gate_indices[start : start + budget]
            extended_gate_indices.update(upcoming)
            budget -= len(upcoming)
        return extended_gate_indices

    # --------- helpers: initial layout ---------

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
            if len(gate_qubits) == SabreLayoutPass._TWO_QUBIT_ARITY and not isinstance(gate, M):
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
