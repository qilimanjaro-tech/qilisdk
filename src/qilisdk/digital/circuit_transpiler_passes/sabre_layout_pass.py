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
from typing import Iterable

from rustworkx import PyGraph

from qilisdk.digital import (
    CZ,
    RX,
    RY,
    RZ,
    SWAP,
    U3,
    Circuit,
    Gate,
    M,
)

from .circuit_transpiler_pass import CircuitTranspilerPass


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
        rng = random.Random(self.seed)  # noqa: S311

        n_logical = circuit.nqubits
        phys_nodes = sorted(int(x) for x in self.topology.node_indices())
        if not phys_nodes:
            raise ValueError("Coupling graph has no nodes.")
        num_phys_nodes = len(phys_nodes)
        if n_logical > num_phys_nodes:
            raise ValueError(f"Coupling graph has {num_phys_nodes} nodes but circuit needs {n_logical} qubits.")
        phys_index = {node: idx for idx, node in enumerate(phys_nodes)}
        max_phys_label_plus_one = max(phys_nodes) + 1

        # Precompute all-pairs shortest-path distances on the coupling graph.
        # We use an internal BFS on the rustworkx graph to avoid relying on version-specific APIs.
        dist = self._all_pairs_shortest_path_unweighted(self.topology, phys_nodes, phys_index)

        # Build the list of 2Q gates and per-qubit indices for the SABRE simulation.
        twoq_indices, twoq_qubits, per_qubit = self._twoq_structure(circuit)

        # edge case: circuits with no 2Q gates -> trivial identity layout
        if not twoq_indices:
            layout = phys_nodes[:n_logical]
            self.last_layout = layout
            self.last_score = 0.0
            if self.context is not None:
                self.context.initial_layout = self.last_layout
            return self._retarget_circuit(circuit, layout, max_phys_label_plus_one)

        # Multi-trial SABRE simulation; keep the best layout according to final cost.
        best_layout: list[int] | None = None
        best_score: float = math.inf

        for _ in range(self.num_trials):
            init_layout = self._random_initial_layout(rng, n_logical, phys_nodes)
            layout, score = self._sabre_simulate_layout(
                init_layout,
                dist,
                twoq_qubits,
                per_qubit,
                rng,
                phys_nodes,
                phys_index,
            )
            if score < best_score:
                best_layout = layout
                best_score = score

        # assert best_layout is not None
        self.last_layout = best_layout
        self.last_score = best_score

        if self.context is not None:
            self.context.initial_layout = self.last_layout or []

        new_circuit = self._retarget_circuit(circuit, best_layout, max_phys_label_plus_one)  # type: ignore[arg-type]

        self.append_circuit_to_context(new_circuit)

        # Return a new circuit with all gates mapped to the chosen physical qubits.
        return new_circuit

    # --------- SABRE core ---------

    def _sabre_simulate_layout(
        self,
        layout: list[int],  # logical -> physical
        dist: list[list[int]],
        twoq_qubits: list[tuple[int, int]],
        per_qubit: list[list[int]],
        rng: random.Random,
        phys_nodes: list[int],
        phys_index: dict[int, int],
    ) -> tuple[list[int], float]:
        """
        Run a SABRE-style simulation on 2Q gates to choose a good final layout.
        This is a *layout-only* variant: we simulate SWAPs on the mapping but do not insert them.
        """
        n_logical = len(layout)
        n_physical = len(phys_nodes)
        # inverse layout: physical -> logical (dense indices)
        inv_layout = self._invert_layout(layout, phys_index)

        scheduled = [False] * len(twoq_qubits)
        pos = [0] * n_logical  # per-qubit pointer to the next 2Q gate index in per_qubit[q]
        decay = [0.0] * n_physical  # physical-qubit penalty stored with dense indexing

        # Sum of executed front distances (diagnostic score)
        score_accum = 0.0

        def advance_front_for(q: int) -> None:
            """Move pos[q] to the first unscheduled 2Q gate on qubit q."""
            L = per_qubit[q]
            i = pos[q]
            while i < len(L) and scheduled[L[i]]:
                i += 1
            pos[q] = i

        def front_set() -> set[int]:
            """Union of first unscheduled 2Q gate index on each qubit."""
            F: set[int] = set()
            for q in range(n_logical):
                if pos[q] < len(per_qubit[q]):
                    F.add(per_qubit[q][pos[q]])
            return F

        # Initialize pointers
        for q in range(n_logical):
            advance_front_for(q)

        # Main loop: continue until all 2Q gates are scheduled
        remaining = len(twoq_qubits)
        while remaining:
            # Try to greedily "apply" any executable front gates (whose mapped endpoints are adjacent).
            progressed = True
            while progressed:
                progressed = False
                F = list(front_set())
                for g_idx in F:
                    if scheduled[g_idx]:
                        continue
                    u, v = twoq_qubits[g_idx]
                    pu, pv = layout[u], layout[v]
                    if dist[phys_index[pu]][phys_index[pv]] == 1:
                        # "Execute" this 2Q gate in the simulation.
                        scheduled[g_idx] = True
                        remaining -= 1
                        progressed = True
                        # Diagnostic score: adjacency cost is 1 by definition
                        score_accum += 1.0
                        # Advance fronts for both logical qubits involved
                        advance_front_for(u)
                        advance_front_for(v)

            if not remaining:
                break  # done

            # No front gate was executable: pick a simulated SWAP using SABRE heuristic.
            F = list(front_set())
            # Build candidate physical edges from neighbors of endpoints of non-executable front gates.
            candidate_edges: set[tuple[int, int]] = set()
            touched_phys: set[int] = set()
            for g_idx in F:
                u, v = twoq_qubits[g_idx]
                pu, pv = layout[u], layout[v]
                touched_phys.add(pu)
                touched_phys.add(pv)
                for nb in self.topology.neighbors(pu):
                    a, b = int(pu), int(nb)
                    if a != b:
                        candidate_edges.add((min(a, b), max(a, b)))
                for nb in self.topology.neighbors(pv):
                    a, b = int(pv), int(nb)
                    if a != b:
                        candidate_edges.add((min(a, b), max(a, b)))

            if not candidate_edges:
                # Graph might be disconnected or trivial; fall back to a random valid swap over touched qubits.
                # This won't crash; it just gives the algorithm a way to keep moving.
                phys_list = sorted(touched_phys) if touched_phys else phys_nodes[:]
                if len(phys_list) >= 2:
                    a, b = rng.sample(phys_list, 2)
                    candidate_edges.add((min(a, b), max(a, b)))
                else:
                    # Degenerate case: nothing to do; break to avoid infinite loop.
                    break

            # Decay relaxation
            for idx in range(n_physical):
                decay[idx] *= self.decay_lambda

            # Evaluate heuristic for each candidate swap
            best_edge: tuple[int, int] | None = None
            best_cost: float = math.inf

            # Precompute lookahead set E (extended)
            E = self._extended_set(twoq_qubits, per_qubit, pos, self.lookahead_size)

            for a, b in candidate_edges:
                # Virtually swap logical assignments at physical nodes a and b
                la = inv_layout[phys_index[a]]
                lb = inv_layout[phys_index[b]]
                # Logical qubit may be "unassigned" if device > logical; ensure we handle that.
                # If la or lb is None, this swap doesn't affect distances; we still allow it.
                # Simulate new layout distances
                # (temporarily mutate layout/inv_layout, compute cost, then revert).
                self._swap_mapping(layout, inv_layout, phys_index, a, b, la, lb)

                cost_front = self._cost_front(F, layout, twoq_qubits, dist, decay, phys_index)
                cost_ext = self._cost_front(E, layout, twoq_qubits, dist, None, phys_index)  # no decay on E
                cost = cost_front + self.beta * cost_ext

                # Revert the swap
                self._swap_mapping(layout, inv_layout, phys_index, a, b, lb, la)

                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best_edge = (a, b)
                elif abs(cost - best_cost) <= 1e-12:
                    # Tie-break randomly for diversification
                    if rng.random() < 0.5:
                        best_edge = (a, b)

            # Apply the chosen swap *to the mapping only* (layout-only SABRE).
            # assert best_edge is not None
            a, b = best_edge  # type: ignore[misc]
            la = inv_layout[phys_index[a]]
            lb = inv_layout[phys_index[b]]
            self._swap_mapping(layout, inv_layout, phys_index, a, b, la, lb)
            # Increase decay on touched physical qubits
            decay[phys_index[a]] += self.decay_delta
            decay[phys_index[b]] += self.decay_delta

        # Final diagnostic cost: re-evaluate sum of distances for the entire 2Q list under final layout
        total_cost = self._cost_front(set(range(len(twoq_qubits))), layout, twoq_qubits, dist, None, phys_index)
        # Combine both measures mildly
        score = 0.5 * total_cost + 0.5 * score_accum
        return layout, float(score)

    # --------- helpers: front / extended set / costs ---------

    def _extended_set(
        self,
        twoq_qubits: list[tuple[int, int]],
        per_qubit: list[list[int]],
        pos: list[int],
        max_size: int,
    ) -> set[int]:
        """
        Collect up to `max_size` upcoming 2Q gates beyond the front, scanning
        forward along each qubit's 2Q list.
        """
        E: set[int] = set()
        if max_size <= 0:
            return E

        # Start from the qubits that currently participate in front gates
        # (i.e., those qubits whose pos[q] points to some gate).
        frontier_qubits = [q for q in range(len(pos)) if pos[q] < len(per_qubit[q])]
        # Scan forward a few steps on each such qubit
        budget = max_size
        for q in frontier_qubits:
            i = pos[q] + 1
            while i < len(per_qubit[q]) and budget > 0:
                g_idx = per_qubit[q][i]
                E.add(g_idx)
                budget -= 1
                i += 1
                if budget == 0:
                    break
            if budget == 0:
                break
        return E

    def _cost_front(
        self,
        idxs: Iterable[int],
        layout: list[int],
        twoq_qubits: list[tuple[int, int]],
        dist: list[list[int]],
        decay: list[float] | None,
        phys_index: dict[int, int],
    ) -> float:
        c = 0.0
        for g_idx in idxs:
            u, v = twoq_qubits[g_idx]
            pu, pv = layout[u], layout[v]
            iu = phys_index[pu]
            iv = phys_index[pv]
            d = dist[iu][iv]
            if d >= 1_000_000_000:
                # Disconnected: incur a large penalty to discourage this layout
                d = 1e6  # type: ignore[assignment]
            if decay is not None:
                c += d * (1.0 + decay[iu] + decay[iv])
            else:
                c += d
        return c

    def _random_initial_layout(
        self,
        rng: random.Random,
        n_logical: int,
        phys_nodes: list[int],
    ) -> list[int]:
        """
        Pick a random injective mapping logical->physical using *existing* graph nodes.
        Returns a list L of length n_logical where L[q_logical] = p_physical.
        """
        if len(phys_nodes) < n_logical:
            raise ValueError(f"Coupling graph has only {len(phys_nodes)} nodes; need ≥ {n_logical}.")
        nodes = phys_nodes[:]  # copy so we can shuffle deterministically
        rng.shuffle(nodes)
        return nodes[:n_logical]

    # --------- helpers: mapping & structure ---------

    @staticmethod
    def _invert_layout(layout: list[int], phys_index: dict[int, int]) -> list[int | None]:
        inv = [None] * len(phys_index)
        for l, p in enumerate(layout):
            idx = phys_index.get(p)
            if idx is not None:
                inv[idx] = l  # type: ignore[call-overload]
        return inv  # type: ignore[return-value]

    @staticmethod
    def _swap_mapping(
        layout: list[int],
        inv_layout: list[int | None],
        phys_index: dict[int, int],
        a: int,
        b: int,
        la: int | None,
        lb: int | None,
    ) -> None:
        """Swap the logical qubits assigned to physical nodes a and b."""
        # Update inverse first
        ia = phys_index[a]
        ib = phys_index[b]
        inv_layout[ia], inv_layout[ib] = lb, la
        # Then forward mapping (only if logicals exist)
        if la is not None:
            layout[la] = b
        if lb is not None:
            layout[lb] = a

    @staticmethod
    def _twoq_structure(circuit: Circuit) -> tuple[list[int], list[tuple[int, int]], list[list[int]]]:
        """
        Extract (a) indices of 2Q gates in circuit.gates,
                (b) their (logical) qubit pairs,
                (c) for each logical qubit, the list of 2Q gate indices touching it.
        """
        n = circuit.nqubits
        twoq_indices: list[int] = []
        twoq_qubits: list[tuple[int, int]] = []
        for idx, g in enumerate(circuit.gates):
            qs = g.qubits
            if len(qs) == 2:
                twoq_indices.append(idx)
                twoq_qubits.append((qs[0], qs[1]))
        per_qubit: list[list[int]] = [[] for _ in range(n)]
        for local_i, (q0, q1) in enumerate(twoq_qubits):
            per_qubit[q0].append(local_i)
            per_qubit[q1].append(local_i)
        return twoq_indices, twoq_qubits, per_qubit

    @staticmethod
    def _all_pairs_shortest_path_unweighted(
        graph: PyGraph,
        phys_nodes: list[int],
        phys_index: dict[int, int],
    ) -> list[list[int]]:
        from collections import deque

        INF = 1_000_000_000
        dist = [[INF] * len(phys_nodes) for _ in phys_nodes]
        for s in phys_nodes:
            s = int(s)
            s_idx = phys_index[s]
            dist[s_idx][s_idx] = 0
            q = deque([s])
            seen = {s}
            while q:
                u = q.popleft()
                u = int(u)
                u_idx = phys_index[u]
                du = dist[s_idx][u_idx]
                for v in graph.neighbors(u):
                    v = int(v)
                    if v not in seen:
                        seen.add(v)
                        v_idx = phys_index[v]
                        dist[s_idx][v_idx] = du + 1
                        q.append(v)
        return dist

    # --------- retargeting to the chosen layout ---------

    def _retarget_circuit(self, circuit: Circuit, layout: list[int], max_phys_label_plus_one: int) -> Circuit:
        # The output circuit must have enough qubits to accommodate the maximum physical index.
        out_n = max(
            circuit.nqubits,
            (max(layout) + 1) if layout else circuit.nqubits,
            max_phys_label_plus_one,
        )
        new_circ = Circuit(out_n)
        for g in circuit.gates:
            mapped = tuple(layout[q] for q in g.qubits)
            new_circ.add(self._retarget_gate(g, mapped))
        return new_circ

    def _retarget_gate(self, g: Gate, new_qubits: tuple[int, ...]) -> Gate:
        """
        Recreate `g` on `new_qubits` using only the public API of the gate classes
        defined in this SDK.
        """
        # 1-qubit basics
        if isinstance(g, RX):
            return RX(new_qubits[0], theta=g.theta)
        if isinstance(g, RY):
            return RY(new_qubits[0], theta=g.theta)
        if isinstance(g, RZ):
            return RZ(new_qubits[0], phi=g.phi)
        if isinstance(g, U3):
            return U3(new_qubits[0], theta=g.theta, phi=g.phi, gamma=g.gamma)

        # 2-qubit basics
        if isinstance(g, CZ):
            return CZ(new_qubits[0], new_qubits[1])
        if isinstance(g, SWAP):
            return SWAP(new_qubits[0], new_qubits[1])

        # Measurement (possibly multi-qubit)
        if isinstance(g, M):
            return M(*new_qubits)

        # If you introduce more gates, add cases above.
        # Falling back to a generic reconstruction is unsafe without a uniform API.
        raise NotImplementedError(
            f"Retargeting not implemented for gate type {type(g).__name__} with arity {g.nqubits}"
        )
