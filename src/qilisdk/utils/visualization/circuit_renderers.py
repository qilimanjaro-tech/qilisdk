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

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Final, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrow, FancyBboxPatch

from qilisdk.digital.gates import Controlled, Gate, M, X
from qilisdk.utils.visualization.style import CircuitStyle

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from qilisdk.digital import Circuit

###############################################################################
# Matplotlib implementation
###############################################################################


class MatplotlibCircuitRenderer:
    """Render a :class:`~qilisdk.digital.Circuit` using *matplotlib*."""

    # Z-order groups -------------------------------------------------------
    _Z: Final = {
        "wire": 1,
        "wire_label": 1,
        "gate": 3,
        "node": 3,
        "bridge": 2,
        "connector": 4,
        "gate_label": 4,
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, circuit: Circuit, ax: Axes | None = None, *, style: CircuitStyle = CircuitStyle()) -> None:
        self.circuit = circuit
        self.style = style
        self._ax = ax or self._make_axes(style.dpi)
        self._end_measure_qubits: set[int] = set()
        self._wires = circuit.nqubits
        # *layer_widths[w][l]* - width (inches) of layer *l* on wire *w*
        self._layer_widths: dict[int, list[float]] = {w: [] for w in range(circuit.nqubits)}

    @property
    def axes(self) -> Axes:
        return self._ax

    def plot(self) -> None:
        """
        Render the circuit on the current axes and show the figure.

        Traverses the circuit gates once, placing and drawing each element,
        deferring final-column measurements as needed, draws wires and finalizes
        the figure.
        """
        self._generate_layer_gate_mapping()
        self._draw_wire_labels()

        # ------------------------------------------------------------------
        # 1. compute last-gate index for each qubit ------------------------
        # ------------------------------------------------------------------
        # done elswere

        # ------------------------------------------------------------------
        # 2. iterate through gates, drawing or deferring measurements -----
        # ------------------------------------------------------------------
        deferred_qubits: set[int] = set()
        self._max_layer_width: list[float] = [self.style.start_pad]
        for layer in self._layer_gate_mapping:
            for _, gate in self._layer_gate_mapping[layer].items():
                if isinstance(gate, M):
                    inline_qubits: list[int] = []
                    for q in gate.target_qubits:
                        if self.last_idx.get(q) == layer:
                            deferred_qubits.add(q)
                            self._end_measure_qubits.add(q)
                        else:
                            inline_qubits.append(q)
                    if inline_qubits:
                        self._draw_inline_measure(inline_qubits, layer=layer)
                    continue

                if isinstance(gate, Controlled):
                    self._draw_controlled_gate(gate, layer=layer)
                    continue

                if gate.name == "SWAP":
                    self._draw_swap_gate(list(gate.target_qubits or []), layer=layer)
                    continue

                # Targets-only (single or multi-qubit) box
                self._draw_targets_gate(
                    label=self._gate_label(gate), targets=list(gate.target_qubits or []), layer=layer
                )
            maxi = max(
                [
                    0.0,
                    *(
                        self._layer_widths[wire][layer]
                        for wire in range(self._wires)
                        if len(self._layer_widths[wire]) - 1 >= layer
                    ),
                ]
            )
            self._max_layer_width.append(maxi)

        # ------------------------------------------------------------------
        # 3. draw any deferred (final-column) measurements -----------------
        # ------------------------------------------------------------------
        if deferred_qubits:
            self._draw_concurrent_measures(sorted(deferred_qubits), layer=layer)

        # ------------------------------------------------------------------
        # final touches -----------------------------------------------------
        # ------------------------------------------------------------------
        self._draw_wires()
        self._finalise_figure()
        # plt.tight_layout()
        plt.show()

    def save(self, filename: str) -> None:  # thin wrapper
        """Save current figure to disk.

        Args:
            filename: Path to save the figure (e.g., 'circuit.png').
        """

        self.axes.figure.savefig(filename, bbox_inches="tight")  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Low-level drawing helpers (private)
    # ------------------------------------------------------------------
    def _generate_layer_gate_mapping(self) -> None:
        self._layer_gate_mapping: dict[int, dict[int, Gate]] = {}
        gate_maping: dict[int, list[Gate]] = {}
        for qubit in range(self.circuit.nqubits):
            gate_maping[qubit] = []
        for gate in self.circuit.gates:
            qubits = gate.qubits
            if len(qubits) == 1:
                gate_maping[qubits[0]].append(gate)
            elif len(qubits) > 1:
                if self.style.layout == "compact":
                    con_qubits = qubits
                elif self.style.layout == "normal":
                    con_qubits = tuple(range(min(qubits), max(qubits) + 1))
                for qubit in con_qubits:
                    gate_maping[qubit].append(gate)

        layer: int = 0
        waiting_list: dict[int, Gate] = {}
        completed = [False] * self.circuit.nqubits
        for q, l in gate_maping.items():
            completed[q] = not bool(l)
        ignore_q: list[int] = []
        self.last_idx: dict[int, int] = {}
        while not all(completed):
            if layer not in self._layer_gate_mapping:
                self._layer_gate_mapping[layer] = {}
            for q in range(self.circuit.nqubits):
                if q in ignore_q:
                    ignore_q.remove(q)
                    continue
                if len(gate_maping[q]) == 0 and not completed[q]:
                    completed[q] = True
                    self.last_idx[q] = layer - 1
                if q in waiting_list or completed[q]:
                    continue
                gate = gate_maping[q][0]
                if gate.nqubits == 1:
                    self._layer_gate_mapping[layer][q] = gate
                    gate_maping[q].pop(0)
                if gate.nqubits > 1:
                    waiting_list[q] = gate
                    qubits = gate.qubits
                    if self.style.layout == "compact":
                        con_qubits = qubits
                    elif self.style.layout == "normal":
                        con_qubits = tuple(range(min(qubits), max(qubits) + 1))
                    if all(key in waiting_list for key in con_qubits) and all(
                        waiting_list[qr] == gate for qr in con_qubits
                    ):
                        self._layer_gate_mapping[layer][q] = gate
                        for c_qubit in con_qubits:
                            gate_maping[c_qubit].pop(0)
                            del waiting_list[c_qubit]

                        if self.style.layout == "compact":
                            for m_qubit in range(min(qubits), q):
                                if m_qubit not in qubits and m_qubit in self._layer_gate_mapping[layer]:
                                    ignore_q.append(m_qubit)
                                    if layer + 1 not in self._layer_gate_mapping:
                                        self._layer_gate_mapping[layer + 1] = {}
                                    self._layer_gate_mapping[layer + 1][m_qubit] = self._layer_gate_mapping[layer][
                                        m_qubit
                                    ]
                                    del self._layer_gate_mapping[layer][m_qubit]
                        ignore_q += [*(m_qubit for m_qubit in range(q + 1, max(qubits) + 1))]
                if len(gate_maping[q]) == 0 and not completed[q]:
                    completed[q] = True
                    self.last_idx[q] = layer
            layer += 1

    def _xpos(self, layer: int) -> float:
        """
        Compute the left x of a given layer.

        With align_layer=True, this ignores *wires* and aligns across all wires.

        Args:
            layer: Column index (0 = the initial label pad entry).

        Returns:
            The x-coordinate (inches) of the left edge of this column.
        """
        return sum(self._max_layer_width[: layer + 1])

    def _reserve(self, width: float, wires: Iterable[int], layer: int) -> None:
        """
        Reserve width in a column for a set of wires.

        Args:
            width: Box width (inches), *excluding* left/right gate margins.
            wires: Wires to update.
            layer: Column index to reserve.
        """
        full_width = width + self.style.gate_margin * 2
        for w in wires:
            layers = self._layer_widths[w]
            if len(layers) > layer:
                layers[layer] = max(layers[layer], full_width)
            else:
                for _ in range(len(layers), layer):
                    layers.append(0.0)
                layers.append(full_width)

    def _place(self, wires: Iterable[int], layer: int, *, min_width: float = 0.0) -> float:
        """
        Choose a column and compute its left x for a set of wires.

        This does *not* draw anything; it optionally reserves a minimal width
        for the selected column to create the column on those wires.

        Args:
            wires: Wires that must participate in this column.
            min_width: Optional minimal content width to reserve (inches).

        Returns:
            A tuple ``(x, layer, wires_sorted)``:
                - x: Left x of the column (including left margin).
                - layer: Column index.
                - wires_sorted: Sorted unique wires used for placement.
        """
        wires = [*range(min(wires), max(wires) + 1)]
        x = self._xpos(layer) + self.style.gate_margin
        if min_width:
            self._reserve(min_width, wires, layer)
        return x

    def _text_width(self, text: str) -> float:
        """
        Measure rendered text width in inches for current DPI/style.

        Args:
            text: Text to measure (mathtext is supported).

        Returns:
            The rendered width in inches.
        """

        t = plt.Text(
            0,
            0,
            text,
            fontproperties=self.style.font,
        )
        self.axes.add_artist(t)
        renderer = self.axes.figure.canvas.get_renderer()  # type: ignore[attr-defined]
        width = t.get_window_extent(renderer=renderer).width / self.style.dpi
        t.remove()
        return width

    # Basic primitives ----------------------------------------------------

    def _draw_targets_gate(
        self,
        *,
        label: str,
        targets: list[int] | None,
        x: float | None = None,
        layer: int | None = None,
        color: str | None = None,
    ) -> tuple[float, int, float]:
        """
        Draw a box gate that touches only *targets* (no controls).

        If ``x``/``layer`` are not provided, the earliest column that all
        targets can share is used.

        Args:
            label: Text label to show inside the box.
            targets: Target qubit indices. If None, defaults to all wires.
            x: Optional left x of the column (from a prior `_place`).
            layer: Optional column index (from a prior `_place`).
            color: Optional box fill/edge color.

        Returns:
            A tuple ``(x, layer, width)`` where:
                - x: Left x used for this column.
                - layer: Column index used.
                - width: Content width (inches) of the box.
        """
        targets = list(targets or range(self._wires))
        t_sorted = sorted(targets)
        a, b = t_sorted[0], t_sorted[-1]

        # Decide layer/x if not given (no-controls: only target wires matter)
        if layer is None or x is None:
            layer = max(len(self._layer_widths[w]) for w in t_sorted)
            x = self._xpos(layer) + self.style.gate_margin

        # Measure and reserve the full width at the given x/layer
        width = max(self._text_width(label) + self.style.gate_pad * 2, self.style.min_gate_w)
        self._reserve(width, t_sorted, layer)

        # Geometry
        y_a = self._ypos(a, n_qubits=self._wires, sep=self.style.wire_sep)
        y_b = self._ypos(b, n_qubits=self._wires, sep=self.style.wire_sep)
        y_bottom = min(y_a, y_b) - self.style.min_gate_h / 2
        height = abs(y_b - y_a) + self.style.min_gate_h
        y_center = (y_a + y_b) / 2.0

        gate_color = color or self.style.theme.primary

        # Box
        self.axes.add_patch(
            FancyBboxPatch(
                (x, y_bottom),
                width,
                height,
                boxstyle=self.style.bulge,
                mutation_scale=0.3,
                facecolor=gate_color,
                edgecolor=gate_color,
                zorder=self._Z["gate"],
            )
        )
        # Label
        self.axes.text(
            x + width / 2,
            y_center,
            label,
            ha="center",
            va="center",
            color=self.style.theme.on_primary,
            fontproperties=self.style.font,
            zorder=self._Z["gate_label"],
        )

        # Visual connectors for multi-targets
        if len(t_sorted) > 1:
            for t in t_sorted:
                y_t = self._ypos(t, n_qubits=self._wires, sep=self.style.wire_sep)
                self.axes.add_patch(
                    Circle(
                        (x + self.style.connector_r, y_t),
                        self.style.connector_r,
                        color=self.style.theme.background,
                        zorder=self._Z["connector"],
                    )
                )
                self.axes.add_patch(
                    Circle(
                        (x + width - self.style.connector_r, y_t),
                        self.style.connector_r,
                        color=self.style.theme.background,
                        zorder=self._Z["connector"],
                    )
                )

        return x, layer, width

    def _draw_control_dot(self, wire: int, x: float) -> None:
        """
        Draw a filled control dot at the given wire/x.

        Args:
            wire: Qubit index.
            x: Column anchor x coordinate.
        """
        y = self._ypos(wire, n_qubits=self._wires, sep=self.style.wire_sep)
        self.axes.add_patch(Circle((x, y), self.style.control_r, color=self.style.theme.accent, zorder=self._Z["node"]))

    def _draw_plus_sign(self, wire: int, x: float) -> None:
        """
        Draw a target ⊕ marker at the given wire/x.

        Args:
            wire: Qubit index.
            x: Column anchor x coordinate.
        """
        y = self._ypos(wire, n_qubits=self._wires, sep=self.style.wire_sep)
        self.axes.add_patch(Circle((x, y), self.style.target_r, color=self.style.theme.accent, zorder=self._Z["node"]))
        self.axes.add_line(
            plt.Line2D(
                (x, x),
                (y - self.style.target_r / 2, y + self.style.target_r / 2),
                lw=1.5,
                color=self.style.theme.background,
                zorder=self._Z["gate_label"],
            )
        )
        self.axes.add_line(
            plt.Line2D(
                (x - self.style.target_r / 2, x + self.style.target_r / 2),
                (y, y),
                lw=1.5,
                color=self.style.theme.background,
                zorder=self._Z["gate_label"],
            )
        )

    def _draw_bridge(self, wire_a: int, wire_b: int, x: float) -> None:
        """
        Draw a vertical bridge line between two wires at x.

        Args:
            wire_a: First wire.
            wire_b: Second wire.
            x: Column x coordinate where the bridge is drawn.
        """
        y1, y2 = (
            self._ypos(wire_a, n_qubits=self._wires, sep=self.style.wire_sep),
            self._ypos(wire_b, n_qubits=self._wires, sep=self.style.wire_sep),
        )
        self.axes.add_line(plt.Line2D([x, x], [y1, y2], color=self.style.theme.accent, zorder=self._Z["bridge"]))

    def _draw_swap_mark(self, wire: int, x: float) -> None:
        """
        Draw one X of a SWAP marker on a given wire at x.

        Args:
            wire: Qubit index.
            x: Column anchor x coordinate.
        """
        y = self._ypos(wire, n_qubits=self._wires, sep=self.style.wire_sep)
        offset = self.style.min_gate_w / 3
        color = self.style.theme.accent
        for xs, ys in (
            ([x + offset, x - offset], [y + self.style.min_gate_h / 4, y - self.style.min_gate_h / 4]),
            ([x - offset, x + offset], [y + self.style.min_gate_h / 4, y - self.style.min_gate_h / 4]),
        ):
            self.axes.add_line(plt.Line2D(xs, ys, color=color, linewidth=2, zorder=self._Z["gate"]))

    def _draw_swap_gate(
        self,
        targets: list[int],
        layer: int,
        *,
        x: float | None = None,
    ) -> float:
        """
        Draw a SWAP between two target wires.

        Args:
            targets: Exactly two wires to swap.
            x: Optional left x (from `_place`).
            layer: Optional column index (from `_place`).

        Returns:
            The anchor x within the column where the swap glyph is centered.
        """
        t_sorted = sorted(targets)

        if x is None:
            x = self._place(t_sorted, layer, min_width=self.style.target_r * 2)
        else:
            self._reserve(self.style.target_r * 2, t_sorted, layer)

        x_anchor = x + self.style.gate_pad

        for t in t_sorted:
            self._draw_swap_mark(t, x_anchor)
        # vertical bridge between the two targets
        self._draw_bridge(t_sorted[0], t_sorted[1], x_anchor)
        return x_anchor

    def _draw_controlled_gate(self, gate: Controlled, layer: int) -> None:
        """
        Draw a controlled gate (controls + targets).

        Handles:
          - MCX family as control dots + ⊕ (no box),
          - Controlled-SWAP by reusing swap glyphs,
          - Generic controlled gates as a box over targets with control stems.

        Args:
            gate: Controlled gate instance.
        """
        targets = list(gate.target_qubits or range(self._wires))
        controls = list(gate.control_qubits or [])
        all_wires = sorted(set(targets + controls))

        # Place a column shared by all involved wires; reserve minimal node width
        x = self._place(all_wires, layer, min_width=self.style.target_r * 2)

        # Controlled-X family (CNOT / multi-controlled X): target glyph, not a box
        if gate.is_modified_from(X):
            x_anchor = x + self.style.gate_pad
            for c in controls:
                self._draw_control_dot(c, x_anchor)
                self._draw_bridge(c, targets[0], x_anchor)
            self._draw_plus_sign(targets[0], x_anchor)
            return

        # Controlled SWAP (Fredkin): reuse the SWAP primitive, then add controls
        if getattr(gate.basic_gate, "name", "") == "SWAP":
            x_anchor = self._draw_swap_gate(targets, x=x, layer=layer)
            for c in controls:
                self._draw_control_dot(c, x_anchor)
                self._draw_bridge(c, targets[0], x_anchor)
            return

        # Generic controlled gate: draw the target box at this same column,
        # then widen control wires for that layer and add stems to the center.
        label = self._gate_label(gate.basic_gate)
        gate_color = self.style.theme.accent
        x_box, layer_box, width = self._draw_targets_gate(
            label=label, targets=targets, x=x, layer=layer, color=gate_color
        )

        # Ensure control wires also reserve the same width for this layer
        extra_controls = [c for c in controls if c not in targets]
        if extra_controls:
            self._reserve(width, extra_controls, layer_box)

        x_center = x_box + width / 2.0
        for c in controls:
            self._draw_control_dot(c, x_center)
            self._draw_bridge(c, targets[0], x_center)

    # Measurements --------------------------------------------------------

    def _draw_inline_measure(self, qubits: list[int], layer: int) -> None:
        """
        Draw measurement boxes interleaved with gates (same column).

        Args:
            qubits: Wires to measure in this column.
        """
        layer = max(len(self._layer_widths[q]) for q in qubits)
        x = self._xpos(layer) + self.style.gate_margin
        self._reserve(self.style.min_gate_w, qubits, layer)
        for q in qubits:
            self._draw_measure_symbol(q, x)

    def _draw_concurrent_measures(self, qubits: list[int], layer: int) -> None:
        """
        Draw a final column of measurements (one shared column).

        Args:
            qubits: Wires to measure concurrently.
        """
        layer = max(len(v) for v in self._layer_widths.values())
        x = self._xpos(layer) + self.style.gate_margin
        self._max_layer_width.append(self.style.min_gate_w)
        for q in qubits:
            self._draw_measure_symbol(q, x)

    def _draw_measure_symbol(self, wire: int, x: float) -> None:
        """
        Draw a measurement glyph at the given wire/x.

        Args:
            wire: Qubit index.
            x: Shared left x for this measurement column.
        """
        y = self._ypos(wire, n_qubits=self._wires, sep=self.style.wire_sep)
        self.axes.add_patch(
            FancyBboxPatch(
                (x, y - self.style.min_gate_h / 2),
                self.style.min_gate_w,
                self.style.min_gate_h,
                boxstyle=self.style.bulge,
                mutation_scale=0.3,
                facecolor=self.style.theme.background,
                edgecolor=self.style.theme.on_background,
                linewidth=1.25,
                zorder=self._Z["gate"],
            )
        )
        self.axes.add_patch(
            Arc(
                (x + self.style.min_gate_w / 2, y - self.style.min_gate_h / 2),
                self.style.min_gate_w * 1.5,
                self.style.min_gate_h,
                theta1=0,
                theta2=180,
                linewidth=1.25,
                color=self.style.theme.on_background,
                zorder=self._Z["gate_label"],
            )
        )
        self.axes.add_patch(
            FancyArrow(
                x + self.style.min_gate_w / 2,
                y - self.style.min_gate_h / 2,
                dx=self.style.min_gate_w * 0.7,
                dy=self.style.min_gate_h * 0.7,
                length_includes_head=True,
                width=0,
                color=self.style.theme.on_background,
                linewidth=1.25,
                zorder=self._Z["gate_label"],
            )
        )

    # Final decoration ----------------------------------------------------

    def _draw_wires(self) -> None:
        """
        Draw horizontal wires up to the last occupied x (plus tail).

        For wires whose last operation is a measurement, the wire stops at the
        measurement edge with no right-hand tail.
        """
        # how far the drawing for this wire actually goes
        x_end = sum(self._max_layer_width)
        for q in range(self._wires):
            y = self._ypos(q, n_qubits=self._wires, sep=self.style.wire_sep)
            # keep the tail only for wires that KEEP going after their last gate
            self.axes.add_line(
                plt.Line2D([0, x_end], [y, y], lw=1, color=self.style.theme.surface_muted, zorder=self._Z["wire"])
            )

    def _draw_wire_labels(self) -> None:
        """Draw wire labels to the left of the drawing."""
        labels = self.style.wire_label or [rf"$q_{{{i}}}$" for i in range(self._wires)]
        widths = [self._text_width(lbl) for lbl in labels]
        self._max_label_width = max(widths)

        for i, label in enumerate(labels):
            y = self._ypos(i, n_qubits=self._wires, sep=self.style.wire_sep)
            self.axes.text(
                -self.style.label_pad,
                y,
                label,
                ha="right",
                va="center",
                fontproperties=self.style.font,
                color=self.style.theme.on_background,
                zorder=self._Z["wire_label"],
            )

    def _finalise_figure(self) -> None:
        """Finalize axes limits, aspect, background, and title."""
        fig = self.axes.figure
        fig.set_facecolor(self.style.theme.background)

        total_length = sum(self._max_layer_width)
        x_end = self.style.padding + total_length

        y_end = self.style.padding + (self._wires - 1) * self.style.wire_sep

        self.axes.set_xlim(
            -self.style.padding - self._max_label_width - self.style.label_pad,
            x_end,
        )
        self.axes.set_ylim(-self.style.padding, y_end)

        if self.style.title:
            self.axes.set_title(
                self.style.title,
                pad=10,
                color=self.style.theme.surface_muted,
                fontdict={"fontsize": self.style.fontsize},
            )

        # In IPython keep figure square so equal aspect ratio does not shrink
        try:
            get_ipython()  # type: ignore
            size = max(self.axes.get_xlim()[1] - self.axes.get_xlim()[0], y_end + self.style.padding)
            fig.set_size_inches(size, size, forward=True)  # type: ignore[union-attr]
        except NameError:
            fig.set_size_inches(  # type: ignore[union-attr]
                self.axes.get_xlim()[1] - self.axes.get_xlim()[0], y_end + self.style.padding, forward=True
            )

        self.axes.set_aspect("equal", adjustable="box")
        self.axes.axis("off")

    # ------------------------------------------------------------------
    # Helpers - human-readable gate labels & π-fractions
    # ------------------------------------------------------------------

    @staticmethod
    def _ypos(index: int, *, n_qubits: int, sep: float) -> float:
        return (n_qubits - 1 - index) * sep

    @staticmethod
    def _pi_fraction(value: float, /, tol: float = 1e-2) -> str:
        """
        Format a float as a π-fraction (mathtext) when close to a rational.

        Args:
            value: Angle value (radians).
            tol: Tolerance for accepting the rational approximation.

        Returns:
            Mathtext string like ``"\\pi/5"`` or fallback decimal.
        """
        coeff = value / np.pi
        frac = Fraction(coeff).limit_denominator(32)
        n, d = frac.numerator, frac.denominator
        if abs(frac - coeff) < tol:
            if n == 0:
                return "0"
            if d == 1:
                return r"\pi" if n == 1 else rf"{n}\pi"
            return rf"\pi/{d}" if n == 1 else rf"{n}\pi/{d}"
        return f"{value:.2f}"

    @staticmethod
    def _with_superscript_dagger(label: str) -> str:
        # Convert trailing dagger to math superscript, e.g. "RX†" -> r"$\mathrm{RX}^{\dagger}$"
        if label.endswith("†"):
            base = label[:-1]
            return rf"$\mathrm{{{base}}}^{{\dagger}}$"
        return label

    @staticmethod
    def _gate_label(gate: Gate) -> str:
        """Build a display label for a (possibly parameterized) gate.

        Args:
            gate: Gate object.

        Returns:
            Label text. Parameterized gates get ``name ( $args$ )``.
        """
        name = MatplotlibCircuitRenderer._with_superscript_dagger(gate.name)
        if gate.is_parameterized and gate.get_parameter_values():
            parameters = ", ".join(
                MatplotlibCircuitRenderer._pi_fraction(value) for value in gate.get_parameter_values()
            )
            return rf"{name} (${parameters}$)"
        return gate.name

    @staticmethod
    def _make_axes(dpi: int) -> Axes:
        """
        Create a new figure and axes with the given DPI.

        Args:
            dpi (int): The DPI of the figure

        Returns:
            A newly created Matplotlib Axes.
        """
        _, ax = plt.subplots(dpi=dpi, constrained_layout=True)
        return ax
