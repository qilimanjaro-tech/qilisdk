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
from math import inf, isclose
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, FancyArrow, FancyBboxPatch
from matplotlib.text import Text

from qilisdk.digital.gates import Controlled, Gate, M, X
from qilisdk.utils.visualization.style import CircuitStyle

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    from qilisdk.digital import Circuit

###############################################################################
# Matplotlib implementation
###############################################################################


class MatplotlibCircuitRenderer:
    """Render a :class:`~qilisdk.digital.Circuit` using *matplotlib*."""

    # Largest canvas the Agg renderer can allocate: 2**16 px per side, and a
    # total area we keep well below it so a huge figure does not exhaust memory
    _MAX_CANVAS_SIDE: Final = 2**16 - 1
    _MAX_CANVAS_PIXELS: Final = 2**25

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
        self._owns_figure = ax is None
        self._ax = ax or self._make_axes(style.dpi)
        self._wires = circuit.nqubits
        # Index of the row currently being drawn
        self._row = 0
        self._text_sizes: dict[str, tuple[float, float]] = {}
        self._texts: list[Text] = []
        # How much bigger than its drawn size the circuit is currently displayed
        self._view_scale = 1.0

    @property
    def axes(self) -> Axes:
        return self._ax

    def plot(self) -> None:
        """
        Render the circuit on the current axes and show the figure.

        Measures the circuit, splits it into rows of bounded width, then draws
        every layer on its row, deferring final-column measurements as needed,
        draws wires and finalizes the figure.
        """
        logger.debug("[CircuitRenderer] Rendering circuit with {} qubits", self.circuit.nqubits)
        self._generate_layer_gate_mapping()
        self._compute_layout()
        self._fit_canvas()
        self._draw_wire_labels()

        for layer in sorted(self._layer_gate_mapping):
            self._row = self._layer_row[layer]
            inline_qubits = self._inline_measures.get(layer)
            if inline_qubits:
                self._draw_inline_measure(inline_qubits, layer=layer)
            for gate in self._layer_gate_mapping[layer].values():
                if isinstance(gate, M):
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

        # Deferred (final-column) measurements ------------------------------
        if self._deferred_qubits:
            self._row = self._layer_row[self._measure_layer]
            self._draw_concurrent_measures(sorted(self._deferred_qubits))

        self._draw_wires()
        self.axes.callbacks.connect("xlim_changed", self._on_view_changed)
        self.axes.callbacks.connect("ylim_changed", self._on_view_changed)
        self.axes.figure.canvas.mpl_connect("resize_event", self._on_view_changed)
        self._finalise_figure()
        plt.show()

    def save(self, filename: str) -> None:  # thin wrapper
        """Save current figure to disk.

        Args:
            filename: Path to save the figure (e.g., 'circuit.png').
        """
        logger.debug("[CircuitRenderer] Saving figure to {}", filename)
        if isinstance(self.axes.figure, Figure):
            self._show_view(*self._drawing_size())
            self.axes.figure.savefig(filename, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Low-level drawing helpers (private)
    # ------------------------------------------------------------------
    def _generate_layer_gate_mapping(self) -> None:
        self._layer_gate_mapping: dict[int, dict[int, Gate]] = {}
        gate_mapping: dict[int, list[Gate]] = {}
        for qubit in range(self.circuit.nqubits):
            gate_mapping[qubit] = []
        for gate in self.circuit.gates:
            qubits = gate.qubits
            if len(qubits) == 1:
                gate_mapping[qubits[0]].append(gate)
            elif len(qubits) > 1:
                if self.style.layout == "compact":
                    con_qubits = qubits
                elif self.style.layout == "normal":
                    con_qubits = tuple(range(min(qubits), max(qubits) + 1))
                for qubit in con_qubits:
                    gate_mapping[qubit].append(gate)

        layer: int = 0
        waiting_list: dict[int, Gate] = {}
        completed = [False] * self.circuit.nqubits
        for q, l in gate_mapping.items():
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
                if len(gate_mapping[q]) == 0 and not completed[q]:
                    completed[q] = True
                    self.last_idx[q] = layer - 1
                if q in waiting_list or completed[q]:
                    continue
                gate = gate_mapping[q][0]
                if gate.nqubits == 1:
                    self._layer_gate_mapping[layer][q] = gate
                    gate_mapping[q].pop(0)
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
                            gate_mapping[c_qubit].pop(0)
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
                        ignore_q += [*range(q + 1, max(qubits) + 1)]
                if len(gate_mapping[q]) == 0 and not completed[q]:
                    completed[q] = True
                    self.last_idx[q] = layer
            layer += 1

    @property
    def _row_height(self) -> float:
        """Vertical distance (inches) between the top wires of two consecutive rows."""
        return (self._wires - 1) * self.style.wire_sep + self.style.row_sep

    def _compute_layout(self) -> None:
        """
        Measure every layer and assign the layers to rows.

        Layer widths have to be known before anything is drawn so that deep
        circuits can be split into several rows instead of being rendered as one
        extremely wide figure that matplotlib cannot allocate a canvas for. The
        pass also decides which measurements are drawn inline and which ones are
        deferred to a shared final column.
        """
        self._deferred_qubits: set[int] = set()
        self._inline_measures: dict[int, list[int]] = {}
        widths: list[float] = []
        for layer in sorted(self._layer_gate_mapping):
            width = 0.0
            for gate in self._layer_gate_mapping[layer].values():
                if isinstance(gate, M):
                    inline = [q for q in gate.target_qubits if self.last_idx.get(q) != layer]
                    self._deferred_qubits.update(q for q in gate.target_qubits if self.last_idx.get(q) == layer)
                    if inline:
                        self._inline_measures.setdefault(layer, []).extend(inline)
                        width = max(width, self.style.min_gate_w)
                    continue
                width = max(width, self._gate_width(gate))
            widths.append(width + self.style.gate_margin * 2 if width else 0.0)

        # The deferred measurements share one extra column after the last layer
        self._measure_layer = len(widths)
        if self._deferred_qubits:
            widths.append(self.style.min_gate_w + self.style.gate_margin * 2)

        self._assign_rows(widths)

        labels = self.style.wire_label or [rf"$q_{{{i}}}$" for i in range(self._wires)]
        self._max_label_width = max((self._text_width(lbl) for lbl in labels), default=0.0)

        self._row_artists: list[list[Artist]] = [[] for _ in self._row_widths]
        self._row_shown = [True for _ in self._row_widths]

        self._title_band = self._text_size(self.style.title)[1] + self.style.padding if self.style.title else 0.0

    def _assign_rows(self, widths: list[float]) -> None:
        """
        Place each layer on a row, wrapping according to ``style.fold``.

        Args:
            widths: Width (inches) of every layer, including its gate margins.
        """
        max_layers = self.style.fold if isinstance(self.style.fold, int) else inf
        max_width = self.style.max_row_width if self.style.fold == "auto" else inf

        self._layer_row: list[int] = []
        self._layer_x: list[float] = []
        self._row_widths: list[float] = []
        row = 0
        x = self.style.start_pad
        layers_in_row = 0
        for width in widths:
            # A layer wider than a whole row still gets drawn, on a row of its own
            if width and layers_in_row and (layers_in_row >= max_layers or x + width > max_width):
                self._row_widths.append(x)
                row += 1
                x = self.style.start_pad
                layers_in_row = 0
            self._layer_row.append(row)
            self._layer_x.append(x)
            x += width
            # An empty layer takes up no room, so it never fills or wraps a row
            if width:
                layers_in_row += 1
        self._row_widths.append(x)
        logger.debug("[CircuitRenderer] Laid out {} layers on {} rows", len(widths), len(self._row_widths))

    def _record(self, artist: Artist) -> None:
        """
        File an artist under the row being drawn, so that row can be hidden later.

        Args:
            artist: The artist that was just added to the axes.
        """
        self._row_artists[self._row].append(artist)
        if isinstance(artist, Text):
            self._texts.append(artist)

    def _on_view_changed(self, _: object) -> None:
        """
        Hide offscreen elements for performance, ensuring that only the visible rows are rendered.

        Also, scale the text labels to maintain their proportion relative to the gates.

        Args:
            _: The axes or event that reported the change, which is not needed.
        """
        self._hide_offscreen_rows()
        self._scale_text()

    def _scale_text(self) -> None:
        """
        Keep the labels in proportion with their gates.
        """
        # What an equal aspect ratio with adjustable="box" will make of the axes
        position = self.axes.get_position(original=True)
        figure_width, figure_height = self.axes.figure.get_size_inches()  # ty:ignore[unresolved-attribute]
        scale = min(
            position.width * figure_width / self.axes.viewLim.width,
            position.height * figure_height / self.axes.viewLim.height,
        )
        if isclose(scale, self._view_scale, rel_tol=1e-3):
            return
        self._view_scale = scale
        fontsize = self.style.font.get_size_in_points() * scale
        for text in self._texts:
            text.set_fontsize(fontsize)

    def _hide_offscreen_rows(self) -> None:
        """
        Hide the rows that have scrolled out of view, for performance.
        """
        bottom, top = self.axes.get_ylim()
        for row, artists in enumerate(self._row_artists):
            offset = row * self._row_height
            shown = -offset - self.style.padding <= top and (self._wires - 1) * self.style.wire_sep - offset >= bottom
            if shown != self._row_shown[row]:
                self._row_shown[row] = shown
                for artist in artists:
                    artist.set_visible(shown)

    def _place(self, layer: int) -> float:
        """
        Compute the x where the content of a column starts, inside its margins.

        Args:
            layer: Column index.

        Returns:
            The x-coordinate (inches) of the left edge of this column's content.
        """
        return self._layer_x[layer] + self.style.gate_margin

    def _box_width(self, label: str) -> float:
        """
        Compute the width of a gate box holding ``label``.

        Args:
            label: Text label shown inside the box.

        Returns:
            The box width in inches.
        """
        return max(self._text_width(label) + self.style.gate_pad * 2, self.style.min_gate_w)

    def _gate_width(self, gate: Gate) -> float:
        """
        Compute the width a gate occupies, mirroring the glyph its drawing helper picks.

        Args:
            gate: Gate to measure.

        Returns:
            The content width in inches, excluding the gate margins.
        """
        if isinstance(gate, Controlled):
            if gate.is_modified_from(X) or getattr(gate.basic_gate, "name", "") == "SWAP":
                return self.style.target_r * 2
            return max(self.style.target_r * 2, self._box_width(self._gate_label(gate.basic_gate)))
        if gate.name == "SWAP":
            return self.style.target_r * 2
        return self._box_width(self._gate_label(gate))

    def _text_width(self, text: str) -> float:
        """
        Measure rendered text width in inches for current DPI/style.

        Args:
            text: Text to measure (mathtext is supported).

        Returns:
            The rendered width in inches.
        """
        return self._text_size(text)[0]

    def _text_size(self, text: str) -> tuple[float, float]:
        """
        Measure rendered text in inches for current DPI/style.

        Results are cached because deep circuits repeat the same few labels
        thousands of times and measuring is the bulk of the rendering cost.

        Args:
            text: Text to measure (mathtext is supported).

        Returns:
            The rendered ``(width, height)`` in inches.
        """
        if text in self._text_sizes:
            return self._text_sizes[text]

        t = plt.Text(
            0,
            0,
            text,
            fontproperties=self.style.font,
        )
        self.axes.add_artist(t)
        renderer = self.axes.figure.canvas.get_renderer()  # ty:ignore[unresolved-attribute]
        extent = t.get_window_extent(renderer=renderer)
        dpi = self.axes.figure.dpi
        t.remove()
        self._text_sizes[text] = (extent.width / dpi, extent.height / dpi)
        return self._text_sizes[text]

    # Basic primitives ----------------------------------------------------

    def _draw_targets_gate(
        self,
        *,
        label: str,
        targets: list[int] | None,
        layer: int,
        color: str | None = None,
    ) -> tuple[float, float]:
        """
        Draw a box gate that touches only *targets* (no controls).

        Args:
            label: Text label to show inside the box.
            targets: Target qubit indices. If None, defaults to all wires.
            layer: Column index the gate belongs to.
            color: Optional box fill/edge color.

        Returns:
            A tuple ``(x, width)`` where:
                - x: Left x used for this column.
                - width: Content width (inches) of the box.
        """
        targets = list(targets or range(self._wires))
        t_sorted = sorted(targets)
        a, b = t_sorted[0], t_sorted[-1]

        x = self._place(layer)
        width = self._box_width(label)

        # Geometry
        y_a = self._ypos(a, n_qubits=self._wires, sep=self.style.wire_sep)
        y_b = self._ypos(b, n_qubits=self._wires, sep=self.style.wire_sep)
        y_bottom = min(y_a, y_b) - self.style.min_gate_h / 2
        height = abs(y_b - y_a) + self.style.min_gate_h
        y_center = (y_a + y_b) / 2.0

        gate_color = color or self.style.theme.primary

        # Box
        self._record(
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
        )

        # Label
        self._record(
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
        )

        # Visual connectors for multi-targets
        if len(t_sorted) > 1:
            for t in t_sorted:
                y_t = self._ypos(t, n_qubits=self._wires, sep=self.style.wire_sep)
                self._record(
                    self.axes.add_patch(
                        Circle(
                            (x + self.style.connector_r, y_t),
                            self.style.connector_r,
                            color=self.style.theme.background,
                            zorder=self._Z["connector"],
                        )
                    )
                )
                self._record(
                    self.axes.add_patch(
                        Circle(
                            (x + width - self.style.connector_r, y_t),
                            self.style.connector_r,
                            color=self.style.theme.background,
                            zorder=self._Z["connector"],
                        )
                    )
                )

        return x, width

    def _draw_control_dot(self, wire: int, x: float) -> None:
        """
        Draw a filled control dot at the given wire/x.

        Args:
            wire: Qubit index.
            x: Column anchor x coordinate.
        """
        y = self._ypos(wire, n_qubits=self._wires, sep=self.style.wire_sep)
        self._record(
            self.axes.add_patch(
                Circle((x, y), self.style.control_r, color=self.style.theme.accent, zorder=self._Z["node"])
            )
        )

    def _draw_plus_sign(self, wire: int, x: float) -> None:
        """
        Draw a target ⊕ marker at the given wire/x.

        Args:
            wire: Qubit index.
            x: Column anchor x coordinate.
        """
        y = self._ypos(wire, n_qubits=self._wires, sep=self.style.wire_sep)
        self._record(
            self.axes.add_patch(
                Circle((x, y), self.style.target_r, color=self.style.theme.accent, zorder=self._Z["node"])
            )
        )
        self._record(
            self.axes.add_line(
                plt.Line2D(
                    (x, x),
                    (y - self.style.target_r / 2, y + self.style.target_r / 2),
                    lw=1.5,
                    color=self.style.theme.background,
                    zorder=self._Z["gate_label"],
                )
            )
        )
        self._record(
            self.axes.add_line(
                plt.Line2D(
                    (x - self.style.target_r / 2, x + self.style.target_r / 2),
                    (y, y),
                    lw=1.5,
                    color=self.style.theme.background,
                    zorder=self._Z["gate_label"],
                )
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
        self._record(
            self.axes.add_line(plt.Line2D([x, x], [y1, y2], color=self.style.theme.accent, zorder=self._Z["bridge"]))
        )

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
            self._record(self.axes.add_line(plt.Line2D(xs, ys, color=color, linewidth=2, zorder=self._Z["gate"])))

    def _draw_swap_gate(self, targets: list[int], layer: int) -> float:
        """
        Draw a SWAP between two target wires.

        Args:
            targets: Exactly two wires to swap.
            layer: Column index the gate belongs to.

        Returns:
            The anchor x within the column where the swap glyph is centered.
        """
        t_sorted = sorted(targets)
        x_anchor = self._place(layer) + self.style.gate_pad

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

        # Controlled-X family (CNOT / multi-controlled X): target glyph, not a box
        if gate.is_modified_from(X):
            x_anchor = self._place(layer) + self.style.gate_pad
            for c in controls:
                self._draw_control_dot(c, x_anchor)
                self._draw_bridge(c, targets[0], x_anchor)
            self._draw_plus_sign(targets[0], x_anchor)
            return

        # Controlled SWAP (Fredkin): reuse the SWAP primitive, then add controls
        if getattr(gate.basic_gate, "name", "") == "SWAP":
            x_anchor = self._draw_swap_gate(targets, layer=layer)
            for c in controls:
                self._draw_control_dot(c, x_anchor)
                self._draw_bridge(c, targets[0], x_anchor)
            return

        # Generic controlled gate: draw the target box at this same column,
        # then add control stems to the center of the box.
        label = self._gate_label(gate.basic_gate)
        gate_color = self.style.theme.accent
        x_box, width = self._draw_targets_gate(label=label, targets=targets, layer=layer, color=gate_color)

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
        x = self._place(layer)
        for q in qubits:
            self._draw_measure_symbol(q, x)

    def _draw_concurrent_measures(self, qubits: list[int]) -> None:
        """
        Draw a final column of measurements (one shared column).

        Args:
            qubits: Wires to measure concurrently.
        """
        x = self._place(self._measure_layer)
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
        self._record(
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
        )
        self._record(
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
        )
        self._record(
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
        )

    # Final decoration ----------------------------------------------------

    def _draw_wires(self) -> None:
        """Draw the horizontal wires of every row, up to the last occupied x of that row."""
        for row, x_end in enumerate(self._row_widths):
            self._row = row
            for q in range(self._wires):
                y = self._ypos(q, n_qubits=self._wires, sep=self.style.wire_sep)
                self._record(
                    self.axes.add_line(
                        plt.Line2D(
                            [0, x_end], [y, y], lw=1, color=self.style.theme.surface_muted, zorder=self._Z["wire"]
                        )
                    )
                )

    def _draw_wire_labels(self) -> None:
        """Draw wire labels to the left of every row."""
        labels = self.style.wire_label or [rf"$q_{{{i}}}$" for i in range(self._wires)]
        for row in range(len(self._row_widths)):
            self._row = row
            for i, label in enumerate(labels):
                y = self._ypos(i, n_qubits=self._wires, sep=self.style.wire_sep)
                self._record(
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
                )

    def _drawing_size(self) -> tuple[float, float]:
        """
        Compute the extent of the whole drawing, every row included.

        Returns:
            The drawing ``(width, height)`` in inches.
        """
        width = self.style.padding * 2 + self._max_label_width + self.style.label_pad + max(self._row_widths)
        height = (
            self.style.padding * 2
            + (self._wires - 1) * self.style.wire_sep
            + (len(self._row_widths) - 1) * self._row_height
        )
        return width, height

    def _show_view(self, width: float, height: float) -> None:
        """
        Point the axes at the top-left ``width`` x ``height`` inches of the drawing.

        Whatever falls outside is still drawn, only out of sight until the figure is
        panned or zoomed out. The figure is fitted around the view, so that the axes
        needs no layout engine and an equal aspect ratio never shrinks the drawing.

        Args:
            width: Width (inches) of the drawing to show.
            height: Height (inches) of the drawing to show.
        """
        x_left = -self.style.padding - self._max_label_width - self.style.label_pad
        y_top = self.style.padding + (self._wires - 1) * self.style.wire_sep
        self.axes.set_xlim(x_left, x_left + width)
        self.axes.set_ylim(y_top - height, y_top)

        # The axes fills the figure, bar the band left on top for the title
        figure = self.axes.figure
        figure.set_size_inches(width, height + self._title_band, forward=True)  # ty:ignore[unresolved-attribute]
        self.axes.set_position((0.0, 0.0, 1.0, height / (height + self._title_band)))

        # The limits above were set against the previous figure size, so the labels are
        # only in proportion once the figure has caught up
        self._scale_text()

    def _fit_canvas(self) -> None:
        """
        Lower the figure DPI if the circuit needs a canvas too large to render.

        Even once folded, a circuit with thousands of gates can ask for a canvas
        that matplotlib cannot allocate; rendering it smaller is friendlier than
        failing with ``MemoryError: std::bad_alloc``. The figure is rebuilt here,
        before anything is drawn, because matplotlib saves and shows a figure at
        the DPI it was created with.

        Raises:
            ValueError: If the circuit cannot be drawn at any usable DPI.
        """
        width, height = self._drawing_size()
        height += self._title_band
        dpi = min(
            self.style.dpi,
            int(self._MAX_CANVAS_SIDE / max(width, height)),
            int((self._MAX_CANVAS_PIXELS / (width * height)) ** 0.5),
        )
        if dpi >= self.style.dpi:
            return
        if dpi < 1:
            raise ValueError(
                f"This circuit is too large to draw: it needs a figure of {width:.0f}x{height:.0f} inches. "
                "Draw fewer gates, or lower 'max_row_width' in the style so the circuit wraps more tightly."
            )
        logger.warning(
            "[CircuitRenderer] Circuit needs a {:.0f}x{:.0f} inch figure; lowering the DPI from {} to {} so it fits.",
            width,
            height,
            self.style.dpi,
            dpi,
        )
        if self._owns_figure:
            plt.close(self.axes.get_figure(root=True))
            self._ax = self._make_axes(dpi)
        else:
            self.axes.figure.set_dpi(dpi)

    def _finalise_figure(self) -> None:
        """Finalize axes limits, aspect, background, and title."""
        fig = self.axes.figure
        fig.set_facecolor(self.style.theme.background)

        if self.style.title:
            self.axes.set_title(
                self.style.title,
                pad=self.style.padding * 72,
                color=self.style.theme.surface_muted,
                fontdict={"fontsize": self.style.fontsize},
            )

        width, height = self._drawing_size()
        try:
            get_ipython()  # type: ignore
        except NameError:
            # A window cannot grow past the screen, so showing a tall circuit whole would
            # shrink it until it is unreadable; open on its first rows instead, and let
            # the figure be panned or zoomed out to reach the rest
            if self.style.max_view_height is not None:
                height = min(height, self.style.max_view_height)
        self._show_view(width, height)

        self.axes.set_aspect("equal", adjustable="box")
        self.axes.axis("off")

    # ------------------------------------------------------------------
    # Helpers - human-readable gate labels & π-fractions
    # ------------------------------------------------------------------

    def _ypos(self, index: int, *, n_qubits: int, sep: float) -> float:
        return (n_qubits - 1 - index) * sep - self._row * self._row_height

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
        return name

    @staticmethod
    def _make_axes(dpi: int) -> Axes:
        """
        Create a new figure and axes with the given DPI.

        Args:
            dpi (int): The DPI of the figure

        Returns:
            A newly created Matplotlib Axes.
        """
        figure = plt.figure(dpi=dpi)
        return figure.add_axes((0.0, 0.0, 1.0, 1.0))
