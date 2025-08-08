"""
Improved and more idiomatic implementation of the matplotlib‑based quantum
circuit renderer.  Public behaviour is unchanged; internal structure is
cleaned‑up, redundant parameters are removed, and pythonic conventions are
followed.  The two public entry points remain:

    >>> renderer = MatRenderer(qc)
    >>> renderer.canvas_plot()
    >>> renderer.save("circuit.png")

The companion configuration models (`Theme`, `StyleConfig`) keep using
Pydantic for runtime validation, which is often valuable in interactive
workflows.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrow, FancyBboxPatch
from pydantic import BaseModel, Field

from qilisdk.digital.gates import BasicGate, Controlled, Gate, M, X

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from qilisdk.digital import Circuit

###############################################################################
# Configuration models
###############################################################################


class Theme(BaseModel):
    """Simple colour theme configuration."""

    bgcolor: str = Field(description="Figure background colour.")
    color: str = Field(description="Primary text colour.")
    wire_color: str = Field(description="Colour of inactive wires.")
    gate_color: str = Field(description="Default gate fill colour.")
    plus_color: str = Field(description="Colour of ⊕ and control dots.")

    class Config:
        """Make *Theme* immutable so that it can be shared safely."""

        frozen = True


light = Theme(
    bgcolor="#FFFFFF",
    color="#000000",
    wire_color="#F0F0F0",
    gate_color="#AC115F",
    plus_color="#5E56A1",
)

dark = Theme(
    bgcolor="#000000",
    color="#FFFFFF",
    wire_color="#FFFFFF",
    gate_color="#FFFFFF",
    plus_color="#5E56A1",
)


class StyleConfig(BaseModel):
    """All visual parameters controlling the appearance of a circuit plot."""

    dpi: int = Field(150, description="Figure DPI.")
    fontsize: int = Field(10, description="Base font size (pt).")
    end_wire_ext: int = Field(2, description="Extra space after last layer.")
    padding: float = Field(0.3, description="Padding around drawing (inches).")
    gate_margin: float = Field(0.15, description="Left/right margin per gate.")
    wire_sep: float = Field(0.5, description="Vertical separation of wires.")
    layer_sep: float = Field(0.5, description="Horizontal separation of layers.")
    gate_pad: float = Field(0.05, description="Padding around gate text.")
    label_pad: float = Field(0.1, description="Padding before wire label.")
    bulge: str = Field("round", description="Box‑style for gate rectangles.")
    align_layer: bool = Field(True, description="Align layers across wires.")
    theme: Theme = Field(light, description="Colour theme.")
    title: Optional[str] = Field(None, description="Figure title.")
    wire_label: Optional[List[Any]] = Field(None, description="Custom wire labels.")

    # ---------------------------------------------------------------------
    # Convenience derived properties – keep as *property* for live updates.
    # ---------------------------------------------------------------------

    @property
    def measure_color(self) -> str:  # noqa: D401 – not @cached_property on purpose
        """Colour of measurement symbol – contrast with gate fill."""

        return "#FFFFFF" if self.theme is dark else "#000000"

    # Aliases – shorter, nicer to read inside drawing code -----------------

    @property
    def bgcolor(self) -> str:
        return self.theme.bgcolor

    @property
    def color(self) -> str:
        return self.theme.color

    @property
    def wire_color(self) -> str:
        return self.theme.wire_color

    @property
    def default_gate_color(self) -> str:
        return self.theme.gate_color


###############################################################################
# Internal helpers
###############################################################################


def _ypos(index: int, *, n_qubits: int, sep: float) -> float:
    """Map **qubit index** to **matplotlib y‑coordinate**."""

    return (n_qubits - 1 - index) * sep


###############################################################################
# Base renderer – shared layer bookkeeping
###############################################################################


class BaseRenderer:
    """Abstract helper that stores layer widths for *MatRenderer* variants."""

    #: Minimum spacing (inches) before the first layer so labels fit nicely.
    _START_PAD: float = 0.1

    def __init__(self, style: StyleConfig, *, n_qubits: int) -> None:
        self.style = style
        self._qwires = n_qubits
        # *layer_widths[w][l]* – width (inches) of layer *l* on wire *w*
        self._layer_widths: dict[int, list[float]] = {
            w: [self._START_PAD] for w in range(n_qubits)
        }

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _xskip(self, wires: Iterable[int], layer: int) -> float:
        """Total horizontal offset needed to start *layer* on *wires*."""

        if self.style.align_layer:
            wires = range(self._qwires)
        return max(sum(self._layer_widths[w][:layer]) for w in wires)

    def _reserve(self, width: float, wires: Iterable[int], layer: int, *, xskip: float = 0.0) -> None:
        """Reserve *width* (plus margins) on *wires* for *layer* at *xskip*."""

        full_width = width + self.style.gate_margin * 2
        for w in wires:
            layers = self._layer_widths[w]
            if len(layers) > layer:
                layers[layer] = max(layers[layer], full_width)
            else:
                gap = xskip - sum(layers) if xskip else 0.0
                layers.append(gap + full_width)

    # ------------------------------------------------------------------
    # Convenience metrics – exposed as *property* so subclasses can use them
    # without redundant locals.
    # ------------------------------------------------------------------

    @property
    def _renderer(self):
        """Matplotlib backend renderer for text size measurements."""

        # Lazily created when first needed – figure must already exist
        return self.axes.figure.canvas.get_renderer()

    # ------------------------------------------------------------------
    # Abstract API expected by subclasses
    # ------------------------------------------------------------------

    @property
    def axes(self) -> Axes:  # noqa: D401 – to be implemented by subclass
        raise NotImplementedError


###############################################################################
# Matplotlib implementation
###############################################################################


class MatRenderer(BaseRenderer):
    """Render a :class:`~qilisdk.digital.Circuit` using *matplotlib*."""

    # Class‑level constants – keep numeric tweaks in one place -------------
    _MIN_GATE_H: float = 0.2
    _MIN_GATE_W: float = 0.2
    _ARROW_LEN: float = 0.06
    _CONNECTOR_R: float = 0.01
    _TARGET_R: float = 0.12
    _CONTROL_R: float = 0.05

    # Z-order groups -------------------------------------------------------
    _Z = {
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

    def __init__(self, qc: Circuit, ax: Axes | None = None, *, style: StyleConfig | None = None) -> None:
        self.qc = qc
        self._ax = ax or self._make_axes(style)
        super().__init__(style or StyleConfig(), n_qubits=qc.nqubits)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def axes(self) -> Axes:  # override *BaseRenderer* stub
        return self._ax

    # Figure‑level helpers -------------------------------------------------

    def canvas_plot(self) -> None:
        """Render the circuit onto the encapsulated *Axes*."""

        self._draw_wire_labels()

        for gate in self.qc.gates:
            if isinstance(gate, M):
                self._handle_measure(gate)
            elif isinstance(gate, Controlled):
                self._draw_multiq_gate(gate)
            else:
                self._draw_singleq_gate(gate)

        self._draw_wires()
        self._finalise_figure()
        plt.tight_layout()
        plt.show()

    def save(self, filename: str, **kwargs) -> None:  # thin wrapper
        """Save current figure to *filename* (passes through to *plt.savefig*)."""

        self.axes.figure.savefig(filename, bbox_inches="tight", **kwargs)

    # ------------------------------------------------------------------
    # Low‑level drawing helpers (private)
    # ------------------------------------------------------------------

    # Shared geometry -----------------------------------------------------

    def _text_width(self, text: str, /, *, fontweight="normal", fontstyle="normal") -> float:
        """Return rendered *text* width in inches for given style."""

        t = plt.Text(0, 0, text, fontsize=self.style.fontsize, fontweight=fontweight,
                     fontfamily="monospace", fontstyle=fontstyle)
        self.axes.add_artist(t)
        width = t.get_window_extent(renderer=self._renderer).width / self.style.dpi
        t.remove()
        return width

    # Basic primitives ----------------------------------------------------

    def _draw_control(self, wire: int, x: float) -> None:
        y = _ypos(wire, n_qubits=self._qwires, sep=self.style.wire_sep)
        self.axes.add_patch(Circle((x, y), self._CONTROL_R, color=self.style.theme.plus_color, zorder=self._Z["node"]))

    def _draw_target(self, wire: int, x: float) -> None:
        y = _ypos(wire, n_qubits=self._qwires, sep=self.style.wire_sep)
        self.axes.add_patch(Circle((x, y), self._TARGET_R, color=self.style.theme.plus_color, zorder=self._Z["node"]))
        self.axes.add_line(plt.Line2D((x, x), (y - self._TARGET_R / 2, y + self._TARGET_R / 2), lw=1.5,
                                      color=self.style.bgcolor, zorder=self._Z["gate_label"]))
        self.axes.add_line(plt.Line2D((x - self._TARGET_R / 2, x + self._TARGET_R / 2), (y, y), lw=1.5,
                                      color=self.style.bgcolor, zorder=self._Z["gate_label"]))

    def _draw_bridge(self, wire_a: int, wire_b: int, x: float) -> None:
        y1, y2 = (
            _ypos(wire_a, n_qubits=self._qwires, sep=self.style.wire_sep),
            _ypos(wire_b, n_qubits=self._qwires, sep=self.style.wire_sep),
        )
        self.axes.add_line(plt.Line2D([x, x], [y1, y2], color=self.style.theme.plus_color, zorder=self._Z["bridge"]))

    def _draw_swap_mark(self, wire: int, x: float) -> None:
        y = _ypos(wire, n_qubits=self._qwires, sep=self.style.wire_sep)
        offset = self._MIN_GATE_W / 3
        for xs, ys in (
            ([x + offset, x - offset], [y + self._MIN_GATE_H / 2, y - self._MIN_GATE_H / 2]),
            ([x - offset, x + offset], [y + self._MIN_GATE_H / 2, y - self._MIN_GATE_H / 2]),
        ):
            self.axes.add_line(plt.Line2D(xs, ys, color=self.style.default_gate_color, linewidth=2, zorder=self._Z["gate"]))

    # Gate‑level drawing ---------------------------------------------------

    def _draw_singleq_gate(self, gate: Gate) -> None:
        wire = gate.target_qubits[0]
        layer = len(self._layer_widths[wire])

        label = self._gate_label(gate)
        width = max(self._text_width(label) + self.style.gate_pad * 2, self._MIN_GATE_W)
        x = self._xskip([wire], layer) + self.style.gate_margin
        y = _ypos(wire, n_qubits=self._qwires, sep=self.style.wire_sep)

        self._reserve(width, [wire], layer)

        # Patch + text
        self.axes.add_patch(FancyBboxPatch((x, y - self._MIN_GATE_H / 2), width, self._MIN_GATE_H,
                                           boxstyle=self.style.bulge, mutation_scale=0.3,
                                           facecolor=self.style.default_gate_color, edgecolor=self.style.default_gate_color,
                                           zorder=self._Z["gate"]))
        self.axes.text(x + width / 2, y, label, ha="center", va="center", fontsize=self.style.fontsize,
                        color=self.style.bgcolor, family="monospace", zorder=self._Z["gate_label"])

    def _draw_multiq_gate(self, gate: Controlled) -> None:
        # In many cases *target_qubits* may be None (global phase) – fall back to all
        targets = list(gate.target_qubits or range(self._qwires))
        controls = list(gate.control_qubits or [])
        wires = sorted(set(targets + controls))
        layer = max(len(self._layer_widths[w]) for w in wires)
        x = self._xskip(wires, layer) + self.style.gate_margin

        # Pre‑reserve minimal width (control/target nodes only need a dot width)
        self._reserve(self._TARGET_R * 2, wires, layer, xskip=x)

        if gate.is_modified_from(X):  # CNOT or multi‑controlled X
            # Controls
            for c in controls:
                self._draw_control(c, x + self.style.gate_pad)
            # Single target assumed for *Controlled* of *X*
            self._draw_target(targets[0], x + self.style.gate_pad)
            # Bridges
            for c in controls:
                self._draw_bridge(c, targets[0], x + self.style.gate_pad)
        elif gate.basic_gate.name == "SWAP":
            # Draw X‑marks and bridge
            for t in targets:
                self._draw_swap_mark(t, x + self.style.gate_pad)
            self._draw_bridge(targets[0], targets[1], x + self.style.gate_pad)
        else:
            # Fallback – generic multi‑q gate rendered as tall rectangle
            adj_targets = sorted(targets)
            label = gate.basic_gate.name
            width = max(self._text_width(label) + self.style.gate_pad * 2, self._MIN_GATE_W)
            height = (adj_targets[-1] - adj_targets[0]) * self.style.wire_sep + self._MIN_GATE_H
            y_bottom = _ypos(adj_targets[0], n_qubits=self._qwires, sep=self.style.wire_sep) - self._MIN_GATE_H / 2

            self._reserve(width, wires, layer, xskip=x)

            gate_color = self.style.theme.plus_color if controls else self.style.default_gate_color

            # Draw gate body
            self.axes.add_patch(FancyBboxPatch((x, y_bottom), width, height, boxstyle=self.style.bulge,
                                               mutation_scale=0.3, facecolor=self.style.default_gate_color,
                                               edgecolor=self.style.default_gate_color, zorder=self._Z["gate"]))
            # Label at centre
            y_center = _ypos(adj_targets[0] + adj_targets[-1] // 2, n_qubits=self._qwires, sep=self.style.wire_sep)
            self.axes.text(x + width / 2, y_center, label, ha="center", va="center", fontsize=self.style.fontsize,
                            color=self.style.bgcolor, family="monospace", zorder=self._Z["gate_label"])

            # Draw connectors inside box for each additional target (visual cues)
            if len(targets) > 1:
                for t in targets:
                    y_t = _ypos(t, n_qubits=self._qwires, sep=self.style.wire_sep)
                    self.axes.add_patch(Circle((x + self._CONNECTOR_R, y_t), self._CONNECTOR_R,
                                               color=self.style.bgcolor, zorder=self._Z["connector"]))
                    self.axes.add_patch(Circle((x + width - self._CONNECTOR_R, y_t), self._CONNECTOR_R,
                                               color=self.style.bgcolor, zorder=self._Z["connector"]))

            # Controls (if any) – bridge to first target
            for c in controls:
                self._draw_control(c, x + width / 2)
                self._draw_bridge(c, targets[0], x + width / 2)

    # Measurements --------------------------------------------------------

    def _handle_measure(self, gate: M) -> None:
        wire = gate.target_qubits[0]
        layer = max(len(self._layer_widths[w]) for w in range(wire + 1))
        x = self._xskip(range(wire + 1), layer) + self.style.gate_margin
        self._reserve(self._MIN_GATE_W, range(wire + 1), layer, xskip=x)

        y = _ypos(wire, n_qubits=self._qwires, sep=self.style.wire_sep)

        # Outer box
        self.axes.add_patch(FancyBboxPatch((x, y - self._MIN_GATE_H / 2), self._MIN_GATE_W, self._MIN_GATE_H,
                                           boxstyle=self.style.bulge, mutation_scale=0.3, facecolor=self.style.bgcolor,
                                           edgecolor=self.style.measure_color, linewidth=1.25, zorder=self._Z["gate"]))
        # Arc and arrow
        self.axes.add_patch(Arc((x + self._MIN_GATE_W / 2, y - self._MIN_GATE_H / 2),
                                self._MIN_GATE_W * 1.5, self._MIN_GATE_H, theta1=0, theta2=180, linewidth=1.25,
                                color=self.style.measure_color, zorder=self._Z["gate_label"]))
        self.axes.add_patch(FancyArrow(x + self._MIN_GATE_W / 2, y - self._MIN_GATE_H / 2,
                                       dx=self._MIN_GATE_W * 0.7, dy=self._MIN_GATE_H * 0.7, width=0.0,
                                       length_includes_head=True, color=self.style.measure_color,
                                       linewidth=1.25, zorder=self._Z["gate_label"]))

    # Final decoration ----------------------------------------------------

    def _draw_wires(self) -> None:
        max_len = max(map(sum, self._layer_widths.values())) + self.style.end_wire_ext * self.style.layer_sep
        for i in range(self._qwires):
            y = _ypos(i, n_qubits=self._qwires, sep=self.style.wire_sep)
            self.axes.add_line(plt.Line2D([0, max_len], [y, y], lw=1, color=self.style.wire_color, zorder=self._Z["wire"]))

    def _draw_wire_labels(self) -> None:
        labels = self.style.wire_label or [fr"$q_{{{i}}}$" for i in range(self._qwires)]
        widths = [self._text_width(lbl) for lbl in labels]
        self._max_label_width = max(widths)

        for i, label in enumerate(labels):
            y = _ypos(i, n_qubits=self._qwires, sep=self.style.wire_sep)
            self.axes.text(-self.style.label_pad, y, label, ha="right", va="center", fontsize=self.style.fontsize,
                           family="monospace", color=self.style.color, zorder=self._Z["wire_label"])

    # Figure layout -------------------------------------------------------

    def _finalise_figure(self) -> None:
        fig = self.axes.figure
        fig.set_facecolor(self.style.bgcolor)

        x_end = self.style.padding + self.style.end_wire_ext * self.style.layer_sep + max(map(sum, self._layer_widths.values()))
        y_end = self.style.padding + (self._qwires - 1) * self.style.wire_sep

        self.axes.set_xlim(
            -self.style.padding - self._max_label_width - self.style.label_pad,
            x_end,
        )
        self.axes.set_ylim(-self.style.padding, y_end)

        if self.style.title:
            self.axes.set_title(self.style.title, pad=10, color=self.style.wire_color,
                                fontdict={"fontsize": self.style.fontsize})

        # In IPython keep figure square so equal aspect ratio does not shrink
        try:
            get_ipython()  # type: ignore
            size = max(self.axes.get_xlim()[1] - self.axes.get_xlim()[0], y_end + self.style.padding)
            fig.set_size_inches(size, size, forward=True)
        except NameError:
            fig.set_size_inches(self.axes.get_xlim()[1] - self.axes.get_xlim()[0], y_end + self.style.padding, forward=True)

        self.axes.set_aspect("equal", adjustable="box")
        self.axes.axis("off")

    # ------------------------------------------------------------------
    # Helpers – human‑readable gate labels & π‑fractions
    # ------------------------------------------------------------------

    @staticmethod
    def _pi_fraction(value: float, /, tol: float = 1e-2) -> str:
        """Return a LaTeX‑friendly string representing *value* as a π‑fraction."""

        coeff = value / np.pi
        if abs(coeff - round(coeff)) < tol:
            n = round(coeff)
            return fr"[{n}\\pi]" if n != 1 else r"[\pi]"
        for d in (2, 3, 4, 6, 8, 12):
            num = round(coeff * d)
            if abs(num / d - coeff) < tol:
                return fr"[{num}\\pi/{d}]" if num != 1 else fr"[\\pi/{d}]"
        return f"[{value:.2f}]"

    def _gate_label(self, gate: Gate) -> str:
        # if gate.is_parameterized and getattr(self, "showarg", True):
        #     return fr"${{{gate.name}}}_{{{self._pi_fraction(gate.parameter_values[0])}}}$"
        return gate.name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_axes(style: StyleConfig | None) -> Axes:
        fig = plt.figure(dpi=(style or StyleConfig()).dpi)
        return fig.add_subplot(111)
