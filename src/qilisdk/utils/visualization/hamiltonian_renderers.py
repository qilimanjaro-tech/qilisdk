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
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib.pyplot as plt
import rustworkx as rx
from loguru import logger
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Wedge

from qilisdk.utils.visualization.style import HamiltonianStyle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap

    from qilisdk.analog.hamiltonian import Hamiltonian

# Number of Pauli operators above which a term can no longer be drawn as a single edge.
_TWO_BODY: Final[int] = 2
# Coefficients whose imaginary part exceeds this are considered genuinely complex.
_IMAG_TOL: Final[float] = 1e-12
# Colour scales narrower than this are widened so that the colour bar stays readable.
_FLAT_SCALE_TOL: Final[float] = 1e-12
# Relative luminance above which dark text is more legible than light text.
_LUMINANCE_MIDPOINT: Final[float] = 0.55
# Number of candidate directions tried when placing a qubit label around its node.
_LABEL_CANDIDATES: Final[int] = 24
# Cosine/sine magnitude above which a label anchors to a side rather than being centred.
_ALIGNMENT_TOL: Final[float] = 0.3

# Order in which local fields are laid out around a node.
_PAULI_ORDER: Final[dict[str, int]] = {"X": 0, "Y": 1, "Z": 2}


class MatplotlibHamiltonianRenderer:
    """Render a :class:`~qilisdk.analog.Hamiltonian` as an interaction graph using *matplotlib*.

    Every qubit becomes a node whose disc is split into one slice per local field acting on it,
    and every two-qubit term becomes an edge between the qubits it couples. Slice and edge
    colours encode the coefficient of the corresponding term, while the edge line style encodes
    the coupling type. Terms acting on three or more qubits are drawn as star-shaped hyperedges.

    Example:
        .. code-block:: python

            from qilisdk.analog import X, Z
            from qilisdk.utils.visualization.hamiltonian_renderers import MatplotlibHamiltonianRenderer

            H = X(0) + 2 * Z(1) + 0.5 * Z(0) * Z(1)
            renderer = MatplotlibHamiltonianRenderer(H)
            renderer.plot()
    """

    _Z: Final = {"edge": 1, "hyperedge": 1, "node": 3, "label": 4}

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        ax: Axes | None = None,
        *,
        style: HamiltonianStyle | None = None,
    ) -> None:
        """Initialize the renderer.

        Args:
            hamiltonian (Hamiltonian): The Hamiltonian to render.
            ax (Axes | None): Axes to draw on. A new figure and axes are created when omitted.
            style (HamiltonianStyle | None): Customization options for the plot appearance.
                Defaults to :class:`HamiltonianStyle`.
        """
        self.hamiltonian = hamiltonian
        self.style = style or HamiltonianStyle()
        self.ax = ax or self._make_axes(self.style)
        # Populated by _collect_terms().
        self._fields: dict[int, dict[str, float]] = {}
        self._couplings: dict[tuple[int, int], dict[str, float]] = {}
        self._multi_body: list[tuple[tuple[int, ...], str, float]] = []
        self._line_styles: dict[str, Any] = {}
        self._offset: float = 0.0
        self._use_magnitude: bool = False

    @property
    def axes(self) -> Axes:
        """The axes the Hamiltonian is drawn on."""
        return self.ax

    def plot(self) -> None:
        """Render the Hamiltonian interaction graph on the current axes.

        Raises:
            ValueError: If the Hamiltonian acts on no qubits and therefore has no graph to draw.
        """
        logger.debug("[HamiltonianRenderer] Rendering Hamiltonian on {} qubits", self.hamiltonian.nqubits)

        if self.hamiltonian.nqubits == 0:
            raise ValueError("Cannot draw a Hamiltonian that does not act on any qubit.")

        self._collect_terms()
        graph = self.build_graph()
        positions = self._compute_positions(graph)
        radius = self._compute_radius(positions)

        field_norm, coupling_norm = self._make_norms()
        cmap = self._make_colormap()

        self._draw_couplings(positions, radius, cmap, coupling_norm)
        self._draw_multi_body(positions, radius, cmap, coupling_norm)
        self._draw_nodes(positions, radius, cmap, field_norm)

        self._setup_axes(positions, radius)
        self._draw_colorbars(cmap, field_norm, coupling_norm)
        self._draw_legend()

        plt.draw()

    def build_graph(self) -> rx.PyGraph:
        """Build the qubit interaction graph backing the drawing.

        Nodes are added in qubit order, so a node's index equals its qubit index. Each node
        payload maps a local field type to its coefficient, and each edge payload maps a
        coupling type to its coefficient.

        Returns:
            rx.PyGraph: Graph with one node per qubit and one edge per coupled qubit pair.
        """
        if not self._fields and not self._couplings and not self._multi_body:
            self._collect_terms()

        graph: rx.PyGraph = rx.PyGraph()
        graph.add_nodes_from([self._fields.get(qubit, {}) for qubit in range(self.hamiltonian.nqubits)])
        for (qubit_a, qubit_b), types in self._couplings.items():
            graph.add_edge(qubit_a, qubit_b, types)
        logger.debug(
            "[HamiltonianRenderer] Built interaction graph with {} nodes and {} edges",
            graph.num_nodes(),
            graph.num_edges(),
        )
        return graph

    def save(self, filename: str) -> None:
        """Save the current figure to disk.

        Args:
            filename (str): Path to save the figure to (e.g. ``'hamiltonian.png'``).
        """
        logger.debug("[HamiltonianRenderer] Saving figure to {}", filename)
        if isinstance(self.ax.figure, Figure):
            self.ax.figure.savefig(filename, bbox_inches="tight")

    def show(self) -> None:  # noqa: PLR6301
        """Show the current figure."""
        plt.show()

    # ------------------------------------------------------------------
    # Term collection
    # ------------------------------------------------------------------
    def _collect_terms(self) -> None:
        """Sort the Hamiltonian terms into local fields, two-qubit couplings and hyperedges."""
        elements = self.hamiltonian.elements
        self._use_magnitude = any(abs(complex(coeff).imag) > _IMAG_TOL for coeff in elements.values())
        if self._use_magnitude:
            logger.warning(
                "[HamiltonianRenderer] Hamiltonian has complex coefficients; colouring by magnitude instead."
            )

        self._fields = {}
        self._couplings = {}
        self._multi_body = []
        self._offset = 0.0

        for operators, coefficient in elements.items():
            value = self._to_scalar(coefficient)
            acting = [op for op in operators if op.name != "I"]
            qubits = sorted({op.qubit for op in acting})

            if not qubits:
                self._offset += value
                continue

            label = "".join(op.name for op in sorted(acting, key=lambda op: (op.qubit, _PAULI_ORDER.get(op.name, 3))))
            if len(qubits) == 1:
                self._fields.setdefault(qubits[0], {})
                self._fields[qubits[0]][label] = self._fields[qubits[0]].get(label, 0.0) + value
            elif len(qubits) == _TWO_BODY:
                pair = (qubits[0], qubits[1])
                self._couplings.setdefault(pair, {})
                self._couplings[pair][label] = self._couplings[pair].get(label, 0.0) + value
            else:
                self._multi_body.append((tuple(qubits), label, value))

        if self._multi_body and not self.style.show_multi_body:
            logger.warning(
                "[HamiltonianRenderer] Skipping {} term(s) acting on more than two qubits.", len(self._multi_body)
            )

        self._build_line_styles()

    def _to_scalar(self, coefficient: complex) -> float:
        """Reduce a term coefficient to the real number used for colouring.

        Args:
            coefficient (complex): Coefficient of a Hamiltonian term.

        Returns:
            float: The magnitude when the Hamiltonian has complex coefficients, the real part otherwise.
        """
        value = complex(coefficient)
        return abs(value) if self._use_magnitude else value.real

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _compute_positions(self, graph: rx.PyGraph) -> dict[int, tuple[float, float]]:
        """Position the qubit nodes and normalize the result into the unit square.

        Args:
            graph (rx.PyGraph): The qubit interaction graph.

        Returns:
            dict[int, tuple[float, float]]: Normalized position per qubit index.

        Raises:
            ValueError: If explicit positions are given but do not cover every qubit.
        """
        if self.style.positions is not None:
            missing = [q for q in range(self.hamiltonian.nqubits) if q not in self.style.positions]
            if missing:
                raise ValueError(f"HamiltonianStyle.positions is missing entries for qubits {missing}.")
            given = self.style.positions
            raw = {qubit: (float(given[qubit][0]), float(given[qubit][1])) for qubit in range(self.hamiltonian.nqubits)}
        else:
            layout = self.style.layout
            if layout == "spring":
                pos = rx.spring_layout(graph, seed=self.style.layout_seed)
            elif layout == "circular":
                pos = rx.circular_layout(graph)
            elif layout == "shell":
                pos = rx.shell_layout(graph)
            elif layout == "spiral":
                pos = rx.spiral_layout(graph)
            else:
                pos = rx.random_layout(graph, seed=self.style.layout_seed)
            raw = {int(node): (float(pos[node][0]), float(pos[node][1])) for node in graph.node_indices()}

        return self._normalize_positions(raw)

    @staticmethod
    def _normalize_positions(raw: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
        """Rescale positions into the unit square while preserving the aspect ratio.

        Args:
            raw (dict[int, tuple[float, float]]): Positions as produced by the layout algorithm.

        Returns:
            dict[int, tuple[float, float]]: Positions rescaled to fit the unit square.
        """
        xs = [x for x, _ in raw.values()]
        ys = [y for _, y in raw.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span <= _FLAT_SCALE_TOL:
            return dict.fromkeys(raw, (0.5, 0.5))
        x_offset = (span - (max(xs) - min(xs))) / 2
        y_offset = (span - (max(ys) - min(ys))) / 2
        return {
            qubit: ((x - min(xs) + x_offset) / span, (y - min(ys) + y_offset) / span) for qubit, (x, y) in raw.items()
        }

    def _compute_radius(self, positions: dict[int, tuple[float, float]]) -> float:
        """Derive the node radius from how tightly the layout packs the nodes.

        Args:
            positions (dict[int, tuple[float, float]]): Normalized qubit positions.

        Returns:
            float: Node radius in normalized layout units.
        """
        points = list(positions.values())
        distances = [math.dist(points[i], points[j]) for i in range(len(points)) for j in range(i + 1, len(points))]
        positive = [d for d in distances if d > _FLAT_SCALE_TOL]
        closest = min(positive) if positive else 1.0
        return max(self.style.node_radius * closest, self.style.min_node_radius)

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    def _make_colormap(self) -> Colormap:
        """Build the colormap used for the coefficient strengths.

        Returns:
            Colormap: The configured colormap, or a gradient built from the theme colours.
        """
        if self.style.colormap is not None:
            return plt.get_cmap(self.style.colormap)
        theme = self.style.theme
        stops = [theme.primary, theme.surface, theme.accent] if self._spans_zero() else [theme.primary, theme.accent]
        return LinearSegmentedColormap.from_list("qilisdk_hamiltonian", stops)

    def _spans_zero(self) -> bool:
        """Whether the drawn coefficients contain both negative and positive values.

        Returns:
            bool: True when a diverging colour scale is appropriate.
        """
        values = self._all_values()
        return bool(values) and min(values) < 0 < max(values)

    def _all_values(self) -> list[float]:
        """Every coefficient that gets a colour.

        Returns:
            list[float]: Local field, coupling and hyperedge coefficients.
        """
        return self._field_values() + self._coupling_values()

    def _field_values(self) -> list[float]:
        """Coefficients of the local field terms.

        Returns:
            list[float]: One value per local field slice.
        """
        return [value for types in self._fields.values() for value in types.values()]

    def _coupling_values(self) -> list[float]:
        """Coefficients of the coupling terms, hyperedges included.

        Returns:
            list[float]: One value per coupling edge or hyperedge.
        """
        values = [value for types in self._couplings.values() for value in types.values()]
        if self.style.show_multi_body:
            values += [value for _, _, value in self._multi_body]
        return values

    def _make_norms(self) -> tuple[Normalize, Normalize]:
        """Build the normalizations mapping coefficients onto the colormap.

        Returns:
            tuple[Normalize, Normalize]: Normalization for the local fields and for the couplings.
                They are the same object unless separate colour scales are requested.
        """
        if self.style.separate_color_scales:
            return self._norm_for(self._field_values()), self._norm_for(self._coupling_values())
        shared = self._norm_for(self._all_values())
        return shared, shared

    @staticmethod
    def _norm_for(values: list[float]) -> Normalize:
        """Build a normalization covering the given values.

        Args:
            values (list[float]): Coefficients the colour scale has to cover.

        Returns:
            Normalize: A symmetric normalization when the values straddle zero, a plain one otherwise.
        """
        if not values:
            return Normalize(vmin=-1.0, vmax=1.0)
        low, high = min(values), max(values)
        if low < 0 < high:
            bound = max(abs(low), abs(high))
            return Normalize(vmin=-bound, vmax=bound)
        if high - low <= _FLAT_SCALE_TOL:
            return Normalize(vmin=low - 0.5, vmax=high + 0.5)
        return Normalize(vmin=low, vmax=high)

    @staticmethod
    def _contrast_color(facecolor: tuple[float, ...]) -> str:
        """Pick a text colour that stays legible on the given fill.

        Args:
            facecolor (tuple[float, ...]): RGB(A) fill colour of the patch the text sits on.

        Returns:
            str: ``'black'`` on light fills, ``'white'`` on dark ones.
        """
        red, green, blue = facecolor[0], facecolor[1], facecolor[2]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        return "black" if luminance > _LUMINANCE_MIDPOINT else "white"

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_nodes(
        self,
        positions: dict[int, tuple[float, float]],
        radius: float,
        cmap: Colormap,
        norm: Normalize,
    ) -> None:
        """Draw one disc per qubit, split into a slice per local field.

        Args:
            positions (dict[int, tuple[float, float]]): Normalized qubit positions.
            radius (float): Node radius in normalized layout units.
            cmap (Colormap): Colormap for the coefficient strengths.
            norm (Normalize): Normalization applied to the local field coefficients.
        """
        style = self.style
        theme = style.theme
        for qubit, (x, y) in positions.items():
            types = sorted(self._fields.get(qubit, {}).items(), key=lambda kv: (_PAULI_ORDER.get(kv[0], 3), kv[0]))
            if not types:
                self.ax.add_patch(
                    Wedge(
                        (x, y),
                        radius,
                        0,
                        360,
                        facecolor=theme.surface_muted,
                        edgecolor=theme.border,
                        zorder=self._Z["node"],
                    )
                )
            else:
                sweep = 360.0 / len(types)
                for index, (label, value) in enumerate(types):
                    start = 90.0 - (index + 1) * sweep
                    facecolor = cmap(norm(value))
                    self.ax.add_patch(
                        Wedge(
                            (x, y),
                            radius,
                            start,
                            start + sweep,
                            facecolor=facecolor,
                            edgecolor=theme.border,
                            zorder=self._Z["node"],
                        )
                    )
                    if style.show_field_labels:
                        mid = math.radians(start + sweep / 2)
                        distance = 0.0 if len(types) == 1 else radius * 0.55
                        self.ax.text(
                            x + distance * math.cos(mid),
                            y + distance * math.sin(mid),
                            label,
                            ha="center",
                            va="center",
                            fontsize=style.field_label_fontsize,
                            color=self._contrast_color(facecolor),
                            fontproperties=style.font,
                            zorder=self._Z["label"],
                        )

            if style.show_qubit_labels:
                angle = self._label_angle(qubit, positions)
                self.ax.text(
                    x + (radius + 0.04) * math.cos(angle),
                    y + (radius + 0.04) * math.sin(angle),
                    f"q{qubit}",
                    ha=self._horizontal_alignment(math.cos(angle)),
                    va=self._vertical_alignment(math.sin(angle)),
                    fontsize=style.qubit_label_fontsize,
                    color=theme.on_background,
                    fontproperties=style.font,
                    zorder=self._Z["label"],
                )

    def _label_angle(self, qubit: int, positions: dict[int, tuple[float, float]]) -> float:
        """Pick the direction around a node that is furthest from every edge leaving it.

        This keeps the qubit label clear of the coupling lines and their labels.

        Args:
            qubit (int): The qubit whose label is being placed.
            positions (dict[int, tuple[float, float]]): Normalized qubit positions.

        Returns:
            float: Angle in radians at which to place the label, measured from the positive x axis.
        """
        origin = positions[qubit]
        occupied: list[float] = []
        for qubit_a, qubit_b in self._couplings:
            if qubit == qubit_a:
                neighbour = qubit_b
            elif qubit == qubit_b:
                neighbour = qubit_a
            else:
                neighbour = None
            if neighbour is not None:
                occupied.append(self._angle_between(origin, positions[neighbour]))
        if self.style.show_multi_body:
            for qubits, _, _ in self._multi_body:
                if qubit in qubits:
                    centroid = (
                        sum(positions[q][0] for q in qubits) / len(qubits),
                        sum(positions[q][1] for q in qubits) / len(qubits),
                    )
                    occupied.append(self._angle_between(origin, centroid))

        below = -math.pi / 2
        if not occupied:
            return below

        candidates = [self._angle_difference(below + index * math.pi / 12, 0.0) for index in range(_LABEL_CANDIDATES)]
        return max(
            candidates,
            key=lambda angle: (
                round(min(abs(self._angle_difference(angle, taken)) for taken in occupied), 3),
                -abs(self._angle_difference(angle, below)),
            ),
        )

    @staticmethod
    def _angle_between(origin: tuple[float, float], target: tuple[float, float]) -> float:
        """Angle of the direction from one point to another.

        Args:
            origin (tuple[float, float]): Point the direction starts from.
            target (tuple[float, float]): Point the direction points to.

        Returns:
            float: Angle in radians, measured from the positive x axis.
        """
        return math.atan2(target[1] - origin[1], target[0] - origin[0])

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        """Signed difference between two angles, wrapped to [-pi, pi].

        Args:
            first (float): First angle in radians.
            second (float): Second angle in radians.

        Returns:
            float: The wrapped difference ``first - second``.
        """
        return (first - second + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _horizontal_alignment(cosine: float) -> str:
        """Horizontal text alignment for a label placed in the given direction.

        Args:
            cosine (float): Cosine of the placement angle.

        Returns:
            str: A matplotlib horizontal alignment.
        """
        if cosine > _ALIGNMENT_TOL:
            return "left"
        if cosine < -_ALIGNMENT_TOL:
            return "right"
        return "center"

    @staticmethod
    def _vertical_alignment(sine: float) -> str:
        """Vertical text alignment for a label placed in the given direction.

        Args:
            sine (float): Sine of the placement angle.

        Returns:
            str: A matplotlib vertical alignment.
        """
        if sine > _ALIGNMENT_TOL:
            return "bottom"
        if sine < -_ALIGNMENT_TOL:
            return "top"
        return "center"

    def _draw_couplings(
        self,
        positions: dict[int, tuple[float, float]],
        radius: float,
        cmap: Colormap,
        norm: Normalize,
    ) -> None:
        """Draw one line per two-qubit coupling term.

        Args:
            positions (dict[int, tuple[float, float]]): Normalized qubit positions.
            radius (float): Node radius in normalized layout units.
            cmap (Colormap): Colormap for the coefficient strengths.
            norm (Normalize): Normalization applied to the coupling coefficients.
        """
        style = self.style
        for (qubit_a, qubit_b), types in self._couplings.items():
            start, end = self._trim(positions[qubit_a], positions[qubit_b], radius)
            ordered = sorted(types.items())
            for index, (label, value) in enumerate(ordered):
                curvature = (index - (len(ordered) - 1) / 2) * style.coupling_curvature
                self.ax.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="-",
                        connectionstyle=f"arc3,rad={curvature}",
                        color=cmap(norm(value)),
                        linestyle=self._line_style(label),
                        linewidth=style.coupling_linewidth,
                        zorder=self._Z["edge"],
                    )
                )
                if style.show_coupling_labels:
                    label_x, label_y = self._label_position(start, end, curvature)
                    self.ax.text(
                        label_x,
                        label_y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=style.coupling_label_fontsize,
                        color=style.theme.on_background,
                        fontproperties=style.font,
                        bbox={"boxstyle": "round,pad=0.15", "fc": style.theme.background, "ec": "none", "alpha": 0.75},
                        zorder=self._Z["label"],
                    )

    def _draw_multi_body(
        self,
        positions: dict[int, tuple[float, float]],
        radius: float,
        cmap: Colormap,
        norm: Normalize,
    ) -> None:
        """Draw terms acting on three or more qubits as a star joined at the term centroid.

        Args:
            positions (dict[int, tuple[float, float]]): Normalized qubit positions.
            radius (float): Node radius in normalized layout units.
            cmap (Colormap): Colormap for the coefficient strengths.
            norm (Normalize): Normalization applied to the coupling coefficients.
        """
        style = self.style
        if not style.show_multi_body:
            return
        for qubits, label, value in self._multi_body:
            centroid = (
                sum(positions[q][0] for q in qubits) / len(qubits),
                sum(positions[q][1] for q in qubits) / len(qubits),
            )
            color = cmap(norm(value))
            for qubit in qubits:
                start, _ = self._trim(positions[qubit], centroid, radius)
                self.ax.add_patch(
                    FancyArrowPatch(
                        start,
                        centroid,
                        arrowstyle="-",
                        connectionstyle="arc3,rad=0",
                        color=color,
                        linestyle=self._line_style(label),
                        linewidth=style.coupling_linewidth,
                        zorder=self._Z["hyperedge"],
                    )
                )
            self.ax.plot(
                *centroid,
                marker="o",
                markersize=style.coupling_linewidth * 2,
                color=color,
                markeredgecolor=style.theme.border,
                zorder=self._Z["hyperedge"],
            )
            if style.show_coupling_labels:
                self.ax.text(
                    centroid[0],
                    centroid[1],
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=style.coupling_label_fontsize,
                    color=style.theme.on_background,
                    fontproperties=style.font,
                    bbox={"boxstyle": "round,pad=0.15", "fc": style.theme.background, "ec": "none", "alpha": 0.75},
                    zorder=self._Z["label"],
                )

    def _build_line_styles(self) -> None:
        """Assign a line style to every coupling type present in the Hamiltonian.

        Types listed in :attr:`HamiltonianStyle.coupling_line_styles` keep their configured style; the
        remaining ones cycle through :attr:`HamiltonianStyle.default_coupling_line_styles`.
        """
        style = self.style
        types = {label for types in self._couplings.values() for label in types}
        types |= {label for _, label, _ in self._multi_body}
        self._line_styles = {
            label: style.coupling_line_styles[label] for label in types if label in style.coupling_line_styles
        }
        # Keep the cycled styles distinct from the ones already claimed by a configured type.
        taken = set(self._line_styles.values())
        fallbacks = [s for s in style.default_coupling_line_styles if s not in taken] or list(
            style.default_coupling_line_styles
        )
        for index, label in enumerate(sorted(types - set(self._line_styles))):
            self._line_styles[label] = fallbacks[index % len(fallbacks)]

    def _line_style(self, coupling_type: str) -> Any:  # noqa: ANN401
        """Resolve the line style used for a coupling type.

        Args:
            coupling_type (str): Pauli string of the coupling, e.g. ``'ZZ'``.

        Returns:
            Any: A matplotlib line style specification.
        """
        return self._line_styles.get(coupling_type, self.style.default_coupling_line_styles[0])

    @staticmethod
    def _trim(
        start: tuple[float, float], end: tuple[float, float], radius: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Pull both endpoints of a segment back to the boundary of the node discs.

        Args:
            start (tuple[float, float]): Centre of the first node.
            end (tuple[float, float]): Centre of the second node.
            radius (float): Node radius in normalized layout units.

        Returns:
            tuple[tuple[float, float], tuple[float, float]]: The trimmed endpoints.
        """
        length = math.dist(start, end)
        if length <= _FLAT_SCALE_TOL:
            return start, end
        step = min(radius, length / 2) / length
        dx, dy = end[0] - start[0], end[1] - start[1]
        return (start[0] + dx * step, start[1] + dy * step), (end[0] - dx * step, end[1] - dy * step)

    @staticmethod
    def _label_position(start: tuple[float, float], end: tuple[float, float], curvature: float) -> tuple[float, float]:
        """Place the label of a matplotlib ``arc3`` connection.

        A straight connection is labelled at its midpoint. A curved one is labelled just outside
        the arc (twice the arc's own bulge), which keeps the labels of parallel couplings apart.

        Args:
            start (tuple[float, float]): Start of the arc.
            end (tuple[float, float]): End of the arc.
            curvature (float): The ``rad`` parameter of the ``arc3`` connection style.

        Returns:
            tuple[float, float]: Point at which to draw the label.
        """
        dx, dy = end[0] - start[0], end[1] - start[1]
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        return mid_x + curvature * dy, mid_y - curvature * dx

    # ------------------------------------------------------------------
    # Figure furniture
    # ------------------------------------------------------------------
    def _setup_axes(self, positions: dict[int, tuple[float, float]], radius: float) -> None:
        """Apply the theme, frame the graph and add the title and identity offset annotation.

        Args:
            positions (dict[int, tuple[float, float]]): Normalized qubit positions.
            radius (float): Node radius in normalized layout units.
        """
        style = self.style
        theme = style.theme
        self.ax.set_facecolor(theme.background)
        if isinstance(self.ax.figure, Figure):
            self.ax.figure.set_facecolor(theme.background)

        margin = radius + 0.12
        xs = [x for x, _ in positions.values()]
        ys = [y for _, y in positions.values()]
        self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax.set_ylim(min(ys) - margin, max(ys) + margin)
        self.ax.set_aspect("equal")
        self.ax.set_axis_off()
        self.ax.set_title(
            style.title or "Hamiltonian",
            fontsize=style.title_fontsize,
            color=theme.on_background,
            fontproperties=style.font,
        )

        if style.show_identity_offset and abs(self._offset) > _FLAT_SCALE_TOL:
            self.ax.text(
                0.0,
                -0.02,
                f"identity offset: {self._offset:+.3g}",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=style.coupling_label_fontsize,
                color=theme.on_background,
                fontproperties=style.font,
            )

        if style.tight_layout and isinstance(self.ax.figure, Figure):
            self.ax.figure.tight_layout()

    def _draw_colorbars(self, cmap: Colormap, field_norm: Normalize, coupling_norm: Normalize) -> None:
        """Add the colour bar(s) describing the coefficient strengths.

        Args:
            cmap (Colormap): Colormap for the coefficient strengths.
            field_norm (Normalize): Normalization used for the local fields.
            coupling_norm (Normalize): Normalization used for the couplings.
        """
        style = self.style
        figure = self.ax.figure
        if not style.show_colorbar or not isinstance(figure, Figure):
            return

        base_label = style.colorbar_label or ("|coefficient|" if self._use_magnitude else "coefficient")
        pairs = (
            [(field_norm, f"local field {base_label}"), (coupling_norm, f"coupling {base_label}")]
            if style.separate_color_scales
            else [(field_norm, base_label)]
        )
        # Each colour bar is inserted between the axes and the previously created one, so build them
        # back to front to keep the first pair closest to the plot.
        for index, (norm, label) in enumerate(reversed(pairs)):
            bar = figure.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=self.ax,
                fraction=0.046,
                # Inner bars need extra padding to fit the label moved to their left below.
                pad=0.04 if index == 0 else 0.12,
                shrink=style.colorbar_shrink,
            )
            if index > 0:
                # The right of an inner bar is taken by its own ticks and the bar next to them, so
                # its label goes on the free side instead of on top of the neighbouring bar.
                bar.ax.yaxis.set_label_position("left")
            bar.set_label(label, color=style.theme.on_background, fontproperties=style.font)
            bar.ax.tick_params(colors=style.theme.on_background, labelsize=style.coupling_label_fontsize)
            for spine in bar.ax.spines.values():
                spine.set_edgecolor(style.theme.border)

    def _draw_legend(self) -> None:
        """Add a legend mapping each coupling type to its line style."""
        style = self.style
        if not style.show_legend:
            return
        labels = sorted({label for types in self._couplings.values() for label in types})
        if style.show_multi_body:
            labels += sorted({label for _, label, _ in self._multi_body} - set(labels))
        if not labels:
            return
        handles = [
            Line2D(
                [],
                [],
                color=style.theme.on_background,
                linestyle=self._line_style(label),
                linewidth=style.coupling_linewidth,
                label=label,
            )
            for label in labels
        ]
        legend = self.ax.legend(
            handles=handles,
            loc=cast("Any", style.legend_loc),
            fontsize=style.legend_fontsize,
            frameon=style.legend_frame,
            facecolor=style.theme.background,
            edgecolor=style.theme.border,
        )
        for text in legend.get_texts():
            text.set_color(style.theme.on_background)

    @staticmethod
    def _make_axes(style: HamiltonianStyle) -> Axes:
        """Create a new figure and axes honouring the style.

        Args:
            style (HamiltonianStyle): Style providing the figure size, DPI and theme.

        Returns:
            Axes: A newly created Matplotlib Axes.
        """
        _, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi, facecolor=style.theme.background)
        return ax
