"""
Module for rendering a quantum circuit using matplotlib library.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import (
    Arc,
    Circle,
    FancyArrow,
    FancyBboxPatch,
)
from pydantic import BaseModel, Field, field_validator

from qilisdk.digital import Circuit
from qilisdk.digital.gates import BasicGate, Controlled, Gate, M, X


class Theme(BaseModel):
    """
    Pydantic V2 model for circuit color themes, containing only common attributes.
    """
    bgcolor: str = Field(..., description="Background color of the circuit.")
    color: str = Field(..., description="Accent color for elements and gate labels.")
    wire_color: str = Field(..., description="Color of the wires.")
    gate_color: str = Field(..., description="Color for gates.")
    plus_color: str = Field(..., description="Color for plus sign.")


light = Theme(
    bgcolor="#FFFFFF",
    color="#000000",
    wire_color="#F0F0F0",
    gate_color="#AC115F",
    plus_color="#5E56A1"
)
dark = Theme(
    bgcolor="#000000",
    color="#FFFFFF",
    wire_color="#FFFFFF",
    gate_color="#FFFFFF",
    plus_color="#5E56A1"
)


class StyleConfig(BaseModel):
    """
    Pydantic V2 model for circuit style configuration.
    """
    dpi: int = Field(150, description="DPI of the figure.")
    fontsize: int = Field(10, description="Fontsize at circuit level.")
    end_wire_ext: int = Field(2, description="Extension of the wire at the end.")
    padding: float = Field(0.3, description="Padding between circuit and figure border.")
    gate_margin: float = Field(0.15, description="Margin space on each side of a gate.")
    wire_sep: float = Field(0.5, description="Separation between wires.")
    layer_sep: float = Field(0.5, description="Separation between layers.")
    gate_pad: float = Field(0.05, description="Padding between gate and gate label.")
    label_pad: float = Field(0.1, description="Padding between wire label and wire.")
    bulge: str = Field("round", description="Bulge style of the gate.")
    align_layer: bool = Field(True, description="Align layers of gates across wires.")
    theme: Theme = Field(light, description="Color theme of the circuit.")
    title: Optional[str] = Field(None, description="Title of the circuit.")
    wire_label: Optional[List[Any]] = Field(None, description="Labels for the wires.")

    @property
    def measure_color(self) -> str:
        # White for dark theme, black otherwise
        return "#FFFFFF" if self.theme is dark else "#000000"

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


class BaseRenderer:
    """
    Base class for rendering quantum circuits with MatRender and TextRender.
    """

    def __init__(self, style: StyleConfig):
        """
        Initialize the base renderer with default values.
        Both Renderers should override these attributes as needed.
        """

        self._qwires = 0
        self._layer_list = []
        self.style = style

    def _get_xskip(self, wire_list: List[int], layer: int) -> float:
        """
        Get the xskip (horizontal value for getting to requested layer) for the gate to be plotted.

        Parameters
        ----------
        wire_list : List[int]
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.

        Returns
        -------
        float
            The maximum xskip value needed to reach the specified layer.
        """

        if self.style.align_layer:
            wire_list = list(range(self._qwires))
        xskip_vals = [sum(self._layer_list.get(w, [])[:layer]) for w in wire_list]
        return max(xskip_vals or [0])

    def _manage_layers(
        self,
        gate_width: float,
        wire_list: List[int],
        layer: int,
        xskip: float = 0,
    ) -> None:
        """
        Manages and updates the layer widths according to the gate's width just plotted.

        Parameters
        ----------
        gate_width : float
            The width of the gate to be plotted.

        wire_list : list
            The list of wires the gate is acting on (control and target).

        layer : int
            The layer the gate is acting on.

        xskip : float, optional
            The horizontal value for getting to requested layer. The default is 0.
        """

        for w in wire_list:
            current = self._layer_list.setdefault(w, [])
            required = gate_width + self.style.gate_margin * 2
            if len(current) > layer:
                current[layer] = max(current[layer], required)
            else:
                gap = xskip - sum(current) if xskip else 0
                current.append(gap + required)


class MatRenderer(BaseRenderer):
    """
    Class to render a quantum circuit using matplotlib.

    Parameters
    ----------
    qc : QuantumCircuit Object
        The quantum circuit to be rendered.

    ax : Axes Object, optional
        The axes object to plot the circuit. The default is None.

    **style
        Additional style arguments to be passed to the `StyleConfig` dataclass.
    """

    def __init__(
        self,
        qc: Circuit,
        ax: Axes = None,
        style: StyleConfig | None = None,
    ) -> None:

        # user defined style
        self.style = style or StyleConfig()

        super().__init__(self.style)
        self._qc = qc
        self._ax = ax
        self._qwires = qc.nqubits

        # default values
        self._min_gate_height = 0.2
        self._min_gate_width = 0.2
        self._default_layers = 2
        self._arrow_lenght = 0.06
        self._connector_r = 0.01
        self._target_node_r = 0.12
        self._control_node_r = 0.05
        self._display_layer_len = 0
        self._start_pad = 0.1
        self._layer_list = {i: [self._start_pad] for i in range(self._qwires)}

        # fig config
        self._zorder = {
            "wire": 1,
            "wire_label": 1,
            "gate": 2,
            "node": 2,
            "bridge": 2,
            "connector": 3,
            "gate_label": 3,
            "node_label": 3,
        }
        if self._ax is None:
            self.fig = plt.figure()
            self._ax = self.fig.add_subplot(111)
            self.fig.set_dpi(self.style.dpi)
        else:
            self.fig = self._ax.get_figure()

    def _ypos(self, index: int) -> float:
        return (self._qwires - 1 - index) * self.style.wire_sep

    def _get_text_width(
        self,
        text: str,
        fontsize: float,
        fontweight: Union[float, str],
        fontfamily: str,
        fontstyle: str,
    ) -> float:
        """
        Get the width of the text to be plotted.

        Parameters
        ----------
        text : str
            The text to be plotted.

        fontsize : float
            The fontsize of the text.

        fontweight : str or float
            The fontweight of the text.

        fontfamily : str
            The fontfamily of the text.

        fontstyle : str
            The fontstyle of the text.

        Returns
        -------
        float
            The width of the text inches.
        """

        text_obj = plt.Text(
            0,
            0,
            text,
            fontsize=fontsize,
            fontweight=fontweight,
            fontfamily=fontfamily,
            fontstyle=fontstyle,
        )
        self._ax.add_artist(text_obj)

        bbox = text_obj.get_window_extent(
            renderer=self._ax.figure.canvas.get_renderer()
        )
        text_obj.remove()

        return bbox.width / self.style.dpi

    def _add_wire(self) -> None:
        """
        Adds the wires to the circuit.
        """
        max_len = max(sum(self._layer_list[i]) for i in range(self._qwires)) + self.style.end_wire_ext * self.style.layer_sep
        for i in range(self._qwires):
            y = self._ypos(i)
            wire = plt.Line2D([0, max_len], [y, y], lw=1, color=self.style.wire_color, zorder=self._zorder["wire"])
            self._ax.add_line(wire)

    def _add_wire_labels(self) -> None:
        """
        Adds the wire labels to the circuit.
        """

        if self.style.wire_label is None:
            self.style.wire_label = [f"$q_{{{i}}}$" for i in range(self._qwires)]
        self.max_label_width = max(
            self._get_text_width(label, self.style.fontsize, "normal", "monospace", "normal")
            for label in self.style.wire_label
        )
        for i, label in enumerate(self.style.wire_label):
            y = self._ypos(i)
            wire_label = plt.Text(-self.style.label_pad, y, label,
                                   fontsize=self.style.fontsize,
                                   fontweight="normal",
                                   fontfamily="monospace",
                                   fontstyle="normal",
                                   verticalalignment="center",
                                   horizontalalignment="right",
                                   zorder=self._zorder["wire_label"],
                                   color=self.style.color)
            self._ax.add_artist(wire_label)

    def _draw_control_node(self, pos: int, xskip: float, color: str) -> None:
        """
        Draw the control node for the multi-qubit gate.

        Parameters
        ----------
        pos : int
            The position of the control node, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node. HEX code or color name supported by matplotlib are valid.
        """
        y = self._ypos(pos)
        control_node = Circle((xskip + self.style.gate_margin + self.style.gate_pad, y),
                               self._control_node_r, color=self.style.theme.plus_color, zorder=self._zorder["node"])
        self._ax.add_artist(control_node)

    def _draw_target_node(self, pos: int, xskip: float, color: str) -> None:
        """
        Draw the target node for the multi-qubit gate.

        Parameters
        ----------
        pos : int
            The position of the target node, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node. HEX code or color name supported by matplotlib are valid.
        """
        y = self._ypos(pos)
        target_node = Circle((xskip + self.style.gate_margin + self.style.gate_pad, y),
                              self._target_node_r, color=self.style.theme.plus_color, zorder=self._zorder["node"])
        vert = plt.Line2D((xskip + self.style.gate_margin + self.style.gate_pad,) * 2,
                          (y - self._target_node_r / 2, y + self._target_node_r / 2), lw=1.5,
                          color=self.style.bgcolor, zorder=self._zorder["node_label"])
        horiz = plt.Line2D((xskip + self.style.gate_margin + self.style.gate_pad - self._target_node_r/2,
                            xskip + self.style.gate_margin + self.style.gate_pad + self._target_node_r/2),
                           (y, y), lw=1.5, color=self.style.bgcolor, zorder=self._zorder["node_label"])
        self._ax.add_artist(target_node)
        self._ax.add_line(vert)
        self._ax.add_line(horiz)

    def _draw_qbridge(
        self, pos1: int, pos2: int, xskip: float, color: str
    ) -> None:
        """
        Draw the bridge between the control and target nodes for the multi-qubit gate.

        Parameters
        ----------
        pos1 : int
            The position of the first node for the bridge, in terms of the wire number.

        pos2 : int
            The position of the second node for the bridge, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the control node. HEX code or color name supported by matplotlib are valid.
        """
        y1 = self._ypos(pos1)
        y2 = self._ypos(pos2)
        bridge = plt.Line2D([xskip + self.style.gate_margin + self.style.gate_pad]*2,
                             [y1, y2], color=self.style.theme.plus_color,
                             zorder=self._zorder["bridge"])
        self._ax.add_line(bridge)

    def _draw_swap_mark(self, pos: int, xskip: int, color: str) -> None:
        """
        Draw the swap mark for the SWAP gate.

        Parameters
        ----------
        pos : int
            The position of the swap mark, in terms of the wire number.

        xskip : float
            The horizontal value for getting to requested layer.

        color : str
            The color of the swap mark.
        """
        y = self._ypos(pos)
        offset = self._min_gate_width / 3
        lines = [
            ([xskip + self.style.gate_margin + self.style.gate_pad + offset,
              xskip + self.style.gate_margin + self.style.gate_pad - offset],
             [y + self._min_gate_height/2, y - self._min_gate_height/2]),
            ([xskip + self.style.gate_margin + self.style.gate_pad - offset,
              xskip + self.style.gate_margin + self.style.gate_pad + offset],
             [y + self._min_gate_height/2, y - self._min_gate_height/2])
        ]
        for xs, ys in lines:
            self._ax.add_line(plt.Line2D(xs, ys, color=color, linewidth=2, zorder=self._zorder["gate"]))

    def to_pi_fraction(self, value: float, tolerance: float = 0.01) -> str:
        """
        Convert a value to a string fraction of pi.

        Parameters
        ----------
        value : float
            The value to be converted.

        tolerance : float, optional
            The tolerance for the fraction. The default is 0.01.

        Returns
        -------
        str
            The value in terms of pi.
        """

        pi_value = value / np.pi
        if abs(pi_value - round(pi_value)) < tolerance:
            num = round(pi_value)
            return f"[{num}\\pi]" if num != 1 else "[\\pi]"

        for denom in [2, 3, 4, 6, 8, 12]:
            fraction_value = pi_value * denom
            if abs(fraction_value - round(fraction_value)) < tolerance:
                num = round(fraction_value)
                return (
                    f"[{num}\\pi/{denom}]" if num != 1 else f"[\\pi/{denom}]"
                )

        return f"[{round(value, 2)}]"

    def _draw_singleq_gate(self, gate: Gate, layer: int) -> None:
        """
        Draw the single qubit gate.

        Parameters
        ----------
        gate : Gate Object
            The gate to be plotted.

        layer : int
            The layer the gate is acting on.
        """

        gate_wire = gate.target_qubits[0]
        if gate.is_parameterized and self.showarg is True:
            pi_frac = self.to_pi_fraction(gate.parameter_values[0])
            text = f"${{{self.text}}}_{{{pi_frac}}}$"
        else:
            text = self.text

        text_width = self._get_text_width(
            text,
            self.fontsize,
            self.fontweight,
            self.fontfamily,
            self.fontstyle,
        )
        gate_width = max(
            text_width + self.style.gate_pad * 2, self._min_gate_width
        )

        gate_text = plt.Text(
            self._get_xskip([gate_wire], layer)
            + self.style.gate_margin
            + gate_width / 2,
            gate_wire * self.style.wire_sep,
            text,
            color=self.fontcolor,
            fontsize=self.fontsize,
            fontweight=self.fontweight,
            fontfamily=self.fontfamily,
            fontstyle=self.fontstyle,
            verticalalignment="center",
            horizontalalignment="center",
            zorder=self._zorder["gate_label"],
        )
        gate_patch = FancyBboxPatch(
            (
                self._get_xskip([gate_wire], layer) + self.style.gate_margin,
                gate_wire * self.style.wire_sep
                - self._min_gate_height / 2,
            ),
            gate_width,
            self._min_gate_height,
            boxstyle=self.style.bulge,
            mutation_scale=0.3,
            facecolor=self.color,
            edgecolor=self.color,
            zorder=self._zorder["gate"],
        )

        self._ax.add_artist(gate_text)
        self._ax.add_patch(gate_patch)
        self._manage_layers(gate_width, [gate_wire], layer)

    def _draw_multiq_gate(self, gate: Gate, layer: int) -> None:
        """
        Draw the multi-qubit gate.

        Parameters
        ----------
        gate : Gate Object
            The gate to be plotted.

        layer : int
            The layer the gate is acting on.
        """

        wire_list = list(
            range(self.merged_wires[0], self.merged_wires[-1] + 1)
        )
        com_xskip = self._get_xskip(wire_list, layer)

        if isinstance(gate, Controlled) and gate.is_modified_from(X):
            self._draw_control_node(gate.control_qubits[0], com_xskip, self.color)
            self._draw_target_node(gate.target_qubits[0], com_xskip, self.color)
            self._draw_qbridge(
                gate.target_qubits[0], gate.control_qubits[0], com_xskip, self.color
            )
            self._manage_layers(
                2 * self.style.gate_pad + self._target_node_r / 3,
                wire_list,
                layer,
                com_xskip,
            )

        elif gate.name == "SWAP":
            self._draw_swap_mark(gate.target_qubits[0], com_xskip, self.color)
            self._draw_swap_mark(gate.target_qubits[1], com_xskip, self.color)
            self._draw_qbridge(
                gate.target_qubits[0], gate.target_qubits[1], com_xskip, self.color
            )
            self._manage_layers(
                2 * (self.style.gate_pad + self._min_gate_width / 3),
                wire_list,
                layer,
                com_xskip,
            )

        elif gate.name == "TOFFOLI":
            self._draw_control_node(gate.control_qubits[0], com_xskip, self.color)
            self._draw_control_node(gate.control_qubits[1], com_xskip, self.color)
            self._draw_target_node(gate.target_qubits[0], com_xskip, self.color)
            self._draw_qbridge(
                gate.target_qubits[0], gate.control_qubits[0], com_xskip, self.color
            )
            self._draw_qbridge(
                gate.target_qubits[0], gate.control_qubits[1], com_xskip, self.color
            )
            self._manage_layers(
                2 * self.style.gate_pad + self._target_node_r / 3,
                wire_list,
                layer,
                com_xskip,
            )

        else:

            adj_targets = sorted(
                    gate.target_qubits
                    if gate.target_qubits is not None
                    else list(
                        range(self._qwires)
                    )  # adaptation for globalphase
                )
            text_width = self._get_text_width(
                gate.basic_gate.name,
                self.fontsize,
                self.fontweight,
                self.fontfamily,
                self.fontstyle,
            )
            gate_width = max(
                text_width + self.style.gate_pad * 2, self._min_gate_width
            )
            xskip = self._get_xskip(wire_list, layer)

            gate_text = plt.Text(
                xskip + self.style.gate_margin + gate_width / 2,
                (adj_targets[0] + adj_targets[-1]) / 2 * self.style.wire_sep,
                gate.basic_gate.name,
                color=self.fontcolor,
                fontsize=self.fontsize,
                fontweight=self.fontweight,
                fontfamily=self.fontfamily,
                fontstyle=self.fontstyle,
                verticalalignment="center",
                horizontalalignment="center",
                zorder=self._zorder["gate_label"],
            )

            gate_patch = FancyBboxPatch(
                (
                    xskip + self.style.gate_margin,
                    adj_targets[0] * self.style.wire_sep
                    - self._min_gate_height / 2,
                ),
                gate_width,
                self._min_gate_height
                + self.style.wire_sep * (adj_targets[-1] - adj_targets[0]),
                boxstyle=self.style.bulge,
                mutation_scale=0.3,
                facecolor=self.color,
                edgecolor=self.color,
                zorder=self._zorder["gate"],
            )

            if gate.target_qubits is not None and len(gate.target_qubits) > 1:
                for i in range(len(gate.target_qubits)):
                    connector_l = Circle(
                        (
                            xskip + self.style.gate_margin + self._connector_r,
                            (adj_targets[i]) * self.style.wire_sep,
                        ),
                        self._connector_r,
                        color=self.fontcolor,
                        zorder=self._zorder["connector"],
                    )
                    connector_r = Circle(
                        (
                            xskip
                            + self.style.gate_margin
                            + gate_width
                            - self._connector_r,
                            (adj_targets[i]) * self.style.wire_sep,
                        ),
                        self._connector_r,
                        color=self.fontcolor,
                        zorder=self._zorder["connector"],
                    )
                    self._ax.add_artist(connector_l)
                    self._ax.add_artist(connector_r)

            # add cbridge if control qubits are present
            if gate.control_qubits:
                for control in gate.control_qubits:
                    self._draw_control_node(
                        control, xskip + text_width / 2, self.color
                    )
                    self._draw_qbridge(
                        control,
                        gate.target_qubits[0],
                        xskip + text_width / 2,
                        self.color,
                    )

            self._ax.add_artist(gate_text)
            self._ax.add_patch(gate_patch)
            self._manage_layers(gate_width, wire_list, layer, xskip)

    def _draw_measure(self, q_pos: int, layer: int) -> None:
        """
        Draw the measurement gate.

        Parameters
        ----------
        c_pos : int
            The position of the classical wire.

        q_pos : int
            The position of the quantum wire.

        layer : int
            The layer the gate is acting on.
        """

        xskip = self._get_xskip(
            list(range(self.merged_wires[-1] + 1)), layer
        )
        measure_box = FancyBboxPatch(
            (
                xskip + self.style.gate_margin,
                q_pos * self.style.wire_sep
                - self._min_gate_height / 2,
            ),
            self._min_gate_width,
            self._min_gate_height,
            boxstyle=self.style.bulge,
            mutation_scale=0.3,
            facecolor=self.style.bgcolor,
            edgecolor=self.style.measure_color,
            linewidth=1.25,
            zorder=self._zorder["gate"],
        )
        arc = Arc(
            (
                xskip + self.style.gate_margin + self._min_gate_width / 2,
                q_pos * self.style.wire_sep
                - self._min_gate_height / 2,
            ),
            self._min_gate_width * 1.5,
            self._min_gate_height * 1,
            angle=0,
            theta1=0,
            theta2=180,
            color=self.style.measure_color,
            linewidth=1.25,
            zorder=self._zorder["gate_label"],
        )
        arrow = FancyArrow(
            xskip + self.style.gate_margin + self._min_gate_width / 2,
            q_pos * self.style.wire_sep
            - self._min_gate_height / 2,
            self._min_gate_width * 0.7,
            self._min_gate_height * 0.7,
            length_includes_head=True,
            head_width=0,
            linewidth=1.25,
            color=self.style.measure_color,
            zorder=self._zorder["gate_label"],
        )

        self._manage_layers(
            self._min_gate_width,
            list(range(0, self.merged_wires[-1] + 1)),
            layer,
            xskip,
        )
        self._ax.add_patch(measure_box)
        self._ax.add_artist(arc)
        self._ax.add_artist(arrow)

    def canvas_plot(self) -> None:
        """
        Plot the quantum circuit.
        """

        self._add_wire_labels()

        for gate in self._qc.gates:

            if isinstance(gate, M):
                self.merged_wires = list(gate.target_qubits)
                self.merged_wires.sort()

                self._draw_measure(
                    gate.target_qubits[0],
                    max(
                        len(self._layer_list[i])
                        for i in range(self.merged_wires[-1] + 1)
                    ),
                )

            self.text = gate.name
            self.color = self.style.default_gate_color
            self.fontsize = self.style.fontsize
            self.fontcolor = self.style.bgcolor
            self.fontweight = "normal"
            self.fontstyle = "normal"
            self.fontfamily = "monospace"
            self.showarg = True

            # multi-qubit gate
            if isinstance(gate, Controlled):
                # If targets=None, it implies globalphase. Adaptation for the renderer: targets=all-qubits.
                self.merged_wires = (
                    list(gate.target_qubits) or list(range(self._qwires))
                )
                if gate.control_qubits:
                    self.merged_wires += list(gate.control_qubits)
                self.merged_wires.sort()

                find_layer = [
                    len(self._layer_list[i])
                    for i in range(
                        self.merged_wires[0], self.merged_wires[-1] + 1
                    )
                ]
                self._draw_multiq_gate(gate, max(find_layer))

            else:
                self._draw_singleq_gate(
                    gate, len(self._layer_list[gate.target_qubits[0]])
                )

        self._add_wire()
        self._fig_config()
        plt.tight_layout()
        plt.show()

    def _fig_config(self) -> None:
        """
        Configure the figure settings.
        """
        self.fig.set_facecolor(self.style.bgcolor)
        xlim = (
            -self.style.padding - self.max_label_width - self.style.label_pad,
            self.style.padding
            + self.style.end_wire_ext * self.style.layer_sep
            + max(sum(self._layer_list[i]) for i in range(self._qwires)),
        )
        ylim = (
            -self.style.padding,
            self.style.padding
            + (self._qwires - 1) * self.style.wire_sep,
        )
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

        if self.style.title is not None:
            self._ax.set_title(
                self.style.title,
                pad=10,
                color=self.style.wire_color,
                fontdict={"fontsize": self.style.fontsize},
            )

        # Adjusting to square dimensions in jupyter to prevent small fig size with equal-aspect cmd
        try:
            get_ipython()
            max_dim = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
            self.fig.set_size_inches(max_dim, max_dim, forward=True)
        except NameError:
            self.fig.set_size_inches(
                xlim[1] - xlim[0], ylim[1] - ylim[0], forward=True
            )
        self._ax.set_aspect("equal", adjustable="box")
        self._ax.axis("off")

    def save(self, filename: str, **kwargs) -> None:
        """
        Save the circuit to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the circuit to.

        **kwargs
            Additional arguments to be passed to `plt.savefig`.
        """
        self.fig.savefig(filename, bbox_inches="tight", **kwargs)
