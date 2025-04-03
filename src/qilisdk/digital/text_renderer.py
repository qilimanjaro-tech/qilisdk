"""
Module for rendering a quantum circuit in text format.

This renderer is adapted from the qutip-qip TextRenderer to work with
a Circuit class (with attribute nqubits and a list of gates) and the following gate types:
- M (measurement) gate,
- Controlled gate (with attributes control_qubits and target_qubits),
- Exponential gate,
- Adjoint gate,
- and default (single- or multi-qubit) gates.
"""

from dataclasses import dataclass
from math import ceil
from typing import List

# from .base_renderer import BaseRenderer, StyleConfig
from .circuit import Circuit
from .gates import Controlled, M

__all__ = [
    "TextRenderer",
]

qutip = {
    "bgcolor": "#FFFFFF",  # White
    "color": "#FFFFFF",  # White
    "wire_color": "#000000",  # Black
    "default_gate": "#000000",  # Black
    "H": "#6270CE",  # Medium Slate Blue
    "SNOT": "#6270CE",  # Medium Slate Blue
    "X": "#CB4BF9",  # Medium Orchid
    "Y": "#CB4BF9",  # Medium Orchid
    "Z": "#CB4BF9",  # Medium Orchid
    "S": "#254065",  # Dark Slate Blue
    "T": "#254065",  # Dark Slate Blue
    "RX": "#5EBDF8",  # Light Sky Blue
    "RY": "#5EBDF8",  # Light Sky Blue
    "RZ": "#5EBDF8",  # Light Sky Blue
    "CNOT": "#3B3470",  # Indigo
    "CPHASE": "#456DB2",  # Steel Blue
    "TOFFOLI": "#3B3470",  # Indigo
    "SWAP": "#3B3470",  # Indigo
    "CX": "#9598F5",  # Light Slate Blue
    "CY": "#9598F5",  # Light Slate Blue
    "CZ": "#9598F5",  # Light Slate Blue
    "CS": "#9598F5",  # Light Slate Blue
    "CT": "#9598F5",  # Light Slate Blue
    "CRX": "#A66DDF",  # Medium Purple
    "CRY": "#A66DDF",  # Medium Purple
    "CRZ": "#A66DDF",  # Medium Purple
    "BERKELEY": "#7648CB",  # Dark Orchid
    "FREDKIN": "#7648CB",  # Dark Orchid
}

light = {
    "bgcolor": "#EEEEEE",  # Light Gray
    "color": "#000000",  # Black
    "wire_color": "#000000",  # Black
    "default_gate": "#D8CDAF",  # Bit Dark Beige
    "H": "#A3C1DA",  # Light Blue
    "SNOT": "#A3C1DA",  # Light Blue
    "X": "#F4A7B9",  # Light Pink
    "Y": "#F4A7B9",  # Light Pink
    "Z": "#F4A7B9",  # Light Pink
    "S": "#D3E2EE",  # Very Light Blue
    "T": "#D3E2EE",  # Very Light Blue
    "RX": "#B3E6E4",  # Light Teal
    "RY": "#B3E6E4",  # Light Teal
    "RZ": "#B3E6E4",  # Light Teal
    "CNOT": "#B7C9F2",  # Light Indigo
    "CPHASE": "#D5E0F2",  # Light Slate Blue
    "TOFFOLI": "#E6CCE6",  # Soft Lavender
    "SWAP": "#FFB6B6",  # Lighter Coral Pink
    "CX": "#E0E2F7",  # Very Light Indigo
    "CY": "#E0E2F7",  # Very Light Indigo
    "CZ": "#E0E2F7",  # Very Light Indigo
    "CS": "#E0E2F7",  # Very Light Indigo
    "CT": "#E0E2F7",  # Very Light Indigo
    "CRX": "#D6C9E8",  # Light Muted Purple
    "CRY": "#D6C9E8",  # Light Muted Purple
    "CRZ": "#D6C9E8",  # Light Muted Purple
    "BERKELEY": "#CDC1E8",  # Light Purple
    "FREDKIN": "#CDC1E8",  # Light Purple
}

dark = {
    "bgcolor": "#000000",  # Black
    "color": "#000000",  # Black
    "wire_color": "#989898",  # Dark Gray
    "default_gate": "#D8BFD8",  # (Thistle)
    "H": "#AFEEEE",  # Pale Turquoise
    "SNOT": "#AFEEEE",  # Pale Turquoise
    "X": "#9370DB",  # Medium Purple
    "Y": "#9370DB",  # Medium Purple
    "Z": "#9370DB",  # Medium Purple
    "S": "#B0E0E6",  # Powder Blue
    "T": "#B0E0E6",  # Powder Blue
    "RX": "#87CEEB",  # Sky Blue
    "RY": "#87CEEB",  # Sky Blue
    "RZ": "#87CEEB",  # Sky Blue
    "CNOT": "#6495ED",  # Cornflower Blue
    "CPHASE": "#8A2BE2",  # Blue Violet
    "TOFFOLI": "#DA70D6",  # Orchid
    "SWAP": "#BA55D3",  # Medium Orchid
    "CX": "#4682B4",  # Steel Blue
    "CY": "#4682B4",  # Steel Blue
    "CZ": "#4682B4",  # Steel Blue
    "CS": "#4682B4",  # Steel Blue
    "CT": "#4682B4",  # Steel Blue
    "CRX": "#7B68EE",  # Medium Slate Blue
    "CRY": "#7B68EE",  # Medium Slate Blue
    "CRZ": "#7B68EE",  # Medium Slate Blue
    "BERKELEY": "#6A5ACD",  # Slate Blue
    "FREDKIN": "#6A5ACD",  # Slate Blue
}


modern = {
    "bgcolor": "#FFFFFF",  # White
    "color": "#FFFFFF",  # White
    "wire_color": "#000000",  # Black
    "default_gate": "#ED9455",  # Slate Orange
    "H": "#C25454",  # Soft Red
    "SNOT": "#C25454",  # Soft Red
    "X": "#4A5D6D",  # Dark Slate Blue
    "Y": "#4A5D6D",  # Dark Slate Blue
    "Z": "#4A5D6D",  # Dark Slate Blue
    "S": "#2C3E50",  # Very Dark Slate Blue
    "T": "#2C3E50",  # Very Dark Slate Blue
    "RX": "#2F4F4F",  # Dark Slate Teal
    "RY": "#2F4F4F",  # Dark Slate Teal
    "RZ": "#2F4F4F",  # Dark Slate Teal
    "CNOT": "#4A6D7C",  # Dark Slate Blue Gray
    "CPHASE": "#5E7D8B",  # Dark Slate Blue
    "TOFFOLI": "#4A4A4A",  # Dark Gray
    "SWAP": "#6A9ACD",  # Slate Blue
    "CX": "#5D8AA8",  # Medium Slate Blue
    "CY": "#5D8AA8",  # Medium Slate Blue
    "CZ": "#5D8AA8",  # Medium Slate Blue
    "CS": "#5D8AA8",  # Medium Slate Blue
    "CT": "#5D8AA8",  # Medium Slate Blue
    "CRX": "#6C5B7B",  # Dark Lavender
    "CRY": "#6C5B7B",  # Dark Lavender
    "CRZ": "#6C5B7B",  # Dark Lavender
    "BERKELEY": "#4A5D6D",  # Dark Slate Blue
    "FREDKIN": "#4A5D6D",  # Dark Slate Blue
}


@dataclass
class StyleConfig:
    """
    Dataclass to store the style configuration for circuit customization.

    Parameters
    ----------
    dpi : int, optional
        DPI of the figure. The default is 150.

    fontsize : int, optional
        Fontsize control at circuit level, including tile and wire labels. The default is 10.

    end_wire_ext : int, optional
        Extension of the wire at the end of the circuit. The default is 2.
        Available to TextRender and MatRender.

    padding : float, optional
        Padding between the circuit and the figure border. The default is 0.3.

    gate_margin : float, optional
        Margin space left on each side of the gate. The default is 0.15.

    wire_sep : float, optional
        Separation between the wires. The default is 0.5.

    layer_sep : float, optional
        Separation between the layers. The default is 0.5.

    gate_pad : float, optional
        Padding between the gate and the gate label. The default is 0.05.
        Available to TextRender and MatRender.

    label_pad : float, optional
        Padding between the wire label and the wire. The default is 0.1.

    bulge : Union[str, bool], optional
        Bulge style of the gate. Renders non-bulge gates if False. The default is True.

    align_layer : bool, optional
        Align the layers of the gates across different wires. The default is False.
        Available to TextRender and MatRender.

    theme : Optional[Union[str, Dict]], optional
        Color theme of the circuit. The default is "qutip".
        The available themes are 'qutip', 'light', 'dark' and 'modern'.

    title : Optional[str], optional
        Title of the circuit. The default is None.

    bgcolor : Optional[str], optional
        Background color of the circuit. The default is None.

    color : Optional[str], optional
        Controls color of acsent elements (eg. cross sign in the target node)
        and set as deafult color of gate-label. Can be overwritten by gate specific color.
        The default is None.

    wire_label : Optional[List], optional
        Labels of the wires. The default is None.
        Available to TextRender and MatRender.

    wire_color : Optional[str], optional
        Color of the wires. The default is None.
    """

    dpi: int = 150
    fontsize: int = 10
    end_wire_ext: int = 2
    padding: float = 0.3
    gate_margin: float = 0.15
    wire_sep: float = 0.5
    layer_sep: float = 0.5
    gate_pad: float = 0.05
    label_pad: float = 0.1
    bulge: str | bool = True
    align_layer: bool = False
    theme: str | dict | None = "qutip"
    title: str | None = None
    bgcolor: str | None = None
    color: str | None = None
    wire_label: str | None = None
    wire_color: str | None = None

    def __post_init__(self):
        if isinstance(self.bulge, bool):
            self.bulge = "round4" if self.bulge else "square"

        self.measure_color = "#000000"
        if self.theme == "qutip":
            self.theme = qutip
        elif self.theme == "light":
            self.theme = light
        elif self.theme == "dark":
            self.theme = dark
            self.measure_color = "#FFFFFF"
        elif self.theme == "modern":
            self.theme = modern
        else:
            raise ValueError(
                f"""Invalid theme: {self.theme},
                Must be selectec from 'qutip', 'light', 'dark', or 'modern'.
                """
            )

        self.bgcolor = self.bgcolor or self.theme["bgcolor"]
        self.color = self.color or self.theme["color"]
        self.wire_color = self.wire_color or self.theme["wire_color"]


class TextRenderer:
    """
    A class to render a quantum circuit in text format.
    This version integrates layer management functionality originally provided
    by the qutip-qip BaseRenderer.
    """

    def __init__(self, circuit: Circuit, **style):
        # User-defined style. We force gate_margin to 0.
        style = {} if style is None else style
        style["gate_margin"] = 0
        self.style = StyleConfig(**style)

        self._qc = circuit
        self._qwires = circuit.nqubits  # only quantum wires are considered
        # One list per wire to manage the width of each gate (layer) on that wire.
        self._layer_list = [[] for _ in range(self._qwires)]
        # Initialize render strings for each wire.
        self._render_strs = {
            "top_frame": ["  "] * self._qwires,
            "mid_frame": ["──"] * self._qwires,
            "bot_frame": ["  "] * self._qwires,
        }

    # ===== Layer management functions (integrated from BaseRenderer) =====

    def _get_xskip(self, wire_list: List[int], layer: int) -> float:
        """
        Compute the horizontal offset (xskip) needed to reach the requested layer.
        If self.style.align_layer is True, all wires are considered.
        """
        if getattr(self.style, "align_layer", False):
            wire_list = list(range(self._qwires))
        xskip = []
        for wire in wire_list:
            xskip.append(sum(self._layer_list[wire][:layer]))
        return max(xskip) if xskip else 0

    def _manage_layers(self, gate_width: float, wire_list: List[int], layer: int, xskip: float = 0) -> None:
        """
        Update each wire's layer width according to the width of the gate just drawn.
        """
        for wire in wire_list:
            if len(self._layer_list[wire]) > layer:
                self._layer_list[wire][layer] = max(
                    self._layer_list[wire][layer], gate_width + self.style.gate_margin * 2
                )
            else:
                temp = xskip - sum(self._layer_list[wire]) if xskip != 0 else 0
                self._layer_list[wire].append(temp + gate_width + self.style.gate_margin * 2)

    # ===== Functions for adjusting wire render strings =====

    def _adjust_layer_pad(self, wire_list: List[int], xskip: int):
        """
        Pad the render strings on the given wires up to xskip.
        """
        for wire in wire_list:
            self._render_strs["top_frame"][wire] += " " * (xskip - len(self._render_strs["top_frame"][wire]))
            self._render_strs["bot_frame"][wire] += " " * (xskip - len(self._render_strs["bot_frame"][wire]))
            self._render_strs["mid_frame"][wire] += "─" * (xskip - len(self._render_strs["mid_frame"][wire]))

    def _add_wire_labels(self):
        """
        Add labels to each wire. If no wire_label is provided in the style,
        default labels (q0, q1, ...) are used.
        """
        if not getattr(self.style, "wire_label", None):
            default_labels = [f"q{i}" for i in range(self._qwires)]
        else:
            default_labels = self.style.wire_label[: self._qwires]

        max_label_len = max(len(label) for label in default_labels)
        for i, label in enumerate(default_labels):
            self._render_strs["mid_frame"][i] = (
                f" {label} " + " " * (max_label_len - len(label)) + ":" + self._render_strs["mid_frame"][i]
            )
            update_len = len(self._render_strs["mid_frame"][i])
            self._render_strs["top_frame"][i] = " " * update_len
            self._render_strs["bot_frame"][i] = " " * update_len
            self._layer_list[i].append(update_len)

    # ===== Gate drawing functions =====

    def _draw_singleq_gate(self, gate_name: str):
        """
        Draw a single-qubit gate with the given name.
        """
        lid_seg = "─" * (ceil(self.style.gate_pad) * 2 + len(gate_name))
        pad = " " * ceil(self.style.gate_pad)
        top_frame = f" ┌{lid_seg}┐ "
        mid_frame = f"─┤{pad}{gate_name}{pad}├─"
        bot_frame = f" └{lid_seg}┘ "
        assert len(top_frame) == len(mid_frame) == len(bot_frame)
        return (top_frame, mid_frame, bot_frame), len(top_frame)

    def _draw_multiq_gate(self, gate, gate_text: str):
        """
        Draw a multi-qubit gate. For controlled gates, adjust the top or bottom
        frames if needed.
        """
        lid_seg = "─" * (ceil(self.style.gate_pad) * 2 + len(gate_text))
        pad = " " * ceil(self.style.gate_pad)
        top_frame = f" ┌{lid_seg}┐ "
        bot_frame = f" └{lid_seg}┘ "
        mid_frame = f" │{pad}{' ' * len(gate_text)}{pad}│ "
        mid_connect = f"─┤{pad}{' ' * len(gate_text)}{pad}├─"
        mid_connect_label = f"─┤{pad}{gate_text}{pad}├─"
        if isinstance(gate, Controlled) and gate.control_qubits:
            sorted_controls = sorted(gate.control_qubits)
            sorted_targets = sorted(gate.target_qubits)
            mid_index = len(bot_frame) // 2
            if sorted_controls[-1] > sorted_targets[0]:
                top_frame = top_frame[:mid_index] + "┴" + top_frame[mid_index + 1 :]
            if sorted_controls[0] < sorted_targets[-1]:
                bot_frame = bot_frame[:mid_index] + "┬" + bot_frame[mid_index + 1 :]
        assert len(top_frame) == len(mid_frame) == len(bot_frame) == len(mid_connect) == len(mid_connect_label)
        return (top_frame, mid_frame, mid_connect, mid_connect_label, bot_frame), len(top_frame)

    def _draw_measurement_gate(self, gate):
        """
        Draw a measurement (M) gate.
        """
        parts, width = self._draw_singleq_gate("M")
        return parts, width

    # ===== Functions for updating render strings =====

    def _update_singleq(self, wire_list, parts):
        """
        Update render strings for a single-qubit gate.
        """
        top_frame, mid_frame, bot_frame = parts
        for wire in wire_list:
            self._render_strs["top_frame"][wire] += top_frame
            self._render_strs["mid_frame"][wire] += mid_frame
            self._render_strs["bot_frame"][wire] += bot_frame

    def _update_target_multiq(self, gate, wire_list: List[int], parts: List[str]):
        """
        Update render strings for the target wires of a multi-qubit gate.
        """
        top_frame, mid_frame, mid_connect, mid_connect_label, bot_frame = parts
        for i, wire in enumerate(wire_list):
            if len(gate.qubits) == 1:
                self._render_strs["top_frame"][wire] += top_frame
                self._render_strs["mid_frame"][wire] += mid_connect_label
                self._render_strs["bot_frame"][wire] += bot_frame
            elif i == 0 and wire in gate.qubits:
                self._render_strs["top_frame"][wire] += mid_frame
                self._render_strs["mid_frame"][wire] += mid_connect_label
                self._render_strs["bot_frame"][wire] += bot_frame
            elif i == len(wire_list) - 1 and wire in gate.qubits:
                self._render_strs["top_frame"][wire] += top_frame
                self._render_strs["mid_frame"][wire] += mid_connect
                self._render_strs["bot_frame"][wire] += mid_frame
            else:
                self._render_strs["top_frame"][wire] += mid_frame
                self._render_strs["mid_frame"][wire] += mid_frame
                self._render_strs["bot_frame"][wire] += mid_frame

    def _update_qbridge(self, gate, wire_list_control: List[int], width: int, is_top: bool):
        """
        Update render strings for the control wires (q-bridge) of a controlled gate.
        """
        bar_conn = " " * (width // 2) + "│" + " " * (width // 2 - 1)
        mid_bar_conn = "─" * (width // 2) + "│" + "─" * (width // 2 - 1)
        node_conn = "─" * (width // 2) + "█" + "─" * (width // 2 - 1)
        for wire in wire_list_control:
            if wire not in getattr(gate, "target_qubits", []):
                if wire in getattr(gate, "control_qubits", []):
                    if wire == wire_list_control[0] or wire == wire_list_control[-1]:
                        self._render_strs["mid_frame"][wire] += node_conn
                        self._render_strs["top_frame"][wire] += bar_conn if not is_top else " " * len(bar_conn)
                        self._render_strs["bot_frame"][wire] += bar_conn if is_top else " " * len(bar_conn)
                    else:
                        self._render_strs["top_frame"][wire] += bar_conn
                        self._render_strs["mid_frame"][wire] += node_conn
                        self._render_strs["bot_frame"][wire] += bar_conn
                else:
                    self._render_strs["top_frame"][wire] += bar_conn
                    self._render_strs["mid_frame"][wire] += mid_bar_conn
                    self._render_strs["bot_frame"][wire] += bar_conn

    def _update_swap_gate(self, wire_list: List[int]):
        """
        Update render strings for a SWAP gate.
        """
        width = 4 * ceil(self.style.gate_pad) + 1
        cross_conn = "─" * (width // 2) + "╳" + "─" * (width // 2)
        bar_conn = " " * (width // 2) + "│" + " " * (width // 2)
        mid_bar_conn = "─" * (width // 2) + "│" + "─" * (width // 2)
        for wire in wire_list:
            if wire == wire_list[-1]:
                self._render_strs["top_frame"][wire] += " " * len(bar_conn)
                self._render_strs["mid_frame"][wire] += cross_conn
                self._render_strs["bot_frame"][wire] += bar_conn
            elif wire == wire_list[0]:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["mid_frame"][wire] += cross_conn
                self._render_strs["bot_frame"][wire] += " " * len(bar_conn)
            else:
                self._render_strs["top_frame"][wire] += bar_conn
                self._render_strs["mid_frame"][wire] += mid_bar_conn
                self._render_strs["bot_frame"][wire] += bar_conn

    # ===== Main layout and output functions =====

    def layout(self):
        """
        Layout the circuit by processing each gate, updating the layer information,
        and building the render strings.
        """
        self._add_wire_labels()

        for gate in self._qc.gates:
            # Determine gate text: use arg_label if provided, else use the gate name.
            gate_text = getattr(gate, "arg_label", None) or gate.name

            if isinstance(gate, M):
                wire_list = sorted(gate.qubits)
                parts, width = self._draw_measurement_gate(gate)
            elif isinstance(gate, Controlled):
                merged_wire = sorted(gate.control_qubits + gate.target_qubits)
                wire_list = list(range(merged_wire[0], merged_wire[-1] + 1))
                parts, width = self._draw_multiq_gate(gate, gate.basic_gate.name)
            elif len(gate.qubits) == 1:
                wire_list = gate.qubits
                parts, width = self._draw_singleq_gate(gate_text)
            elif gate.name == "SWAP":
                wire_list = sorted(gate.qubits)
                width = 4 * ceil(self.style.gate_pad) + 1
            else:
                wire_list = list(range(min(gate.qubits), max(gate.qubits) + 1))
                parts, width = self._draw_multiq_gate(gate, gate_text)

            layer = max(len(self._layer_list[i]) for i in wire_list)
            xskip = self._get_xskip(wire_list, layer)
            self._adjust_layer_pad(wire_list, xskip)
            self._manage_layers(width, wire_list, layer, xskip)

            if isinstance(gate, M):
                self._update_singleq(gate.qubits, parts)
            elif len(gate.qubits) == 1:
                self._update_singleq(wire_list, parts)
            elif gate.name == "SWAP":
                self._update_swap_gate(wire_list)
            else:
                sorted_targets = sorted(gate.target_qubits)
                self._update_target_multiq(gate, list(range(sorted_targets[0], sorted_targets[-1] + 1)), parts)
                if isinstance(gate, Controlled) and gate.control_qubits:
                    sorted_controls = sorted(gate.control_qubits)
                    is_top = sorted_controls[-1] > sorted_targets[0]
                    is_bot = sorted_controls[0] < sorted_targets[-1]
                    if is_top:
                        self._update_qbridge(
                            gate, list(range(sorted_targets[0], sorted_controls[-1] + 1)), width, is_top
                        )
                    if is_bot:
                        self._update_qbridge(
                            gate, list(range(sorted_controls[0], sorted_targets[-1] + 1)), width, not is_bot
                        )

        max_layer_len = max(sum(layer) for layer in self._layer_list)
        self._adjust_layer_pad(list(range(self._qwires)), max_layer_len + getattr(self.style, "end_wire_ext", 0))
        self.print_circuit()

    def print_circuit(self):
        """
        Print the rendered circuit to standard output.
        Qubit 0 is printed at the top.
        """
        for i in range(self._qwires):
            print(self._render_strs["top_frame"][i])
            print(self._render_strs["mid_frame"][i])
            print(self._render_strs["bot_frame"][i])

    def save(self, filename: str):
        """
        Save the rendered circuit to a text file.
        Qubit 0 is saved at the top.
        """
        if not filename.endswith(".txt"):
            filename += ".txt"
        with open(filename, "w", encoding="utf-8") as file:
            for i in range(self._qwires):
                file.write(self._render_strs["top_frame"][i] + "\n")
                file.write(self._render_strs["mid_frame"][i] + "\n")
                file.write(self._render_strs["bot_frame"][i] + "\n")
