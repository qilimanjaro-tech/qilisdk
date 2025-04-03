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
import itertools

import plotly.graph_objects as go

from .circuit import Circuit
from .gates import Adjoint, Controlled, Exponential, M


class PlotlyRenderer:
    def __init__(self, config: dict | None = None) -> None:
        self.config = {
            # Font size for labels
            "font_size": 26,
            # Thickness of drawn lines
            "line_width": 2,
            # Width of each column
            "column_width": 0.6,
            # Distance between wires
            "wire_spacing": 0.5,
            # Size of the boxes (square: width = height)
            "box_size": 0.4,
        }
        if config:
            self.config.update(config)

    def draw(self, circuit: Circuit, filename: str | None = None) -> None:
        nqubits = circuit.nqubits

        # Step 1: Group gates into columns (non-overlapping sets)
        columns = []
        current_column = []
        occupied = set()

        for gate in circuit.gates:
            gate_qubits = set(gate.qubits)
            if occupied & gate_qubits:
                columns.append(current_column)
                current_column = [gate]
                occupied = set(gate.qubits)
            else:
                current_column.append(gate)
                occupied.update(gate.qubits)
        if current_column:
            columns.append(current_column)

        column_width = self.config["column_width"]
        wire_spacing = self.config["wire_spacing"]
        box_size = self.config["box_size"]
        line_width = self.config["line_width"]
        font_size = self.config["font_size"]

        # Compute figure extents using col_width and box_size
        num_cols = len(columns)
        if num_cols > 0:
            x_last_center = (num_cols - 1) * column_width
            # left margin
            x_min = -(box_size + 1)
            # right margin
            x_max = x_last_center + box_size + 1
        else:
            x_min = -0.5
            x_max = 0.5

        if nqubits > 0:
            y_top = 0
            y_bottom = (nqubits - 1) * wire_spacing
        else:
            y_top = 0
            y_bottom = 0
        # margin above top wire
        y_min = y_top - 0.5 * wire_spacing
        # margin below bottom wire
        y_max = y_bottom + 0.5 * wire_spacing

        fig = go.Figure()

        # Draw wires
        for q in range(nqubits):
            y = q * wire_spacing
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=(len(columns) - 1) * column_width + 0.5,
                y0=y,
                y1=y,
                line={"color": "black", "width": line_width},
                layer="below",
            )

        def add_box(x: float, y: float, label: str) -> None:
            fig.add_shape(
                type="rect",
                x0=x - box_size / 2,
                x1=x + box_size / 2,
                y0=y - box_size / 2,
                y1=y + box_size / 2,
                fillcolor="white",
                line={"color": "black", "width": line_width},
            )
            fig.add_annotation(
                x=x, y=y, text=label, showarrow=False, font={"size": font_size}, xanchor="center", yanchor="middle"
            )

        def add_plus(x: float, y: float) -> None:
            size = box_size * 0.2
            fig.add_shape(
                type="line",
                x0=x - size,
                x1=x + size,
                y0=y - size,
                y1=y + size,
                line={"color": "black", "width": line_width},
            )
            fig.add_shape(
                type="line",
                x0=x - size,
                x1=x + size,
                y0=y + size,
                y1=y - size,
                line={"color": "black", "width": line_width},
            )

        def add_dot(x: float, y: float) -> None:
            r = box_size * 0.125
            fig.add_shape(type="circle", x0=x - r, x1=x + r, y0=y - r, y1=y + r, fillcolor="black", line_color="black")

        for col_idx, col in enumerate(columns):
            x = col_idx * column_width
            for gate in col:
                qs = gate.qubits
                if isinstance(gate, M):
                    qs_sorted = sorted(qs)
                    contiguous = all(q2 == q1 + 1 for q1, q2 in itertools.pairwise(qs_sorted))
                    if contiguous and len(qs_sorted) > 1:
                        y_top = qs_sorted[0] * wire_spacing
                        y_bottom = qs_sorted[-1] * wire_spacing
                        fig.add_shape(
                            type="rect",
                            x0=x - box_size / 2,
                            x1=x + box_size / 2,
                            y0=y_top - box_size / 2,
                            y1=y_bottom + box_size / 2,
                            fillcolor="white",
                            line={"color": "black", "width": line_width},
                        )
                        fig.add_annotation(
                            x=x,
                            y=(y_top + y_bottom) / 2,
                            text=gate.name,
                            showarrow=False,
                            font={"size": font_size},
                            xanchor="center",
                            yanchor="middle",
                        )
                    else:
                        for q in qs:
                            y = q * wire_spacing
                            add_box(x, y, gate.name)
                elif isinstance(gate, Controlled):
                    ctrl = gate.control_qubits
                    tgt = gate.target_qubits[0]
                    ys = [q * wire_spacing for q in (*ctrl, tgt)]
                    y_0, y_1 = min(ys), max(ys)
                    fig.add_shape(type="line", x0=x, x1=x, y0=y_0, y1=y_1, line={"color": "black", "width": line_width})
                    for q in ctrl:
                        add_dot(x, q * wire_spacing)
                    if gate.basic_gate.name == "X":
                        add_plus(x, tgt * wire_spacing)
                    else:
                        add_box(x, tgt * wire_spacing, gate.basic_gate.name)
                elif isinstance(gate, Exponential):
                    for q in qs:
                        y = q * wire_spacing
                        add_box(x, y, f"e<sup>{gate.name[2:]}</sup>")
                elif isinstance(gate, Adjoint):
                    for q in qs:
                        y = q * wire_spacing
                        add_box(x, y, f"{gate.name[0]}<sup>{gate.name[1]}</sup>")
                else:
                    for q in qs:
                        y = q * wire_spacing
                        add_box(x, y, gate.name)

        fig.update_layout(
            xaxis={"range": [x_min, x_max], "visible": False},
            yaxis={"range": [y_max, y_min], "visible": False},
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            plot_bgcolor="rgba(240,240,255,1)",
            showlegend=False,
        )
        fig.update_yaxes(scaleanchor="x")

        if filename:
            fig.write_image(filename)
        else:
            fig.show()
