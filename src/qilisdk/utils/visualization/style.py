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

from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib.font_manager as fm
from pydantic import BaseModel, Field

from .themes import Theme, light

_DEFAULT_FONT_PATH = Path(__file__).parent / "PlusJakartaSans-SemiBold.ttf"


class Style(BaseModel):
    # --- FontProperties-mapped fields (mirror matplotlib.font_manager.FontProperties) ---
    # If `fontfname` exists, it takes precedence and loads the exact TTF.
    theme: Theme = Field(default=light, description="Colour theme.")
    fontfamily: str | list[str] | None = Field(
        default=None, description="Font family name(s), e.g. 'Outfit' or ['Outfit', 'DejaVu Sans']."
    )
    fontstyle: Literal["normal", "italic", "oblique"] = Field(
        default="normal", description="Font style: 'normal', 'italic', or 'oblique'."
    )
    fontvariant: Literal["normal", "small-caps"] = Field(
        default="normal", description="Font variant: typically 'normal' or 'small-caps'."
    )
    fontweight: str | int = Field(
        default="normal", description="Font weight: 'normal', 'bold', 'light', or numeric (100-900)."
    )
    fontstretch: str | int = Field(
        default="normal", description="Width/condensation: 'ultra-condensed'..'ultra-expanded' or numeric."
    )
    fontsize: float | str = Field(
        default=10, description="Font size in pt or keywords like 'small', 'medium', 'large'."
    )
    fontfname: str | None = Field(
        default=str(_DEFAULT_FONT_PATH), description="Absolute path to the TTF/OTF file. If present, overrides family."
    )
    math_fontfamily: str | None = Field(default=None, description="Math text family, e.g. 'dejavusans', 'cm', or None.")
    dpi: int = Field(default=150, description="Figure DPI.")
    title: str | None = Field(default=None, description="Figure title.")

    @property
    def font(self) -> fm.FontProperties:
        """
        Construct a Matplotlib FontProperties from the configured fields.
        If `fontfname` points to a real file, it is used (and overrides family).
        """
        return fm.FontProperties(
            family=self.fontfamily,
            style=self.fontstyle,
            variant=self.fontvariant,
            weight=self.fontweight,
            stretch=self.fontstretch,
            size=self.fontsize,
            fname=self.fontfname,
            math_fontfamily=self.math_fontfamily,
        )


class CircuitStyle(Style):
    """All visual parameters controlling the appearance of a circuit plot."""

    end_wire_ext: int = Field(default=2, description="Extra space after last layer.")
    padding: float = Field(default=0.3, description="Padding around drawing (inches).")
    gate_margin: float = Field(default=0.15, description="Left/right margin per gate.")
    wire_sep: float = Field(default=0.5, description="Vertical separation of wires.")
    layer_sep: float = Field(default=0.5, description="Horizontal separation of layers.")
    gate_pad: float = Field(default=0.05, description="Padding around gate text.")
    label_pad: float = Field(default=0.1, description="Padding before wire label.")
    bulge: str = Field(default="round", description="Box-style for gate rectangles.")
    align_layer: bool = Field(default=True, description="Align layers across wires.")

    wire_label: list[Any] | None = Field(default=None, description="Custom wire labels.")
    start_pad: float = Field(
        default=0.1, description="Minimum spacing (inches) before the first layer so wire labels fit."
    )
    min_gate_h: float = Field(default=0.2, description="Minimum gate box height (inches).")
    min_gate_w: float = Field(default=0.2, description="Minimum gate box width (inches).")
    connector_r: float = Field(
        default=0.01, description="Radius (inches) of small connector dots on multi-target gates."
    )
    target_r: float = Field(default=0.12, description="Radius (inches) of ⊕ target circle and SWAP half-width.")
    control_r: float = Field(default=0.05, description="Radius (inches) of a filled control dot.")

    layout: Literal["normal", "compact"] = Field(
        default="normal",
        description="If 'compact' minimizes the layers to highlight circuit depth, if 'normal' conserves the order of the circuit",
    )


class ScheduleStyle(Style):
    """
    Customization options for matplotlib schedule plots, with theme support.
    """

    # Figure and axes
    figsize: Optional[tuple] = Field(default=(8, 5), description="Figure size in inches (width, height).")
    grid: bool = Field(default=True, description="Whether to show grid lines on the plot.")
    grid_style: dict[str, Any] = Field(
        default_factory=lambda: {"linestyle": "--", "color": "#e0e0e0", "alpha": 0.7},
        description="Style dictionary for grid lines (linestyle, color, alpha, etc.).",
    )

    # Title and labels
    title_fontsize: int = Field(default=16, description="Font size for the plot title.")
    xlabel: str = Field(default="time (dt)", description="Label for the x-axis.")
    ylabel: str = Field(default="coefficient value", description="Label for the y-axis.")
    label_fontsize: int = Field(default=14, description="Font size for axis labels.")

    # Legend
    legend_loc: str = Field(
        default="best", description="Location of the legend (matplotlib string, e.g. 'best', 'upper right')."
    )
    legend_fontsize: int = Field(default=12, description="Font size for legend text.")
    legend_frame: bool = Field(default=True, description="Whether to draw a frame around the legend.")

    # Line style
    line_styles: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom line style dictionary for each Hamiltonian (e.g. {label: {color, linestyle, linewidth}}).",
    )
    default_line_style: dict[str, Any] = Field(
        default_factory=lambda: {"linestyle": "-", "linewidth": 2},
        description="Default line style for Hamiltonians not in line_styles.",
    )

    # Marker style
    marker: Optional[str] = Field(
        default=None, description="Matplotlib marker style for data points (e.g. 'o', 's', None for no marker)."
    )
    marker_size: int = Field(default=6, description="Size of markers if used.")

    # Ticks
    xtick_fontsize: int = Field(default=12, description="Font size for x-axis tick labels.")
    ytick_fontsize: int = Field(default=12, description="Font size for y-axis tick labels.")
    tick_color: Optional[str] = Field(
        default=None, description="Color for tick labels (None uses theme.on_background)."
    )

    # Misc
    tight_layout: bool = Field(default=True, description="Whether to use matplotlib's tight_layout for figure spacing.")
