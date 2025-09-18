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


class CircuitStyle(BaseModel):
    """All visual parameters controlling the appearance of a circuit plot."""

    # --- FontProperties-mapped fields (mirror matplotlib.font_manager.FontProperties) ---
    # If `fontfname` exists, it takes precedence and loads the exact TTF.
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
    end_wire_ext: int = Field(default=2, description="Extra space after last layer.")
    padding: float = Field(default=0.3, description="Padding around drawing (inches).")
    gate_margin: float = Field(default=0.15, description="Left/right margin per gate.")
    wire_sep: float = Field(default=0.5, description="Vertical separation of wires.")
    layer_sep: float = Field(default=0.5, description="Horizontal separation of layers.")
    gate_pad: float = Field(default=0.05, description="Padding around gate text.")
    label_pad: float = Field(default=0.1, description="Padding before wire label.")
    bulge: str = Field(default="round", description="Box-style for gate rectangles.")
    align_layer: bool = Field(default=True, description="Align layers across wires.")
    theme: Theme = Field(default=light, description="Colour theme.")
    title: str | None = Field(default=None, description="Figure title.")
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


class ScheduleStyle(BaseModel):
    """
    Customization options for matplotlib schedule plots, with theme support.
    """

    theme: Theme = Field(default=light, description="Colour theme.")

    # Figure and axes
    figsize: Optional[tuple] = (8, 5)
    dpi: int = 150
    grid: bool = True
    grid_style: dict[str, Any] = Field(default_factory=lambda: {"linestyle": "--", "color": "#e0e0e0", "alpha": 0.7})

    # Title and labels
    title_fontsize: int = 16
    xlabel: str = "time (dt)"
    ylabel: str = "coefficient value"
    label_fontsize: int = 14

    # Legend
    legend_loc: str = "best"
    legend_fontsize: int = 12
    legend_frame: bool = True
    # Line style
    line_styles: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )  # e.g. {"Hinit": {"color": "red", "linestyle": "-", "linewidth": 2}}
    default_line_style: dict[str, Any] = Field(default_factory=lambda: {"linestyle": "-", "linewidth": 2})

    # Marker style
    marker: Optional[str] = None
    marker_size: int = 6

    # Font
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

    # Ticks
    xtick_fontsize: int = 12
    ytick_fontsize: int = 12
    tick_color: Optional[str] = None  # If None, use theme.on_background

    # Misc
    tight_layout: bool = True

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
