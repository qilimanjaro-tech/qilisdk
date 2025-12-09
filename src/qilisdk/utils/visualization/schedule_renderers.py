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

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from qilisdk.analog.schedule import Schedule
    from qilisdk.core.variables import Number

from qilisdk.utils.visualization.style import ScheduleStyle


class MatplotlibScheduleRenderer:
    """Render a Schedule using matplotlib, with theme support."""

    def __init__(
        self,
        schedule: Schedule,
        ax: plt.Axes | None = None,
        *,
        style: ScheduleStyle | None = None,
    ) -> None:
        self.schedule = schedule
        self.style = style or ScheduleStyle()
        self.ax = ax or self._make_axes(self.style.dpi, self.style)

    def plot(self, ax: plt.Axes | None = None) -> None:
        """
        Plot the schedule coefficients for each Hamiltonian over time.
        Args:
            ax (plt.Axes | None): The matplotlib axes to plot on. Default is None.
        """
        style = self.style
        theme = style.theme
        facecolor = theme.background
        title_color = theme.on_background
        label_color = theme.on_background
        legend_facecolor = theme.surface
        legend_edgecolor = theme.border
        tick_color = theme.on_background

        # Set axes and figure background to theme
        self.ax.set_facecolor(facecolor)
        if hasattr(ax, "figure"):
            self.ax.figure.set_facecolor(facecolor)
        plots: dict[str, list[Number]] = {}
        T = self.schedule.T
        dt = self.schedule.dt
        hamiltonians = self.schedule.hamiltonians
        times = list(np.linspace(0, T, int(1 / dt), dtype=float))  # [i * dt for i in range(int((T + dt) / dt))]
        for t in self.schedule.tlist:
            if t not in times:
                times.append(t)
        times = sorted(times)
        for h in hamiltonians:
            coef = self.schedule.coefficients[h]
            plots[h] = [coef[float(t)] for t in times]

        # Generate gradient colors between primary and accent
        def hex_to_rgb(hex_color: str) -> tuple[int, ...]:
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        def rgb_to_hex(rgb: tuple[int, ...]) -> str:
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        def gradient_colors(start_hex: str, end_hex: str, n: int) -> list[str]:
            start_rgb = hex_to_rgb(start_hex)
            end_rgb = hex_to_rgb(end_hex)
            colors = []
            for i in range(n):
                ratio = i / max(n - 1, 1)
                rgb = tuple(int(start_rgb[j] + (end_rgb[j] - start_rgb[j]) * ratio) for j in range(3))
                colors.append(rgb_to_hex(rgb))
            return colors

        n_hams = len(hamiltonians)
        grad_colors = gradient_colors(theme.primary, theme.accent, n_hams)

        for idx, h in enumerate(hamiltonians):
            line_style = style.line_styles.get(h, style.default_line_style)
            marker = style.marker
            # If no color specified, use gradient color
            if "color" not in line_style:
                color = grad_colors[idx]
                line_style = {**line_style, "color": color}
            self.ax.plot(times, plots[h], label=h, marker=marker, markersize=style.marker_size, **line_style)
        if style.grid:
            grid_style = dict(style.grid_style)
            if "color" not in grid_style:
                grid_style["color"] = theme.surface_muted
            self.ax.grid(**grid_style)
        leg = self.ax.legend(
            loc=style.legend_loc,
            fontsize=style.legend_fontsize,
            frameon=style.legend_frame,
            facecolor=legend_facecolor,
            edgecolor=legend_edgecolor,
        )
        # Set legend text color to match theme text color
        if leg:
            for text in leg.get_texts():
                text.set_color(title_color)
        self.ax.set_title(
            self.style.title or "Schedule Plot",
            fontsize=style.title_fontsize,
            color=title_color,
            fontweight=style.fontweight,
            family=style.fontfamily,
        )
        self.ax.set_xlabel(
            style.xlabel,
            fontsize=style.label_fontsize,
            color=label_color,
            fontweight=style.fontweight,
            family=style.fontfamily,
        )
        self.ax.set_ylabel(
            style.ylabel,
            fontsize=style.label_fontsize,
            color=label_color,
            fontweight=style.fontweight,
            family=style.fontfamily,
        )
        self.ax.tick_params(axis="x", labelsize=style.xtick_fontsize, colors=tick_color)
        self.ax.tick_params(axis="y", labelsize=style.ytick_fontsize, colors=tick_color)
        if style.tight_layout:
            plt.tight_layout()
        plt.draw()

    def save(self, filename: str) -> None:  # thin wrapper
        """Save current figure to disk.

        Args:
            filename: Path to save the figure (e.g., 'circuit.png').
        """

        self.ax.figure.savefig(filename, bbox_inches="tight")  # type: ignore[union-attr]

    def show(self) -> None:  # noqa: PLR6301
        """Show the current figure."""

        plt.show()

    @staticmethod
    def _make_axes(dpi: int, style: ScheduleStyle) -> plt.Axes:
        """
        Create a new figure and axes with the given DPI.

        Args:
            style: Optional style configuration (for DPI).

        Returns:
            A newly created Matplotlib Axes.
        """
        _, ax = plt.subplots(figsize=style.figsize, dpi=dpi or style.dpi, facecolor=style.theme.background)
        return ax
