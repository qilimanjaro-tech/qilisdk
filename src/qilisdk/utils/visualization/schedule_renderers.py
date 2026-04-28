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
from matplotlib.figure import Figure

from qilisdk.analog.hamiltonian import Hamiltonian

if TYPE_CHECKING:
    from qilisdk.analog.schedule import Schedule
    from qilisdk.core import QTensor
    from qilisdk.core.types import Number

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

    def setup_axes(self) -> None:
        style = self.style
        theme = style.theme
        title_color = theme.on_background
        label_color = theme.on_background
        legend_facecolor = theme.surface
        legend_edgecolor = theme.border
        tick_color = theme.on_background

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
            self.style.title or "Schedule Eigenvalues",
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

    def plot(self, ax: plt.Axes | None = None) -> None:
        """
        Plot the schedule coefficients for each Hamiltonian over time.
        Args:
            ax (plt.Axes | None): The matplotlib axes to plot on. Default is None.
        """
        style = self.style
        theme = style.theme
        facecolor = theme.background

        # Set axes and figure background to theme
        self.ax.set_facecolor(facecolor)
        if hasattr(ax, "figure"):
            self.ax.figure.set_facecolor(facecolor)
        plots: dict[str, list[Number]] = {}
        hamiltonians = self.schedule.hamiltonians
        times = self.schedule.tlist
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
            self.ax.plot(
                times,
                plots[h],
                label=h,
                marker=marker,
                markersize=style.marker_size,
                **line_style,  # ty:ignore[invalid-argument-type]
            )

        self.setup_axes()

        plt.draw()

    def save(self, filename: str) -> None:  # thin wrapper
        """Save current figure to disk.

        Args:
            filename: Path to save the figure (e.g., 'circuit.png').
        """
        if isinstance(self.ax.figure, Figure):
            self.ax.figure.savefig(filename, bbox_inches="tight")

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


class MatplotlibEigenvalueRenderer(MatplotlibScheduleRenderer):
    """Render the eigenvalues Schedule using matplotlib, with theme support."""

    def __init__(
        self,
        schedule: Schedule,
        ax: plt.Axes | None = None,
        *,
        style: ScheduleStyle | None = None,
        levels: int = 2,
        intermediate_states: list[QTensor] | None = None,
        show_overlaps: bool = True,
    ) -> None:
        self.schedule: Schedule = schedule
        self.style = style or ScheduleStyle(xlabel="Time", ylabel="Eigenvalue")
        self.ax = ax or self._make_axes(self.style.dpi, self.style)
        self.levels = levels
        self.intermediate_states = intermediate_states
        self.show_overlaps = show_overlaps

    def _calculate_eigenvalues_and_overlaps(
        self, hamiltonians: dict[str, Hamiltonian], times: list[float]
    ) -> tuple[list[list[float]], list[list[QTensor]], list[float]]:
        full_eigenvalues = []
        full_eigenstates = []
        actual_expectation_values = []
        for i, t in enumerate(times):
            full_hamiltonian = sum(
                self.schedule.coefficients[h][float(t)] * self.schedule.hamiltonians[h] for h in hamiltonians
            )
            if not isinstance(full_hamiltonian, Hamiltonian):
                raise ValueError(f"Expected full_hamiltonian to be a Hamiltonian, got {type(full_hamiltonian)}")
            as_qtensor = full_hamiltonian.to_qtensor()
            vals, vecs = as_qtensor.eig()

            full_eigenvalues.append([float(ev.real) for ev in vals[: self.levels]])
            full_eigenstates.append(list(vecs[: self.levels]))

            # Also plot the expectation value if we have intermediate states
            if self.intermediate_states:
                state = self.intermediate_states[i]
                exp_val = state.expectation_value(as_qtensor)
                actual_expectation_values.append(float(exp_val.real))

        return full_eigenvalues, full_eigenstates, actual_expectation_values

    @staticmethod
    def _calculate_overlaps(
        state: QTensor,
        eigenstates: list[QTensor],
        eigenvalues: list[list[float]],
        time_index: int,
        eigen_range: float,
        sig_figs: int,
    ) -> list[tuple[float, float]]:
        overlaps = []
        for j, eig in enumerate(eigenstates):
            overlap = 100.0 * state.fidelity(eig)
            y_loc = eigenvalues[j][time_index]
            if overlap > 10 ** (-sig_figs):
                overlaps.append((y_loc, overlap))

        # Group nearby overlaps together to avoid clutter
        grouped_overlaps: list[tuple[float, float]] = []
        for overlap in overlaps:
            found_group = False
            for idx, grouped_overlap in enumerate(grouped_overlaps):
                if abs(grouped_overlap[0] - overlap[0]) < 0.05 * eigen_range:
                    # If within 5% of the eigenvalue range, group them together by averaging the y location and summing the overlap percentage
                    new_y_loc = (grouped_overlap[0] + overlap[0]) / 2
                    new_overlap = grouped_overlap[1] + overlap[1]
                    grouped_overlaps[idx] = (new_y_loc, new_overlap)
                    found_group = True
                    break
            if not found_group:
                grouped_overlaps.append(overlap)

        return grouped_overlaps

    def plot(self, ax: plt.Axes | None = None) -> None:
        """
        Plot the schedule coefficients for each Hamiltonian over time.

        Args:
            ax (plt.Axes | None): The matplotlib axes to plot on. Default is None.

        Raises:
            ValueError: If the full Hamiltonian cannot be constructed or is not a Hamiltonian instance.
        """
        style = self.style
        theme = style.theme
        facecolor = theme.background

        # Set axes and figure background to theme
        self.ax.set_facecolor(facecolor)
        if hasattr(ax, "figure"):
            self.ax.figure.set_facecolor(facecolor)
        plots: dict[str, list[Number]] = {}
        hamiltonians: dict[str, Hamiltonian] = self.schedule.hamiltonians
        times = self.schedule.tlist
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

        # Plot the eigenvalues of the full Hamiltonian as solid lines
        full_eigenvalues, full_eigenstates, actual_expectation_values = self._calculate_eigenvalues_and_overlaps(
            hamiltonians, times
        )

        min_eigenvalue = min(min(evs) for evs in full_eigenvalues)
        max_eigenvalue = max(max(evs) for evs in full_eigenvalues)
        eigen_range = max_eigenvalue - min_eigenvalue

        color = grad_colors[-1] if grad_colors else theme.accent
        full_eigenvalues = list(zip(*full_eigenvalues))  # transpose to get eigenvalues over time
        # only show the id for the first one
        for idx, evs in enumerate(full_eigenvalues):
            label = "Eigenvalues" if idx == 0 and self.intermediate_states else None
            self.ax.plot(
                times,
                evs,
                label=label,
                linestyle="--",
                color=color,
            )

        if self.intermediate_states and actual_expectation_values:
            self.ax.plot(
                times,
                actual_expectation_values,
                label="State Expectation Value",
                linestyle="-",
                color="black",
                zorder=10,
            )

            # Every 10% of the way through, write the overlap with each of the eigenstates at that time
            if self.show_overlaps:
                time_steps = list(range(0, len(times), max(1, len(times) // 9)))
                time_steps.append(len(times) - 1)
                for i in time_steps:
                    t = times[i]
                    eigenstates = full_eigenstates[i]
                    state = self.intermediate_states[i]

                    _sig_figs = 3
                    grouped_overlaps = self._calculate_overlaps(
                        state, eigenstates[: self.levels], full_eigenvalues[: self.levels], i, eigen_range, _sig_figs
                    )

                    # Plot each overlap with an arrow pointing to the eigenvalue, and annotate with the percentage
                    for overlap in grouped_overlaps:
                        overlap_text = f"{overlap[1]:.{_sig_figs}f}%"
                        y_loc = overlap[0]
                        self.ax.annotate(
                            overlap_text,
                            xy=(t, y_loc),
                            xytext=(t, y_loc + 0.5),
                            arrowprops={"arrowstyle": "->", "color": theme.on_background},
                            color=theme.on_background,
                            bbox={"boxstyle": "round,pad=0.2", "fc": theme.surface, "ec": "none", "alpha": 0.8},
                            fontsize=style.label_fontsize * 0.4,
                            ha="center",
                            va="bottom",
                            zorder=15,
                        )

        self.setup_axes()
        plt.draw()
