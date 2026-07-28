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

from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from qilisdk.utils.visualization.style import DatasetStyle

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from qilisdk.ml.datasets.dataset import DatasetSample

_VALID_STYLES = ("1d", "2d", "3d")
_MIN_2D_COMPONENTS = 2
_MIN_3D_COMPONENTS = 3

class MatplotlibDatasetRenderer:
    """Render a dataset sample using matplotlib, with theme support.

    This is the common drawing class shared by every dataset's
    :meth:`~qilisdk.ml.datasets.dataset.Dataset.draw` method. The kind of plot
    is chosen with ``style``:

    * ``"1d"`` -- each component of the series plotted against the sample index.
    * ``"2d"`` -- a phase portrait. For series with two or more components the
      first two are plotted against each other; a one-dimensional series is
      delay-embedded as ``x(t)`` vs ``x(t + delay)``.
    * ``"3d"`` -- a three-dimensional phase portrait. Series with three or more
      components use the first three, lower-dimensional series are delay-embedded.
    """

    def __init__(
        self,
        sample: DatasetSample,
        style: str = "1d",
        *,
        config: DatasetStyle | None = None,
        labels: tuple[str, ...] | None = None,
        title: str | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        if style not in _VALID_STYLES:
            raise ValueError(f"style must be one of {_VALID_STYLES}, got {style!r}.")
        self.sample = sample
        self.mode = style
        self.style = config or DatasetStyle()
        self.labels = labels
        self.title = title
        self.series = self._as_matrix(sample.inputs)
        self.ax = ax or self._make_axes(self.mode, self.style)

    @staticmethod
    def _as_matrix(inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Make a sample's inputs into a 2-D array.

        Args:
            inputs (NDArray[np.float64]): The raw series, one row per time step.

        Returns:
            NDArray[np.float64]: The series shaped ``(n_points, n_components)``.
        """
        arr = np.asarray(inputs, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _cmap(self) -> LinearSegmentedColormap:
        theme = self.style.theme
        return LinearSegmentedColormap.from_list("qili_time", [theme.primary, theme.accent])

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, ...]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, ...]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _gradient_colors(self, n: int) -> list[str]:
        start = self._hex_to_rgb(self.style.theme.primary)
        end = self._hex_to_rgb(self.style.theme.accent)
        colors = []
        for i in range(n):
            ratio = i / max(n - 1, 1)
            rgb = tuple(int(start[j] + (end[j] - start[j]) * ratio) for j in range(3))
            colors.append(self._rgb_to_hex(rgb))
        return colors

    def _embed(self, dims: int) -> list[NDArray[np.float64]]:
        """Delay-embed the first component into ``dims`` coordinates.

        Args:
            dims (int): Number of embedding coordinates to produce.

        Returns:
            list[NDArray[np.float64]]: One equal-length array per coordinate,
            successively shifted by the configured delay.

        Raises:
            ValueError: If the series is too short for the requested embedding.
        """
        lag = max(1, self.style.delay)
        col = self.series[:, 0]
        span = (dims - 1) * lag
        if len(col) <= span:
            raise ValueError(
                f"series of length {len(col)} is too short for a {dims}-D delay embedding with delay {lag}."
            )
        stop = len(col) - span
        return [col[i * lag : i * lag + stop] for i in range(dims)]

    def _component_label(self, index: int, embedded: bool, lag: int) -> str:
        base = self.labels[0] if self.labels else "x"
        if embedded:
            if index == 0:
                return f"{base}(t)"
            return f"{base}(t + {index * lag})"
        if self.labels and index < len(self.labels):
            return self.labels[index]
        return f"x{index}"

    def plot(self, ax: plt.Axes | None = None) -> None:
        """Render the sample onto the renderer's axes."""
        if ax is not None:
            self.ax = ax
        logger.debug("[DatasetRenderer] Rendering sample as {} ({} points)", self.mode, len(self.series))
        if self.mode == "1d":
            self._plot_1d()
        elif self.mode == "2d":
            self._plot_2d()
        else:
            self._plot_3d()
        self._setup_axes()
        plt.draw()

    def _plot_1d(self) -> None:
        style = self.style
        n, d = self.series.shape
        t = np.arange(n)
        colors = self._gradient_colors(d)
        line_style = dict(style.line_style)
        line_style.pop("color", None)
        for i in range(d):
            self.ax.plot(
                t,
                self.series[:, i],
                label=self._component_label(i, embedded=False, lag=style.delay),
                color=colors[i],
                marker=style.marker,
                markersize=style.marker_size,
                **line_style,
            )
        self._xlabel = style.xlabel or "step"
        self._ylabel = style.ylabel or "value"
        self._show_legend = d > 1 or self.labels is not None

    def _plot_2d(self) -> None:
        style = self.style
        d = self.series.shape[1]
        embedded = d < _MIN_2D_COMPONENTS
        if embedded:
            x, y = self._embed(2)
            self._xlabel = style.xlabel or self._component_label(0, True, style.delay)
            self._ylabel = style.ylabel or self._component_label(1, True, style.delay)
        else:
            x, y = self.series[:, 0], self.series[:, 1]
            self._xlabel = style.xlabel or self._component_label(0, False, style.delay)
            self._ylabel = style.ylabel or self._component_label(1, False, style.delay)
        self._draw_trajectory((x, y))
        self._show_legend = False

    def _plot_3d(self) -> None:
        style = self.style
        d = self.series.shape[1]
        embedded = d < _MIN_3D_COMPONENTS
        if embedded:
            x, y, z = self._embed(3)
            self._xlabel = style.xlabel or self._component_label(0, True, style.delay)
            self._ylabel = style.ylabel or self._component_label(1, True, style.delay)
            self._zlabel = style.zlabel or self._component_label(2, True, style.delay)
        else:
            x, y, z = self.series[:, 0], self.series[:, 1], self.series[:, 2]
            self._xlabel = style.xlabel or self._component_label(0, False, style.delay)
            self._ylabel = style.ylabel or self._component_label(1, False, style.delay)
            self._zlabel = style.zlabel or self._component_label(2, False, style.delay)
        self._draw_trajectory((x, y, z))
        self._show_legend = False

    def _draw_trajectory(self, coords: tuple[NDArray[np.float64], ...]) -> None:
        style = self.style
        n = len(coords[0])
        if style.trajectory_style == "line" or not style.color_by_time:
            color = self.style.theme.primary
            line_style = dict(style.line_style)
            line_style.pop("color", None)
            if style.trajectory_style == "line":
                self.ax.plot(*coords, color=color, **line_style)
            else:
                self.ax.scatter(*coords, color=color, s=style.point_size)
            return
        scatter = self.ax.scatter(*coords, c=np.arange(n), cmap=self._cmap(), s=style.point_size)
        if style.colorbar and hasattr(self.ax, "figure"):
            cbar = self.ax.figure.colorbar(scatter, ax=self.ax, pad=0.1)
            cbar.set_label("step", color=self.style.theme.on_background)
            cbar.ax.yaxis.set_tick_params(color=self.style.theme.on_background)
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=self.style.theme.on_background)

    def _setup_axes(self) -> None:
        style = self.style
        theme = style.theme
        text_color = theme.on_background

        facecolor = theme.background
        self.ax.set_facecolor(facecolor)
        if hasattr(self.ax, "figure"):
            self.ax.figure.set_facecolor(facecolor)

        if style.grid:
            grid_style = dict(style.grid_style)
            if "color" not in grid_style:
                grid_style["color"] = theme.surface_muted
            self.ax.grid(**grid_style)

        self.ax.set_title(
            style.title or self.title or "Dataset",
            fontsize=style.title_fontsize,
            color=text_color,
            fontproperties=style.font,
        )
        self.ax.set_xlabel(self._xlabel, fontsize=style.label_fontsize, color=text_color, fontproperties=style.font)
        self.ax.set_ylabel(self._ylabel, fontsize=style.label_fontsize, color=text_color, fontproperties=style.font)
        self.ax.tick_params(axis="x", labelsize=style.xtick_fontsize, colors=text_color)
        self.ax.tick_params(axis="y", labelsize=style.ytick_fontsize, colors=text_color)

        if self.mode == "3d":
            self.ax.set_zlabel(  # ty:ignore[unresolved-attribute]
                self._zlabel, fontsize=style.label_fontsize, color=text_color, fontproperties=style.font
            )
            self.ax.tick_params(axis="z", labelsize=style.ytick_fontsize, colors=text_color)  # ty:ignore[invalid-argument-type]

        if getattr(self, "_show_legend", False):
            leg = self.ax.legend(
                loc=cast("Any", style.legend_loc),
                fontsize=style.legend_fontsize,
                frameon=style.legend_frame,
                facecolor=theme.surface,
                edgecolor=theme.border,
            )
            if leg:
                for text in leg.get_texts():
                    text.set_color(text_color)

        if style.tight_layout:
            plt.tight_layout()

    def save(self, filename: str) -> None:
        """Save the current figure to disk (format inferred from the extension)."""
        logger.debug("[DatasetRenderer] Saving figure to {}", filename)
        if isinstance(self.ax.figure, Figure):
            self.ax.figure.savefig(filename, bbox_inches="tight")

    def show(self) -> None:  # noqa: PLR6301
        """Show the current figure."""
        plt.show()

    @staticmethod
    def _make_axes(mode: str, style: DatasetStyle) -> plt.Axes:
        """Create a new figure and axes appropriate for the requested plot mode.

        Args:
            mode (str): The plot mode, one of ``"1d"``, ``"2d"`` or ``"3d"``.
            style (DatasetStyle): Style configuration (for figure size and DPI).

        Returns:
            plt.Axes: A newly created Matplotlib Axes (3-D projection for ``"3d"``).
        """
        if mode == "3d":
            fig = plt.figure(figsize=style.figsize, dpi=style.dpi, facecolor=style.theme.background)
            return cast("plt.Axes", fig.add_subplot(111, projection="3d"))
        _, ax = plt.subplots(figsize=style.figsize, dpi=style.dpi, facecolor=style.theme.background)
        return ax
