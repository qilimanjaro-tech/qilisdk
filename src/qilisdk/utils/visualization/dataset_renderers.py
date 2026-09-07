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

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from qilisdk.utils.visualization.style import DatasetStyle

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from qilisdk.ml.datasets.dataset import DatasetSample

    # An array to plot, optionally paired with a label for the axis or line. A transform may return any number of these.
    Channel: TypeAlias = tuple[str, NDArray[np.float64]] | NDArray[np.float64]

    # A callable that takes the full series and returns a sequence of channels to plot.
    Transform: TypeAlias = Callable[[NDArray[np.float64]], Sequence[Channel]]

_VALID_STYLES = ("1d", "2d", "3d")
_MODE_NDIM = {"2d": 2, "3d": 3}
_MIN_2D_COMPONENTS = 2
_MIN_3D_COMPONENTS = 3
_MAX_SERIES_NDIM = 2
_LABELLED_CHANNEL_LEN = 2


class MatplotlibDatasetRenderer:
    """Render a dataset sample using matplotlib, with theme support.

    This is the common drawing class shared by every dataset's
    :meth:`~qilisdk.ml.datasets.dataset.Dataset.draw` method. The series to plot
    is taken either from a :class:`~qilisdk.ml.datasets.dataset.DatasetSample`
    (whose ``inputs`` are used) or from a raw array, so that a single component
    of a sample -- or any series of the same shape -- can be plotted directly.

    What is drawn on each axis is decided by a *transform*: a callable mapping
    the full ``(n_points, n_components)`` series to the coordinates to plot. A
    transform returns the channels for the current mode -- two arrays for a
    ``"2d"`` view, three for ``"3d"``, or any number of lines for ``"1d"`` --
    each optionally paired with an axis label. It is free to reshape, slice or
    delay-embed the data rather than merely selecting columns, so the view is
    fully general: e.g. ``lambda d: [("x", d[:, 0]), ("z", d[:, 2])]`` picks the
    Lorenz ``x``--``z`` plane, and ``lambda d: [("P(t)", d[17:, 0]),
    ("P(t-17)", d[:-17, 0])]`` builds a delay embedding. When no transform is
    given, a sensible default is used per mode:

    * ``"1d"`` -- every component plotted against the sample index.
    * ``"2d"`` -- the first two components; a one-dimensional series is
      delay-embedded as ``x(t)`` vs ``x(t + delay)``.
    * ``"3d"`` -- the first three components; lower-dimensional series are
      delay-embedded.
    """

    def __init__(
        self,
        sample: DatasetSample | NDArray[np.floating[Any]],
        style: str = "1d",
        *,
        config: DatasetStyle | None = None,
        labels: tuple[str, ...] | None = None,
        title: str | None = None,
        transform: Transform | None = None,
        ax: plt.Axes | None = None,
    ) -> None:
        """
        Initialize a renderer for a dataset sample.

        Args:
            sample (DatasetSample | NDArray[np.floating[Any]]): The sample to plot, or a raw array holding the series itself, one row per time step.
            style (str): The plot mode, one of ``"1d"``, ``"2d"`` or ``"3d"``. Defaults to ``"1d"``.
            config (DatasetStyle | None): Optional style configuration. Defaults to ``None`` (use dataset defaults).
            labels (tuple[str, ...] | None): Optional labels for the components of the series. Defaults to ``None`` (use dataset defaults).
            title (str | None): Optional title for the plot. Defaults to ``None`` (use dataset defaults).
            transform (Transform | None): Optional callable mapping the full series to the coordinates to plot. Defaults to ``None`` (use dataset defaults).
            ax (plt.Axes | None): Optional Matplotlib Axes to draw on. Defaults to ``None`` (create a new figure and axes).

        Raises:
            ValueError: If ``style`` is not one of ``"1d"``, ``"2d"``, or ``"3d"``.
        """
        if style not in _VALID_STYLES:
            raise ValueError(f"style must be one of {_VALID_STYLES}, got {style!r}.")
        self.sample = sample
        self.mode = style
        self.style = config or DatasetStyle()
        self.labels = labels
        self.title = title
        self.transform = transform
        self.series = self._as_matrix(sample)
        self.ax = ax or self._make_axes(self.mode, self.style)

    @staticmethod
    def _as_matrix(sample: DatasetSample | NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Make the series to plot into a 2-D array.

        Accepts either a :class:`~qilisdk.ml.datasets.dataset.DatasetSample` (in
        which case its ``inputs`` are plotted) or a raw array holding the series
        itself, one row per time step.

        Args:
            sample (DatasetSample | NDArray[np.floating[Any]]): The sample or raw series.

        Returns:
            NDArray[np.float64]: The series shaped ``(n_points, n_components)``.

        Raises:
            ValueError: If the series is empty or has more than two dimensions.
        """
        inputs = sample.inputs if hasattr(sample, "inputs") else sample
        arr = np.asarray(inputs, dtype=np.float64)
        if arr.ndim > _MAX_SERIES_NDIM:
            raise ValueError(
                f"series must be at most {_MAX_SERIES_NDIM}-dimensional (n_points, n_components), got shape {arr.shape}."
            )
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            raise ValueError("series is empty, nothing to plot.")
        return arr

    def _cmap(self) -> LinearSegmentedColormap:
        """
        Build a colormap for coloring points by time, from the theme's primary to accent color.

        Returns:
            LinearSegmentedColormap: The colormap for time-based coloring.
        """
        theme = self.style.theme
        return LinearSegmentedColormap.from_list("qili_time", [theme.primary, theme.accent])

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple[int, ...]:
        """
        Convert a hex color string to an RGB tuple.

        Args:
            hex_color (str): The hex color string (e.g., ``"#RRGGBB"``).

        Returns:
            tuple[int, ...]: The corresponding RGB tuple.
        """
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def _rgb_to_hex(rgb: tuple[int, ...]) -> str:
        """
        Convert an RGB tuple to a hex color string.

        Args:
            rgb (tuple[int, ...]): The RGB tuple (e.g., ``(255, 0, 0)``).

        Returns:
            str: The corresponding hex color string (e.g., ``"#RRGGBB"``).
        """
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _gradient_colors(self, n: int) -> list[str]:
        """
        Generate a list of colors forming a gradient from the theme's primary to accent color.

        Args:
            n (int): The number of colors to generate.

        Returns:
            list[str]: A list of hex color strings forming the gradient.
        """
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
        """
        Build a label for a component of the series, either from the dataset's
        component labels or by default naming.

        Args:
            index (int): The index of the component.
            embedded (bool): Whether the component is part of a delay embedding.
            lag (int): The delay used for embedding, in sampled points.

        Returns:
            str: The label for the component, e.g. ``"x(t)"``, ``"x(t + 5)"``, or a dataset-provided label.
        """
        base = self.labels[0] if self.labels else "x"
        if embedded:
            if index == 0:
                return f"{base}(t)"
            return f"{base}(t + {index * lag})"
        if self.labels and index < len(self.labels):
            return self.labels[index]
        return f"x{index}"

    def plot(self, ax: plt.Axes | None = None) -> None:
        """
        Render the sample onto the renderer's axes.

        Args:
            ax (plt.Axes | None): Optional Matplotlib Axes to draw on. If not provided, the renderer's own axes are used.
        """
        if ax is not None:
            self.ax = ax
        logger.debug("[DatasetRenderer] Rendering sample as {} ({} points)", self.mode, len(self.series))
        channels = self._resolve_channels()
        if self.mode == "1d":
            self._plot_lines(channels)
        else:
            self._plot_trajectory(channels)
        self._setup_axes()
        plt.draw()

    def _resolve_channels(self) -> list[tuple[str, NDArray[np.float64]]]:
        """
        Run the transform (or the per-mode default) and validate its output.

        Returns:
            list[tuple[str, NDArray[np.float64]]]: The labelled coordinate arrays
            to plot -- one per line for ``"1d"``, or one per axis for ``"2d"``/``"3d"``.

        Raises:
            ValueError: If the transform yields the wrong number of channels for
                the mode, or trajectory coordinates of unequal length.
        """
        raw = list(self.transform(self.series)) if self.transform is not None else self._default_channels()
        channels = self._normalise_channels(raw)

        expected = _MODE_NDIM.get(self.mode)
        if expected is not None and len(channels) != expected:
            raise ValueError(
                f"a {self.mode} plot needs exactly {expected} coordinates, but the transform returned {len(channels)}."
            )
        if not channels:
            raise ValueError("the transform returned no channels to plot.")
        if expected is not None and len({len(values) for _, values in channels}) > 1:
            raise ValueError("the transform returned coordinates of unequal length.")
        return channels

    def _default_channels(self) -> list[tuple[str, NDArray[np.float64]]]:
        """
        Build the default channels for the current mode from the raw series.

        Returns:
            list[tuple[str, NDArray[np.float64]]]: Every component as a line for
            ``"1d"``; otherwise the first components (or a delay embedding of a
            one-dimensional series), one labelled array per axis.
        """
        d = self.series.shape[1]
        if self.mode == "1d":
            return [
                (self._component_label(i, embedded=False, lag=self.style.delay), self.series[:, i]) for i in range(d)
            ]
        dims = _MODE_NDIM[self.mode]
        min_components = _MIN_2D_COMPONENTS if self.mode == "2d" else _MIN_3D_COMPONENTS
        if d < min_components:
            coords = self._embed(dims)
            return [
                (self._component_label(i, embedded=True, lag=max(1, self.style.delay)), coords[i]) for i in range(dims)
            ]
        return [
            (self._component_label(i, embedded=False, lag=self.style.delay), self.series[:, i]) for i in range(dims)
        ]

    def _normalise_channels(self, raw: Sequence[Channel]) -> list[tuple[str, NDArray[np.float64]]]:
        """
        Coerce a transform's output to ``(label, values)`` pairs.

        Each item is either a ``(label, array)`` pair or a bare array, which is
        then labelled positionally from the dataset's component labels.

        Args:
            raw (Sequence[Channel]): The raw channels returned by a transform.

        Returns:
            list[tuple[str, NDArray[np.float64]]]: The labelled coordinate arrays.
        """
        channels: list[tuple[str, NDArray[np.float64]]] = []
        for i, item in enumerate(raw):
            if isinstance(item, tuple) and len(item) == _LABELLED_CHANNEL_LEN and isinstance(item[0], str):
                label, values = item
            else:
                values = cast("NDArray[np.float64]", item)
                label = self.labels[i] if self.labels and i < len(self.labels) else f"x{i}"
            channels.append((label, np.asarray(values, dtype=np.float64)))
        return channels

    def _plot_lines(self, channels: list[tuple[str, NDArray[np.float64]]]) -> None:
        """
        Plot a set of lines on the renderer's axes, one per channel.

        Args:
            channels (list[tuple[str, NDArray[np.float64]]]): The labelled coordinate arrays to plot, one per line.
        """
        style = self.style
        colors = self._gradient_colors(len(channels))
        line_style = dict(style.line_style)
        line_style.pop("color", None)
        for i, (label, values) in enumerate(channels):
            self.ax.plot(
                np.arange(len(values)),
                values,
                label=label,
                color=colors[i],
                marker=style.marker,
                markersize=style.marker_size,
                **line_style,
            )
        self._xlabel = style.xlabel or "step"
        self._ylabel = style.ylabel or "value"
        self._show_legend = len(channels) > 1 or self.labels is not None

    def _plot_trajectory(self, channels: list[tuple[str, NDArray[np.float64]]]) -> None:
        """
        Plot a trajectory on the renderer's axes, using the first two or three channels as coordinates.

        Args:
            channels (list[tuple[str, NDArray[np.float64]]]): The labelled coordinate arrays to plot, one per axis.
        """
        style = self.style
        self._xlabel = style.xlabel or channels[0][0]
        self._ylabel = style.ylabel or channels[1][0]
        if self.mode == "3d":
            self._zlabel = style.zlabel or channels[2][0]
        self._draw_coords(tuple(values for _, values in channels))
        self._show_legend = False

    def _draw_coords(self, coords: tuple[NDArray[np.float64], ...]) -> None:
        """
        Draw the trajectory on the axes, either as a line or a scatter plot.

        Args:
            coords (tuple[NDArray[np.float64], ...]): The coordinate arrays to plot, one per axis.
        """
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
        """
        Configure the axes with titles, labels, grid, and legend according to the style and theme.
        """
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

    def show(self) -> None:  # ruff: ignore[no-self-use]
        """Show the current figure."""
        plt.show()

    @staticmethod
    def _make_axes(mode: str, style: DatasetStyle) -> plt.Axes:
        """
        Create a new figure and axes appropriate for the requested plot mode.

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
