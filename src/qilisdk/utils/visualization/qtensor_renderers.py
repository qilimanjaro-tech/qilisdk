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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d

from qilisdk.utils.visualization.style import QTensorStyle

if TYPE_CHECKING:
    from qilisdk.core import QTensor


###############################################################################
# Matplotlib implementation
###############################################################################

# the type of axes is a 3d axes:
AXIS_TYPE = mplot3d.axes3d.Axes3D


class MatplotlibQTensorRenderer:
    """Render a :class:`~qilisdk.core.QTensor` using *matplotlib*."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, qtensor: QTensor, ax: AXIS_TYPE | None = None, *, style: QTensorStyle = QTensorStyle()) -> None:
        self.qtensor = qtensor
        self.style = style
        self._ax = ax or self._make_axes(style.dpi)

    @property
    def axes(self) -> AXIS_TYPE:
        return self._ax

    def plot(self) -> None:
        """
        Render the QTensor on the current axes.

        Raises:
            ValueError: If the QTensor is not a single-qubit state vector (ket or bra).
            ValueError: If the QTensor has more than one qubit.
        """

        if not self.qtensor.is_ket() and not self.qtensor.is_bra():
            raise ValueError("Drawing is only supported for state vectors (kets or bras)")
        if self.qtensor.nqubits != 1:
            raise ValueError(
                "Drawing is only supported for single-qubit states: consider using .partial_trace([i]) to reduce to a single qubit i"
            )

        logger.debug("[QTensorRenderer] Rendering single-qubit state on Bloch sphere")

        # Get the values from the style
        sphere_points = self.style.sphere_points
        sphere_color = self.style.sphere_color
        arrow_color = self.style.arrow_color
        arrow_length_ratio = self.style.arrow_length_ratio
        reference_point_distance = self.style.reference_point_distance
        draw_reference_points = self.style.draw_reference_points
        font_size = self.style.fontsize
        rotation_style = self.style.rotation_style
        figure_title = self.style.title or "Bloch Sphere Visualization"
        draw_center_circle = self.style.draw_center_circle

        # Better mouse rotation style
        mpl.rcParams["axes3d.mouserotationstyle"] = rotation_style

        # Set up the plot
        ax = self.axes
        u = np.linspace(0, 2 * np.pi, sphere_points)
        v = np.linspace(0, np.pi, sphere_points)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color=sphere_color, alpha=0.1)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(figure_title)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))

        # Fix the aspect ratio to be equal
        ax.set_box_aspect([1, 1, 1])

        # Draw the arrow
        coeffs = self.qtensor.dense().flatten()
        x = 2 * np.real(coeffs[0] * np.conj(coeffs[1]))
        y = 2 * np.imag(coeffs[0] * np.conj(coeffs[1]))
        z = np.abs(coeffs[0]) ** 2 - np.abs(coeffs[1]) ** 2
        ax.quiver(0, 0, 0, x, y, z, color=arrow_color, arrow_length_ratio=arrow_length_ratio)

        # Draw some key points for reference
        if draw_reference_points:
            ax.text(0, 0, reference_point_distance, "|0⟩", fontsize=font_size, ha="center")
            ax.text(0, 0, -reference_point_distance, "|1⟩", fontsize=font_size, ha="center")
            ax.text(reference_point_distance, 0, 0, "|+⟩", fontsize=font_size, ha="center")
            ax.text(-reference_point_distance, 0, 0, "|-⟩", fontsize=font_size, ha="center")
            ax.text(0, reference_point_distance, 0, "|+i⟩", fontsize=font_size, ha="center")
            ax.text(0, -reference_point_distance, 0, "|-i⟩", fontsize=font_size, ha="center")

        # Draw a circle around the centre
        if draw_center_circle:
            u = np.linspace(0, 2 * np.pi, sphere_points)
            radius = 1.0
            x = radius * np.cos(u)
            y = radius * np.sin(u)
            z = np.zeros_like(x)
            ax.plot(x, y, z, color="b", linestyle="--", alpha=0.5)

        # Hide the axes
        ax.set_axis_off()

        # Draw the plot
        plt.draw()

    def save(self, filename: str) -> None:
        """Save current figure to disk.

        Args:
            filename: Path to save the figure (e.g., 'circuit.png').
        """
        logger.debug("[QTensorRenderer] Saving figure to {}", filename)
        if isinstance(self.axes.figure, Figure):
            self.axes.figure.savefig(filename, bbox_inches="tight")

    def show(self) -> None:  # noqa: PLR6301
        """Show the current figure."""
        plt.show()

    @staticmethod
    def _make_axes(dpi: int) -> AXIS_TYPE:
        """
        Create a new figure and axes with the given DPI.

        Args:
            dpi (int): The DPI of the figure

        Returns:
            A newly created Matplotlib Axes.

        Raises:
            TypeError: If the created axes is not of the expected type.
        """
        _, ax = plt.subplots(dpi=dpi, constrained_layout=True, subplot_kw={"projection": "3d"})
        if not isinstance(ax, AXIS_TYPE):
            raise TypeError(f"Expected axes of type {AXIS_TYPE}, but got {type(ax)}")
        return ax
