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

import pathlib
from typing import TYPE_CHECKING, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

DataArray: TypeAlias = NDArray[np.floating]
DimensionAxisArray: TypeAlias = NDArray[np.floating | np.integer]
Measurement: TypeAlias = tuple[DataArray, list["Dimension"]]


@yaml.register_class
class Dimension:
    """Stores dimension labels and their coordinate values."""

    def __init__(self, labels: list[str], values: list[DimensionAxisArray]) -> None:
        if len(labels) != len(values):
            raise ValueError("labels and values must have the same length")

        self.labels = labels
        self.values = values

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(labels={self.labels!r}, values={self.values!r})"


@yaml.register_class
class PulseExperimentResult(FunctionalResult):
    def __init__(self, measurements: list[Measurement]) -> None:
        self.measurements = measurements

    def __iter__(self) -> Iterator[Measurement]:
        """Iterate over measurements.

        Returns:
            Iterator[Measurement]: Iterator over tuples containing the data array and a list of dimension dictionaries.
        """
        return iter(self.measurements)

    def __len__(self) -> int:
        return len(self.measurements)

    def get(self, measurement: int = 0) -> Measurement:
        """Retrieves data and dimensions for a specified measurement.

        Args:
            measurement (int, optional): The index of the measurement. Defaults to 0.

        Returns:
            tuple[Measurement]: A tuple containing the data array and a list of dimension dictionaries.
        """

        return self.measurements[measurement]

    def plot_S21(self, measurement: int = 0, save_to: str | None = None) -> None:
        """Plots the S21 parameter from the experiment results.

        Args:
            measurement (int | str, optional): The index or name of the measurement. Defaults to 0.

        Raises:
            NotImplementedError: If the data has more than 2 dimensions.
        """

        def decibels(s21: np.ndarray) -> np.ndarray:
            """Convert result values from s21 into dB

            Returns:
                np.ndarray: The converted dB values.
            """
            return 20 * np.log10(np.abs(s21))

        def plot_1d(s21: np.ndarray, dims: list[Dimension]) -> None:
            """Plot 1d"""
            x_labels, x_values = dims[0].labels, dims[0].values

            fig, ax1 = plt.subplots()
            ax1.set_xlabel(x_labels[0])
            ax1.set_ylabel(r"$|S_{21}|$")
            ax1.plot(x_values[0], s21, ".")

            if len(x_labels) > 1:
                # Create secondary x-axis
                ax2 = ax1.twiny()

                # Set labels
                ax2.set_xlabel(x_labels[1])
                ax2.set_xlim(min(x_values[1]), max(x_values[1]))

                # Set tick locations
                ax2_ticks = np.linspace(min(x_values[1]), max(x_values[1]), num=6)
                ax2.set_xticks(ax2_ticks)

                # Force scientific notation
                ax2.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

            if save_to:
                fig.savefig(pathlib.Path(save_to))

            plt.show()

        # pylint: disable=too-many-locals
        def plot_2d(s21: np.ndarray, dims: list[Dimension]) -> None:
            """Plot 2d"""
            x_labels, x_values = dims[0].labels, dims[0].values
            y_labels, y_values = dims[1].labels, dims[1].values

            # Create x and y edge arrays by extrapolating the edges
            x_edges = np.linspace(x_values[0].min(), x_values[0].max(), len(x_values[0]) + 1)
            y_edges = np.linspace(y_values[0].min(), y_values[0].max(), len(y_values[0]) + 1)

            fig, ax1 = plt.subplots()
            ax1.set_xlabel(x_labels[0])
            ax1.set_ylabel(y_labels[0])

            # Force scientific notation
            ax1.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))

            mesh = ax1.pcolormesh(x_edges, y_edges, s21.T, cmap="viridis", shading="auto")
            fig.colorbar(mesh, ax=ax1)

            if len(x_labels) > 1:
                # Create secondary x-axis
                ax2 = ax1.twiny()

                # Set labels
                ax2.set_xlabel(x_labels[1])
                ax2.set_xlim(min(x_values[1]), max(x_values[1]))

                # Set tick locations
                ax2_ticks = np.linspace(min(x_values[1]), max(x_values[1]), num=6)
                ax2.set_xticks(ax2_ticks)

                # Force scientific notation
                ax2.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
            if len(y_labels) > 1:
                ax3 = ax1.twinx()
                ax3.set_ylabel(y_labels[1])
                ax3.set_ylim(min(y_values[1]), max(y_values[1]))

                # Set tick locations
                ax3_ticks = np.linspace(min(y_values[1]), max(y_values[1]), num=6)
                ax3.set_xticks(ax3_ticks)

                # Force scientific notation
                ax3.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

            if save_to:
                fig.savefig(pathlib.Path(save_to))

            plt.tight_layout()
            plt.show()

        data, dims = self.get(measurement=measurement)

        # Calculate S21
        s21 = data[..., 0] + 1j * data[..., 1]
        s21 = decibels(s21)

        n_dimensions = len(s21.shape)
        if n_dimensions == 1:
            plot_1d(s21, dims)
        elif n_dimensions == 2:
            plot_2d(s21, dims)
        else:
            raise NotImplementedError("3D and higher dimension plots are not supported yet.")
