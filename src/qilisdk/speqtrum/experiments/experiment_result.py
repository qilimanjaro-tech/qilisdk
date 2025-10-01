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
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class Dimension:
    def __init__(self, labels: list[str], values: list[np.ndarray]) -> None:
        self.labels = labels
        self.values = values


@yaml.register_class
class ExperimentResult(FunctionalResult):
    plot_title: ClassVar[str]

    def __init__(self, qubit: int, data: np.ndarray, dims: list[Dimension]) -> None:
        self.qubit = qubit
        self.data = data
        self.dims = dims

    @property
    def s21(self) -> np.ndarray:
        return self.data[..., 0] + 1j * self.data[..., 1]

    @property
    def s21_modulus(self) -> np.ndarray:
        return np.abs(self.s21)

    @property
    def s21_db(self) -> np.ndarray:
        return 20 * np.log10(self.s21_modulus)

    def plot(self, save_to: str | None = None) -> None:
        """Plots the S21 parameter from the experiment results.

        Args:
            save_to (str, optional): If given, save the plot the the specified path. Defaults to None.

        Raises:
            NotImplementedError: If the data has more than 2 dimensions.
        """

        def save_figure(figure: Figure, save_to: str | Path) -> None:
            save_to = Path(save_to)

            # If a directory was given, append the default filename
            if save_to.is_dir():
                save_to /= f"{self.plot_title}_qubit{self.qubit}.png"

            save_to.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_to)

        def plot_1d(s21: np.ndarray, dims: list[Dimension]) -> None:
            """Plot 1d"""
            x_labels, x_values = dims[0].labels, dims[0].values

            fig, ax1 = plt.subplots()
            ax1.set_title(f"{self.plot_title} - Qubit {self.qubit}")
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
                save_figure(fig, save_to)

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
            ax1.set_title(f"{self.plot_title} - Qubit {self.qubit}")
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
                save_figure(fig, save_to)

            plt.tight_layout()
            plt.show()

        n_dimensions = len(self.s21_modulus.shape)
        if n_dimensions == 1:
            plot_1d(self.s21_modulus, self.dims)
        elif n_dimensions == 2:  # noqa: PLR2004
            plot_2d(self.s21_modulus, self.dims)
        else:
            raise NotImplementedError("3D and higher dimension plots are not supported yet.")


@yaml.register_class
class RabiExperimentResult(ExperimentResult):
    plot_title: ClassVar[str] = "Rabi"


@yaml.register_class
class T1ExperimentResult(ExperimentResult):
    plot_title: ClassVar[str] = "T1"
