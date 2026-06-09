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
from typing import Callable, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from qilisdk.functionals.functional_result import FunctionalResult
from qilisdk.yaml import yaml


@yaml.register_class
class Dimension:
    """Represents a labeled dimension in an experiment sweep.

    A `Dimension` defines one or more sweep parameters, such as drive
    amplitude, frequency, or delay time, together with their associated
    numerical values.
    """

    def __init__(self, labels: list[str], values: list[np.ndarray]) -> None:
        """Initialize a Dimension object.

        Args:
            labels (list[str]): Labels describing each dimension (e.g. ``["Drive amplitude"]``).
            values (list[np.ndarray]): Numerical arrays for the corresponding parameter values.
        """
        self.labels = labels
        self.values = values


class DimOverride:
    """Label and value transform override for a single plot dimension."""

    def __init__(self, label: str = "", transform: Callable = lambda x: x) -> None:
        """Initialize a DimOverride.

        Args:
            label (str): Axis label to use instead of the dimension's default.
            transform (Callable): Function applied to the dimension's values before plotting.
        """
        self.label = label
        self.transform = transform


@yaml.register_class
class ExperimentResult(FunctionalResult):
    """Base class for storing and visualizing experiment results.

    This class defines common utilities for handling experimental data,
    including computation of S21 parameters and automatic 1D or 2D plotting.
    Subclasses provide specific sweep parameters and plot titles.
    """

    plot_title: ClassVar[str]
    """Default plot title; subclasses provide the concrete label."""

    dims_override: ClassVar[dict[int, DimOverride]] = {}
    """Per-dimension label and value transform overrides; keyed by dimension index."""

    fit_by_default: ClassVar[bool] = False
    """Whether to perform fitting by default when plotting; can be overridden by subclasses if needed."""

    def __init__(self, qubit: int, data: np.ndarray, dims: list[Dimension]) -> None:
        """Initialize an experiment result.

        Args:
            qubit (int): The qubit index on which the experiment was performed.
            data (np.ndarray): Raw experimental data array.
            dims (list[Dimension]): Sweep dimensions of the experiment.
        """
        self.qubit = qubit
        self.data = data
        self.dims = dims

    @property
    def s21(self) -> np.ndarray:
        """Complex S21 transmission parameter.

        Returns:
            np.ndarray: The complex-valued S21 response computed as ``Re + i * Im``.
        """
        return self.data[..., 0] + 1j * self.data[..., 1]

    @property
    def s21_modulus(self) -> np.ndarray:
        """Magnitude of the S21 parameter.

        Returns:
            np.ndarray: The absolute value of the S21 parameter.
        """
        return np.abs(self.s21)

    @property
    def s21_db(self) -> np.ndarray:
        """Magnitude of S21 in decibels (dB).

        Returns:
            np.ndarray: ``20 * log10(abs(S21))`` expressed in dB.
        """
        return 20 * np.log10(self.s21_modulus)

    @staticmethod
    def add_fit(
        x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None, db: bool = False
    ) -> None:
        """
        Fit a user-provided function to the experimental data.

        This should be implemented by subclasses to provide specific fitting functionality relevant to the experiment type.

        Args:
            x_values (np.ndarray): The independent variable data (e.g., frequencies, drive durations).
            y_values (np.ndarray): The dependent variable data (e.g., measured signal).
            initial_guess (list[float] | None): Optional initial guess for the fit parameters. The specific parameters depend on the fit model used by the subclass.
            db (bool): Whether the data is in dB scale. This can affect how the fit is performed and plotted.
        """

    def _save_figure(self, figure: Figure, save_to: str | Path) -> None:
        """Save the figure to disk, handling both file and directory paths.

        Args:
            figure (Figure): The Matplotlib figure to save.
            save_to (str | Path): The path or directory where the figure should be saved.
        """
        save_to = Path(save_to)
        if save_to.is_dir():
            save_to /= f"{self.plot_title}_qubit{self.qubit}.png"
        save_to.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_to)

    def _plot_1d(
        self,
        s21: np.ndarray,
        dims: list[Dimension],
        db: bool = False,
        fit: bool = False,
        save_to: str | None = None,
        initial_guess: list[float] | None = None,
    ) -> None:
        """Plot 1D S21 data.

        Args:
            s21 (np.ndarray): The S21 data to plot, either in linear or dB scale.
            dims (list[Dimension]): The dimensions of the experiment, used for labeling axes.
            db (bool): Whether the data is in dB scale, affecting axis labels and fit behavior.
            fit (bool): Whether to perform and plot the fit using the `add_fit` method.
            save_to (str | None): Optional path or directory to save the figure.
            initial_guess (list[float] | None): Optional initial guess passed to `add_fit`.
        """
        x_labels, x_values = dims[0].labels, dims[0].values
        x_override = self.dims_override.get(0)
        y_override = self.dims_override.get(1)

        fig, ax1 = plt.subplots()
        ax1.set_title(f"{self.plot_title} - Qubit {self.qubit}")
        ax1.set_xlabel(x_override.label if x_override and x_override.label else x_labels[0])
        default_y_label = r"$|S_{21}|$ ∝ Voltage" if not db else r"$|S_{21}|$ (dB)"
        ax1.set_ylabel(y_override.label if y_override and y_override.label else default_y_label)
        x_values = [x_override.transform(x) for x in x_values] if x_override else x_values
        ax1.plot(x_values[0], s21, ".")

        if len(x_labels) > 1:
            ax2 = ax1.twiny()
            ax2.set_xlabel(x_labels[1])
            ax2.set_xlim(min(x_values[1]), max(x_values[1]))
            ax2.set_xticks(np.linspace(min(x_values[1]), max(x_values[1]), num=6))
            ax2.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        if fit:
            self.add_fit(x_values[0], s21, initial_guess=initial_guess, db=db)

        if save_to:
            self._save_figure(fig, save_to)

        plt.show()
        plt.close(fig)

    def _plot_2d(
        self,
        s21: np.ndarray,
        dims: list[Dimension],
        db: bool = False,
        fit: bool = False,
        save_to: str | None = None,
        initial_guess: list[float] | None = None,
    ) -> None:
        """Plot 2D S21 data as a color mesh.

        Args:
            s21 (np.ndarray): The 2D S21 data to plot, either in linear or dB scale.
            dims (list[Dimension]): The dimensions of the experiment, used for labeling axes.
            db (bool): Whether the data is in dB scale, affecting axis labels and fit behavior.
            fit (bool): Whether to perform and plot the fit using the `add_fit` method.
            save_to (str | None): Optional path or directory to save the figure.
            initial_guess (list[float] | None): Optional initial guess passed to `add_fit`.
        """
        x_labels, x_values = dims[0].labels, dims[0].values
        y_labels, y_values = dims[1].labels, dims[1].values
        x_override = self.dims_override.get(0)
        y_override = self.dims_override.get(1)
        z_override = self.dims_override.get(2)

        x_values = [x_override.transform(x) for x in x_values] if x_override else x_values
        y_values = [y_override.transform(y) for y in y_values] if y_override else y_values

        x_edges = np.linspace(x_values[0].min(), x_values[0].max(), len(x_values[0]) + 1)
        y_edges = np.linspace(y_values[0].min(), y_values[0].max(), len(y_values[0]) + 1)

        fig, ax1 = plt.subplots()
        ax1.set_title(f"{self.plot_title} - Qubit {self.qubit}")
        ax1.set_xlabel(x_override.label if x_override and x_override.label else x_labels[0])
        ax1.set_ylabel(y_override.label if y_override and y_override.label else y_labels[0])
        ax1.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))

        mesh = ax1.pcolormesh(x_edges, y_edges, s21.T, cmap="viridis", shading="auto")
        z_label = z_override.label if z_override and z_override.label else r"$|S_{21}|$"
        colorbar_label = z_label + " ∝ Voltage" if not db else z_label + " (dB)"
        fig.colorbar(mesh, ax=ax1, label=colorbar_label)

        if len(x_labels) > 1:
            ax2 = ax1.twiny()
            ax2.set_xlabel(x_labels[1])
            ax2.set_xlim(min(x_values[1]), max(x_values[1]))
            ax2.set_xticks(np.linspace(min(x_values[1]), max(x_values[1]), num=6))
            ax2.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

        if len(y_labels) > 1:
            ax3 = ax1.twinx()
            ax3.set_ylabel(y_labels[1])
            ax3.set_ylim(min(y_values[1]), max(y_values[1]))
            ax3.set_yticks(np.linspace(min(y_values[1]), max(y_values[1]), num=6))
            ax3.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        if fit:
            self.add_fit(x_values[0], s21, initial_guess=initial_guess, db=db)

        if save_to:
            self._save_figure(fig, save_to)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def plot(
        self,
        save_to: str | None = None,
        initial_guess: list[float] | None = None,
        db: bool = False,
        fit: bool | None = None,
    ) -> None:
        """Plot the S21 parameter from experiment results.

        Automatically detects whether the dataset is 1D or 2D and creates
        the appropriate figure. Optionally saves the figure to disk.

        Args:
            save_to (str | None): Optional path or directory to save the
                generated plot. If a directory is provided, the filename is
                automatically generated as ``{plot_title}_qubit{qubit}.png``.
            initial_guess (list[float] | None): Optional initial guess for the fit parameters, passed to the `add_fit` method.
            db (bool): Whether to plot the S21 data in dB scale.
            fit (bool | None): Whether to perform and plot the fit using the `add_fit` method. If None, the class-level `fit_by_default` is used.

        Raises:
            NotImplementedError: If the experiment data has more than 2 dimensions.
        """
        to_plot = self.s21_db if db else self.s21_modulus
        n_dimensions = len(to_plot.shape)
        should_fit = fit if fit is not None else self.fit_by_default
        if n_dimensions == 1:
            self._plot_1d(to_plot, self.dims, db=db, fit=should_fit, save_to=save_to, initial_guess=initial_guess)
        elif n_dimensions == 2:  # noqa: PLR2004
            self._plot_2d(to_plot, self.dims, db=db, fit=should_fit, save_to=save_to, initial_guess=initial_guess)
        else:
            raise NotImplementedError("3D and higher dimension plots are not supported yet.")


@yaml.register_class
class RabiExperimentResult(ExperimentResult):
    """Result container for Rabi experiments."""

    plot_title: ClassVar[str] = "Rabi"
    """Default title for Rabi experiment plots."""

    @staticmethod
    def add_fit(
        x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None, db: bool = False
    ) -> None:
        """
        Fit a sinusoidal curve to the Rabi experiment data.

        Args:
            x_values (np.ndarray): The independent variable data (e.g., drive durations in nanoseconds).
            y_values (np.ndarray): The dependent variable data (e.g., measured signal).
            initial_guess (list[float] | None): Optional initial guess for the fit parameters [a, f_rabi, phi, b].
                If None, a default guess is generated based on the data.
            db (bool): Whether the data is in dB scale. Non-finite dB values (e.g. -inf at zero crossings)
                are excluded from the fit.
        """

        def _rabi_model(t: np.ndarray, a: float, f_rabi: float, phi: float, b: float) -> np.ndarray:
            return a * np.cos(2 * np.pi * f_rabi * t + phi) + b

        y_linear = 10 ** (y_values / 20) if db else y_values

        if initial_guess is None:
            amplitude_guess = (y_linear.max() - y_linear.min()) / 2
            baseline_guess = y_linear.mean()
            fft_freqs = np.fft.rfftfreq(len(x_values), d=(x_values[1] - x_values[0]))
            fft_magnitudes = np.abs(np.fft.rfft(y_linear - baseline_guess))
            fft_magnitudes[0] = 0
            freq_guess = float(fft_freqs[np.argmax(fft_magnitudes)])
            initial_guess = [amplitude_guess, freq_guess, 0.0, baseline_guess]

        popt, _ = curve_fit(_rabi_model, x_values, y_linear, p0=initial_guess)
        a_fit, f_rabi_fit, phi_fit, b_fit = popt
        t_fit = np.linspace(min(x_values), max(x_values), 1000)
        y_fit = _rabi_model(t_fit, a_fit, f_rabi_fit, phi_fit, b_fit)
        if db:
            y_fit = 20 * np.log10(np.abs(y_fit))
        plt.plot(t_fit, y_fit, label=f"Fit: f_rabi={f_rabi_fit * 1e3:.2f} MHz")
        plt.legend()


@yaml.register_class
class T1ExperimentResult(ExperimentResult):
    """Result container for T1 relaxation experiments."""

    plot_title: ClassVar[str] = "T1"
    """Default title for T1 experiment plots."""

    dims_override: ClassVar[dict[int, DimOverride]] = {
        0: DimOverride(label=r"Time ($\mu$s)", transform=lambda x: x * 1e-3),
    }
    """Override x-axis to display in microseconds with appropriate label."""

    fit_by_default: ClassVar[bool] = True
    """T1 experiments typically benefit from fitting, so we set this to True by default."""

    @staticmethod
    def add_fit(
        x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None, db: bool = False
    ) -> None:
        """
        Fit an exponential decay curve to the T1 experiment data.

        Args:
            x_values (list[np.ndarray]): The independent variable data (e.g., delay times).
            y_values (np.ndarray): The dependent variable data (e.g., measured signal).
            initial_guess (list[float] | None): Optional initial guess for the fit parameters [a, t1, b]. If None, a default guess is generated based on the data.
            db (bool): Whether to plot the fit in dB scale.
        """

        def _t1_decay_model(t: np.ndarray, a: float, t1: float, b: float) -> np.ndarray:
            """Exponential decay model for T1 measurement.

            Args:
                t (np.ndarray): Time array (in microseconds).
                a (float): Amplitude of the decay.
                t1 (float): T1 relaxation time (in microseconds).
                b (float): Baseline offset.

            Returns:
                np.ndarray: The modeled decay curve values at time t.
            """
            return a * np.exp(-t / t1) + b

        y_linear = 10 ** (y_values / 20) if db else y_values

        if initial_guess is None:
            initial_guess = [y_linear.max() - y_linear.min(), (x_values.max() - x_values.min()) / 3, y_linear.min()]

        popt, pcov = curve_fit(_t1_decay_model, x_values, y_linear, p0=initial_guess)
        a_fit, t1_fit, b_fit = popt
        t1_err = np.sqrt(np.diag(pcov))[1]
        t_fit = np.linspace(min(x_values), max(x_values), 100)
        y_fit = _t1_decay_model(t_fit, a_fit, t1_fit, b_fit)
        if db:
            y_fit = 20 * np.log10(np.abs(y_fit))
        plt.plot(t_fit, y_fit, label=f"Fit: T1={t1_fit:.2f} ± {t1_err:.2f} μs")
        plt.legend()


@yaml.register_class
class T2ExperimentResult(ExperimentResult):
    """Result container for T2 dephasing experiments."""

    plot_title: ClassVar[str] = "T2"
    """Default title for T2 experiment plots."""

    @staticmethod
    def add_fit(
        x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None, db: bool = False
    ) -> None:
        """
        Fit a decaying sinusoid curve to the T2 experiment data.

        Args:
            x_values (list[np.ndarray]): The independent variable data (e.g., delay times).
            y_values (np.ndarray): The dependent variable data (e.g., measured signal).
            initial_guess (list[float] | None): Optional initial guess for the fit parameters [a, t2, f_detune, phi, b]. If None, a default guess is generated based on the data.
            db (bool): Whether to plot the fit in dB scale.
        """

        def _t2_decay_model(t: np.ndarray, a: float, t2: float, f_detune: float, phi: float, b: float) -> np.ndarray:
            """Decaying sinusoid model for T2 measurement.

            Args:
                t (np.ndarray): Time array (in microseconds).
                a (float): Amplitude of the decay.
                t2 (float): T2 dephasing time (in microseconds).
                f_detune (float): Detuning frequency of the oscillations (in MHz).
                phi (float): Phase offset of the oscillations (in radians).
                b (float): Baseline offset.
            Returns:
                np.ndarray: The modeled decaying sinusoid values at time t.
            """
            return a * np.exp(-t / t2) * np.cos(2 * np.pi * f_detune * t + phi) + b

        y_linear = 10 ** (y_values / 20) if db else y_values

        if initial_guess is None:
            initial_guess = [
                y_linear.max() - y_linear.min(),
                (x_values.max() - x_values.min()) / 3,
                1.0,
                0.0,
                y_linear.min(),
            ]
        popt, _ = curve_fit(_t2_decay_model, x_values, y_linear, p0=initial_guess)
        a_fit, t2_fit, f_detune_fit, phi_fit, b_fit = popt
        t_fit = np.linspace(min(x_values), max(x_values), 100)
        y_fit = _t2_decay_model(t_fit, a_fit, t2_fit, f_detune_fit, phi_fit, b_fit)
        if db:
            y_fit = 20 * np.log10(np.abs(y_fit))
        plt.plot(t_fit, y_fit, label=f"Fit: T2={t2_fit:.2f} ns, f_detune={f_detune_fit:.2f} MHz")
        plt.legend()


@yaml.register_class
class TwoTonesAtFluxBiasExperimentResult(ExperimentResult):
    """Result container for TwoTones experiments."""

    plot_title: ClassVar[str] = "Two Tones At Flux Bias"
    """Default title for TwoTones at flux bias experiment plots."""

    dims_override: ClassVar[dict[int, DimOverride]] = {
        0: DimOverride(label=r"Frequency (GHz)", transform=lambda x: x * 1e-9),
    }
    """Override x-axis to display in GHz with appropriate label."""

    @staticmethod
    def add_fit(
        x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None, db: bool = False
    ) -> None:
        """
        Fit a Lorentzian curve to the TwoTones at flux bias experiment data.

        Args:
            x_values (list[np.ndarray]): The independent variable data (e.g., frequencies).
            y_values (np.ndarray): The dependent variable data (e.g., measured signal).
            initial_guess (list[float] | None): Optional initial guess for the fit parameters [f0, gamma, A, B]. If None, a default guess is generated based on the data.
            db (bool): Whether to plot the fit in dB scale.
        """

        def _lorentzian(f: np.ndarray, f0: float, gamma: float, a: float, b: float) -> np.ndarray:
            """Lorentzian model for TwoTones at flux bias measurement.

            Args:
                f (np.ndarray): Frequency array (in GHz).
                f0 (float): Resonance frequency (in GHz).
                gamma (float): Full width at half maximum (FWHM) of the resonance (in GHz).
                a (float): Amplitude of the Lorentzian peak.
                b (float): Baseline offset.

            Returns:
                np.ndarray: The modeled Lorentzian curve values at frequency f.
            """
            return b + a / (1 + ((f - f0) / (gamma / 2)) ** 2)

        y_linear = 10 ** (y_values / 20) if db else y_values

        if initial_guess is None:
            initial_guess = [
                float(x_values[np.argmax(y_linear)]),
                float((x_values.max() - x_values.min()) / 20),
                float(y_linear.max() - y_linear.min()),
                float(y_linear.min()),
            ]
        popt, _ = curve_fit(_lorentzian, x_values, y_linear, p0=initial_guess)
        f0_fit, gamma_fit, a_fit, b_fit = popt
        f_fit = np.linspace(min(x_values), max(x_values), 1000)
        y_fit = _lorentzian(f_fit, f0_fit, gamma_fit, a_fit, b_fit)
        if db:
            y_fit = 20 * np.log10(np.abs(y_fit))
        plt.plot(f_fit, y_fit, label=f"Fit: f0={f0_fit:.2f} GHz, gamma={gamma_fit:.2f} GHz")
        plt.legend()


@yaml.register_class
class TwoTonesVsFluxBiasExperimentResult(ExperimentResult):
    """Result container for TwoTones experiments swept vs flux bias."""

    plot_title: ClassVar[str] = "Two Tones Vs Flux Bias"
    """Default title for TwoTones vs flux bias experiment plots."""

    dims_override: ClassVar[dict[int, DimOverride]] = {
        0: DimOverride(label=r"$\Phi_z~(\Phi_0)$"),
        1: DimOverride(label=r"LO Frequency (GHz)", transform=lambda y: y * 1e-9),
    }
    """Override axis labels and convert y-axis from Hz to GHz."""
