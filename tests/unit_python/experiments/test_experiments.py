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

from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
from loguru import logger

from qilisdk.experiments import Dimension, ExperimentResult

_FIT_WARNING = (
    "[ExperimentResult] Fitting is only implemented for amplitude plots. Ignoring fit request for non-amplitude plot."
)


class RecordingResult(ExperimentResult):
    """Experiment result that records every `add_fit` call, standing in for a backend-defined result."""

    plot_title = "recording_experiment"

    fit_calls: ClassVar[list[tuple[np.ndarray, np.ndarray, list[float] | None]]] = []

    @staticmethod
    def add_fit(x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None) -> None:
        RecordingResult.fit_calls.append((x_values, y_values, initial_guess))


class FitByDefaultResult(RecordingResult):
    """Experiment result that fits without being asked to."""

    plot_title = "fit_by_default_experiment"
    fit_by_default = True


def _relabel(label: str):
    """Build a dims_override callable that renames a dimension."""
    return lambda dim: Dimension(labels=[label, *dim.labels[1:]], values=dim.values)


class OverriddenResult(RecordingResult):
    """Experiment result overriding every plotted dimension label."""

    plot_title = "overridden_experiment"
    dims_override: ClassVar[list] = [_relabel("x override"), _relabel("y override"), _relabel("z override")]


@pytest.fixture(autouse=True)
def _clear_fit_calls():
    RecordingResult.fit_calls.clear()


@pytest.fixture
def captured_figures(monkeypatch):
    """Capture the figure of every `plt.show()` call instead of displaying it."""
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    return figures


def _data_1d(n: int = 8) -> np.ndarray:
    real = np.linspace(0.1, 1.0, n)
    imag = np.linspace(-0.5, 0.5, n)
    return np.stack([real, imag], axis=-1)


def _data_2d(n: int = 5, m: int = 4) -> np.ndarray:
    real = np.linspace(0.1, 1.0, n * m).reshape(n, m)
    imag = np.linspace(-0.5, 0.5, n * m).reshape(n, m)
    return np.stack([real, imag], axis=-1)


def test_dimension_initialization():
    dim = Dimension(labels=["Drive amplitude"], values=[np.array([0.1, 0.2, 0.3])])
    assert dim.labels == ["Drive amplitude"]
    assert np.array_equal(dim.values[0], np.array([0.1, 0.2, 0.3]))


def test_dimension_printing():
    dim = Dimension(labels=["Freq"], values=[np.array([1, 2])])
    assert repr(dim) == "Dimension(labels=['Freq'], values=[array([1, 2])])"


def test_experiment_result_init():
    data = np.array([[1, 2], [3, 4]])
    qubit = 0
    averages = 1000
    dims = [Dimension(labels=["Freq"], values=[np.array([1, 2])])]

    exp_result = ExperimentResult(qubit=qubit, averages=averages, data=data, dims=dims)

    assert exp_result.qubit == qubit
    assert exp_result.averages == averages
    assert np.array_equal(exp_result.data, data)
    assert exp_result.dims == dims


def test_experiment_s21_computation():
    data = np.array([[1, 2], [3, 4]])

    exp_result = ExperimentResult(qubit=0, averages=1000, data=data, dims=[])

    s21 = exp_result.s21
    expected_s21 = np.array([1 + 2j, 3 + 4j])
    assert np.allclose(s21, expected_s21)

    s21_modulus = exp_result.s21_modulus
    expected_modulus = np.abs(expected_s21)
    assert np.allclose(s21_modulus, expected_modulus)

    s21_db = exp_result.s21_db
    expected_db = 20 * np.log10(expected_modulus)
    assert np.allclose(s21_db, expected_db)

    s21_phase = exp_result.s21_phase
    expected_phase = np.unwrap(np.angle(expected_s21))
    assert np.allclose(s21_phase, expected_phase)


def test_add_fit_is_a_no_op_on_the_base_class():
    x_values = np.array([1.0, 2.0])
    y_values = np.array([0.1, 0.2])

    assert ExperimentResult.add_fit(x_values, y_values) is None
    assert ExperimentResult.add_fit(x_values, y_values, initial_guess=[1.0]) is None


def test_plot_1d_defaults(captured_figures):
    dims = [Dimension(labels=["Drive duration (ns)"], values=[np.arange(8)])]
    result = RecordingResult(qubit=2, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    assert len(captured_figures) == 1
    axes = captured_figures[0].axes[0]
    assert axes.get_title() == "recording_experiment - Qubit 2"
    assert axes.get_xlabel() == "Drive duration (ns)"
    assert axes.get_ylabel() == "Amplitude (V)"
    # `fit_by_default` is False, so no fit is attempted unless requested.
    assert RecordingResult.fit_calls == []


def test_plot_1d_fit_and_connect_points(captured_figures):
    x_values = np.arange(8)
    dims = [Dimension(labels=["Drive duration (ns)"], values=[x_values])]
    data = _data_1d()
    result = RecordingResult(qubit=0, averages=1000, data=data, dims=dims)

    result.plot(fit=True, connect_points=True, initial_guess=[1.0, 2.0])

    assert len(RecordingResult.fit_calls) == 1
    fitted_x, fitted_y, initial_guess = RecordingResult.fit_calls[0]
    assert np.array_equal(fitted_x, x_values)
    assert np.allclose(fitted_y, np.abs(data[..., 0] + 1j * data[..., 1]))
    assert initial_guess == [1.0, 2.0]
    # The dashed connecting line is drawn in addition to the markers.
    assert len(captured_figures[0].axes[0].lines) == 2


def test_plot_1d_fit_by_default(captured_figures):
    dims = [Dimension(labels=["Wait duration (ns)"], values=[np.arange(8)])]
    result = FitByDefaultResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    assert len(RecordingResult.fit_calls) == 1
    assert len(captured_figures[0].axes[0].lines) == 1


@pytest.mark.parametrize(
    ("plot_type", "expected_label"),
    [("amplitude", "Amplitude (V)"), ("phase", "Phase (rad)"), ("db", "Amplitude (dB)")],
)
def test_plot_1d_plot_types(captured_figures, plot_type, expected_label):
    dims = [Dimension(labels=["Frequency (Hz)"], values=[np.arange(8)])]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot(plot_type=plot_type)

    assert captured_figures[0].axes[0].get_ylabel() == expected_label


def test_plot_1d_fit_on_non_amplitude_plot_warns(monkeypatch, captured_figures):
    warnings = []
    monkeypatch.setattr(logger, "warning", warnings.append)
    dims = [Dimension(labels=["Frequency (Hz)"], values=[np.arange(8)])]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot(fit=True, plot_type="phase")

    assert warnings == [_FIT_WARNING]
    assert RecordingResult.fit_calls == []
    assert len(captured_figures) == 1


def test_plot_1d_secondary_x_axis(captured_figures):
    dims = [
        Dimension(
            labels=["Frequency (Hz)", "Flux bias (V)"],
            values=[np.linspace(4.0e9, 5.0e9, 8), np.linspace(-1.0, 1.0, 8)],
        )
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    figure = captured_figures[0]
    assert len(figure.axes) == 2
    assert figure.axes[1].get_xlabel() == "Flux bias (V)"


def test_plot_1d_dimension_overrides(captured_figures):
    dims = [Dimension(labels=["Frequency (Hz)"], values=[np.arange(8)])]
    result = OverriddenResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    axes = captured_figures[0].axes[0]
    assert axes.get_xlabel() == "x override"
    assert axes.get_ylabel() == "y override"


def test_plot_1d_y_override_skipped_for_non_amplitude(captured_figures):
    dims = [Dimension(labels=["Frequency (Hz)"], values=[np.arange(8)])]
    result = OverriddenResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot(plot_type="db")

    axes = captured_figures[0].axes[0]
    assert axes.get_xlabel() == "x override"
    assert axes.get_ylabel() == "Amplitude (dB)"


def test_plot_2d_defaults(captured_figures):
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[np.linspace(-0.5, 0.5, 5)]),
        Dimension(labels=["Frequency (Hz)"], values=[np.linspace(4.0e9, 5.0e9, 4)]),
    ]
    result = RecordingResult(qubit=1, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    figure = captured_figures[0]
    axes = figure.axes[0]
    assert axes.get_title() == "recording_experiment - Qubit 1"
    assert axes.get_xlabel() == "Flux bias (V)"
    assert axes.get_ylabel() == "Frequency (Hz)"
    # The second axes is the colorbar.
    assert figure.axes[1].get_ylabel() == "Amplitude (V)"


def test_plot_2d_dimension_overrides(captured_figures):
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[np.linspace(-0.5, 0.5, 5)]),
        Dimension(labels=["Frequency (Hz)"], values=[np.linspace(4.0e9, 5.0e9, 4)]),
    ]
    result = OverriddenResult(qubit=0, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    figure = captured_figures[0]
    assert figure.axes[0].get_xlabel() == "x override"
    assert figure.axes[0].get_ylabel() == "y override"
    assert figure.axes[1].get_ylabel() == "z override"


def test_plot_2d_z_override_skipped_for_non_amplitude(captured_figures):
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[np.linspace(-0.5, 0.5, 5)]),
        Dimension(labels=["Frequency (Hz)"], values=[np.linspace(4.0e9, 5.0e9, 4)]),
    ]
    result = OverriddenResult(qubit=0, averages=1000, data=_data_2d(), dims=dims)

    result.plot(plot_type="phase")

    assert captured_figures[0].axes[1].get_ylabel() == "Phase (rad)"


def test_plot_2d_secondary_axes(captured_figures):
    dims = [
        Dimension(
            labels=["Flux bias (V)", "Flux current (A)"],
            values=[np.linspace(-0.5, 0.5, 5), np.linspace(-1.0, 1.0, 5)],
        ),
        Dimension(
            labels=["Frequency (Hz)", "IF frequency (Hz)"],
            values=[np.linspace(4.0e9, 5.0e9, 4), np.linspace(1.0e8, 2.0e8, 4)],
        ),
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    labels = {axes.get_xlabel() for axes in captured_figures[0].axes} | {
        axes.get_ylabel() for axes in captured_figures[0].axes
    }
    assert "Flux current (A)" in labels
    assert "IF frequency (Hz)" in labels


def test_plot_3d_is_not_supported(captured_figures):
    data = np.ones((2, 2, 2, 2))
    dims = [
        Dimension(labels=["Dim1"], values=[np.array([1, 2])]),
        Dimension(labels=["Dim2"], values=[np.array([3, 4])]),
        Dimension(labels=["Dim3"], values=[np.array([5, 6])]),
    ]
    result = RecordingResult(qubit=0, averages=1000, data=data, dims=dims)

    with pytest.raises(NotImplementedError, match="3D and higher"):
        result.plot()

    assert captured_figures == []


def test_plot_saves_to_file(tmp_path, captured_figures):
    dims = [Dimension(labels=["Frequency (Hz)"], values=[np.arange(8)])]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)
    save_to = tmp_path / "nested" / "figure.png"

    result.plot(save_to=str(save_to))

    assert save_to.is_file()
    assert len(captured_figures) == 1


def test_plot_saves_to_directory_using_the_plot_title(tmp_path, captured_figures):
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[np.linspace(-0.5, 0.5, 5)]),
        Dimension(labels=["Frequency (Hz)"], values=[np.linspace(4.0e9, 5.0e9, 4)]),
    ]
    result = RecordingResult(qubit=4, averages=1000, data=_data_2d(), dims=dims)

    result.plot(save_to=str(tmp_path))

    assert (tmp_path / "recording_experiment_qubit4.png").is_file()
    assert len(captured_figures) == 1


def test_experiment_printing():
    data = np.array([[1, 2], [3, 4]])
    qubit = 0
    averages = 1000
    dims = [Dimension(labels=["Freq"], values=[np.array([1, 2])])]

    exp_result = ExperimentResult(qubit=qubit, averages=averages, data=data, dims=dims)

    expected_str = (
        "ExperimentResult(qubit=0, averages=1000, data=[[1 2]\n [3 4]], "
        "dims=[Dimension(labels=['Freq'], values=[array([1, 2])])])"
    )
    assert str(exp_result) == expected_str
