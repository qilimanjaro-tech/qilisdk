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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

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


class WideFitResult(ExperimentResult):
    """Result whose `add_fit` draws through pyplot, over a grid wider than the measured points.

    Both are what a real fit does: the signature hands the implementer no axes to draw on, and a
    fitted curve is normally evaluated on a denser grid that reaches a little past the data.
    """

    plot_title = "wide_fit_experiment"

    @staticmethod
    def add_fit(x_values: np.ndarray, y_values: np.ndarray, initial_guess: list[float] | None = None) -> None:
        span = np.ptp(x_values)
        wider = np.linspace(x_values.min() - 0.1 * span, x_values.max() + 0.1 * span, 50)
        plt.plot(wider, np.full_like(wider, y_values.mean()), "-", label="fit")


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


def _mesh_edges(figure) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y cell edges of the colour mesh drawn on a figure, in drawn order."""
    coordinates = figure.axes[0].collections[0].get_coordinates()
    return np.asarray(coordinates[0, :, 0]), np.asarray(coordinates[:, 0, 1])


def _expected_edges(values: np.ndarray) -> np.ndarray:
    """Cell edges placing every swept value on its own cell: midpoints, outer edges half a step out."""
    return np.concatenate(
        [
            [values[0] - (values[1] - values[0]) / 2],
            (values[:-1] + values[1:]) / 2,
            [values[-1] + (values[-1] - values[-2]) / 2],
        ]
    )


def _axes_by_label(figure, label: str):
    """Return the axes carrying `label` on either of its axis labels."""
    return next(axes for axes in figure.axes if label in {axes.get_xlabel(), axes.get_ylabel()})


def _positions(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Where the swept values sit across the axes, as a fraction of the axis width."""
    low, high = limits
    return (values - low) / (high - low)


def _title_and_figure_top(figure) -> tuple[float, float]:
    """Return the top of the axes title and the top of the figure canvas, in display units."""
    FigureCanvasAgg(figure)  # `plot` closed the figure, so re-attach a renderer to measure it
    figure.canvas.draw()
    title = figure.axes[0].title.get_window_extent(figure.canvas.get_renderer())
    return title.y1, figure.bbox.y1


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


def test_plot_2d_descending_sweep_is_not_mirrored(captured_figures):
    """A sweep played downwards is painted where it was measured, not flipped about its midpoint."""
    flux = np.arange(0.515, 0.48, -0.0035)
    frequency = np.linspace(-0.2e9, -0.05e9, 4)
    data = np.zeros((len(flux), len(frequency), 2))
    data[0, :, 0] = 50.0  # a single bright column, measured at the very first flux point
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[flux]),
        Dimension(labels=["Frequency (Hz)"], values=[frequency]),
    ]
    result = RecordingResult(qubit=0, averages=1000, data=data, dims=dims)

    result.plot()

    figure = captured_figures[0]
    x_edges, _ = _mesh_edges(figure)
    np.testing.assert_allclose(x_edges, _expected_edges(flux))
    drawn = np.asarray(figure.axes[0].collections[0].get_array()).reshape(len(frequency), len(flux))
    bright = int(np.argmax(drawn[0]))
    assert min(x_edges[bright], x_edges[bright + 1]) < flux[0] < max(x_edges[bright], x_edges[bright + 1])


def test_plot_2d_cells_span_a_full_sweep_step(captured_figures):
    """Each cell is one sweep step wide and centred on its point, so the map is not compressed."""
    flux = np.linspace(-0.5, 0.5, 5)
    frequency = np.linspace(4.0e9, 5.0e9, 4)
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[flux]),
        Dimension(labels=["Frequency (Hz)"], values=[frequency]),
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    x_edges, y_edges = _mesh_edges(captured_figures[0])
    np.testing.assert_allclose(np.diff(x_edges), 0.25)
    np.testing.assert_allclose(np.diff(y_edges), 1.0e9 / 3)
    np.testing.assert_allclose([x_edges[0], x_edges[-1]], [-0.625, 0.625])


def test_plot_2d_non_uniform_sweep_keeps_its_spacing(captured_figures):
    """A non-uniformly spaced sweep keeps its spacing instead of being redrawn as an even grid."""
    flux = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    frequency = np.array([4.0e9, 4.1e9, 4.3e9, 4.7e9])
    dims = [
        Dimension(labels=["Flux bias (V)"], values=[flux]),
        Dimension(labels=["Frequency (Hz)"], values=[frequency]),
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    x_edges, y_edges = _mesh_edges(captured_figures[0])
    np.testing.assert_allclose(x_edges, [0.5, 1.5, 3.0, 6.0, 12.0, 20.0])
    np.testing.assert_allclose(y_edges, _expected_edges(frequency))


def test_plot_2d_secondary_axes_track_the_primary_ones(captured_figures):
    """Secondary axes that run opposite to their primary still label the same positions."""
    flux = np.linspace(-0.5, 0.5, 5)
    current = np.linspace(1.0, -1.0, 5)  # ramped down while the bias it sets is swept up
    frequency = np.linspace(4.0e9, 5.0e9, 4)
    if_frequency = np.linspace(2.0e8, 1.0e8, 4)
    dims = [
        Dimension(labels=["Flux bias (V)", "Flux current (A)"], values=[flux, current]),
        Dimension(labels=["Frequency (Hz)", "IF frequency (Hz)"], values=[frequency, if_frequency]),
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    figure = captured_figures[0]
    primary = figure.axes[0]
    secondary_x = _axes_by_label(figure, "Flux current (A)")
    secondary_y = _axes_by_label(figure, "IF frequency (Hz)")
    assert secondary_x.get_xlim()[0] > secondary_x.get_xlim()[1]
    assert secondary_y.get_ylim()[0] > secondary_y.get_ylim()[1]
    np.testing.assert_allclose(_positions(current, secondary_x.get_xlim()), _positions(flux, primary.get_xlim()))
    np.testing.assert_allclose(
        _positions(if_frequency, secondary_y.get_ylim()), _positions(frequency, primary.get_ylim())
    )


def test_plot_1d_secondary_x_axis_tracks_the_primary_one(captured_figures):
    """The 1D secondary axis labels the same positions as the primary, whichever way it runs."""
    frequency = np.linspace(4.0e9, 5.0e9, 8)
    flux = np.linspace(1.0, -1.0, 8)  # descending while the frequency ascends
    dims = [Dimension(labels=["Frequency (Hz)", "Flux bias (V)"], values=[frequency, flux])]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    figure = captured_figures[0]
    secondary = _axes_by_label(figure, "Flux bias (V)")
    assert secondary.get_xlim()[0] > secondary.get_xlim()[1]
    np.testing.assert_allclose(_positions(flux, secondary.get_xlim()), _positions(frequency, figure.axes[0].get_xlim()))


def test_plot_1d_fit_is_drawn_on_the_primary_axes(captured_figures):
    """A fit drawn through pyplot must land on the primary axes, not on the secondary twin."""
    dims = [
        Dimension(
            labels=["Frequency (Hz)", "Flux bias (V)"],
            values=[np.linspace(4.0e9, 5.0e9, 8), np.linspace(1.0, -1.0, 8)],
        )
    ]
    result = WideFitResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot(fit=True)

    figure = captured_figures[0]
    primary = figure.axes[0]
    secondary = _axes_by_label(figure, "Flux bias (V)")
    assert [line.get_label() for line in primary.lines].count("fit") == 1
    assert [line.get_label() for line in secondary.lines] == []


def test_plot_1d_secondary_axis_tracks_a_fit_wider_than_the_data(captured_figures):
    """A fit reaching past the data widens the primary axis, and the secondary must follow it."""
    frequency = np.linspace(4.0e9, 5.0e9, 8)
    bias = np.linspace(1.0, -1.0, 8)
    dims = [Dimension(labels=["Frequency (Hz)", "Flux bias (V)"], values=[frequency, bias])]
    result = WideFitResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot(fit=True)

    figure = captured_figures[0]
    primary_limits = figure.axes[0].get_xlim()
    secondary = _axes_by_label(figure, "Flux bias (V)")
    # The fit has to be on the primary axis and to have stretched it past the swept range, otherwise
    # the alignment below would hold trivially -- autoscale margins alone would not prove anything.
    span = np.ptp(frequency)
    assert primary_limits[0] <= frequency.min() - 0.1 * span
    assert primary_limits[1] >= frequency.max() + 0.1 * span
    np.testing.assert_allclose(_positions(bias, secondary.get_xlim()), _positions(frequency, primary_limits))


def test_plot_1d_secondary_x_axis_of_a_motionless_sweep(captured_figures):
    """A primary that never moves sets no scale, so the secondary falls back to its own values."""
    dims = [
        Dimension(
            labels=["Frequency (Hz)", "Flux bias (V)"],
            values=[np.full(8, 4.0e9), np.linspace(-1.0, 1.0, 8)],
        )
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    secondary = _axes_by_label(captured_figures[0], "Flux bias (V)")
    np.testing.assert_allclose(secondary.get_xlim(), (-1.0, 1.0))


def test_plot_1d_title_fits_inside_the_figure(captured_figures):
    """A secondary axis label must not push the title off the top of the canvas."""
    dims = [
        Dimension(
            labels=["Frequency (Hz)", "Flux bias (V)"],
            values=[np.linspace(4.0e9, 5.0e9, 8), np.linspace(1.0, -1.0, 8)],
        )
    ]
    result = RecordingResult(qubit=0, averages=1000, data=_data_1d(), dims=dims)

    result.plot()

    title_top, figure_top = _title_and_figure_top(captured_figures[0])
    assert title_top <= figure_top


def test_plot_2d_saves_the_figure_it_lays_out(tmp_path, captured_figures):
    """The saved file must get the same fitted layout as the displayed figure, not the untidied one."""
    positions_when_saved = []
    original_savefig = Figure.savefig

    def recording_savefig(self, *args, **kwargs):
        positions_when_saved.append(self.axes[0].get_position().bounds)
        return original_savefig(self, *args, **kwargs)

    dims = [
        Dimension(
            labels=["Flux bias (V)", "Flux current (A)"], values=[np.linspace(-0.5, 0.5, 5), np.linspace(1.0, -1.0, 5)]
        ),
        Dimension(
            labels=["Frequency (Hz)", "IF frequency (Hz)"],
            values=[np.linspace(4.0e9, 5.0e9, 4), np.linspace(2.0e8, 1.0e8, 4)],
        ),
    ]
    result = RecordingResult(qubit=1, averages=1000, data=_data_2d(), dims=dims)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Figure, "savefig", recording_savefig)
        result.plot(save_to=str(tmp_path / "map.png"))

    assert positions_when_saved == [captured_figures[0].axes[0].get_position().bounds]


def test_plot_2d_title_fits_inside_the_figure(captured_figures):
    """Two twin axis labels must not push the title off the top of the canvas."""
    dims = [
        Dimension(
            labels=["Flux bias (V)", "Flux current (A)"], values=[np.linspace(-0.5, 0.5, 5), np.linspace(1.0, -1.0, 5)]
        ),
        Dimension(
            labels=["Frequency (Hz)", "IF frequency (Hz)"],
            values=[np.linspace(4.0e9, 5.0e9, 4), np.linspace(2.0e8, 1.0e8, 4)],
        ),
    ]
    result = RecordingResult(qubit=1, averages=1000, data=_data_2d(), dims=dims)

    result.plot()

    title_top, figure_top = _title_and_figure_top(captured_figures[0])
    assert title_top <= figure_top


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
