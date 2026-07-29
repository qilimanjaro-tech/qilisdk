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

import matplotlib as mpl

mpl.use("Agg")  # headless backend so draw() never opens a window

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from qilisdk.ml.datasets import (
    NARMA,
    Dataset,
    DatasetSample,
    HenonMap,
    LogisticMap,
    Lorenz,
    MackeyGlass,
    SantaFeLaser,
)
from qilisdk.ml.datasets.dataset import build_prediction_sample
from qilisdk.utils.visualization import DatasetStyle, dark
from qilisdk.utils.visualization.dataset_renderers import MatplotlibDatasetRenderer

# (dataset factory, expected feature dimension)
DATASETS = [
    (lambda: NARMA(order=10, seed=1), 1),
    (lambda: NARMA(order=15, seed=1), 1),
    (lambda: MackeyGlass(tau=17.0), 1),
    (SantaFeLaser, 1),
    (Lorenz, 3),
    (HenonMap, 2),
    (LogisticMap, 1),
]


@pytest.mark.parametrize(("factory", "features"), DATASETS)
def test_generate_shapes_and_finiteness(factory, features):
    dataset = factory()
    assert isinstance(dataset, Dataset)

    npoints = 500
    sample = dataset.generate(npoints)

    assert isinstance(sample, DatasetSample)
    assert sample.inputs.shape == (npoints, features)
    assert sample.targets.shape == (npoints, features)
    assert sample.inputs.dtype == np.float64
    assert len(sample) == npoints
    assert np.all(np.isfinite(sample.inputs))
    assert np.all(np.isfinite(sample.targets))


@pytest.mark.parametrize(("factory", "features"), DATASETS)
def test_sample_is_unpackable(factory, features):
    inputs, targets = factory().generate(50)
    assert inputs.shape == (50, features)
    assert targets.shape == (50, features)


def test_narma_is_reproducible_with_seed():
    a = NARMA(seed=42).generate(200)
    b = NARMA(seed=42).generate(200)
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.targets, b.targets)


def test_narma_different_seeds_differ():
    a = NARMA(seed=1).generate(200)
    b = NARMA(seed=2).generate(200)
    assert not np.array_equal(a.inputs, b.inputs)


def test_narma_input_range_respected():
    inputs, _ = NARMA(input_range=(0.0, 0.5), seed=0).generate(1000)
    assert inputs.min() >= 0.0
    assert inputs.max() <= 0.5


def test_narma_high_order_stays_bounded():
    # order > 10 squashes the update with tanh to prevent divergence.
    _, targets = NARMA(order=20, seed=0).generate(2000)
    assert np.all(np.isfinite(targets))
    assert np.abs(targets).max() <= 1.0


def test_horizon_alignment():
    # targets[t] must equal the series horizon steps ahead of inputs[t].
    horizon = 3
    inputs, targets = LogisticMap(horizon=horizon).generate(100)
    assert np.allclose(inputs[horizon:, 0], targets[:-horizon, 0])


def test_deterministic_systems_are_repeatable():
    for factory, _ in [(MackeyGlass, 1), (Lorenz, 3), (HenonMap, 2), (LogisticMap, 1), (SantaFeLaser, 1)]:
        first = factory().generate(100)
        second = factory().generate(100)
        assert np.array_equal(first.inputs, second.inputs)


def test_santa_fe_laser_intensity_non_negative():
    inputs, targets = SantaFeLaser().generate(1000)
    assert np.all(inputs >= 0.0)
    assert np.all(targets >= 0.0)


def test_chaotic_series_have_nontrivial_variation():
    for dataset in [MackeyGlass(tau=17.0), Lorenz(), HenonMap(), LogisticMap(), SantaFeLaser()]:
        inputs, _ = dataset.generate(2000)
        assert inputs.std() > 0.0


@pytest.mark.parametrize(("factory", "_features"), DATASETS)
def test_invalid_npoints_raises(factory, _features):
    dataset = factory()
    with pytest.raises(ValueError, match="npoints"):
        dataset.generate(0)


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="order"):
        NARMA(order=0)
    with pytest.raises(ValueError, match="tau"):
        MackeyGlass(tau=-1.0)
    with pytest.raises(ValueError, match="x0"):
        LogisticMap(x0=2.0)


@pytest.mark.parametrize("factory", [Lorenz, MackeyGlass, SantaFeLaser])
def test_invalid_dt_raises(factory):
    with pytest.raises(ValueError, match="dt"):
        factory(dt=0.0)


@pytest.mark.parametrize("factory", [Lorenz, MackeyGlass, SantaFeLaser])
def test_invalid_sample_every_raises(factory):
    with pytest.raises(ValueError, match="sample_every"):
        factory(sample_every=0)


def test_seed_property():
    assert NARMA(seed=7).seed == 7
    assert NARMA().seed is None


def test_narma_order_property():
    assert NARMA(order=12).order == 12


def test_build_prediction_sample_invalid_horizon_raises():
    series = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="horizon"):
        build_prediction_sample(series, horizon=0)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(("factory", "_features"), DATASETS)
@pytest.mark.parametrize("style", ["1d", "2d", "3d"])
def test_draw_all_styles_saves_file(factory, _features, style, tmp_path):
    sample = factory().generate(200)
    out = tmp_path / f"plot_{style}.png"
    factory().draw(sample, style=style, filepath=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.parametrize(("factory", "_features"), DATASETS)
def test_draw_default_style_saves_file(factory, _features, tmp_path):
    # No explicit style falls back to each dataset's natural default mode.
    sample = factory().generate(200)
    out = tmp_path / "plot_default.png"
    factory().draw(sample, filepath=str(out))
    assert out.exists()


def test_draw_is_a_classmethod(tmp_path):
    sample = MackeyGlass(tau=17.0).generate(200)
    # Callable straight off the class (no instance), as in the usage example.
    out = tmp_path / "mg.png"
    MackeyGlass.draw(sample, style="1d", filepath=str(out))
    assert out.exists()


def test_draw_respects_config(tmp_path):
    sample = Lorenz().generate(300)
    config = DatasetStyle(theme=dark, trajectory_style="line", colorbar=False)
    out = tmp_path / "lorenz_dark.png"
    Lorenz().draw(sample, style="3d", config=config, filepath=str(out))
    assert out.exists()


def test_draw_invalid_style_raises():
    sample = MackeyGlass(tau=17.0).generate(50)
    with pytest.raises(ValueError, match="style"):
        MackeyGlass.draw(sample, style="4d", filepath=None)


@pytest.mark.parametrize(("factory", "_features"), DATASETS)
@pytest.mark.parametrize("style", ["1d", "2d", "3d"])
def test_draw_accepts_raw_inputs_array(factory, _features, style, tmp_path):
    # draw() takes the series directly, so half of an unpacked sample can be plotted.
    inputs, _ = factory().generate(200)
    out = tmp_path / f"inputs_{style}.png"
    factory().draw(inputs, style=style, filepath=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_draw_raw_targets_array_matches_sample_of_same_series(tmp_path):
    # Passing an array plots exactly that array: targets drawn raw must match a
    # DatasetSample whose inputs are those same targets.
    _, targets = Lorenz().generate(200)
    raw = tmp_path / "raw.png"
    wrapped = tmp_path / "wrapped.png"
    Lorenz.draw(targets, style="3d", filepath=str(raw))
    Lorenz.draw(DatasetSample(inputs=targets, targets=targets), style="3d", filepath=str(wrapped))
    assert raw.read_bytes() == wrapped.read_bytes()


def test_draw_accepts_one_dimensional_array(tmp_path):
    # A single component sliced out of a multi-component sample is 1-D, not (n, 1).
    inputs, _ = Lorenz().generate(200)
    column = inputs[:, 0]
    assert column.ndim == 1
    out = tmp_path / "column.png"
    Lorenz.draw(column, style="1d", filepath=str(out))
    assert out.exists()


def test_draw_rejects_higher_dimensional_array():
    series = np.zeros((10, 2, 2))
    with pytest.raises(ValueError, match="at most 2-dimensional"):
        MackeyGlass.draw(series, style="1d", filepath=None)


def test_draw_rejects_empty_array():
    series = np.zeros((0, 1))
    with pytest.raises(ValueError, match="empty"):
        MackeyGlass.draw(series, style="1d", filepath=None)


def test_draw_too_short_series_for_embedding_raises():
    sample = LogisticMap().generate(2)
    config = DatasetStyle(delay=5)
    with pytest.raises(ValueError, match="delay embedding"):
        LogisticMap.draw(sample, style="3d", config=config, filepath=None)


def test_draw_without_filepath_shows_the_figure(monkeypatch):
    # With no filepath the figure is shown interactively rather than written to disk.
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    sample = LogisticMap().generate(50)
    LogisticMap().draw(sample, style="1d", filepath=None)
    assert shown == [True]


def test_draw_labels_fall_back_to_positional_for_unlabelled_components(tmp_path):
    # MackeyGlass names a single component ("x"), but a raw array may carry more than the
    # dataset labels: the surplus components fall back to positional "x<i>" labels.
    inputs, _ = MackeyGlass(tau=17.0).generate(100)
    two_components = np.hstack([inputs, inputs * 2.0])
    MackeyGlass.draw(two_components, style="1d", filepath=str(tmp_path / "two_components.png"))
    ax = plt.gcf().axes[0]
    assert [line.get_label() for line in ax.lines] == ["x", "x1"]


def test_renderer_plots_onto_supplied_axes():
    # plot(ax=...) retargets the renderer onto a caller-owned axes, so a dataset can be
    # drawn into an existing figure (e.g. one panel of a subplot grid).
    inputs, _ = LogisticMap().generate(50)
    _, (left, right) = plt.subplots(1, 2)
    renderer = MatplotlibDatasetRenderer(inputs, "1d")
    own_axes = renderer.ax
    renderer.plot(ax=right)
    assert renderer.ax is right
    assert right.lines
    assert not left.lines
    assert not own_axes.lines  # nothing was drawn on the axes the renderer made for itself


@pytest.mark.parametrize("style", ["2d", "3d"])
def test_draw_scatter_without_color_by_time(style, tmp_path):
    # A scatter trajectory that is not coloured by time uses one flat theme colour, and so
    # needs no colour bar (which would otherwise add a second axes to the figure).
    sample = Lorenz().generate(200)
    config = DatasetStyle(trajectory_style="scatter", color_by_time=False)
    out = tmp_path / f"flat_scatter_{style}.png"
    Lorenz().draw(sample, style=style, config=config, filepath=str(out))
    assert out.exists()
    fig = plt.gcf()
    assert len(fig.axes) == 1
    assert fig.axes[0].collections


def test_draw_grid_style_without_color_uses_theme_color(tmp_path):
    # grid_style need not name a colour; the theme's muted surface colour is filled in.
    sample = LogisticMap().generate(50)
    config = DatasetStyle(theme=dark, grid_style={"linestyle": ":"})
    LogisticMap().draw(sample, style="1d", config=config, filepath=str(tmp_path / "themed_grid.png"))
    gridline = plt.gcf().axes[0].xaxis.get_gridlines()[0]
    assert to_rgba(gridline.get_color()) == to_rgba(dark.surface_muted)


def test_draw_grid_style_color_is_respected(tmp_path):
    # An explicit grid colour must win over the theme default.
    sample = LogisticMap().generate(50)
    config = DatasetStyle(theme=dark, grid_style={"linestyle": ":", "color": "#ff00ff"})
    LogisticMap().draw(sample, style="1d", config=config, filepath=str(tmp_path / "explicit_grid.png"))
    gridline = plt.gcf().axes[0].xaxis.get_gridlines()[0]
    assert to_rgba(gridline.get_color()) == to_rgba("#ff00ff")
