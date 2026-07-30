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


def test_mackey_glass_registers_attractor_transforms():
    # The tau delay is folded into the dataset's transforms rather than into
    # style.delay, so drawing 2-D/3-D uses a delay embedding out of the box.
    assert set(MackeyGlass._DRAW_TRANSFORMS) == {"2d", "3d"}
    resolved = MackeyGlass._resolve_draw_style("2d", None)
    assert resolved.title == "Mackey-Glass attractor"
    assert resolved.delay == 1  # untouched global default; the delay lives in the transform


def test_explicit_config_field_overrides_per_mode_default():
    # A field the caller sets explicitly wins over the dataset's per-mode default.
    resolved = MackeyGlass._resolve_draw_style("2d", DatasetStyle(title="custom"))
    assert resolved.title == "custom"


def test_unrelated_config_field_keeps_per_mode_default():
    # Setting an unrelated field must not wipe out the dataset's per-mode title.
    resolved = MackeyGlass._resolve_draw_style("2d", DatasetStyle(theme=dark))
    assert resolved.title == "Mackey-Glass attractor"
    assert resolved.theme == dark


def test_dataset_without_per_mode_defaults_returns_config_unchanged():
    # A dataset that declares no per-mode defaults leaves the caller's config intact.
    assert not LogisticMap._DRAW_STYLE_DEFAULTS
    config = DatasetStyle(delay=9)
    assert LogisticMap._resolve_draw_style("2d", config) is config


def test_mackey_glass_attractor_embeds_with_tau_delay(tmp_path):
    # End-to-end: the 2-D transform plots (P(t), P(t - 17)).
    inputs, _ = MackeyGlass(tau=17.0).generate(500)
    MackeyGlass.draw(inputs, style="2d", filepath=str(tmp_path / "attractor.png"))
    ax = plt.gcf().axes[0]
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], inputs[17:, 0])  # P(t)
    assert np.allclose(offsets[:, 1], inputs[:-17, 0])  # P(t - 17)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("P(t)", "P(t - 17)")


def test_mackey_glass_delay_folded_into_transform_ignores_style_delay(tmp_path):
    # Because the delay lives in the transform, changing style.delay does not
    # change the attractor embedding.
    inputs, _ = MackeyGlass(tau=17.0).generate(500)
    MackeyGlass.draw(inputs, style="2d", config=DatasetStyle(delay=1), filepath=str(tmp_path / "a.png"))
    offsets = plt.gcf().axes[0].collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], inputs[17:, 0])
    assert np.allclose(offsets[:, 1], inputs[:-17, 0])


def test_mackey_glass_3d_transform_embeds_three_delays(tmp_path):
    # The 3-D transform embeds the series at delays 0, 17 and 34.
    inputs, _ = MackeyGlass(tau=17.0).generate(500)
    MackeyGlass.draw(inputs, style="3d", filepath=str(tmp_path / "mg3d.png"))
    xs, ys, zs = plt.gcf().axes[0].collections[0]._offsets3d
    assert np.allclose(xs, inputs[34:, 0])
    assert np.allclose(ys, inputs[17:-17, 0])
    assert np.allclose(zs, inputs[:-34, 0])


def test_lorenz_2d_defaults_to_xz_projection(tmp_path):
    # Lorenz declares an x-z projection for its 2-D view, not the default x-y.
    inputs, _ = Lorenz().generate(300)
    Lorenz.draw(inputs, style="2d", filepath=str(tmp_path / "lorenz_xz.png"))
    ax = plt.gcf().axes[0]
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], inputs[:, 0])  # x
    assert np.allclose(offsets[:, 1], inputs[:, 2])  # z (not y)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "z"


def test_lorenz_1d_and_3d_use_all_three_components(tmp_path):
    # 1-D shows three lines (x, y, z); 3-D plots the full attractor.
    inputs, _ = Lorenz().generate(300)
    Lorenz.draw(inputs, style="1d", filepath=str(tmp_path / "lorenz_1d.png"))
    assert [line.get_label() for line in plt.gcf().axes[0].lines] == ["x", "y", "z"]
    Lorenz.draw(inputs, style="3d", filepath=str(tmp_path / "lorenz_3d.png"))
    assert plt.gcf().axes[0].collections  # a 3-D trajectory was drawn


def test_draw_accepts_custom_transform(tmp_path):
    # A caller-supplied transform overrides the dataset's default view.
    inputs, _ = Lorenz().generate(300)
    Lorenz.draw(
        inputs,
        style="2d",
        transform=lambda d: [("y", d[:, 1]), ("z", d[:, 2])],
        filepath=str(tmp_path / "lorenz_yz.png"),
    )
    ax = plt.gcf().axes[0]
    offsets = ax.collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], inputs[:, 1])  # y
    assert np.allclose(offsets[:, 1], inputs[:, 2])  # z
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("y", "z")


def test_custom_transform_may_return_bare_arrays(tmp_path):
    # Channels need not be labelled; bare arrays fall back to positional labels.
    inputs, _ = Lorenz().generate(200)

    def first_two(d):
        return [d[:, 0], d[:, 1]]

    Lorenz.draw(inputs, style="2d", transform=first_two, filepath=str(tmp_path / "bare.png"))
    ax = plt.gcf().axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x", "y")


def test_transform_may_reshape_beyond_column_selection(tmp_path):
    # A transform is a general data transformation, not just column selection:
    # here it delay-embeds a 3-D series' first component into a 2-D portrait.
    inputs, _ = Lorenz().generate(200)

    def delay_embed(d):
        return [("x(t)", d[5:, 0]), ("x(t-5)", d[:-5, 0])]

    Lorenz.draw(inputs, style="2d", transform=delay_embed, filepath=str(tmp_path / "embed.png"))
    offsets = plt.gcf().axes[0].collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], inputs[5:, 0])
    assert np.allclose(offsets[:, 1], inputs[:-5, 0])


def test_transform_with_wrong_channel_count_raises():
    # A 2-D plot needs exactly two coordinates.
    inputs, _ = Lorenz().generate(200)
    with pytest.raises(ValueError, match="needs exactly 2 coordinates"):
        Lorenz.draw(inputs, style="2d", transform=lambda d: [("x", d[:, 0])], filepath=None)


def test_transform_with_unequal_lengths_raises():
    inputs, _ = Lorenz().generate(200)
    with pytest.raises(ValueError, match="unequal length"):
        Lorenz.draw(inputs, style="2d", transform=lambda d: [("x", d[:, 0]), ("z", d[:-5, 2])], filepath=None)


def test_one_dimensional_transform_can_select_components(tmp_path):
    # In 1-D any number of lines may be returned, e.g. a subset of components.
    inputs, _ = Lorenz().generate(200)
    Lorenz.draw(inputs, style="1d", transform=lambda d: [("z only", d[:, 2])], filepath=str(tmp_path / "z.png"))
    lines = plt.gcf().axes[0].lines
    assert len(lines) == 1
    assert np.allclose(lines[0].get_ydata(), inputs[:, 2])
