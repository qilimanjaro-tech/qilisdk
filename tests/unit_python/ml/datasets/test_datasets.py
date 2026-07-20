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

import numpy as np
import pytest

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
    with pytest.raises(ValueError, match="npoints"):
        factory().generate(0)


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
    with pytest.raises(ValueError, match="horizon"):
        build_prediction_sample(np.arange(10, dtype=np.float64), horizon=0)
