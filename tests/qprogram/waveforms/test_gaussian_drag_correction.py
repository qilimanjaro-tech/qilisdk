import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Gaussian
from qilisdk.qprogram.waveforms.gaussian_drag_correction import GaussianDragCorrection


@pytest.fixture
def gaussian_drag_correction():
    return GaussianDragCorrection(1.0, 100, 0.5, 2.5)


def test_init(gaussian_drag_correction: GaussianDragCorrection):
    # test init method
    assert math.isclose(gaussian_drag_correction.amplitude, 1.0)
    assert gaussian_drag_correction.duration == 100
    assert math.isclose(gaussian_drag_correction.num_sigmas, 0.5)
    assert math.isclose(gaussian_drag_correction.drag_coefficient, 2.5)


def test_envelope_matches_drag_formula(
    gaussian_drag_correction: GaussianDragCorrection
):
    resolution = 5
    sigma = gaussian_drag_correction.duration / gaussian_drag_correction.num_sigmas
    mu = gaussian_drag_correction.duration / 2
    x = np.arange(gaussian_drag_correction.duration / resolution) * resolution

    base_gaussian = Gaussian(
        amplitude=gaussian_drag_correction.amplitude,
        duration=gaussian_drag_correction.duration,
        num_sigmas=gaussian_drag_correction.num_sigmas,
    ).envelope(resolution=resolution)

    expected = (
        -gaussian_drag_correction.drag_coefficient * (x - mu) / sigma**2
    ) * base_gaussian

    np.testing.assert_allclose(
        gaussian_drag_correction.envelope(resolution=resolution), expected
    )


def test_get_duration(gaussian_drag_correction: GaussianDragCorrection):
    assert gaussian_drag_correction.get_duration() == gaussian_drag_correction.duration

