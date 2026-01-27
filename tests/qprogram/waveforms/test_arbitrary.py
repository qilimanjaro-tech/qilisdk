import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Arbitrary

samples = np.concatenate([np.linspace(0, 10_000, 1000), np.linspace(10_000, 0, 1000)])


@pytest.fixture
def arbitrary():
    return Arbitrary(samples=samples)


def test_init(arbitrary: Arbitrary):
    # test init method
    assert np.allclose(arbitrary.samples, samples)


def test_envelope(arbitrary: Arbitrary):
    # test envelope method
    assert np.allclose(arbitrary.envelope(), samples)


def test_envelope_with_higher_resolution(arbitrary: Arbitrary):
    # test envelope method
    resolution = 100
    envelope = arbitrary.envelope(resolution=resolution)
    assert math.isclose(envelope[0], samples[0], abs_tol=1e-9)
    assert math.isclose(envelope[-1], samples[-1], abs_tol=1e-9)
    assert max(envelope) == max(samples)
    assert min(envelope) == min(envelope)
    assert np.allclose(envelope, np.array([
        0.00000000e+00, 1.11111111e+03, 2.22222222e+03, 3.33333333e+03,
        4.44444444e+03, 5.55555556e+03, 6.66666667e+03, 7.77777778e+03,
        8.88888889e+03, 1.00000000e+04, 1.00000000e+04, 8.88888889e+03,
        7.77777778e+03, 6.66666667e+03, 5.55555556e+03, 4.44444444e+03,
        3.33333333e+03, 2.22222222e+03, 1.11111111e+03, 0.00000000e+00
    ]))


def test_envelope_with_constant_values():
    resolution = 20_000
    arbitrary = Arbitrary(samples=np.ones(resolution))
    envelope = arbitrary.envelope(resolution=resolution)
    assert np.allclose(envelope, np.ones(200))


def test_envelope_with_higher_resolution_raises_error(arbitrary: Arbitrary):
    resolution = 20_000
    with pytest.raises(ValueError, match=r"Resolution is too high compared to the waveform duration."):
        _ = arbitrary.envelope(resolution=resolution)


def test_get_duration_method(arbitrary: Arbitrary):
    assert arbitrary.get_duration() == 2000
