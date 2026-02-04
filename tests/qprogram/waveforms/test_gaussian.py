import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Gaussian


@pytest.fixture
def gaussian():
    return Gaussian(amplitude=1, duration=10, num_sigmas=2.5)


def test_init(gaussian: Gaussian):
    """Test __init__ method"""
    assert gaussian.amplitude == 1
    assert gaussian.duration == 10
    assert math.isclose(gaussian.num_sigmas, 2.5)


def test_envelope(gaussian: Gaussian):
    """Test envelope method"""

    # calculate envelope
    sigma = gaussian.duration / gaussian.num_sigmas
    mu = gaussian.duration / 2
    x = np.arange(5) * 2
    envelope = np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
    norm = np.amax(np.real(envelope))
    envelope = envelope - envelope[0]
    corr_norm = np.amax(np.real(envelope))
    envelope = envelope * norm / corr_norm

    assert np.allclose(gaussian.envelope(resolution=2), envelope)


def test_envelope_method_with_zero_amplitude(gaussian: Gaussian):
    """Test envelope method when amplitude is zero."""
    gaussian.amplitude = 0
    assert np.allclose(gaussian.envelope(), np.zeros(10))
