import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import SuddenNetZero


@pytest.fixture
def snz():
    return SuddenNetZero(amplitude=1, duration=10, b=0.1, t_phi=2)


def test_init(snz: SuddenNetZero):
    """Test __init__ method"""
    assert snz.amplitude == 1
    assert snz.duration == 10
    assert math.isclose(snz.b, 0.1)
    assert snz.t_phi == 2


def test_envelope(snz: SuddenNetZero):
    """Test the SNZ envelope construction logic."""
    resolution = 2
    duration = snz.duration
    t_phi = snz.t_phi
    b = snz.b
    amp = snz.amplitude

    half_pulse_t = int((duration - 2 - t_phi) / 2 / resolution)
    N = round(duration / resolution)

    expected = np.zeros(N)
    expected[:half_pulse_t] = amp
    expected[half_pulse_t] = b * amp
    expected[half_pulse_t + 1 + t_phi] = -b * amp
    expected[half_pulse_t + 2 + t_phi:] = -amp

    actual = snz.envelope(resolution=resolution)

    assert np.allclose(actual, expected)


def test_get_duration(snz: SuddenNetZero):
    """Test that the duration is retrieved properly."""
    assert snz.get_duration() == snz.duration
