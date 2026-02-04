import numpy as np
import pytest
from scipy.special import erf

from qilisdk.qprogram.waveforms import FlatTop


@pytest.fixture
def flat_top():
    return FlatTop(amplitude=1, duration=30, smooth_duration=10, buffer=1)


def test_init(flat_top: FlatTop):
    """Test __init__ method"""
    assert flat_top.amplitude == 1
    assert flat_top.duration == 30
    assert flat_top.smooth_duration == 10
    assert flat_top.buffer == 1


def test_envelope(flat_top: FlatTop):
    """Test envelope method"""

    # calculate envelope
    x = np.arange(-flat_top.duration / 2, flat_top.duration / 2 + 1, 2)
    A = flat_top.amplitude
    sigma = flat_top.smooth_duration
    buf = flat_top.buffer
    dur = flat_top.duration
    envelope = (
        0.5 * A * np.real((erf(4 * (x + (dur / 2 - buf)) / sigma - 2) - erf(4 * (x - (dur / 2 - buf)) / sigma + 2)))
    )

    assert np.allclose(flat_top.envelope(resolution=2), envelope)


def test_get_duration(flat_top: FlatTop):
    """Test that the duration is retrieved properly."""
    assert flat_top.get_duration() == flat_top.duration
