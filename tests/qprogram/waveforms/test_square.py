import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Square


@pytest.fixture
def square():
    return Square(amplitude=0.5, duration=100)


def test_init(square: Square):
    """Test __init__ method"""
    assert math.isclose(square.amplitude, 0.5)
    assert square.duration == 100


def test_envelope_method(square: Square):
    """Test envelope method"""
    envelope = 0.5 * np.ones(round(100 / 1))
    assert np.allclose(square.envelope(), envelope)


def test_get_duration_method(square: Square):
    """Test get_duration method"""
    assert square.get_duration() == 100
