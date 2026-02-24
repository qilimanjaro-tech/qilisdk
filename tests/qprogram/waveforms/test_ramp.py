import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Ramp


@pytest.fixture
def ramp_wf() -> Ramp:
    return Ramp(from_amplitude=0.0, to_amplitude=1.0, duration=100)


def test_init(ramp_wf: Ramp):
    """Test __init__ method"""
    assert math.isclose(ramp_wf.from_amplitude, 0.0)
    assert math.isclose(ramp_wf.to_amplitude, 1.0)
    assert ramp_wf.duration == 100


def test_envelope_method(ramp_wf: Ramp):
    """Test envelope method"""
    expected_envelope = np.linspace(ramp_wf.from_amplitude, ramp_wf.to_amplitude, ramp_wf.duration)
    envelope = ramp_wf.envelope()
    assert np.allclose(envelope, expected_envelope)


def test_get_duration_method(ramp_wf: Ramp):
    """Test get_duration method"""
    assert ramp_wf.get_duration() == 100
