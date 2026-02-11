import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Chained, Ramp, Square


@pytest.fixture
def chained_wf() -> Chained:
    return Chained(
        waveforms=[
            Ramp(from_amplitude=0.0, to_amplitude=1.0, duration=100),
            Square(amplitude=1.0, duration=200),
            Ramp(from_amplitude=1.0, to_amplitude=0.0, duration=100),
        ]
    )


def test_init(chained_wf: Chained):
    """Test __init__ method"""
    assert isinstance(chained_wf.waveforms, list)
    assert len(chained_wf.waveforms) == 3


def test_envelope_method(chained_wf: Chained):
    """Test envelope method"""
    expected_envelope = np.concatenate(
        [
            Ramp(from_amplitude=0.0, to_amplitude=1.0, duration=100).envelope(),
            Square(amplitude=1.0, duration=200).envelope(),
            Ramp(from_amplitude=1.0, to_amplitude=0.0, duration=100).envelope(),
        ]
    )
    envelope = chained_wf.envelope()
    assert np.allclose(envelope, expected_envelope)


def test_get_duration_method(chained_wf: Chained):
    """Test get_duration method"""
    assert chained_wf.get_duration() == 400
