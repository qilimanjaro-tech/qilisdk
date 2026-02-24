import math

import numpy as np
import pytest

from qilisdk.qprogram.waveforms import Gaussian, GaussianDragCorrection, IQDrag, IQPair, Square


@pytest.fixture
def iq_pair() -> IQPair:
    return IQPair(
        I=Square(amplitude=0.5, duration=100),
        Q=Square(amplitude=1.0, duration=100),
    )


def test_init(iq_pair: IQPair):
    """Test __init__ method."""
    assert math.isclose(getattr(iq_pair.I, "amplitude"), 0.5)
    assert getattr(iq_pair.I, "duration") == 100
    assert math.isclose(getattr(iq_pair.Q, "amplitude"), 1.0)
    assert getattr(iq_pair.Q, "duration") == 100


def test_get_duration_method(iq_pair: IQPair):
    """Test get_duration method."""
    assert iq_pair.get_duration() == 100


def test_getters_return_original_waveforms(iq_pair: IQPair):
    """IQPair accessors should expose the stored waveforms unchanged."""
    assert iq_pair.get_I() is iq_pair.I
    assert iq_pair.get_Q() is iq_pair.Q


def test_iq_pair_with_different_durations_throws_error():
    """Test that waveforms of an IQ pair must have the same duration."""
    with pytest.raises(ValueError, match=r"Waveforms of an IQ pair must have the same duration."):
        IQPair(
            I=Square(amplitude=0.5, duration=200),
            Q=Square(amplitude=1.0, duration=100),
        )


def test_iq_pair_without_waveform_type_throws_error():
    """Test that waveforms of an IQ pair must have Waveform type."""
    with pytest.raises(TypeError, match=r"Waveform inside IQPair must have Waveform type."):
        IQPair(
            I=IQPair(
                I=Square(amplitude=0.5, duration=100),
                Q=Square(amplitude=1.0, duration=100),
            ),
            Q=IQPair(
                I=Square(amplitude=0.5, duration=100),
                Q=Square(amplitude=1.0, duration=100),
            ),
        )

    with pytest.raises(TypeError, match=r"Waveform inside IQPair must have Waveform type."):
        IQPair(I=Square(amplitude=0.5, duration=100), Q="not a waveform")


def test_drag_method():
    """Test __init__ method"""
    drag = IQDrag(drag_coefficient=0.4, amplitude=0.7, duration=40, num_sigmas=2)
    gaus = Gaussian(amplitude=0.7, duration=40, num_sigmas=2)
    corr = GaussianDragCorrection(
        amplitude=gaus.amplitude,
        duration=gaus.duration,
        num_sigmas=gaus.num_sigmas,
        drag_coefficient=0.4,
    )

    assert isinstance(drag, IQDrag)
    assert isinstance(drag.get_I(), Gaussian)
    assert isinstance(drag.get_Q(), GaussianDragCorrection)
    assert np.allclose(drag.get_I().envelope(), gaus.envelope())
    assert np.allclose(drag.get_Q().envelope(), corr.envelope())
