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

from qilisdk.noise import BitFlip, Depolarizing, KrausChannel, PauliChannel, PhaseFlip
from qilisdk.noise.protocols import AttachmentScope


def test_pauli_channel_properties_and_kraus():
    x_rate = 0.25
    y_rate = 0.1
    z_rate = 0.15
    identity_rate = 1.0 - (x_rate + y_rate + z_rate)
    channel = PauliChannel(pX=x_rate, pY=y_rate, pZ=z_rate)

    assert np.isclose(channel.pX, x_rate)
    assert np.isclose(channel.pY, y_rate)
    assert np.isclose(channel.pZ, z_rate)

    kraus = channel.as_kraus()
    assert isinstance(kraus, KrausChannel)
    assert len(kraus.operators) == 4

    expected_identity = np.sqrt(identity_rate) * np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    expected_x = np.sqrt(x_rate) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    expected_y = np.sqrt(y_rate) * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    expected_z = np.sqrt(z_rate) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    np.testing.assert_allclose(kraus.operators[0].dense(), expected_identity)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_x)
    np.testing.assert_allclose(kraus.operators[2].dense(), expected_y)
    np.testing.assert_allclose(kraus.operators[3].dense(), expected_z)

    # need to re.escape the regex
    with pytest.raises(ValueError, match=r"pX must be in \[0, 1\]."):
        PauliChannel(pX=-0.1)
    with pytest.raises(ValueError, match=r"pX \+ pY \+ pZ must be <= 1."):
        PauliChannel(pX=0.6, pY=0.6)

    jump_operators = channel.as_lindblad_from_duration(duration=1.0)
    assert len(jump_operators.jump_operators) == 3
    with pytest.raises(ValueError, match="Duration must be positive"):
        channel.as_lindblad_from_duration(duration=-1.0)


def test_bit_flip_probability_and_scopes():
    noise = BitFlip(probability=0.2)

    assert np.isclose(noise.probability, 0.2)
    assert np.isclose(noise.pX, 0.2)
    assert np.isclose(noise.pY, 0.0)
    assert np.isclose(noise.pZ, 0.0)

    scopes = BitFlip.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes

    kraus = noise.as_kraus()
    assert len(kraus.operators) == 2

    expected_x = np.sqrt(0.2) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_x)

    with pytest.raises(ValueError):  # noqa: PT011
        BitFlip(probability=1.2)


def test_phase_flip_probability_and_scopes():
    noise = PhaseFlip(probability=0.3)

    assert np.isclose(noise.probability, 0.3)
    assert np.isclose(noise.pX, 0.0)
    assert np.isclose(noise.pY, 0.0)
    assert np.isclose(noise.pZ, 0.3)
    scopes = PhaseFlip.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes

    kraus = noise.as_kraus()
    assert len(kraus.operators) == 2

    expected_z = np.sqrt(0.3) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_z)

    with pytest.raises(ValueError):  # noqa: PT011
        PhaseFlip(probability=-0.1)


def test_depolarizing_probability_distribution():
    noise = Depolarizing(probability=0.3)

    scopes = Depolarizing.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_GATE_TYPE_PER_QUBIT in scopes

    assert np.isclose(noise.probability, 0.3)
    assert np.isclose(noise.pX, 0.1)
    assert np.isclose(noise.pY, 0.1)
    assert np.isclose(noise.pZ, 0.1)

    with pytest.raises(ValueError):  # noqa: PT011
        Depolarizing(probability=-0.1)
