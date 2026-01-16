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

from qilisdk.noise import BitFlip, KrausChannel, PauliChannel, PhaseFlip
from qilisdk.noise.depolarizing import Depolarizing
from qilisdk.noise.protocols import AttachmentScope


def test_pauli_channel_properties_and_kraus():
    channel = PauliChannel(pX=0.25, pY=0.0, pZ=0.0)

    assert channel.pX == 0.25
    assert channel.pY == 0.0
    assert channel.pZ == 0.0

    kraus = channel.as_kraus()
    assert isinstance(kraus, KrausChannel)
    assert len(kraus.operators) == 2

    expected_identity = np.sqrt(0.75) * np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    expected_x = np.sqrt(0.25) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    np.testing.assert_allclose(kraus.operators[0].dense(), expected_identity)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_x)

    with pytest.raises(ValueError):  # noqa: PT011
        PauliChannel(pX=-0.1)
    with pytest.raises(ValueError):  # noqa: PT011
        PauliChannel(pX=0.6, pY=0.6)


def test_bit_flip_probability_and_scopes():
    noise = BitFlip(probability=0.2)

    assert noise.probability == 0.2
    assert noise.pZ == 0.2
    assert noise.pX == 0.0
    assert AttachmentScope.PER_GATE_TYPE in BitFlip.allowed_scopes()

    kraus = noise.as_kraus()
    assert len(kraus.operators) == 2

    expected_z = np.sqrt(0.2) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_z)

    with pytest.raises(ValueError):  # noqa: PT011
        BitFlip(probability=1.2)


def test_phase_flip_probability_and_scopes():
    noise = PhaseFlip(probability=0.3)

    assert noise.probability == 0.3
    assert noise.pX == 0.3
    assert noise.pZ == 0.0
    assert AttachmentScope.PER_GATE_TYPE in PhaseFlip.allowed_scopes()

    kraus = noise.as_kraus()
    assert len(kraus.operators) == 2

    expected_x = np.sqrt(0.3) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    np.testing.assert_allclose(kraus.operators[1].dense(), expected_x)

    with pytest.raises(ValueError):  # noqa: PT011
        PhaseFlip(probability=-0.1)


def test_depolarizing_probability_distribution():
    noise = Depolarizing(probability=0.3)

    assert noise.probability == 0.3
    assert noise.pX == pytest.approx(0.1)
    assert noise.pY == pytest.approx(0.1)
    assert noise.pZ == pytest.approx(0.1)

    with pytest.raises(ValueError):  # noqa: PT011
        Depolarizing(probability=-0.1)
