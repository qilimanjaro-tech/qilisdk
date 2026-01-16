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

from qilisdk.noise import AmplitudeDamping, Dephasing


def test_dephasing_init_lindblad_and_kraus():
    with pytest.raises(ValueError):  # noqa: PT011
        Dephasing(Tphi=0.0)

    noise = Dephasing(Tphi=2.0)
    assert noise.Tphi == 2.0

    generator = noise.as_lindblad()
    expected_l = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    assert len(generator.jump_operators) == 1
    np.testing.assert_allclose(generator.jump_operators[0].dense(), expected_l)

    kraus = noise.as_kraus(duration=0.0)
    assert len(kraus.operators) == 2
    np.testing.assert_allclose(kraus.operators[0].dense(), np.eye(2, dtype=complex))
    np.testing.assert_allclose(kraus.operators[1].dense(), np.zeros((2, 2), dtype=complex))

    with pytest.raises(ValueError):  # noqa: PT011
        noise.as_kraus(duration=-1.0)


def test_amplitude_damping_init_lindblad_and_kraus():
    with pytest.raises(ValueError):  # noqa: PT011
        AmplitudeDamping(T1=-1.0)

    noise = AmplitudeDamping(T1=4.0)
    assert noise.T1 == 4.0

    generator = noise.as_lindblad()
    expected_l = 0.5 * np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    assert len(generator.jump_operators) == 1
    np.testing.assert_allclose(generator.jump_operators[0].dense(), expected_l)

    kraus = noise.as_kraus(duration=0.0)
    assert len(kraus.operators) == 2
    np.testing.assert_allclose(kraus.operators[0].dense(), np.eye(2, dtype=complex))
    np.testing.assert_allclose(kraus.operators[1].dense(), np.zeros((2, 2), dtype=complex))

    with pytest.raises(ValueError):  # noqa: PT011
        noise.as_kraus(duration=-0.5)
