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

from qilisdk.core import QTensor
from qilisdk.noise import AmplitudeDamping, Dephasing, LindbladGenerator
from qilisdk.noise.utils import _sigma_plus


def test_dephasing_init_lindblad_and_kraus():
    with pytest.raises(ValueError, match=r"Tphi must be > 0."):
        Dephasing(Tphi=0.0)

    noise = Dephasing(Tphi=2.0)
    assert noise.Tphi == 2.0

    generator = noise.as_lindblad()
    expected_l = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    assert len(generator.rates) == 1
    assert len(generator.jump_operators) == 1
    assert len(generator.jump_operators_with_rates) == 1
    np.testing.assert_allclose(generator.jump_operators_with_rates[0].dense(), expected_l)

    kraus = noise.as_kraus(duration=0.0)
    assert len(kraus.operators) == 2
    np.testing.assert_allclose(kraus.operators[0].dense(), np.eye(2, dtype=complex))
    np.testing.assert_allclose(kraus.operators[1].dense(), np.zeros((2, 2), dtype=complex))

    with pytest.raises(ValueError, match=r"duration must be >= 0."):
        noise.as_kraus(duration=-1.0)


def test_amplitude_damping_init_lindblad_and_kraus():
    with pytest.raises(ValueError, match=r"T1 must be > 0."):
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


def test_lindblad_bad_rates():
    operator = QTensor(_sigma_plus())
    with pytest.raises(ValueError, match=r"Length of rates must match"):
        LindbladGenerator(jump_operators=[operator], rates=[])

    gen = LindbladGenerator(jump_operators=[], rates=[])
    gen._rates = [0.1]
    with pytest.raises(ValueError, match=r"Length of rates must match"):
        gen.jump_operators_with_rates
