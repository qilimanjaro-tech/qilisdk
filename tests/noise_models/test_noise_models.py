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

from qilisdk.noise_models.analog_noise import DissipationNoise
from qilisdk.noise_models.common_noise import ParameterNoise
from qilisdk.noise_models.digital_noise import KrausNoise
from qilisdk.noise_models.noise_model import NoiseBase, NoiseModel, NoiseType


class DummyNoise(NoiseBase):
    @property
    def noise_type(self) -> NoiseType:
        return NoiseType.DIGITAL


def test_noise_base_is_abstract():
    with pytest.raises(TypeError):
        NoiseBase()


def test_noise_model_defaults():
    model = NoiseModel()
    assert model.noise_passes == []
    assert model.noise_model_types() == []


def test_noise_model_add_and_types():
    model = NoiseModel()
    noise = DummyNoise()
    model.add(noise)
    assert model.noise_passes == [noise]
    assert model.noise_model_types() == [NoiseType.DIGITAL]


def test_noise_model_type_unique():
    model = NoiseModel([ParameterNoise(), ParameterNoise()])
    assert set(model.noise_model_types()) == {NoiseType.PARAMETER}


def test_parameter_noise_properties():
    noise = ParameterNoise(affected_parameters=["theta", "phi"], noise_std=0.25)
    assert noise.noise_type == NoiseType.PARAMETER
    assert noise.affected_parameters == ["theta", "phi"]
    assert np.isclose(noise.noise_std, 0.25)


def test_dissipation_noise_properties():
    noise = DissipationNoise(jump_operators=[])
    assert noise.noise_type == NoiseType.ANALOG
    assert noise.jump_operators == []


def test_kraus_noise_properties():
    noise = KrausNoise(kraus_operators=[], affected_qubits=[0, 2], affected_gates=[])
    assert noise.noise_type == NoiseType.DIGITAL
    assert noise.kraus_operators == []
    assert noise.affected_qubits == [0, 2]
    assert noise.affected_gates == []
