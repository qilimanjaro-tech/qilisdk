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

from qilisdk.core.qtensor import QTensor
from qilisdk.noise_models.analog_noise import DissipationNoise
from qilisdk.noise_models.common_noise import ParameterNoise
from qilisdk.noise_models.digital_noise import (
    AmplitudeDampingNoise,
    BitFlipNoise,
    DephasingNoise,
    DepolarizingNoise,
    KrausNoise,
)
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


def test_bit_flip_noise_properties():
    noise = BitFlipNoise(qubit=0, probability=0.3, affected_gates=[])
    assert noise.noise_type == NoiseType.DIGITAL
    assert len(noise.kraus_operators) == 2
    assert noise.affected_qubits == [0]
    assert noise.affected_gates == []


def test_bit_flip_noise_invalid_probability():
    with pytest.raises(ValueError, match=r"The probability must be in the range \[0, 1\]."):
        BitFlipNoise(qubit=0, probability=1.5, affected_gates=[])


def test_bit_flip_noise_invalid_qubit():
    with pytest.raises(ValueError, match="Invalid qubit index: -1"):
        BitFlipNoise(qubit=-1, probability=0.2, affected_gates=[])


def test_invalid_kraus_operators():
    K0 = np.array([[1, 0], [0, 1]])
    K1 = np.array([[1, 0], [0, 1]])  # Invalid: does not satisfy completeness relation
    kraus_ops = [QTensor(K0), QTensor(K1)]
    with pytest.raises(ValueError, match=r"Kraus operators do not satisfy the completeness relation."):
        KrausNoise(kraus_operators=kraus_ops, affected_qubits=[0], affected_gates=[])


def test_dephasing_noise_properties():
    noise = DephasingNoise(qubit=1, probability=0.2, affected_gates=[])
    assert noise.noise_type == NoiseType.DIGITAL
    assert len(noise.kraus_operators) == 2
    assert noise.affected_qubits == [1]
    assert noise.affected_gates == []


def test_depolarizing_noise_properties():
    noise = DepolarizingNoise(qubit=0, probability=0.1, affected_gates=[])
    assert noise.noise_type == NoiseType.DIGITAL
    assert len(noise.kraus_operators) == 4
    assert noise.affected_qubits == [0]
    assert noise.affected_gates == []


def test_amplitude_damping_noise_properties():
    noise = AmplitudeDampingNoise(qubit=2, gamma=0.3, affected_gates=[])
    assert noise.noise_type == NoiseType.DIGITAL
    assert len(noise.kraus_operators) == 2
    assert noise.affected_qubits == [2]
    assert noise.affected_gates == []


def test_kraus_noise_properties():
    K0 = np.sqrt(0.7) * np.array([[1, 0], [0, 1]])
    K1 = np.sqrt(0.3) * np.array([[0, 1], [1, 0]])
    kraus_ops = [QTensor(K0), QTensor(K1)]
    noise = KrausNoise(kraus_operators=kraus_ops, affected_qubits=[0], affected_gates=[])
    assert noise.noise_type == NoiseType.DIGITAL
    assert noise.kraus_operators == kraus_ops
    assert noise.affected_qubits == [0]
    assert noise.affected_gates == []
