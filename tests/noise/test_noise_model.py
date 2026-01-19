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

import pytest

from qilisdk.digital.gates import X
from qilisdk.noise import BitFlip, Dephasing, NoiseModel, OffsetPerturbation, PauliChannel
from qilisdk.noise.parameter_pertubation import ParameterPerturbation
from qilisdk.noise.protocols import AttachmentScope


class GlobalOnlyPerturbation(ParameterPerturbation):
    def perturb(self, value): ...

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        return frozenset({AttachmentScope.GLOBAL})


class NoScopePerturbation(ParameterPerturbation):
    def perturb(self, value): ...

    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        return frozenset({})


class NoScopePauliChannel(PauliChannel):
    @classmethod
    def allowed_scopes(cls) -> frozenset[AttachmentScope]:
        return frozenset({})


def test_noise_model_add_global_noise():
    noise_model = NoiseModel()
    noise = Dephasing(t_phi=1.0)

    noise_model.add(noise)

    assert noise_model.global_noise == [noise]
    assert len(noise_model.per_qubit_noise) == 0
    assert len(noise_model.per_gate_noise) == 0


def test_noise_model_add_per_qubit_noise():
    noise_model = NoiseModel()
    noise = Dephasing(t_phi=1.0)

    noise_model.add(noise, qubits=[2])

    assert noise_model.global_noise == []
    assert noise_model.per_qubit_noise[2] == [noise]


def test_noise_model_add_per_gate_noise():
    noise_model = NoiseModel()
    noise = BitFlip(probability=0.1)

    noise_model.add(noise, gate=X)

    assert noise_model.per_gate_noise[X] == [noise]


def test_noise_model_add_per_gate_per_qubit_noise():
    noise_model = NoiseModel()
    noise = BitFlip(probability=0.1)

    noise_model.add(noise, gate=X, qubits=[1, 3])

    assert noise_model.per_gate_per_qubit_noise[X, 1] == [noise]
    assert noise_model.per_gate_per_qubit_noise[X, 3] == [noise]


def test_noise_model_add_global_perturbation():
    noise_model = NoiseModel()
    perturbation = OffsetPerturbation(offset=0.5)

    noise_model.add(perturbation, parameter="theta")

    assert noise_model.global_perturbations["theta"] == [perturbation]


def test_noise_model_add_per_gate_perturbation():
    noise_model = NoiseModel()
    perturbation = OffsetPerturbation(offset=0.5)

    noise_model.add(perturbation, gate=X, parameter="theta")

    assert noise_model.per_gate_perturbations[X, "theta"] == [perturbation]


def test_noise_model_rejects_invalid_arguments():
    noise_model = NoiseModel()
    noise = Dephasing(t_phi=1.0)

    with pytest.raises(ValueError, match="cannot be applied to parameters"):
        noise_model.add(noise, parameter="theta")

    with pytest.raises(ValueError, match="requires a parameter name"):
        noise_model.add(OffsetPerturbation(offset=0.1))

    with pytest.raises(ValueError, match="cannot be applied to specific qubits"):
        noise_model.add(OffsetPerturbation(offset=0.1), parameter="theta", qubits=[1])


def test_noise_model_scope_validation():
    noise_model = NoiseModel()

    with pytest.raises(ValueError, match="per_gate_type"):
        noise_model.add(Dephasing(t_phi=1.0), gate=X)

    with pytest.raises(ValueError, match="per_gate_type"):
        noise_model.add(GlobalOnlyPerturbation(), gate=X, parameter="theta")

    with pytest.raises(ValueError, match="global"):
        noise_model.add(NoScopePerturbation(), parameter="theta")

    with pytest.raises(ValueError, match="per_gate_type"):
        noise_model.add(NoScopePerturbation(), gate=X, parameter="theta")

    with pytest.raises(ValueError, match="global"):
        noise_model.add(NoScopePauliChannel())

    with pytest.raises(ValueError, match="per_qubit"):
        noise_model.add(NoScopePauliChannel(), qubits=[0])

    with pytest.raises(ValueError, match="per_gate_type"):
        noise_model.add(NoScopePauliChannel(), gate=X)

    with pytest.raises(ValueError, match="per_gate_type_per_qubit"):
        noise_model.add(NoScopePauliChannel(), gate=X, qubits=[0])
