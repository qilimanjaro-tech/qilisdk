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

from qilisdk.noise.gaussian_pertubation import GaussianPerturbation
from qilisdk.noise.noise import Noise
from qilisdk.noise.noise_abc import NoiseABC
from qilisdk.noise.offset_pertubation import OffsetPerturbation
from qilisdk.noise.parameter_pertubation import ParameterPerturbation
from qilisdk.noise.protocols import AttachmentScope, HasAllowedScopes
from qilisdk.noise.readout_assignment import ReadoutAssignment


class DummyRng:
    def __init__(self, value: float) -> None:
        self._value = value
        self.called_with: tuple[float, float] | None = None

    def normal(self, mean: float, std: float) -> float:
        self.called_with = (mean, std)
        return self._value


def test_noise_allowed_scopes_and_protocols():
    scopes = Noise.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_QUBIT in scopes
    assert AttachmentScope.PER_GATE_TYPE not in scopes

    noise = Noise()
    assert isinstance(noise, NoiseABC)
    assert isinstance(noise, HasAllowedScopes)


def test_parameter_perturbation_allowed_scopes_and_abstract():
    scopes = ParameterPerturbation.allowed_scopes()
    assert AttachmentScope.GLOBAL in scopes
    assert AttachmentScope.PER_GATE_TYPE in scopes
    assert AttachmentScope.PER_QUBIT not in scopes

    with pytest.raises(TypeError):
        ParameterPerturbation()


def test_offset_perturbation_properties_and_perturb():
    perturbation = OffsetPerturbation(offset=0.5)

    assert perturbation.offset == 0.5
    assert perturbation.perturb(1.25) == pytest.approx(1.75)


def test_gaussian_perturbation_properties_and_perturb():
    perturbation = GaussianPerturbation(mean=1.0, std=0.5)
    rng = DummyRng(0.25)
    perturbation._rng = rng

    assert perturbation.mean == 1.0
    assert perturbation.std == 0.5
    assert perturbation.perturb(2.0) == pytest.approx(2.25)
    assert rng.called_with == (1.0, 0.5)

    with pytest.raises(ValueError):  # noqa: PT011
        GaussianPerturbation(std=-0.1)


def test_readout_assignment_properties_and_validation():
    assignment = ReadoutAssignment(p01=0.1, p10=0.2)

    assert assignment.p01 == 0.1
    assert assignment.p10 == 0.2

    with pytest.raises(ValueError):  # noqa: PT011
        ReadoutAssignment(p01=-0.1, p10=0.2)
    with pytest.raises(ValueError):  # noqa: PT011
        ReadoutAssignment(p01=0.1, p10=1.2)
