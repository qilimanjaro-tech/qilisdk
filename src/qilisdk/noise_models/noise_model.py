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
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from qilisdk.yaml import yaml


class NoiseType(Enum):
    DIGITAL = "Digital Noise"
    ANALOG = "Analog Noise"
    PARAMETER = "Parameter Noise"


@yaml.register_class
class NoiseBase(ABC):
    """Generic Noise Class"""

    @property
    @abstractmethod
    def noise_type(self) -> NoiseType: ...


class NoiseModel:
    def __init__(self, noise_passes: list[NoiseBase] | None = None) -> None:
        self._noise_passes: list[NoiseBase] = noise_passes or []

    @property
    def noise_passes(self) -> list[NoiseBase]:
        return self._noise_passes

    def noise_model_types(self) -> list[NoiseType]:
        return list({noise.noise_type for noise in self._noise_passes})

    def add(self, noise: NoiseBase) -> None:
        self._noise_passes.append(noise)
